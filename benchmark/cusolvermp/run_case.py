"""Run one JAXMg benchmark case.

``run_suite.py`` starts this module once per case with ``srun``. Each Python
process owns one GPU. Input creation and validation are not timed. The first
call includes compilation; the remaining calls use the same compiled solver.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile
import time

# XLA reads allocator settings when the GPU client is initialized.  Set these
# defaults before importing JAX; an explicit launcher override remains valid.
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "vmm")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.99")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from benchmark.cusolvermp.model import BenchmarkCase, BenchmarkConfig, ProcessGrid


def _arguments() -> argparse.Namespace:
    """Read the command-line description of one fresh benchmark case.

    Returns:
        Parsed configuration path, solver, dtype, grid, matrix size, tile size,
        and destination JSON path.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routine", choices=("potrs", "lu_solve"), required=True)
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64", "complex64", "complex128"),
        required=True,
    )
    parser.add_argument("--grid", type=ProcessGrid.parse, required=True)
    parser.add_argument("--matrix-size", type=int, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write a complete JSON result without exposing a partial file.

    Args:
        path: Final result path written by rank zero.
        payload: JSON-serialisable benchmark result.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _global_numpy(value: jax.Array) -> np.ndarray:
    """Materialize a small distributed result as one NumPy array.

    Args:
        value: A JAX array that may be distributed across hosts.

    Returns:
        The complete global value on every host when gathering is required.
    """
    if not value.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value)


def _global_max_seconds(elapsed: float) -> float:
    """Return the largest elapsed time reported by any process.

    Args:
        elapsed: Local elapsed wall-clock seconds.

    Returns:
        The slowest rank's duration, used as the distributed solve time.
    """
    gathered = multihost_utils.process_allgather(jnp.asarray(elapsed), tiled=False)
    return float(np.max(np.asarray(gathered)))


def _make_mesh(grid: ProcessGrid) -> Mesh:
    """Construct the row-major JAX mesh used by cuSOLVERMp.

    Args:
        grid: Requested process-grid shape.

    Returns:
        A mesh with ``pr`` and ``pc`` axes.

    Raises:
        RuntimeError: If the visible GPU count does not match the grid.
    """
    devices = np.asarray(jax.devices("gpu"), dtype=object)
    if devices.size != grid.processes:
        raise RuntimeError(
            f"grid {grid} requires {grid.processes} GPUs, but JAX sees {devices.size}"
        )
    return Mesh(devices.reshape(grid.rows, grid.cols), ("pr", "pc"))


def _make_input_factory(
    *, matrix_size: int, dtype: jnp.dtype, mesh: Mesh
):
    """Build a compiled factory for the diagonal system used by a case.

    Args:
        matrix_size: Global square matrix dimension.
        dtype: JAX dtype for both matrix and right-hand side.
        mesh: Mesh determining the output sharding.

    Returns:
        A zero-argument jitted function returning sharded ``(A, b)`` arrays.
    """
    a_sharding = NamedSharding(mesh, P("pr", "pc"))
    b_sharding = NamedSharding(mesh, P("pr", None))

    def make_inputs():
        """Return ``diag(1, ..., N)`` and an all-ones single-column RHS."""
        diagonal = jnp.arange(1, matrix_size + 1, dtype=dtype)
        a = jnp.diag(diagonal)
        b = jnp.ones((matrix_size, 1), dtype=dtype)
        return a, b

    return jax.jit(make_inputs, out_shardings=(a_sharding, b_sharding))


def _wait_for_inputs(a: jax.Array, b: jax.Array) -> None:
    """Wait until both input arrays are ready before starting the timer.

    Args:
        a: Sharded coefficient matrix.
        b: Sharded or replicated right-hand side.
    """
    for leaf in jax.tree_util.tree_leaves((a, b)):
        leaf.block_until_ready()


def _validate(
    *, out: jax.Array, status: jax.Array, status_size: int, dtype_name: str
) -> tuple[float, list[int]]:
    """Validate the diagonal solution and native status from every rank.

    Args:
        out: Returned solution vector or matrix.
        status: Concatenated native status values.
        status_size: Number of status values emitted by one rank.
        dtype_name: Benchmark dtype, used to select numerical tolerance.

    Returns:
        The largest absolute solution error and one native return code per rank.

    Raises:
        AssertionError: If the solution, status shape, or native status fails.
    """
    solution = _global_numpy(out).reshape(-1)
    expected = 1.0 / np.arange(1, solution.size + 1, dtype=np.float64)
    maximum_error = float(np.max(np.abs(solution - expected)))
    tolerance = 5e-4 if dtype_name in ("float32", "complex64") else 1e-10
    if not np.allclose(solution, expected, rtol=tolerance, atol=tolerance):
        raise AssertionError(f"solution validation failed: max_abs_error={maximum_error}")

    words = _global_numpy(status).reshape(-1)
    if words.size % status_size:
        raise AssertionError(f"unexpected native status size {words.size}")
    rank_codes = [int(value) for value in words[::status_size]]
    if any(rank_codes):
        raise AssertionError(f"non-zero native status codes: {rank_codes}")
    return maximum_error, rank_codes


def _solver_for(routine: str):
    """Load the requested public solver and its per-rank status layout.

    Args:
        routine: ``"potrs"`` or ``"lu_solve"``.

    Returns:
        The public JAXMg solver and its native status-vector length.
    """
    if routine == "potrs":
        from jaxmg import potrs as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_POTRS_STATUS_SIZE as status_size,
        )
    else:
        from jaxmg import lu_solve as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_LU_SOLVE_STATUS_SIZE as status_size,
        )
    return solver, status_size


def _check_case(case: BenchmarkCase) -> None:
    """Check that a case is valid for one-process-per-GPU execution.

    Args:
        case: Requested benchmark case.

    Raises:
        ValueError: If the matrix dimension requires padding.
        RuntimeError: If JAX's processes or visible GPUs do not match the grid.
    """
    if case.needs_matrix_padding:
        raise ValueError(
            f"{case.case_id} needs matrix padding; N must be divisible by "
            f"{case.alignment_quantum}"
        )
    if jax.process_count() != case.grid.processes:
        raise RuntimeError(
            f"expected {case.grid.processes} processes, got {jax.process_count()}"
        )
    if len(jax.local_devices(backend="gpu")) != 1:
        raise RuntimeError("JAXMg requires one visible GPU per Python process")


def _phase(iteration: int, cold_runs: int, warm_runs: int) -> tuple[str, int, int]:
    """Describe the cold or warm phase for a zero-based iteration.

    Args:
        iteration: Zero-based call index.
        cold_runs: Number of cold calls, currently one.
        warm_runs: Number of warm calls after compilation.

    Returns:
        Phase name, one-based phase index, and total calls in that phase.
    """
    if iteration < cold_runs:
        return "cold", iteration + 1, cold_runs
    return "warm", iteration - cold_runs + 1, warm_runs


def _solve_once(
    *,
    solver,
    a: jax.Array,
    b: jax.Array,
    tile_size: int,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, float]:
    """Run one solver call and measure the slowest participating rank.

    Args:
        solver: Public JAXMg POTRS or LU-solve callable.
        a: Sharded coefficient matrix.
        b: Right-hand side matching ``a``.
        tile_size: cuSOLVERMp tile width.
        mesh: Distributed process mesh.

    Returns:
        The solution, native status array, and maximum elapsed seconds.
    """
    started = time.perf_counter()
    out, status = solver(
        a,
        b,
        T_A=tile_size,
        mesh=mesh,
        matrix_specs=P("pr", "pc"),
        return_status=True,
        pad=True,
    )
    out.block_until_ready()
    status.block_until_ready()
    return out, status, _global_max_seconds(time.perf_counter() - started)


def _package_versions() -> dict[str, str | None]:
    """Return installed package versions recorded with each result.

    Returns:
        Package-name to version mapping. Optional CUDA packages are recorded as
        ``None`` when their metadata is unavailable.
    """
    versions: dict[str, str | None] = {}
    for package in ("jax", "jaxlib", "jaxmg", "nvidia-cusolvermp-cu12"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Run the configured case and return its rank-zero JSON payload.

    Args:
        args: Parsed case description from ``_arguments``.

    Returns:
        Complete case metadata, timings, validation results, and package data.
    """
    solver, status_size = _solver_for(args.routine)

    config = BenchmarkConfig.load(args.config)
    case = BenchmarkCase(
        args.routine, args.dtype, args.grid, args.matrix_size, args.tile_size
    )
    _check_case(case)

    dtype = getattr(jnp, case.dtype)
    mesh = _make_mesh(case.grid)
    make_inputs = _make_input_factory(
        matrix_size=case.matrix_size, dtype=dtype, mesh=mesh
    )
    timings: list[float] = []
    maximum_error = 0.0
    rank_codes: list[int] = []
    for iteration in range(config.cold_runs + config.warm_runs):
        phase, phase_iteration, phase_total = _phase(
            iteration, config.cold_runs, config.warm_runs
        )
        if jax.process_index() == 0:
            print(
                f"{case.case_id}: {phase} {phase_iteration}/{phase_total} start",
                flush=True,
            )
        a, b = make_inputs()
        _wait_for_inputs(a, b)
        multihost_utils.sync_global_devices(f"{case.case_id}_{iteration}_start")
        out, status, elapsed = _solve_once(
            solver=solver,
            a=a,
            b=b,
            tile_size=case.tile_size,
            mesh=mesh,
        )
        multihost_utils.sync_global_devices(f"{case.case_id}_{iteration}_stop")
        timings.append(elapsed)
        maximum_error, rank_codes = _validate(
            out=out, status=status, status_size=status_size, dtype_name=case.dtype
        )
        if jax.process_index() == 0:
            print(
                f"{case.case_id}: {phase} {phase_iteration}/{phase_total} "
                f"complete ({timings[-1]:.3f} s)",
                flush=True,
            )
        del a, b, out, status
        gc.collect()

    cold = timings[: config.cold_runs]
    warm = timings[config.cold_runs :]
    return {
        **case.to_dict(config),
        "status": "passed",
        "gpu_name": config.gpu_name,
        "visible_memory_mib": config.visible_memory_mib,
        "allocator": os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"],
        "allocator_fraction": float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]),
        "cold_seconds": cold,
        "warm_seconds": warm,
        "warm_median_seconds": float(np.median(warm)),
        "warm_min_seconds": float(np.min(warm)),
        "warm_max_seconds": float(np.max(warm)),
        "max_abs_error": maximum_error,
        "native_status_codes": rank_codes,
        "jax_version": jax.__version__,
        "package_versions": _package_versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def main() -> None:
    """Initialize distributed JAX, run the case, and write one result on rank zero."""
    args = _arguments()
    # Slurm supplies rank and coordinator metadata. This makes the one GPU
    # assigned to each process explicit to JAX.
    local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(local_device_ids=[local_id])
    payload = _run(args)

    # Finish validation on every rank before rank zero writes the result.
    # Normal process exit releases the distributed runtime.
    multihost_utils.sync_global_devices("jaxmg_benchmark_case_complete")
    if jax.process_index() == 0:
        _atomic_json(Path(args.output), payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
