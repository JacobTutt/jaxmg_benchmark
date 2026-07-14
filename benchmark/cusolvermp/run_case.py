"""Run one distributed JAXMg solver benchmark in a fresh process group.

This module is launched by ``srun`` with one Python process per GPU.  Input
construction and validation are outside the timed region.  The first solver
call records compilation plus execution; the next three calls reuse the same
compiled configuration and report warm execution time.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routine", choices=("potrs", "lu_solve"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64", "complex64", "complex128"), required=True)
    parser.add_argument("--grid", type=ProcessGrid.parse, required=True)
    parser.add_argument("--matrix-size", type=int, required=True)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Write a complete result atomically so interrupted cases are obvious."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _global_numpy(value: jax.Array) -> np.ndarray:
    """Materialize a small global result from a possibly multi-host array."""
    if not value.is_fully_addressable:
        return np.asarray(multihost_utils.process_allgather(value, tiled=True))
    return np.asarray(value)


def _global_max_seconds(elapsed: float) -> float:
    """Use the slowest rank as the distributed wall-clock duration."""
    gathered = multihost_utils.process_allgather(jnp.asarray(elapsed), tiled=False)
    return float(np.max(np.asarray(gathered)))


def _make_mesh(grid: ProcessGrid) -> Mesh:
    """Construct the explicit row-major JAX mesh expected by cuSOLVERMp."""
    devices = np.asarray(jax.devices("gpu"), dtype=object)
    if devices.size != grid.processes:
        raise RuntimeError(
            f"grid {grid} requires {grid.processes} GPUs, but JAX sees {devices.size}"
        )
    return Mesh(devices.reshape(grid.rows, grid.cols), ("pr", "pc"))


def _make_input_factory(
    *, matrix_size: int, dtype: jnp.dtype, mesh: Mesh
):
    """Compile deterministic diagonal A and vector B creation outside timing."""
    a_sharding = NamedSharding(mesh, P("pr", "pc"))
    b_sharding = NamedSharding(mesh, P("pr", None))

    def make_inputs():
        diagonal = jnp.arange(1, matrix_size + 1, dtype=dtype)
        a = jnp.diag(diagonal)
        b = jnp.ones((matrix_size, 1), dtype=dtype)
        return a, b

    return jax.jit(make_inputs, out_shardings=(a_sharding, b_sharding))


def _synchronize_inputs(a: jax.Array, b: jax.Array) -> None:
    for leaf in jax.tree_util.tree_leaves((a, b)):
        leaf.block_until_ready()


def _validate(
    *, out: jax.Array, status: jax.Array, status_size: int, dtype_name: str
) -> tuple[float, list[int]]:
    """Validate the diagonal solve and every rank's native status word."""
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


def _run(args: argparse.Namespace) -> dict[str, object]:
    """Execute the configured cold and warm solver calls."""
    if args.routine == "potrs":
        from jaxmg import potrs as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_POTRS_STATUS_SIZE as status_size,
        )
    else:
        from jaxmg import lu_solve as solver
        from jaxmg._cusolvermp_status import (
            _CUSOLVERMP_LU_SOLVE_STATUS_SIZE as status_size,
        )

    config = BenchmarkConfig.load(args.config)
    case = BenchmarkCase(
        args.routine, args.dtype, args.grid, args.matrix_size, args.tile_size
    )
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
        raise RuntimeError("JAXMg requires exactly one visible GPU per Python process")

    dtype = getattr(jnp, case.dtype)
    mesh = _make_mesh(case.grid)
    make_inputs = _make_input_factory(
        matrix_size=case.matrix_size, dtype=dtype, mesh=mesh
    )
    timings: list[float] = []
    maximum_error = 0.0
    rank_codes: list[int] = []
    total_runs = config.cold_runs + config.warm_runs
    for iteration in range(total_runs):
        a, b = make_inputs()
        _synchronize_inputs(a, b)
        multihost_utils.sync_global_devices(f"{case.case_id}_{iteration}_start")
        started = time.perf_counter()
        out, status = solver(
            a,
            b,
            T_A=case.tile_size,
            mesh=mesh,
            matrix_specs=P("pr", "pc"),
            return_status=True,
            pad=True,
        )
        out.block_until_ready()
        status.block_until_ready()
        multihost_utils.sync_global_devices(f"{case.case_id}_{iteration}_stop")
        timings.append(_global_max_seconds(time.perf_counter() - started))
        maximum_error, rank_codes = _validate(
            out=out, status=status, status_size=status_size, dtype_name=case.dtype
        )
        del a, b, out, status
        gc.collect()

    cold = timings[: config.cold_runs]
    warm = timings[config.cold_runs :]
    package_versions = {}
    for package in ("jax", "jaxlib", "jaxmg", "nvidia-cusolvermp-cu12"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
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
        "package_versions": package_versions,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def main() -> None:
    args = _arguments()
    # Slurm supplies rank and coordinator metadata, while local_device_ids
    # enforces the supported one-process-per-GPU execution model.
    local_id = int(os.environ.get("SLURM_LOCALID", "0"))
    jax.distributed.initialize(local_device_ids=[local_id])
    payload = _run(args)

    # Keep every worker alive until its peers have completed validation.  An
    # explicit jax.distributed.shutdown() adds a second internal barrier which
    # can time out when near-limit device work finishes unevenly across ranks.
    # Normal process teardown releases the distributed runtime after this
    # benchmark-level completion barrier.
    multihost_utils.sync_global_devices("jaxmg_benchmark_case_complete")
    if jax.process_index() == 0:
        _atomic_json(Path(args.output), payload)
        print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
