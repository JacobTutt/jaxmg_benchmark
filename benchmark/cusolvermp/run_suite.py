"""Run every selected case for one solver, dtype, and process grid.

Each case receives its own ``srun`` process group. This keeps an out-of-memory
failure or an NCCL error from affecting the next matrix size.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmark.cusolvermp.model import (
    BenchmarkCase,
    BenchmarkConfig,
    ProcessGrid,
    planned_sizes,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--routine", choices=("potrs", "lu_solve"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64", "complex64", "complex128"), required=True)
    parser.add_argument("--grid", type=ProcessGrid.parse, required=True)
    parser.add_argument("--tile-size", type=int, action="append")
    parser.add_argument("--matrix-size", type=int, action="append")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="rerun cases with an existing failed result file",
    )
    return parser.parse_args()


def _existing_pass(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _write_failure(path: Path, case: BenchmarkCase, returncode: int | str,
                   elapsed: float) -> None:
    """Record a failed case in the same place as successful JSON records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "case_id": case.case_id,
                "routine": case.routine,
                "dtype": case.dtype,
                "process_rows": case.grid.rows,
                "process_cols": case.grid.cols,
                "matrix_size": case.matrix_size,
                "tile_size": case.tile_size,
                "status": "failed",
                "returncode": returncode,
                "elapsed_seconds": elapsed,
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )


def _selected_tiles(args: argparse.Namespace, config: BenchmarkConfig) -> tuple[int, ...]:
    """Return requested tile sizes after checking the configuration."""
    tiles = tuple(args.tile_size or config.tiles)
    if set(tiles) - set(config.tiles):
        raise ValueError("every selected tile size must be configured")
    return tiles


def _selected_sizes(
    args: argparse.Namespace,
    config: BenchmarkConfig,
    *,
    tile_size: int,
) -> tuple[int, ...]:
    """Return either the standard size sweep or explicit aligned dimensions."""
    if not args.matrix_size:
        return planned_sizes(
            config,
            dtype=args.dtype,
            grid=args.grid,
            tile_size=tile_size,
        )

    sizes = tuple(
        sorted(
            size
            for size in set(args.matrix_size)
            if not BenchmarkCase(
                args.routine, args.dtype, args.grid, size, tile_size
            ).needs_matrix_padding
        )
    )
    return sizes


def _selected_cases(
    args: argparse.Namespace, config: BenchmarkConfig
) -> list[BenchmarkCase]:
    """Build the ordered case list printed before the suite starts."""
    cases = []
    for tile_size in _selected_tiles(args, config):
        sizes = _selected_sizes(args, config, tile_size=tile_size)
        cases.extend(
            BenchmarkCase(args.routine, args.dtype, args.grid, size, tile_size)
            for size in sizes
        )
    if not cases:
        raise ValueError("no requested matrix sizes fit the selected grid and tiles")
    return cases


def _srun_command(
    *,
    case: BenchmarkCase,
    config: BenchmarkConfig,
    config_path: Path,
    output: Path,
) -> list[str]:
    """Build the ``srun`` command for a single fresh benchmark case."""
    nodes = case.grid.processes // config.gpus_per_node
    return [
        "srun",
        "--nodes",
        str(nodes),
        "--ntasks",
        str(case.grid.processes),
        "--ntasks-per-node",
        str(config.gpus_per_node),
        # Keep the host cores assigned by the outer Slurm allocation. NCCL and
        # the JAX runtime both need CPU progress while CUDA work is in flight.
        "--cpus-per-task",
        str(config.cpus_per_gpu),
        sys.executable,
        "-u",
        "-m",
        "benchmark.cusolvermp.run_case",
        "--config",
        str(config_path),
        "--routine",
        case.routine,
        "--dtype",
        case.dtype,
        "--grid",
        str(case.grid),
        "--matrix-size",
        str(case.matrix_size),
        "--tile-size",
        str(case.tile_size),
        "--output",
        str(output),
    ]


def _run_case(
    *,
    command: list[str],
    output: Path,
    log_path: Path,
    case: BenchmarkCase,
    timeout_seconds: int,
) -> None:
    """Run one case, preserving its combined log and a failure JSON if needed."""
    started = time.perf_counter()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                timeout=timeout_seconds,
                check=False,
            )
        if completed.returncode != 0 and not _existing_pass(output):
            _write_failure(output, case, completed.returncode, time.perf_counter() - started)
    except subprocess.TimeoutExpired:
        _write_failure(output, case, "timeout", time.perf_counter() - started)


def main() -> None:
    args = _arguments()
    config = BenchmarkConfig.load(args.config)
    if args.routine not in config.routines or args.dtype not in config.dtypes:
        raise ValueError("routine or dtype is not enabled by the configuration")
    if args.grid not in config.grids:
        raise ValueError(f"grid {args.grid} is not enabled by the configuration")

    suite = f"{args.routine}/{args.dtype}/{args.grid}"
    root = Path(args.output_root)
    result_dir = root / "cases" / args.routine / args.dtype / str(args.grid)
    log_dir = root / "logs" / args.routine / args.dtype / str(args.grid)
    result_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cases = _selected_cases(args, config)
    print(f"suite={suite} cases={len(cases)}", flush=True)
    for index, case in enumerate(cases, start=1):
        output = result_dir / f"{case.case_id}.json"
        if _existing_pass(output):
            print(f"[{index}/{len(cases)}] skip passed {case.case_id}", flush=True)
            continue
        if output.exists() and not args.retry_failures:
            print(f"[{index}/{len(cases)}] skip recorded failure {case.case_id}", flush=True)
            continue
        command = _srun_command(
            case=case,
            config=config,
            config_path=Path(args.config).resolve(),
            output=output.resolve(),
        )
        print(f"[{index}/{len(cases)}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        log_path = log_dir / f"{case.case_id}.log"
        _run_case(
            command=command,
            output=output,
            log_path=log_path,
            case=case,
            timeout_seconds=config.case_timeout_seconds,
        )


if __name__ == "__main__":
    main()
