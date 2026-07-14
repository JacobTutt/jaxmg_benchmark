"""Run one logical benchmark suite as isolated ``srun`` case steps."""

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
    parser.add_argument("--retry-failures", action="store_true")
    return parser.parse_args()


def _existing_pass(path: Path) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "passed"
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def _write_failure(path: Path, case: BenchmarkCase, returncode: int | str,
                   elapsed: float) -> None:
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

    selected_tiles = tuple(args.tile_size or config.tiles)
    if set(selected_tiles) - set(config.tiles):
        raise ValueError("every selected tile size must be configured")
    planned_by_tile = {
        tile: planned_sizes(config, dtype=args.dtype, grid=args.grid, tile_size=tile)
        for tile in selected_tiles
    }
    selected_sizes = set(args.matrix_size or ())
    if selected_sizes:
        # An explicit matrix-size request is allowed outside the default ladder
        # when it is a valid no-padding case for the selected tile size. This
        # supports targeted memory-limit checks without mutating the standard
        # reproducible sweep configuration.
        planned_by_tile = {
            tile: tuple(
                sorted(
                    size
                    for size in selected_sizes
                    if not BenchmarkCase(
                        args.routine, args.dtype, args.grid, size, tile
                    ).needs_matrix_padding
                )
            )
            for tile in selected_tiles
        }
        planned_by_tile = {
            tile: sizes for tile, sizes in planned_by_tile.items() if sizes
        }
        if not planned_by_tile:
            raise ValueError(
                "no selected matrix size is tile-aligned for the selected grid"
            )
    cases = [
        BenchmarkCase(args.routine, args.dtype, args.grid, size, tile)
        for tile, sizes in planned_by_tile.items()
        for size in sizes
    ]
    print(f"suite={suite} cases={len(cases)}", flush=True)
    for index, case in enumerate(cases, start=1):
        output = result_dir / f"{case.case_id}.json"
        if _existing_pass(output):
            print(f"[{index}/{len(cases)}] skip passed {case.case_id}", flush=True)
            continue
        if output.exists() and not args.retry_failures:
            print(f"[{index}/{len(cases)}] skip recorded failure {case.case_id}", flush=True)
            continue
        command = [
            "srun", "--nodes", str(case.grid.processes // config.gpus_per_node),
            "--ntasks", str(case.grid.processes), "--ntasks-per-node", str(config.gpus_per_node),
            # Distributed CUDA/NCCL work needs host progress threads.  Assign
            # the per-GPU CPU share reserved by the outer Slurm allocation.
            "--cpus-per-task", str(config.cpus_per_gpu), sys.executable, "-u", "-m",
            "benchmark.cusolvermp.run_case", "--config", str(Path(args.config).resolve()),
            "--routine", case.routine, "--dtype", case.dtype, "--grid", str(case.grid),
            "--matrix-size", str(case.matrix_size), "--tile-size", str(case.tile_size),
            "--output", str(output.resolve()),
        ]
        print(f"[{index}/{len(cases)}] {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        started = time.perf_counter()
        log_path = log_dir / f"{case.case_id}.log"
        try:
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                    timeout=config.case_timeout_seconds,
                    check=False,
                )
            if completed.returncode != 0 and not _existing_pass(output):
                _write_failure(output, case, completed.returncode,
                               time.perf_counter() - started)
        except subprocess.TimeoutExpired:
            _write_failure(output, case, "timeout", time.perf_counter() - started)


if __name__ == "__main__":
    main()
