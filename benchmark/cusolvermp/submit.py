"""Create or submit the Isambard logical-suite job matrix.

Dry-run is the default. Pass ``--submit`` only after reviewing the printed
commands and case counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from benchmark.cusolvermp.model import BenchmarkConfig, ProcessGrid, planned_sizes


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/isambard_gh200.toml")
    parser.add_argument("--script", default="slurm/isambard_suite.sbatch")
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--routine", action="append", choices=("potrs", "lu_solve"))
    parser.add_argument("--dtype", action="append", choices=("float32", "float64", "complex64", "complex128"))
    parser.add_argument("--grid", action="append", type=ProcessGrid.parse)
    parser.add_argument("--submit", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    config_path = Path(args.config).resolve()
    script_path = Path(args.script).resolve()
    config = BenchmarkConfig.load(config_path)
    routines = tuple(args.routine or config.routines)
    dtypes = tuple(args.dtype or config.dtypes)
    grids = tuple(args.grid or config.grids)

    submitted = 0
    total_cases = 0
    for routine in routines:
        for dtype in dtypes:
            for grid in grids:
                if grid not in config.grids:
                    raise ValueError(f"grid {grid} is not configured")
                nodes = grid.processes // config.gpus_per_node
                case_count = sum(
                    len(planned_sizes(config, dtype=dtype, grid=grid, tile_size=tile))
                    for tile in config.tiles
                )
                total_cases += case_count
                job_name = f"jm_{routine[:2]}_{dtype.replace('complex', 'c').replace('float', 'f')}_{grid}"
                command = [
                    "sbatch", "--job-name", job_name, "--nodes", str(nodes),
                    "--ntasks", str(nodes), "--ntasks-per-node", "1",
                    "--cpus-per-task", "288", "--gres", "gpu:4", "--mem", "400G",
                    "--time", "24:00:00", "--output", "slurm-logs/%x-%j.out",
                    "--error", "slurm-logs/%x-%j.err", "--export",
                    (
                        "ALL,JAXMG_BENCHMARK_CONFIG=" + str(config_path)
                        + ",JAXMG_BENCHMARK_ROUTINE=" + routine
                        + ",JAXMG_BENCHMARK_DTYPE=" + dtype
                        + ",JAXMG_BENCHMARK_GRID=" + str(grid)
                        + ",JAXMG_BENCHMARK_OUTPUT_ROOT=" + str(Path(args.output_root).resolve())
                    ),
                    str(script_path),
                ]
                print(f"cases={case_count:3d} {' '.join(command)}")
                if args.submit:
                    Path("slurm-logs").mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, check=True)
                    submitted += 1
    print(
        f"logical_suites={len(routines) * len(dtypes) * len(grids)} "
        f"fresh_process_cases={total_cases} submitted={submitted}"
    )


if __name__ == "__main__":
    main()

