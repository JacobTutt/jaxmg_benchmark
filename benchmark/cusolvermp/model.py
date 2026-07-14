"""Describe the benchmark cases before they are submitted.

The benchmark only uses matrix sizes that fit the selected process grid and
tile size without padding. This module also records the memory that JAXMg can
calculate in advance. cuSOLVERMp allocates its own routine-specific workspace,
so that part is intentionally left out of the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, lcm, sqrt
from pathlib import Path
import tomllib
from typing import Iterable


DTYPE_BYTES = {
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}
@dataclass(frozen=True, order=True)
class ProcessGrid:
    """A row-major cuSOLVERMp process grid."""

    rows: int
    cols: int

    @classmethod
    def parse(cls, value: str) -> "ProcessGrid":
        """Parse a grid written as ``ROWSxCOLS``."""
        try:
            rows, cols = (int(part) for part in value.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid process grid {value!r}; expected ROWSxCOLS") from exc
        if rows < 1 or cols < 1:
            raise ValueError("process-grid dimensions must be positive")
        return cls(rows, cols)

    @property
    def processes(self) -> int:
        """Return the number of participating GPU processes."""
        return self.rows * self.cols

    def __str__(self) -> str:
        return f"{self.rows}x{self.cols}"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Static hardware and sweep configuration loaded from TOML."""

    gpu_name: str
    visible_memory_mib: int
    allocator_fraction: float
    gpus_per_node: int
    cpus_per_gpu: int
    tiles: tuple[int, ...]
    dtypes: tuple[str, ...]
    routines: tuple[str, ...]
    baseline_sizes: tuple[int, ...]
    frontier_fractions: tuple[float, ...]
    grids: tuple[ProcessGrid, ...]
    cold_runs: int
    warm_runs: int
    case_timeout_seconds: int
    suite_walltime: str
    suite_memory: str

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load and validate a benchmark configuration file."""
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        hardware, sweep = raw["hardware"], raw["sweep"]
        execution, slurm = raw["execution"], raw["slurm"]
        config = cls(
            gpu_name=str(hardware["gpu_name"]),
            visible_memory_mib=int(hardware["visible_memory_mib"]),
            allocator_fraction=float(hardware["allocator_fraction"]),
            gpus_per_node=int(hardware["gpus_per_node"]),
            cpus_per_gpu=int(hardware["cpus_per_gpu"]),
            tiles=tuple(int(value) for value in sweep["tiles"]),
            dtypes=tuple(str(value) for value in sweep["dtypes"]),
            routines=tuple(str(value) for value in sweep["routines"]),
            baseline_sizes=tuple(int(value) for value in sweep["baseline_sizes"]),
            frontier_fractions=tuple(float(value) for value in sweep["frontier_fractions"]),
            grids=tuple(ProcessGrid.parse(value) for value in sweep["grids"]),
            cold_runs=int(execution["cold_runs"]),
            warm_runs=int(execution["warm_runs"]),
            case_timeout_seconds=int(execution["case_timeout_seconds"]),
            suite_walltime=str(slurm["suite_walltime"]),
            suite_memory=str(slurm["suite_memory"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject inconsistent configurations before submitting jobs."""
        if (
            self.visible_memory_mib < 1
            or self.gpus_per_node < 1
            or self.cpus_per_gpu < 1
        ):
            raise ValueError("hardware memory, GPU count, and CPUs per GPU must be positive")
        if not 0 < self.allocator_fraction <= 1:
            raise ValueError("allocator_fraction must be in (0, 1]")
        if set(self.dtypes) - set(DTYPE_BYTES):
            raise ValueError(f"unsupported dtypes: {set(self.dtypes) - set(DTYPE_BYTES)}")
        if set(self.routines) - {"potrs", "lu_solve"}:
            raise ValueError("routines must be potrs and/or lu_solve")
        if any(tile < 1 for tile in self.tiles):
            raise ValueError("tile sizes must be positive")
        if self.cold_runs != 1 or self.warm_runs < 1:
            raise ValueError("the suite requires one cold run and at least one warm run")
        if not self.suite_walltime or not self.suite_memory:
            raise ValueError("Slurm wall time and memory must be configured")
        for grid in self.grids:
            if grid.processes % self.gpus_per_node:
                raise ValueError(f"grid {grid} does not occupy whole nodes")

    @property
    def visible_bytes_per_gpu(self) -> int:
        """Return scheduler-visible device memory in bytes."""
        return self.visible_memory_mib * 1024**2

    @property
    def allocator_budget_per_gpu(self) -> int:
        """Return the configured VMM allocator budget per GPU."""
        return floor(self.visible_bytes_per_gpu * self.allocator_fraction)


@dataclass(frozen=True)
class MemoryEstimate:
    """Known per-GPU allocations for one benchmark case."""

    local_matrix_bytes: int
    rhs_capacity_bytes: int
    redistribution_scratch_bytes: int
    pivot_bytes: int
    allocator_budget_bytes: int

    @property
    def known_total_bytes(self) -> int:
        return sum(
            (
                self.local_matrix_bytes,
                self.rhs_capacity_bytes,
                self.redistribution_scratch_bytes,
                self.pivot_bytes,
            )
        )

    @property
    def known_budget_fraction(self) -> float:
        return self.known_total_bytes / self.allocator_budget_bytes


@dataclass(frozen=True)
class BenchmarkCase:
    """One fresh-process benchmark case."""

    routine: str
    dtype: str
    grid: ProcessGrid
    matrix_size: int
    tile_size: int

    @property
    def case_id(self) -> str:
        return (
            f"{self.routine}__{self.dtype}__g{self.grid}__"
            f"n{self.matrix_size}__t{self.tile_size}"
        )

    @property
    def alignment_quantum(self) -> int:
        """Smallest N increment avoiding distributed matrix padding."""
        return self.tile_size * lcm(self.grid.rows, self.grid.cols)

    @property
    def needs_matrix_padding(self) -> bool:
        return self.matrix_size % self.alignment_quantum != 0

    def estimate_memory(self, config: BenchmarkConfig) -> MemoryEstimate:
        """Estimate known per-rank matrix, RHS, scratch, and pivot storage."""
        if self.needs_matrix_padding:
            raise ValueError(f"case {self.case_id} requires matrix padding")
        itemsize = DTYPE_BYTES[self.dtype]
        local_rows = self.matrix_size // self.grid.rows
        local_cols = self.matrix_size // self.grid.cols
        # The native redistribution uses three equally sized tile buffers:
        # receive, send, and saved local data. It is sized for whichever
        # process-grid direction has the longer local slab.
        return MemoryEstimate(
            local_matrix_bytes=local_rows * local_cols * itemsize,
            rhs_capacity_bytes=local_rows * self.tile_size * itemsize,
            redistribution_scratch_bytes=(
                3 * self.tile_size * max(local_rows, local_cols) * itemsize
            ),
            pivot_bytes=local_cols * 8 if self.routine == "lu_solve" else 0,
            allocator_budget_bytes=config.allocator_budget_per_gpu,
        )

    def to_dict(self, config: BenchmarkConfig) -> dict[str, object]:
        """Return a self-contained record of the case and its memory estimate."""
        memory = self.estimate_memory(config)
        return {
            "case_id": self.case_id,
            "routine": self.routine,
            "dtype": self.dtype,
            "process_rows": self.grid.rows,
            "process_cols": self.grid.cols,
            "processes": self.grid.processes,
            "matrix_size": self.matrix_size,
            "tile_size": self.tile_size,
            "alignment_quantum": self.alignment_quantum,
            "matrix_padding": False,
            "local_matrix_bytes": memory.local_matrix_bytes,
            "global_matrix_bytes": self.matrix_size**2 * DTYPE_BYTES[self.dtype],
            "local_matrix_budget_fraction": (
                memory.local_matrix_bytes / memory.allocator_budget_bytes
            ),
            "rhs_capacity_bytes": memory.rhs_capacity_bytes,
            "redistribution_scratch_bytes": memory.redistribution_scratch_bytes,
            "pivot_bytes": memory.pivot_bytes,
            "known_total_bytes": memory.known_total_bytes,
            "allocator_budget_bytes": memory.allocator_budget_bytes,
            "known_budget_fraction": memory.known_budget_fraction,
            "solver_workspace_bytes": None,
        }


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round a baseline dimension upwards to a valid no-padding dimension."""
    return (value + multiple - 1) // multiple * multiple


def _near_limit_sizes(
    config: BenchmarkConfig,
    *,
    dtype: str,
    grid: ProcessGrid,
    tile_size: int,
) -> set[int]:
    """Return aligned dimensions near the matrix-only allocator limit.

    This is a guide for the sweep, not a promise that a case fits. The native
    cuSOLVERMp workspace is routine-specific and cannot be known in advance.
    """
    matrix_limit = floor(
        sqrt(config.allocator_budget_per_gpu * grid.processes / DTYPE_BYTES[dtype])
    )
    quantum = tile_size * lcm(grid.rows, grid.cols)
    return {
        floor(sqrt(fraction) * matrix_limit) // quantum * quantum
        for fraction in config.frontier_fractions
    }


def planned_sizes(
    config: BenchmarkConfig,
    *,
    dtype: str,
    grid: ProcessGrid,
    tile_size: int,
) -> tuple[int, ...]:
    """Return small-to-large aligned dimensions for one configuration.

    The shared baseline covers small and medium problems. Near the expected
    memory limit, the configured fractions give a more detailed sweep.
    """
    quantum = tile_size * lcm(grid.rows, grid.cols)
    near_limit = {
        size
        for size in _near_limit_sizes(
            config, dtype=dtype, grid=grid, tile_size=tile_size
        )
        if size >= quantum
    }
    if not near_limit:
        return ()

    first_near_limit = min(near_limit)
    baseline = {
        aligned
        for size in config.baseline_sizes
        if (aligned := _round_up_to_multiple(size, quantum)) < first_near_limit
    }
    return tuple(sorted(baseline | near_limit))


def iter_cases(config: BenchmarkConfig, *, routines: Iterable[str] | None = None,
               dtypes: Iterable[str] | None = None,
               grids: Iterable[ProcessGrid] | None = None) -> Iterable[BenchmarkCase]:
    """Yield configured cases in deterministic order."""
    for routine in tuple(routines or config.routines):
        for dtype in tuple(dtypes or config.dtypes):
            for grid in tuple(grids or config.grids):
                for tile_size in config.tiles:
                    for matrix_size in planned_sizes(
                        config, dtype=dtype, grid=grid, tile_size=tile_size
                    ):
                        yield BenchmarkCase(routine, dtype, grid, matrix_size, tile_size)
