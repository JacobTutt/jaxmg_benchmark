"""Analytical case planning for the cuSOLVERMp benchmark suite.

The planner selects dimensions that need no padding for the distributed input
matrix and estimates allocations known before execution. cuSOLVERMp's opaque,
routine-specific workspace is deliberately reported as unknown.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, gcd, sqrt
from pathlib import Path
import tomllib
from typing import Iterable


DTYPE_BYTES = {
    "float32": 4,
    "float64": 8,
    "complex64": 8,
    "complex128": 16,
}


def _lcm(left: int, right: int) -> int:
    return left * right // gcd(left, right)


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
    tiles: tuple[int, ...]
    dtypes: tuple[str, ...]
    routines: tuple[str, ...]
    baseline_sizes: tuple[int, ...]
    frontier_fractions: tuple[float, ...]
    grids: tuple[ProcessGrid, ...]
    cold_runs: int
    warm_runs: int
    case_timeout_seconds: int

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkConfig":
        """Load and validate a benchmark configuration file."""
        with Path(path).open("rb") as stream:
            raw = tomllib.load(stream)
        hardware, sweep, execution = raw["hardware"], raw["sweep"], raw["execution"]
        config = cls(
            gpu_name=str(hardware["gpu_name"]),
            visible_memory_mib=int(hardware["visible_memory_mib"]),
            allocator_fraction=float(hardware["allocator_fraction"]),
            gpus_per_node=int(hardware["gpus_per_node"]),
            tiles=tuple(int(value) for value in sweep["tiles"]),
            dtypes=tuple(str(value) for value in sweep["dtypes"]),
            routines=tuple(str(value) for value in sweep["routines"]),
            baseline_sizes=tuple(int(value) for value in sweep["baseline_sizes"]),
            frontier_fractions=tuple(float(value) for value in sweep["frontier_fractions"]),
            grids=tuple(ProcessGrid.parse(value) for value in sweep["grids"]),
            cold_runs=int(execution["cold_runs"]),
            warm_runs=int(execution["warm_runs"]),
            case_timeout_seconds=int(execution["case_timeout_seconds"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject inconsistent configurations before submitting jobs."""
        if self.visible_memory_mib < 1 or self.gpus_per_node < 1:
            raise ValueError("hardware memory and GPU count must be positive")
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
        return sum((self.local_matrix_bytes, self.rhs_capacity_bytes,
                    self.redistribution_scratch_bytes, self.pivot_bytes))

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
        return (f"{self.routine}__{self.dtype}__g{self.grid}__"
                f"n{self.matrix_size}__t{self.tile_size}")

    @property
    def alignment_quantum(self) -> int:
        """Smallest N increment avoiding distributed matrix padding."""
        return self.tile_size * _lcm(self.grid.rows, self.grid.cols)

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
        memory = self.estimate_memory(config)
        return {
            "case_id": self.case_id, "routine": self.routine, "dtype": self.dtype,
            "process_rows": self.grid.rows, "process_cols": self.grid.cols,
            "processes": self.grid.processes, "matrix_size": self.matrix_size,
            "tile_size": self.tile_size, "alignment_quantum": self.alignment_quantum,
            "matrix_padding": False, "local_matrix_bytes": memory.local_matrix_bytes,
            "global_matrix_bytes": self.matrix_size**2 * DTYPE_BYTES[self.dtype],
            "local_matrix_budget_fraction": (
                memory.local_matrix_bytes / memory.allocator_budget_bytes
            ),
            "rhs_capacity_bytes": memory.rhs_capacity_bytes,
            "redistribution_scratch_bytes": memory.redistribution_scratch_bytes,
            "pivot_bytes": memory.pivot_bytes, "known_total_bytes": memory.known_total_bytes,
            "allocator_budget_bytes": memory.allocator_budget_bytes,
            "known_budget_fraction": memory.known_budget_fraction,
            "solver_workspace_bytes": None,
        }


def planned_sizes(config: BenchmarkConfig, *, dtype: str, grid: ProcessGrid,
                  tile_size: int) -> tuple[int, ...]:
    """Plan aligned baseline and raw-memory-frontier matrix sizes.

    Baseline points are retained only below the first 85% frontier point. This
    prevents the shared default ladder from duplicating or obscuring the
    deliberately denser high-utilisation sweep.
    """
    itemsize = DTYPE_BYTES[dtype]
    quantum = tile_size * _lcm(grid.rows, grid.cols)
    raw_limit = floor(sqrt(config.allocator_budget_per_gpu * grid.processes / itemsize))
    frontier = {
        floor(sqrt(fraction) * raw_limit) // quantum * quantum
        for fraction in config.frontier_fractions
    }
    first_frontier = min(size for size in frontier if size >= quantum)
    values = {
        ((size + quantum - 1) // quantum) * quantum
        for size in config.baseline_sizes
        if ((size + quantum - 1) // quantum) * quantum < first_frontier
        and ((size + quantum - 1) // quantum) * quantum <= raw_limit
    }
    values.update(frontier)
    return tuple(sorted(values))


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
