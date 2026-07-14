"""CPU-only tests for benchmark planning and memory accounting."""

from pathlib import Path
import unittest

from benchmark.cusolvermp.model import (
    BenchmarkCase,
    BenchmarkConfig,
    ProcessGrid,
    planned_sizes,
)


CONFIG = Path(__file__).parents[1] / "configs" / "isambard_gh200.toml"


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = BenchmarkConfig.load(CONFIG)

    def test_isambard_allocator_budget(self):
        self.assertEqual(self.config.visible_memory_mib, 97871)
        self.assertEqual(self.config.cpus_per_gpu, 72)
        self.assertAlmostEqual(
            self.config.allocator_budget_per_gpu / 1024**3,
            94.6216,
            places=3,
        )

    def test_alignment_quantum_uses_grid_lcm(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(6, 2), 24576, 1024)
        self.assertEqual(case.alignment_quantum, 6144)
        self.assertFalse(case.needs_matrix_padding)

    def test_misaligned_case_is_rejected_by_memory_model(self):
        case = BenchmarkCase("potrs", "float32", ProcessGrid(4, 1), 5000, 1024)
        self.assertTrue(case.needs_matrix_padding)
        with self.assertRaises(ValueError):
            case.estimate_memory(self.config)

    def test_lu_adds_int64_pivots(self):
        potrs = BenchmarkCase("potrs", "float64", ProcessGrid(4, 1), 16384, 1024)
        lu = BenchmarkCase("lu_solve", "float64", ProcessGrid(4, 1), 16384, 1024)
        self.assertEqual(potrs.estimate_memory(self.config).pivot_bytes, 0)
        self.assertEqual(lu.estimate_memory(self.config).pivot_bytes, 16384 * 8)

    def test_all_planned_sizes_need_no_matrix_padding(self):
        for grid in self.config.grids:
            for dtype in self.config.dtypes:
                for tile in self.config.tiles:
                    for size in planned_sizes(
                        self.config, dtype=dtype, grid=grid, tile_size=tile
                    ):
                        case = BenchmarkCase("potrs", dtype, grid, size, tile)
                        self.assertFalse(case.needs_matrix_padding, case.case_id)

    def test_large_grid_has_baseline_bridges_before_frontier(self):
        """Shared baselines fill the multi-node gap without entering 85% sweep."""
        sizes = planned_sizes(
            self.config,
            dtype="float32",
            grid=ProcessGrid(4, 4),
            tile_size=256,
        )
        self.assertTrue({327680, 393216, 458752, 524288}.issubset(sizes))
        self.assertEqual(sizes[-8], 586752)
        self.assertNotIn(589824, sizes)


if __name__ == "__main__":
    unittest.main()
