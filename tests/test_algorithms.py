"""Independent small-instance and invariant checks for the study algorithms."""

from itertools import combinations
import unittest

import numpy as np

from kmeans_study.algorithms import exact_kmeans_1d, run_kmeans


def enumerated_optimum(points, k):
    """Exhaustively enumerate contiguous partitions; no DP recurrence reused."""
    values = np.sort(np.asarray(points, dtype=float))
    best_cost = np.inf
    for cuts in combinations(range(1, len(values)), k - 1):
        boundaries = (0,) + cuts + (len(values),)
        total = 0.0
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment = values[start:end]
            total += float(np.sum((segment - np.mean(segment)) ** 2))
        best_cost = min(best_cost, total)
    return best_cost


class ExactKmeansTests(unittest.TestCase):
    def test_matches_exhaustive_partitions(self):
        rng = np.random.default_rng(309)
        for size in range(2, 9):
            for _ in range(4):
                points = rng.integers(-8, 9, size=size).astype(float)
                rng.shuffle(points)
                for k in range(1, size + 1):
                    with self.subTest(points=points, k=k):
                        result = exact_kmeans_1d(points, k)
                        self.assertAlmostEqual(result.cost, enumerated_optimum(points, k), places=10)
                        self.assertAlmostEqual(result.cost, float(np.sum((points - result.centers[result.labels]) ** 2)))

    def test_known_optimum_and_input_order(self):
        points = np.array([11.0, 0.0, 10.0, 1.0])
        result = exact_kmeans_1d(points, 2)
        self.assertEqual(result.cost, 1.0)
        np.testing.assert_allclose(result.centers, [0.5, 10.5])
        np.testing.assert_array_equal(result.labels, [1, 0, 1, 0])

    def test_translation_and_narrow_distant_clusters(self):
        points = np.array([0.0, 0.125, 1e8, 1e8 + 0.125])
        base = exact_kmeans_1d(points, 2)
        shifted = exact_kmeans_1d(points + 1e12, 2)
        self.assertAlmostEqual(base.cost, 0.015625)
        self.assertAlmostEqual(shifted.cost, base.cost)

    def test_duplicate_locations_and_zero_optimum(self):
        for k in [1, 2, 5]:
            result = exact_kmeans_1d(np.full(5, 7.0), k)
            self.assertEqual(result.cost, 0.0)
        self.assertEqual(exact_kmeans_1d([0, 0, 10, 10], 3).cost, 0.0)


class LloydTests(unittest.TestCase):
    def test_monotonicity_and_optimal_lower_bound(self):
        rng = np.random.default_rng(508)
        for _ in range(8):
            points = rng.normal(size=30)
            for k in [1, 3, 8]:
                optimum = exact_kmeans_1d(points, k).cost
                for method in ["random", "kmeans++"]:
                    result = run_kmeans(points, k, method, seed=17)
                    self.assertLessEqual(result.final_cost, result.initial_cost + 1e-10)
                    self.assertGreaterEqual(result.final_cost + 1e-10, optimum)
                    self.assertTrue(result.converged)
                    self.assertAlmostEqual(result.final_cost, float(np.sum((points - result.centers[result.labels]) ** 2)), places=10)
                    for cluster in range(k):
                        assigned = points[result.labels == cluster]
                        if assigned.size:
                            self.assertAlmostEqual(result.centers[cluster], float(np.mean(assigned)), places=9)

    def test_reproducible_non_timing_results(self):
        points = np.random.default_rng(1).normal(size=100)
        for method in ["random", "kmeans++"]:
            first = run_kmeans(points, 6, method, seed=18)
            second = run_kmeans(points, 6, method, seed=18)
            np.testing.assert_array_equal(first.centers, second.centers)
            np.testing.assert_array_equal(first.labels, second.labels)
            self.assertEqual(first.initial_cost, second.initial_cost)
            self.assertEqual(first.final_cost, second.final_cost)
            self.assertEqual(first.iterations, second.iterations)

    def test_duplicates_empty_centers_and_single_point(self):
        for method in ["random", "kmeans++"]:
            for points, k in [([8.0], 1), ([2.0] * 7, 5), ([0, 0, 10, 10], 4)]:
                result = run_kmeans(points, k, method, seed=4)
                self.assertEqual(result.final_cost, 0.0)
                self.assertTrue(result.converged)
                self.assertTrue(np.all(np.isfinite(result.centers)))

    def test_translation_preserves_results(self):
        points = np.array([0, 1, 2, 8, 9, 10, 18, 19, 20], dtype=float)
        for method in ["random", "kmeans++"]:
            base = run_kmeans(points, 3, method, 16)
            shifted = run_kmeans(points + 1e12, 3, method, 16)
            self.assertEqual(base.final_cost, shifted.final_cost)
            self.assertEqual(base.initial_cost, shifted.initial_cost)
            np.testing.assert_array_equal(base.labels, shifted.labels)

    def test_iteration_cap_and_work_accounting(self):
        points = np.arange(50, dtype=float) ** 1.2
        for method in ["random", "kmeans++"]:
            result = run_kmeans(points, 4, method, 5, max_iterations=1, tolerance=0)
            self.assertEqual(result.iterations, 1)
            self.assertEqual(result.lloyd_distance_evaluations, 50 * 4 * 2)
            self.assertEqual(result.initialization_distance_evaluations, 0 if method == "random" else 50 * 3)
            self.assertGreaterEqual(result.total_seconds, 0)
            self.assertAlmostEqual(result.total_seconds, result.initialization_seconds + result.lloyd_seconds)

    def test_invalid_inputs(self):
        for points, k in [([], 1), ([[1, 2]], 1), ([1, np.nan], 1), ([1, np.inf], 1), ([1, 2], 0), ([1, 2], 3), ([1, 2], 1.5), ([1, 2], True)]:
            for function in [exact_kmeans_1d, lambda x, k: run_kmeans(x, k, "random", 0)]:
                with self.subTest(points=points, k=k), self.assertRaises(ValueError):
                    function(points, k)
        with self.assertRaises(ValueError):
            run_kmeans([1, 2], 1, "other", 0)
        with self.assertRaises(ValueError):
            run_kmeans([1, 2], 1, "random", 0, tolerance=-1)
        with self.assertRaises(ValueError):
            run_kmeans([1, 2], 1, "random", 0, max_iterations=0)


if __name__ == "__main__":
    unittest.main()
