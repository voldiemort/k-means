"""Observable contracts of the synthetic data generators."""

import json
from pathlib import Path
import unittest

import numpy as np

from kmeans_study.datasets import FAMILIES, generate_dataset


class DatasetTests(unittest.TestCase):
    def test_all_families_are_finite_and_reproducible(self):
        for family in FAMILIES:
            with self.subTest(family=family):
                first = generate_dataset(family, 200, 4, {}, 17)
                second = generate_dataset(family, 200, 4, {}, 17)
                other = generate_dataset(family, 200, 4, {}, 18)
                self.assertEqual(first.shape, (200,))
                self.assertTrue(np.isfinite(first).all())
                np.testing.assert_array_equal(first, second)
                self.assertFalse(np.array_equal(first, other))

    def test_uniform_respects_explicit_bounds(self):
        points = generate_dataset("uniform", 1000, 4, {"low": 12, "high": 13}, 1)
        self.assertTrue(np.all((12 <= points) & (points < 13)))

    def test_balanced_components_have_prescribed_populations(self):
        points = generate_dataset("balanced", 103, 4, {"gap": 100, "std": 0.001}, 3)
        counts = np.histogram(points, bins=[-200, -100, 0, 100, 200])[0]
        np.testing.assert_array_equal(counts, [26, 26, 26, 25])

    def test_size_imbalance_preserves_every_component_and_total(self):
        points = generate_dataset("unequal_sizes", 400, 4,
                                  {"gap": 100, "std": 0.001, "size_ratio": 32}, 4)
        counts = np.histogram(points, bins=[-200, -100, 0, 100, 200])[0]
        self.assertEqual(int(counts.sum()), 400)
        self.assertTrue(np.all(counts > 0))
        self.assertTrue(np.all(np.diff(counts) < 0))
        self.assertGreater(counts[0] / counts[-1], 20)

    def test_spread_ratio_changes_variances_while_preserving_rms_scale(self):
        points = generate_dataset("unequal_spreads", 40000, 4,
                                  {"gap": 100, "std": 1, "spread_ratio": 8}, 5)
        components = [points[(points > left) & (points < right)]
                      for left, right in [(-200, -100), (-100, 0), (0, 100), (100, 200)]]
        deviations = np.array([np.std(component) for component in components])
        self.assertAlmostEqual(deviations[-1] / deviations[0], 8, delta=0.3)
        self.assertAlmostEqual(float(np.mean(deviations**2)), 1, delta=0.04)

    def test_outliers_are_taken_from_fixed_total_budget(self):
        points = generate_dataset("outliers", 200, 4,
                                  {"gap": 8, "std": 0.001,
                                   "outlier_fraction": 0.1, "outlier_distance": 10000}, 8)
        self.assertEqual(len(points), 200)
        self.assertEqual(int(np.count_nonzero(np.abs(points) > 20)), 20)
        self.assertEqual(int(np.count_nonzero(points < -20)), 10)
        self.assertEqual(int(np.count_nonzero(points > 20)), 10)

    def test_invalid_inputs_are_rejected(self):
        invalid = [
            ("missing", 100, 4, {}),
            ("uniform", 0, 4, {}),
            ("uniform", 10, 11, {}),
            ("uniform", 10.5, 4, {}),
            ("uniform", 10, True, {}),
            ("uniform", 10, 4, {"low": 2, "high": 1}),
            ("balanced", 10, 4, {"std": 0}),
            ("balanced", 10, 4, {"gap": -1}),
            ("balanced", 10, 4, {"std": float("nan")}),
            ("balanced", 10, 4, {"std": "1"}),
            ("balanced", 10, 4, {"variance": 1}),
            ("unequal_sizes", 100, 4, {"size_ratio": 0.5}),
            ("unequal_spreads", 100, 4, {"spread_ratio": 0}),
            ("outliers", 10, 4, {"outlier_fraction": 0.9}),
            ("outliers", 10, 4, {"outlier_fraction": 0.001}),
            ("outliers", 100, 4, {"outlier_fraction": 1}),
            ("outliers", 100, 4, {"outlier_distance": -1}),
        ]
        for family, n, k, parameters in invalid:
            with self.subTest(family=family, n=n, k=k, parameters=parameters):
                with self.assertRaises(ValueError):
                    generate_dataset(family, n, k, parameters, 0)

    def test_checked_in_configurations_generate_valid_instances(self):
        config_directory = Path(__file__).resolve().parents[1] / "configs"
        for config_path in config_directory.glob("*.json"):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            ids = [setting["id"] for setting in config["settings"]]
            self.assertEqual(len(ids), len(set(ids)))
            for setting in config["settings"]:
                with self.subTest(config=config_path.name, setting=setting["id"]):
                    points = generate_dataset(setting["family"], setting["n"], setting["k"],
                                              setting["parameters"], 123)
                    self.assertEqual(len(points), setting["n"])


if __name__ == "__main__":
    unittest.main()
