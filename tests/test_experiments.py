"""End-to-end checks of persisted studies and append-only replication."""

import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kmeans_study.algorithms import run_kmeans
from kmeans_study.experiments import deterministic_seed, run_experiment


def _rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _key(row):
    return row["setting_id"], int(row["dataset_index"]), row["method"], int(row["attempt"])


def _without_measurements(row):
    # Timing and invocation identity necessarily differ across fresh/resumed runs.
    return {key: value for key, value in row.items()
            if not key.endswith("_seconds") and key not in {"session_id", "record_sha256"}}


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = {
            "name": "resume-test",
            "master_seed": 913,
            "datasets_per_setting": 2,
            "initializations": 3,
            "max_iterations": 300,
            "tolerance": 1e-10,
            "settings": [{"id": "small_uniform", "family": "uniform", "n": 24,
                          "k": 3, "parameters": {"low": -5, "high": 5}}],
        }

    def run_study(self, config=None, directory=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_experiment(config or self.config, directory or self.directory)

    def test_idempotent_resume_preserves_saved_records(self):
        first = self.run_study()
        results_before = (self.directory / "results.csv").read_bytes()
        datasets_before = (self.directory / "datasets.csv").read_bytes()
        second = self.run_study()
        self.assertEqual(first["new_attempts"], 12)
        self.assertEqual(second["new_attempts"], 0)
        self.assertEqual(results_before, (self.directory / "results.csv").read_bytes())
        self.assertEqual(datasets_before, (self.directory / "datasets.csv").read_bytes())
        self.assertFalse((self.directory / ".run.lock").exists())
        manifest = json.loads((self.directory / "manifest.json").read_text())
        self.assertEqual(len(manifest["sessions"]), 2)
        self.assertTrue(all(session["completed"] for session in manifest["sessions"]))

    def test_attempt_topup_is_append_only_and_matches_fresh_full_run(self):
        self.run_study()
        before = (self.directory / "results.csv").read_bytes()
        initial_rows = {_key(row): row for row in _rows(self.directory / "results.csv")}
        expanded = copy.deepcopy(self.config)
        expanded["initializations"] = 5
        topup = self.run_study(expanded)
        self.assertEqual(topup["new_attempts"], 8)
        self.assertTrue((self.directory / "results.csv").read_bytes().startswith(before))
        resumed = {_key(row): row for row in _rows(self.directory / "results.csv")}
        self.assertEqual(len(resumed), 20)
        for key, row in initial_rows.items():
            self.assertEqual(resumed[key], row)
        fresh_directory = self.directory / "fresh"
        self.run_study(expanded, fresh_directory)
        fresh = {_key(row): row for row in _rows(fresh_directory / "results.csv")}
        self.assertEqual({key: _without_measurements(row) for key, row in resumed.items()},
                         {key: _without_measurements(row) for key, row in fresh.items()})

    def test_dataset_topup_preserves_existing_points_and_results(self):
        self.run_study()
        original_rows = {_key(row): row for row in _rows(self.directory / "results.csv")}
        paths = [self.directory / row["path"] for row in _rows(self.directory / "datasets.csv")]
        original_data = {path: path.read_bytes() for path in paths}
        expanded = copy.deepcopy(self.config)
        expanded["datasets_per_setting"] = 3
        result = self.run_study(expanded)
        self.assertEqual(result["new_attempts"], 6)
        new_rows = {_key(row): row for row in _rows(self.directory / "results.csv")}
        self.assertEqual(len(new_rows), 18)
        for key, row in original_rows.items():
            self.assertEqual(new_rows[key], row)
        for path, contents in original_data.items():
            self.assertEqual(path.read_bytes(), contents)

    def test_persisted_runs_match_direct_independent_calls(self):
        self.run_study()
        datasets = {int(row["dataset_index"]): row
                    for row in _rows(self.directory / "datasets.csv")}
        for row in _rows(self.directory / "results.csv"):
            metadata = datasets[int(row["dataset_index"])]
            points = np.load(self.directory / metadata["path"], allow_pickle=False)
            expected_seed = deterministic_seed(913, "small_uniform", int(row["dataset_index"]),
                                               row["method"], int(row["attempt"]), "initialization")
            self.assertEqual(int(row["seed"]), expected_seed)
            direct = run_kmeans(points, 3, row["method"], expected_seed)
            self.assertEqual(float(row["initial_cost"]), direct.initial_cost)
            self.assertEqual(float(row["final_cost"]), direct.final_cost)
            self.assertEqual(int(row["iterations"]), direct.iterations)
            self.assertEqual(row["converged"], str(direct.converged))
            optimum = float(metadata["optimum_cost"])
            self.assertEqual(float(row["initial_ratio"]), direct.initial_cost / optimum)
            self.assertEqual(float(row["final_ratio"]), direct.final_cost / optimum)

    def test_changed_fixed_configuration_and_reduced_counts_are_rejected(self):
        self.run_study()
        before = (self.directory / "results.csv").read_bytes()
        for field, value in [("master_seed", 914), ("tolerance", 1e-5),
                             ("initializations", 2), ("datasets_per_setting", 1)]:
            changed = copy.deepcopy(self.config)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.run_study(changed)
            self.assertEqual(before, (self.directory / "results.csv").read_bytes())
            self.assertFalse((self.directory / ".run.lock").exists())

    def test_saved_point_modification_is_rejected(self):
        self.run_study()
        metadata = _rows(self.directory / "datasets.csv")[0]
        path = self.directory / metadata["path"]
        points = np.load(path, allow_pickle=False)
        points[0] += 1
        np.save(path, points, allow_pickle=False)
        with self.assertRaisesRegex(ValueError, "modified"):
            self.run_study()
        self.assertFalse((self.directory / ".run.lock").exists())

    def test_modified_result_cost_is_rejected(self):
        self.run_study()
        path = self.directory / "results.csv"
        rows = _rows(path)
        rows[0]["final_cost"] = str(float(rows[0]["final_cost"]) + 0.5)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            self.run_study()

    def test_modified_optimum_metadata_is_rejected(self):
        self.run_study()
        path = self.directory / "datasets.csv"
        rows = _rows(path)
        rows[0]["optimum_cost"] = str(float(rows[0]["optimum_cost"]) + 0.5)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            self.run_study()

    def test_duplicate_attempt_records_are_rejected(self):
        self.run_study()
        path = self.directory / "results.csv"
        with path.open("a", newline="", encoding="utf-8") as stream:
            stream.write(path.read_text(encoding="utf-8").splitlines()[1] + "\n")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.run_study()

    def test_exclusive_lock_prevents_concurrent_writers(self):
        lock = self.directory / ".run.lock"
        lock.write_text("another writer", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "running"):
            self.run_study()
        self.assertEqual(lock.read_text(), "another writer")

    def test_seed_derivation_is_stable_and_namespaced(self):
        expected = deterministic_seed(913, "setting", 0, "random", 0)
        self.assertEqual(expected, deterministic_seed(913, "setting", 0, "random", 0))
        alternatives = [deterministic_seed(913, "setting", 0, "kmeans++", 0),
                        deterministic_seed(913, "setting", 0, "random", 1),
                        deterministic_seed(913, "setting", 1, "random", 0),
                        deterministic_seed(914, "setting", 0, "random", 0),
                        deterministic_seed(913, "setting", "0", "random", 0)]
        self.assertNotIn(expected, alternatives)
        self.assertEqual(len(set(alternatives)), len(alternatives))
        self.assertGreaterEqual(expected, 0)
        self.assertLess(expected, 2**64)


if __name__ == "__main__":
    unittest.main()
