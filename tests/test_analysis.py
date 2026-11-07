"""Tests of independent dataset weighting and undefined objective ratios."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from kmeans_study.analysis import METRICS, analyse_results, summarise_results


def make_attempt(dataset_index, attempt, ratio, converged=True):
    row = {metric: 1.0 for metric in METRICS}
    row.update(setting_id="example", dataset_index=dataset_index, method="random",
               attempt=attempt, converged=converged, initial_ratio=ratio, final_ratio=ratio)
    return row


def make_metadata(dataset_index, optimum=1.0):
    return dict(setting_id="example", dataset_index=dataset_index, family="uniform",
                n=20, k=2, optimum_cost=optimum)


class AnalysisTests(unittest.TestCase):
    def test_setting_statistics_weight_datasets_equally(self):
        results = pd.DataFrame([
            make_attempt(0, 0, 1.0),
            make_attempt(1, 0, 3.0), make_attempt(1, 1, 5.0), make_attempt(1, 2, 7.0),
        ])
        metadata = pd.DataFrame([make_metadata(0), make_metadata(1)])
        per_dataset, per_setting = summarise_results(results, metadata)
        self.assertEqual(per_dataset.loc[per_dataset.dataset_index == 1, "final_ratio_median"].item(), 5.0)
        summary = per_setting.iloc[0]
        self.assertEqual(summary["final_ratio_mean"], 3.0)
        self.assertEqual(summary["final_ratio_q1"], 2.0)
        self.assertEqual(summary["final_ratio_q3"], 4.0)
        self.assertEqual(summary["final_ratio_count"], 2)
        self.assertEqual(summary["attempts"], 4)

    def test_zero_optimum_excludes_ratio_but_preserves_cost(self):
        results = pd.DataFrame([make_attempt(0, 0, 0.0, False), make_attempt(1, 0, 2.0)])
        metadata = pd.DataFrame([make_metadata(0, 0.0), make_metadata(1)])
        per_dataset, per_setting = summarise_results(results, metadata)
        zero_row = per_dataset[per_dataset.dataset_index == 0].iloc[0]
        self.assertTrue(np.isnan(zero_row["final_ratio_mean"]))
        self.assertEqual(zero_row["final_ratio_count"], 0)
        self.assertEqual(zero_row["final_cost_count"], 1)
        self.assertEqual(per_setting.iloc[0]["final_ratio_mean"], 2.0)
        self.assertEqual(per_setting.iloc[0]["nonconverged_attempts"], 1)

    def test_unmatched_dataset_rejected(self):
        with self.assertRaisesRegex(ValueError, "matching dataset"):
            summarise_results(pd.DataFrame([make_attempt(1, 0, 1.0)]),
                              pd.DataFrame([make_metadata(0)]))

    def test_duplicate_attempt_rejected(self):
        attempt = make_attempt(0, 0, 1.0)
        with self.assertRaisesRegex(ValueError, "duplicate attempts"):
            summarise_results(pd.DataFrame([attempt, attempt]), pd.DataFrame([make_metadata(0)]))

    def test_csv_boolean_strings_are_parsed(self):
        attempt = make_attempt(0, 0, 1.0)
        attempt["converged"] = "False"
        _, summary = summarise_results(pd.DataFrame([attempt]), pd.DataFrame([make_metadata(0)]))
        self.assertEqual(summary.iloc[0]["nonconverged_attempts"], 1)

    def test_analysis_writes_tables_findings_and_figures(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            rows = []
            for method in ("random", "kmeans++"):
                for dataset in (0, 1):
                    row = make_attempt(dataset, 0, 1.0 + dataset)
                    row["method"] = method
                    rows.append(row)
            pd.DataFrame(rows).to_csv(output / "results.csv", index=False)
            metadata = pd.DataFrame([make_metadata(0), make_metadata(1)])
            metadata["path"] = "example.npy"
            metadata.to_csv(output / "datasets.csv", index=False)
            np.save(output / "example.npy", np.array([0.0, 1.0, 2.0]))
            manifest = {"last_requested_config": {
                "name": "partial-test", "datasets_per_setting": 2,
                "initializations": 2, "settings": [{"id": "example"}],
            }}
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            analyse_results(output)
            self.assertEqual(len(pd.read_csv(output / "summary_per_dataset.csv")), 4)
            self.assertEqual(len(pd.read_csv(output / "summary_per_setting.csv")), 2)
            self.assertEqual(len(list((output / "figures").glob("*.png"))), 4)
            findings = (output / "findings.md").read_text(encoding="utf-8")
            self.assertIn("within-dataset", findings)
            self.assertIn("partial-test", findings)
            self.assertIn("0 datasets and 4 algorithm attempts are missing", findings)


if __name__ == "__main__":
    unittest.main()
