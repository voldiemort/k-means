"""Dataset-aware summaries and figures for the one-dimensional seeding study.

Attempts on one dataset share the same points. The setting summary therefore
gives equal weight to each dataset, rather than treating every attempt as an
independent dataset. Its quartiles describe the distribution of dataset means;
the per-dataset CSV retains quartiles of individual attempts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter
import numpy as np
import pandas as pd


METHODS = ("random", "kmeans++")
COLOURS = {"random": "#4477AA", "kmeans++": "#CC6677"}
METRICS = (
    "initial_cost", "final_cost", "initial_ratio", "final_ratio", "iterations",
    "initialization_seconds", "lloyd_seconds", "total_seconds",
    "initialization_distance_evaluations", "lloyd_distance_evaluations",
)
DATASET_KEYS = ["setting_id", "dataset_index"]


def summarise_results(
    results: pd.DataFrame, datasets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return attempt distributions per dataset and dataset-mean distributions.

    Undefined ratios (including zero-optimum cases) remain missing and are
    excluded only from ratio summaries. Their raw objective costs remain usable.
    A ``*_count`` column records the number of valid observations for a metric.
    """
    required = set(DATASET_KEYS + ["method", "attempt", "converged", *METRICS])
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Results are missing columns: {sorted(missing)}")
    dataset_required = set(DATASET_KEYS + ["family", "n", "k", "optimum_cost"])
    if dataset_required - set(datasets.columns):
        raise ValueError("Dataset metadata is missing required columns")
    if results.empty:
        raise ValueError("Cannot analyse an empty experiment")
    if datasets.duplicated(DATASET_KEYS).any():
        raise ValueError("Dataset metadata contains duplicate dataset keys")
    if results.duplicated(DATASET_KEYS + ["method", "attempt"]).any():
        raise ValueError("Results contain duplicate attempts")
    if not set(results["method"]).issubset(METHODS):
        raise ValueError("Results contain an unknown initialisation method")

    results = results.copy()
    for metric in METRICS:
        results[metric] = pd.to_numeric(results[metric], errors="raise")
        results[metric] = results[metric].replace([np.inf, -np.inf], np.nan)
    if results["converged"].dtype != bool:
        values = results["converged"].astype(str).str.lower()
        if not values.isin(["true", "false", "1", "0"]).all():
            raise ValueError("Convergence values must be booleans")
        results["converged"] = values.isin(["true", "1"])

    metadata = datasets[DATASET_KEYS + ["family", "n", "k", "optimum_cost"]]
    merged = results.merge(metadata, on=DATASET_KEYS, how="left", validate="many_to_one", indicator=True)
    if (merged["_merge"] != "both").any():
        raise ValueError("Some results do not have matching dataset metadata")
    # A ratio has no defined value when OPT is zero, even if a CSV incorrectly
    # supplies 0 or infinity. Keep cost results without inventing a ratio.
    merged.loc[merged["optimum_cost"] <= 0, ["initial_ratio", "final_ratio"]] = np.nan
    rows = []
    keys = DATASET_KEYS + ["method"]
    for key, group in merged.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(group.iloc[0][["family", "n", "k", "optimum_cost"]].to_dict())
        row["attempts"] = len(group)
        row["nonconverged_attempts"] = int((~group["converged"]).sum())
        for metric in METRICS:
            values = group[metric].dropna()
            row.update(_distribution(values, metric))
        rows.append(row)
    per_dataset = pd.DataFrame(rows)

    rows = []
    for key, group in per_dataset.groupby(["setting_id", "method"], sort=True):
        row = dict(zip(["setting_id", "method"], key))
        row.update(group.iloc[0][["family", "n", "k"]].to_dict())
        row["datasets"] = len(group)
        row["attempts"] = int(group["attempts"].sum())
        row["nonconverged_attempts"] = int(group["nonconverged_attempts"].sum())
        row["distribution_unit"] = "within_dataset_mean"
        for metric in METRICS:
            row.update(_distribution(group[f"{metric}_mean"].dropna(), metric))
        rows.append(row)
    return per_dataset, pd.DataFrame(rows)


def _distribution(values: pd.Series, prefix: str) -> dict:
    """Summarise finite observations, leaving empty distributions undefined."""
    return {
        f"{prefix}_count": int(len(values)),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q1": float(values.quantile(0.25)),
        f"{prefix}_q3": float(values.quantile(0.75)),
    }


def analyse_results(output_dir: Path) -> None:
    """Read a completed run and write summaries, reproducible figures and notes."""
    output_dir = Path(output_dir)
    results = pd.read_csv(output_dir / "results.csv")
    datasets = pd.read_csv(output_dir / "datasets.csv")
    per_dataset, per_setting = summarise_results(results, datasets)
    per_dataset.to_csv(output_dir / "summary_per_dataset.csv", index=False)
    per_setting.to_csv(output_dir / "summary_per_setting.csv", index=False)
    figures = output_dir / "figures"
    figures.mkdir(exist_ok=True)
    # Mask invalid ratios consistently in tables and plots.
    results = results.merge(
        datasets[DATASET_KEYS + ["optimum_cost"]], on=DATASET_KEYS,
        validate="many_to_one",
    )
    results[["initial_ratio", "final_ratio"]] = results[["initial_ratio", "final_ratio"]].replace(
        [np.inf, -np.inf], np.nan
    )
    results.loc[results["optimum_cost"] <= 0, ["initial_ratio", "final_ratio"]] = np.nan
    _plot_distributions(results, figures)
    _plot_quality(per_dataset, figures)
    _plot_runtime(per_dataset, figures)
    _plot_examples(output_dir, datasets, figures)
    _write_findings(output_dir, results, datasets, per_setting)


def _facet_grid(settings: list[str], title: str):
    columns = min(4, len(settings))
    rows = math.ceil(len(settings) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.2 * rows), squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(title, fontsize=13)
    for axis in axes.flat[len(settings):]:
        axis.set_visible(False)
    return figure, list(axes.flat)


def _plot_distributions(results: pd.DataFrame, figures: Path) -> None:
    settings = sorted(results["setting_id"].unique())
    figure, axes = _facet_grid(settings, "Final cost / OPT: descriptive distributions of individual attempts")
    for setting, axis in zip(settings, axes):
        selected = results[results["setting_id"] == setting]
        data = [selected.loc[selected["method"] == method, "final_ratio"].dropna().to_numpy()
                for method in METHODS]
        if any(len(values) for values in data):
            boxes = axis.boxplot(data, tick_labels=METHODS, patch_artist=True,
                                 flierprops={"markersize": 2, "alpha": 0.35})
            for patch, method in zip(boxes["boxes"], METHODS):
                patch.set_facecolor(COLOURS[method])
                patch.set_alpha(0.65)
            axis.axhline(1, color="#555555", linestyle=":", linewidth=0.8)
            if all((values > 0).all() for values in data):
                axis.set_yscale("log")
        else:
            axis.text(0.5, 0.5, "No defined ratios", ha="center", transform=axis.transAxes)
        axis.set_title(setting, fontsize=9, wrap=True)
        axis.set_ylabel("Final cost / OPT (log scale)")
        axis.set_xlabel(f"{selected['dataset_index'].nunique()} datasets; pooled attempts")
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(figures / "final_ratio_distributions.png", dpi=160)
    plt.close(figure)


def _plot_quality(per_dataset: pd.DataFrame, figures: Path) -> None:
    settings = sorted(per_dataset["setting_id"].unique())
    figure, axes = _facet_grid(settings, "Seeding and final quality: median and quartiles of dataset means")
    for setting, axis in zip(settings, axes):
        selected = per_dataset[per_dataset["setting_id"] == setting]
        for offset, method in zip((-0.04, 0.04), METHODS):
            method_rows = selected[selected["method"] == method]
            x = np.array([0.0, 1.0]) + offset
            values = [method_rows[f"{stage}_ratio_mean"].dropna() for stage in ("initial", "final")]
            if any(value.empty for value in values):
                continue
            median = np.array([value.median() for value in values])
            q1 = np.array([value.quantile(0.25) for value in values])
            q3 = np.array([value.quantile(0.75) for value in values])
            axis.errorbar(x, median, yerr=[median - q1, q3 - median], marker="o",
                          color=COLOURS[method], label=method, capsize=3)
        axis.set_xticks([0, 1], ["Initial", "Final"])
        axis.set_title(setting, fontsize=9, wrap=True)
        axis.set_ylabel("Cost / OPT (log scale)")
        if axis.lines:
            axis.set_yscale("log")
            axis.legend(fontsize=8)
        axis.grid(axis="y", alpha=0.2)
    figure.savefig(figures / "quality_before_after.png", dpi=160)
    plt.close(figure)


def _plot_runtime(per_dataset: pd.DataFrame, figures: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    figure.suptitle("Runtime comparison: each point pairs means on the same dataset")
    for axis, metric, title in zip(
        axes, ("total_seconds_mean", "initialization_seconds_mean"),
        ("Total: initialisation + Lloyd", "Initialisation only"),
    ):
        paired = per_dataset.pivot(index=DATASET_KEYS + ["family"], columns="method", values=metric)
        if not set(METHODS).issubset(paired.columns):
            axis.text(0.5, 0.5, "Both methods required", ha="center", transform=axis.transAxes)
            continue
        paired = paired.dropna(subset=list(METHODS))
        paired = paired[(paired[list(METHODS)] > 0).all(axis=1)]
        for family, group in paired.groupby(level="family"):
            axis.scatter(group["random"], group["kmeans++"], label=family, s=15, alpha=0.6)
        if not paired.empty:
            lower = paired[list(METHODS)].min().min() * 0.8
            upper = paired[list(METHODS)].max().max() * 1.2
            axis.plot([lower, upper], [lower, upper], linestyle=":", color="#444444", label="Equal time")
            axis.set_xscale("log")
            axis.set_yscale("log")
            # Default log formatting can label every minor tick on short ranges,
            # making neighbouring scientific-notation labels overlap.
            axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=8))
            axis.xaxis.set_major_formatter(LogFormatterSciNotation(
                base=10, labelOnlyBase=False, minor_thresholds=(np.inf, np.inf)
            ))
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(title)
        axis.set_xlabel("Random: mean seconds per attempt")
        axis.set_ylabel("K-means++: mean seconds per attempt")
        axis.grid(alpha=0.2)
    if axes[0].collections:
        axes[0].legend(fontsize=7)
    figure.savefig(figures / "runtime_comparison.png", dpi=160)
    plt.close(figure)


def _plot_examples(output_dir: Path, datasets: pd.DataFrame, figures: Path) -> None:
    if "path" not in datasets:
        return
    examples = datasets.sort_values(DATASET_KEYS).groupby("family", sort=True).first().reset_index()
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, max(3, 1.5 * len(examples))),
                                squeeze=False, constrained_layout=True)
    any_loaded = False
    for (_, example), axis in zip(examples.iterrows(), axes.flat):
        path = Path(str(example["path"]))
        if not path.is_absolute():
            path = output_dir / path
        if not path.exists() or path.suffix != ".npy":
            axis.set_visible(False)
            continue
        points = np.load(path, allow_pickle=False)
        axis.plot(points, np.zeros_like(points), "|", alpha=0.4, markersize=13)
        axis.set_yticks([])
        axis.set_title(f"{example['family']}: {example['setting_id']}, n={example['n']}, k={example['k']}", fontsize=9)
        axis.set_xlabel("Coordinate (separate scale for each example)")
        any_loaded = True
    if any_loaded:
        figure.suptitle("One saved dataset per instance family")
        figure.savefig(figures / "example_instances.png", dpi=160)
    plt.close(figure)


def _write_findings(
    output_dir: Path, results: pd.DataFrame, datasets: pd.DataFrame,
    per_setting: pd.DataFrame,
) -> None:
    manifest_path = output_dir / "manifest.json"
    run_name = output_dir.name
    config = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = manifest.get("last_requested_config", manifest.get("original_config", {}))
        run_name = config.get("name", run_name)
    nonconverged = int(per_setting["nonconverged_attempts"].sum())
    undefined = int(results["final_ratio"].isna().sum())
    lines = [
        f"# Experiment findings: {run_name}", "",
        "This file is generated from the saved raw results. Interpret pilot and smoke runs as checks, not final evidence.", "",
        f"The run contains {len(datasets)} datasets across {datasets['setting_id'].nunique()} settings "
        f"and {len(results)} algorithm attempts. There are {nonconverged} attempts that reached the iteration limit without convergence.", "",
        _completeness_note(config, results, datasets), "",
        "## How to read the summaries", "",
        "`summary_per_dataset.csv` gives means, medians and quartiles across attempts on each fixed dataset. "
        "`summary_per_setting.csv` gives means, medians and quartiles across those dataset means, weighting datasets equally. "
        "Count columns record valid observations. These quartiles are descriptive distributions, not confidence intervals.", "",
        "The final-ratio boxplots pool individual attempts within each setting for display. Those attempts share datasets "
        "and must not be treated as independent dataset replications. The before/after figure instead uses dataset means.", "",
        f"There are {undefined} attempts with undefined final ratios. A zero optimum makes cost / OPT undefined; "
        "such ratios are excluded from summaries while objective costs are retained. Inspect the saved optimum and raw costs for these cases.", "",
        "## Quality by setting", "",
        "Entries below are means of within-dataset mean final ratios. Lower is better; 1 is optimal. "
        "They describe this experiment and do not establish or disprove the general expected approximation guarantee.", "",
        "| Setting | Random | K-means++ |", "|---|---:|---:|",
    ]
    for setting, group in per_setting.groupby("setting_id", sort=True):
        means = group.set_index("method")["final_ratio_mean"]
        values = [_format_number(means.get(method, np.nan)) for method in METHODS]
        lines.append(f"| {setting} | {values[0]} | {values[1]} |")
    lines.extend([
        "", "## Timing interpretation", "",
        "Total algorithm time includes initialisation and Lloyd iterations. Dataset generation, exact optimisation, "
        "CSV writing and figure generation are outside that measurement. Initialisation time is also reported separately. "
        "Wall-clock results depend on the machine, background activity and instrumentation; they imply no universal speed guarantee. "
        "Distance-evaluation counts provide a complementary measure, not a complete model of computation.", "",
        "The experiment compares equal numbers of attempts. It does not evaluate best-of-many restarts or equal-time-budget selection. "
        "Synthetic component labels are not assumed to define the optimal clustering; quality is measured against the exact objective optimum.",
    ])
    (output_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_number(value: float) -> str:
    return f"{value:.5g}" if pd.notna(value) else "undefined"


def _completeness_note(config: dict, results: pd.DataFrame, datasets: pd.DataFrame) -> str:
    """Check coverage against the latest requested run, including top-up attempts."""
    required = {"settings", "datasets_per_setting", "initializations"}
    if not required.issubset(config):
        return "Run completeness could not be checked because no full requested configuration was found in the manifest."
    expected_datasets = {
        (setting["id"], index)
        for setting in config["settings"]
        for index in range(config["datasets_per_setting"])
    }
    present_datasets = set(datasets[DATASET_KEYS].itertuples(index=False, name=None))
    counts = results.groupby(DATASET_KEYS + ["method"])["attempt"].agg(set)
    expected_attempts = set(range(config["initializations"]))
    missing_attempts = sum(
        len(expected_attempts - counts.get((setting, index, method), set()))
        for setting, index in expected_datasets
        for method in METHODS
    )
    missing_datasets = len(expected_datasets - present_datasets)
    if missing_datasets or missing_attempts:
        return (
            f"**Incomplete run:** compared with the latest requested configuration, {missing_datasets} datasets "
            f"and {missing_attempts} algorithm attempts are missing. Summaries describe only the available observations."
        )
    return "All datasets and algorithm attempts requested by the latest configuration are present."
