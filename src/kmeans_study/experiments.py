"""Persist reproducible experiments and resume only missing attempts.

Each dataset has its own generation seed and exact optimum. Initialisation seeds
depend on setting, dataset, method, and attempt, rather than execution order.
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
from time import perf_counter

import numpy as np

from .algorithms import exact_kmeans_1d, run_kmeans
from .datasets import generate_dataset

DATASET_FIELDS = ["setting_id", "dataset_index", "family", "n", "k", "dataset_seed",
                  "optimum_cost", "exact_seconds", "path", "sha256", "record_sha256"]
RESULT_FIELDS = ["setting_id", "dataset_index", "method", "attempt", "seed",
                 "initial_cost", "final_cost", "initial_ratio", "final_ratio",
                 "iterations", "converged", "initialization_seconds", "lloyd_seconds",
                 "total_seconds", "initialization_distance_evaluations",
                 "lloyd_distance_evaluations", "session_id", "record_sha256"]


def deterministic_seed(master_seed: int, *parts: object) -> int:
    """Derive stable 64-bit seeds without Python's process-randomised hash."""
    payload = json.dumps([master_seed, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def load_config(path: Path) -> dict:
    """Read and validate the experiment structure before creating output."""
    config = json.loads(path.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Validate settings and counts for both the CLI and Python entry points."""
    required = {"name", "master_seed", "datasets_per_setting", "initializations", "settings"}
    if not required <= config.keys():
        raise ValueError(f"Configuration missing {sorted(required - config.keys())}")
    for key in ("datasets_per_setting", "initializations", "max_iterations"):
        value = config.get(key, 300)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config["master_seed"]) is not int or config["master_seed"] < 0:
        raise ValueError("master_seed must be a non-negative integer")
    tolerance = config.get("tolerance", 1e-10)
    if not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if not isinstance(config["settings"], list) or not config["settings"]:
        raise ValueError("settings must be a nonempty list")
    identifiers = set()
    for setting in config["settings"]:
        identifier = setting["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError("Setting IDs must be unique and contain only letters, digits, _ or -")
        identifiers.add(identifier)
        # This also validates family-specific parameters and feasibility.
        generate_dataset(setting["family"], setting["n"], setting["k"],
                         setting.get("parameters", {}), 0)


def _signature(config: dict) -> str:
    fixed = {key: value for key, value in config.items()
             if key not in {"initializations", "datasets_per_setting"}}
    digest = hashlib.sha256(json.dumps(fixed, sort_keys=True).encode())
    for filename in ("algorithms.py", "datasets.py", "experiments.py"):
        digest.update(Path(__file__).with_name(filename).read_bytes())
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        expected_fields = DATASET_FIELDS if path.name == "datasets.csv" else RESULT_FIELDS
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Unexpected CSV header in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Incomplete CSV row in {path}; restore it before resuming")
    if any(row["record_sha256"] != _row_digest(row) for row in rows):
        raise ValueError(f"CSV integrity check failed in {path}; restore it before resuming")
    return rows


def _row_digest(row: dict) -> str:
    """Detect accidental edits to scientific records, including their metadata."""
    values = {key: str(value) for key, value in row.items() if key != "record_sha256"}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _append_row(path: Path, fields: list[str], row: dict) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow({**row, "record_sha256": _row_digest(row)})
        stream.flush()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _environment() -> dict:
    versions = {}
    for name in ("numpy", "pandas", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "packages": versions}


def run_experiment(config: dict, output_dir: Path) -> dict:
    """Run or resume a study, refusing changed algorithms, data, or settings.

Increasing replication counts is supported. Times from separate invocations are
identified by session_id; do not mix environments when interpreting wall time.
An exclusive lock prevents two processes from appending to the same study.
"""
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = output_dir / ".run.lock"
    try:
        lock_stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError(f"{lock} exists. If no study is running, remove this stale lock.") from error
    try:
        with lock_stream:
            lock_stream.write(f"Started {datetime.now(timezone.utc).isoformat()}\n")
            lock_stream.flush()
            return _run_locked(config, output_dir)
    finally:
        lock.unlink(missing_ok=True)


def _run_locked(config: dict, output_dir: Path) -> dict:
    started = perf_counter()
    manifest_path = output_dir / "manifest.json"
    signature = _signature(config)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != signature:
            raise ValueError("Configuration or algorithm source changed. Choose a new output directory.")
        previous = manifest["last_requested_config"]
        if any(config[key] < previous[key] for key in ("initializations", "datasets_per_setting")):
            raise ValueError("Replication counts cannot be reduced in an existing study")
    else:
        if any((output_dir / name).exists() for name in ("results.csv", "datasets.csv")):
            raise ValueError("Existing data has no manifest; choose a clean output directory")
        manifest = {"signature": signature, "original_config": config, "sessions": []}
    session_id = len(manifest["sessions"])
    session = {"id": session_id, "started_utc": datetime.now(timezone.utc).isoformat(),
               "environment": _environment(), "completed": False}
    manifest["sessions"].append(session)
    manifest["last_requested_config"] = config
    _write_json(manifest_path, manifest)

    dataset_path = output_dir / "datasets.csv"
    results_path = output_dir / "results.csv"
    dataset_rows = _read_rows(dataset_path)
    stored = {(row["setting_id"], int(row["dataset_index"])): row for row in dataset_rows}
    result_rows = _read_rows(results_path)
    completed = {(row["setting_id"], int(row["dataset_index"]), row["method"], int(row["attempt"]))
                 for row in result_rows}
    if len(completed) != len(result_rows) or len(stored) != len(dataset_rows):
        raise ValueError("Duplicate records found; refusing an ambiguous resume")
    new_attempts = 0
    for setting in config["settings"]:
        identifier = setting["id"]
        for dataset_index in range(config["datasets_per_setting"]):
            key = (identifier, dataset_index)
            seed = deterministic_seed(config["master_seed"], identifier, dataset_index, "dataset")
            if key in stored:
                metadata = stored[key]
                saved_path = output_dir / metadata["path"]
                if hashlib.sha256(saved_path.read_bytes()).hexdigest() != metadata["sha256"]:
                    raise ValueError(f"Saved dataset was modified: {saved_path}")
                points = np.load(saved_path, allow_pickle=False)
                optimum = float(metadata["optimum_cost"])
            else:
                points = generate_dataset(setting["family"], setting["n"], setting["k"],
                                          setting.get("parameters", {}), seed)
                exact_started = perf_counter()
                exact = exact_kmeans_1d(points, setting["k"])
                exact_seconds = perf_counter() - exact_started
                optimum = exact.cost
                relative_path = Path("datasets") / identifier / f"dataset_{dataset_index:04d}.npy"
                saved_path = output_dir / relative_path
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(saved_path, points, allow_pickle=False)
                metadata = dict(setting_id=identifier, dataset_index=dataset_index,
                                family=setting["family"], n=setting["n"], k=setting["k"],
                                dataset_seed=seed, optimum_cost=optimum, exact_seconds=exact_seconds,
                                path=relative_path.as_posix(),
                                sha256=hashlib.sha256(saved_path.read_bytes()).hexdigest())
                _append_row(dataset_path, DATASET_FIELDS, metadata)
            for attempt in range(config["initializations"]):
                # Alternate which method runs first to reduce systematic order effects.
                methods = ("random", "kmeans++") if (dataset_index + attempt) % 2 == 0 else ("kmeans++", "random")
                for method in methods:
                    record_key = (*key, method, attempt)
                    if record_key in completed:
                        continue
                    run_seed = deterministic_seed(config["master_seed"], identifier, dataset_index,
                                                  method, attempt, "initialization")
                    result = run_kmeans(points, setting["k"], method, run_seed,
                                        max_iterations=config.get("max_iterations", 300),
                                        tolerance=config.get("tolerance", 1e-10))
                    row = asdict(result)
                    del row["centers"], row["labels"]
                    if result.final_cost < optimum - 1e-7 * max(1.0, abs(optimum)):
                        raise ArithmeticError("A heuristic cost is below the exact optimum")
                    if result.final_cost > result.initial_cost + 1e-7 * max(1.0, result.initial_cost):
                        raise ArithmeticError("Lloyd iterations increased the objective")
                    row.update(setting_id=identifier, dataset_index=dataset_index, method=method,
                               attempt=attempt, seed=run_seed, session_id=session_id,
                               initial_ratio=result.initial_cost / optimum if optimum > 0 else "",
                               final_ratio=result.final_cost / optimum if optimum > 0 else "")
                    _append_row(results_path, RESULT_FIELDS, row)
                    new_attempts += 1
            print(f"{identifier}: dataset {dataset_index + 1}/{config['datasets_per_setting']} complete", flush=True)
    session.update(completed=True, elapsed_seconds=perf_counter() - started, new_attempts=new_attempts,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    _write_json(manifest_path, manifest)
    print(f"Completed {new_attempts} new attempts in {session['elapsed_seconds']:.2f}s", flush=True)
    return session
