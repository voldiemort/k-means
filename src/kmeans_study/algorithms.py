"""Auditable one-dimensional K-means implementations used by the study.

All objectives are sums of squared point-to-centre distances. The two heuristics
share exactly the same Lloyd updates; only their initialisation differs.
"""

from dataclasses import dataclass
from numbers import Integral
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ExactResult:
    """Globally optimal cost, centres, and labels in the input point order."""

    cost: float
    centers: np.ndarray
    labels: np.ndarray


@dataclass(frozen=True)
class RunResult:
    """One initialisation and Lloyd run, including timing and work counts.

    Distance counts include each scalar squared point-to-centre distance
    evaluated during seeding or Lloyd assignments. They exclude validation,
    centre means, and the exact solver, so they complement rather than replace
    elapsed time. Initial objective measurement reuses the first Lloyd
    assignment; the final assignment is included in both time and work counts.
    """

    initial_cost: float
    final_cost: float
    centers: np.ndarray
    labels: np.ndarray
    iterations: int
    converged: bool
    initialization_seconds: float
    lloyd_seconds: float
    total_seconds: float
    initialization_distance_evaluations: int
    lloyd_distance_evaluations: int


def _validate_points(points: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("points must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must contain only finite values")
    if isinstance(k, bool) or not isinstance(k, Integral) or not 1 <= k <= values.size:
        raise ValueError("k must be an integer between 1 and the number of points")
    # A translation improves numerical accuracy without changing any distances.
    with np.errstate(over="ignore", invalid="ignore"):
        span = values.max() - values.min()
    if not np.isfinite(span) or span > np.sqrt(np.finfo(np.float64).max / values.size):
        raise ValueError("point range is too large for finite squared-distance costs")
    return values


def exact_kmeans_1d(points: np.ndarray, k: int) -> ExactResult:
    """Find the exact 1-D optimum by contiguous-partition dynamic programming.

    An optimal clustering of sorted one-dimensional points has contiguous
    clusters. For each prefix endpoint, the recurrence considers every previous
    split and adds the squared error of the final segment. This costs O(kn²)
    time and O(kn + n) memory.

    Segment errors use vectorised Welford updates rather than subtracting
    prefix sums of squares. This avoids catastrophic cancellation for narrow
    clusters far from the origin. The optimum is exact combinatorially, subject
    to ordinary floating-point rounding.
    """
    values = _validate_points(points, k)
    origin = values[0]
    order = np.argsort(values, kind="stable")
    sorted_points = values[order] - origin
    size = values.size
    costs = np.full((k + 1, size + 1), np.inf)
    costs[0, 0] = 0.0
    splits = np.full((k + 1, size + 1), -1, dtype=np.int64)
    segment_means = np.zeros(size)
    segment_errors = np.zeros(size)
    starts = np.arange(size)

    for end in range(1, size + 1):
        lengths = end - starts[:end]
        delta = sorted_points[end - 1] - segment_means[:end]
        segment_means[:end] += delta / lengths
        segment_errors[:end] += delta * (sorted_points[end - 1] - segment_means[:end])
        np.maximum(segment_errors[:end], 0.0, out=segment_errors[:end])
        for groups in range(1, min(k, end) + 1):
            first_split = groups - 1
            candidates = costs[groups - 1, first_split:end] + segment_errors[first_split:end]
            best_split = first_split + int(np.argmin(candidates))
            costs[groups, end] = candidates[best_split - first_split]
            splits[groups, end] = best_split

    centers = np.empty(k)
    sorted_labels = np.empty(size, dtype=np.int64)
    end = size
    for groups in range(k, 0, -1):
        start = int(splits[groups, end])
        centers[groups - 1] = np.mean(sorted_points[start:end])
        sorted_labels[start:end] = groups - 1
        end = start
    labels = np.empty(size, dtype=np.int64)
    labels[order] = sorted_labels
    # Recompute from the reconstructed partition, avoiding accumulated DP error.
    residuals = sorted_points - centers[sorted_labels]
    return ExactResult(float(residuals @ residuals), centers + origin, labels)


def _initial_centers(values: np.ndarray, k: int, method: str, rng: np.random.Generator):
    if method == "random":
        return values[rng.choice(values.size, size=k, replace=False)].copy(), 0

    centers = np.empty(k)
    selected = np.zeros(values.size, dtype=bool)
    selected_index = int(rng.integers(values.size))
    centers[0] = values[selected_index]
    selected[selected_index] = True
    nearest_squared = np.full(values.size, np.inf)
    for center_index in range(1, k):
        squared = (values - centers[center_index - 1]) ** 2
        np.minimum(nearest_squared, squared, out=nearest_squared)
        potential = float(nearest_squared.sum())
        if potential == 0.0:
            # All distinct locations are covered. Additional centres cannot
            # change the objective; choose an unused input index uniformly.
            selected_index = int(rng.choice(np.flatnonzero(~selected)))
        else:
            selected_index = int(rng.choice(values.size, p=nearest_squared / potential))
        centers[center_index] = values[selected_index]
        selected[selected_index] = True
    return centers, values.size * (k - 1)


def run_kmeans(
    points: np.ndarray,
    k: int,
    method: str,
    seed: int,
    max_iterations: int = 300,
    tolerance: float = 1e-10,
) -> RunResult:
    """Run random-point or standard single-candidate D²-seeded Lloyd K-means.

    ``method`` is ``random`` or ``kmeans++``. Input indices are sampled without
    replacement for random initialisation; duplicate coordinates are allowed.
    Ties always go to the lowest centre index. Empty centres retain their old
    position: this deterministic policy cannot increase the objective and is
    identical for both initialisers.

    Each iteration updates the means then reassigns all points. Termination
    occurs when assignments are unchanged or the maximum absolute centre
    movement is at most ``tolerance`` (in input units), with an iteration cap.
    Returned labels and final cost always refer to the returned centres.
    Timings cover initialisation and Lloyd only, excluding input validation and
    random-generator setup. A nonconverged capped run is explicitly flagged.
    """
    values = _validate_points(points, k)
    if method not in {"random", "kmeans++"}:
        raise ValueError("method must be 'random' or 'kmeans++'")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    origin = values[0]
    values = values - origin
    rng = np.random.default_rng(seed)

    initialization_start = perf_counter()
    centers, initialization_distances = _initial_centers(values, k, method, rng)
    lloyd_start = perf_counter()
    initialization_seconds = lloyd_start - initialization_start
    squared_distances = (values[:, None] - centers[None, :]) ** 2
    labels = np.argmin(squared_distances, axis=1)
    initial_cost = float(squared_distances[np.arange(values.size), labels].sum())
    converged = False
    for iterations in range(1, max_iterations + 1):
        populations = np.bincount(labels, minlength=k)
        sums = np.bincount(labels, weights=values, minlength=k)
        updated_centers = centers.copy()
        occupied = populations > 0
        updated_centers[occupied] = sums[occupied] / populations[occupied]
        movement = float(np.max(np.abs(updated_centers - centers)))
        centers = updated_centers
        squared_distances = (values[:, None] - centers[None, :]) ** 2
        updated_labels = np.argmin(squared_distances, axis=1)
        unchanged = np.array_equal(labels, updated_labels)
        labels = updated_labels
        if unchanged or movement <= tolerance:
            converged = True
            break
    final_cost = float(squared_distances[np.arange(values.size), labels].sum())
    lloyd_seconds = perf_counter() - lloyd_start
    return RunResult(
        initial_cost=initial_cost,
        final_cost=final_cost,
        centers=centers + origin,
        labels=labels,
        iterations=iterations,
        converged=converged,
        initialization_seconds=initialization_seconds,
        lloyd_seconds=lloyd_seconds,
        total_seconds=initialization_seconds + lloyd_seconds,
        initialization_distance_evaluations=initialization_distances,
        lloyd_distance_evaluations=values.size * k * (iterations + 1),
    )
