"""Reproducible one-dimensional synthetic instances.

Every Gaussian component has a predetermined positive population. Component
membership is a generation device, never a claim about the optimal partition.
The returned observations are shuffled so input ordering carries no labels.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Mapping

import numpy as np


FAMILIES = (
    "uniform",
    "balanced",
    "overlapping",
    "unequal_sizes",
    "unequal_spreads",
    "outliers",
)

_DEFAULTS = {
    "uniform": {"low": -16.0, "high": 16.0},
    "balanced": {"gap": 8.0, "std": 1.0},
    "overlapping": {"gap": 2.0, "std": 1.0},
    "unequal_sizes": {"gap": 8.0, "std": 1.0, "size_ratio": 8.0},
    "unequal_spreads": {"gap": 4.0, "std": 1.0, "spread_ratio": 8.0},
    "outliers": {
        "gap": 8.0,
        "std": 1.0,
        "outlier_fraction": 0.02,
        "outlier_distance": 24.0,
    },
}


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _component_counts(total: int, weights: np.ndarray) -> np.ndarray:
    """Reserve one point per component, then apportion the remainder.

    The largest-remainder rule fixes the exact total. Ties go to the earlier
    component, making allocation independent of the random number generator.
    """
    component_count = len(weights)
    quotas = (total - component_count) * weights / weights.sum()
    counts = np.floor(quotas).astype(int) + 1
    remaining = total - int(counts.sum())
    order = np.argsort(-(quotas - np.floor(quotas)), kind="stable")
    counts[order[:remaining]] += 1
    return counts


def generate_dataset(
    family: str,
    n: int,
    k: int,
    parameters: Mapping[str, float],
    seed: int,
) -> np.ndarray:
    """Generate exactly ``n`` finite observations using ``default_rng(seed)``.

    ``k`` is the requested number of clustering centres and, except for the
    uniform family, also the number of Gaussian core components. Component
    means are equally spaced by ``gap`` and centred on zero. Thus varying k
    preserves local gap/std, rather than squeezing more groups into a fixed
    interval. ``std`` is the reference standard deviation, never a variance.

    Family parameters (omitted keys take the defaults in this module):

    * uniform: ``low < high`` bounds an independent uniform sample.
    * balanced / overlapping: ``gap > 0``, ``std > 0``; equal populations.
      These share a mechanism but use distinct gap settings in the experiments.
    * unequal_sizes: additionally ``size_ratio >= 1``. Geometric population
      weights run from size_ratio down to 1. Mandatory points and integer
      rounding mean realised population ratios can differ from weight ratios.
    * unequal_spreads: additionally ``spread_ratio >= 1``. Geometric component
      standard deviations have this maximum/minimum ratio and are normalised
      so their unweighted root-mean-square is ``std``. Equal populations keep
      overall within-component variance approximately fixed across the sweep.
    * outliers: ``0 <= outlier_fraction < 1`` removes a rounded fraction of
      observations from the core budget. There must still be at least one
      point per core component. Outliers alternate beyond the left/right
      outer means by ``outlier_distance * std`` and have standard deviation
      std. A positive fraction must allocate at least one outlier.

    No scaling based on a realised sample or on algorithm performance is used.
    Unknown parameter names and invalid or non-finite values are rejected.
    """
    if family not in _DEFAULTS:
        raise ValueError(f"Unknown family {family!r}; expected one of {FAMILIES}")
    n = _positive_integer(n, "n")
    k = _positive_integer(k, "k")
    if k > n:
        raise ValueError("k cannot exceed n")
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping")
    unknown = set(parameters) - set(_DEFAULTS[family])
    if unknown:
        raise ValueError(f"Unknown parameters for {family}: {sorted(unknown)}")
    values = dict(_DEFAULTS[family])
    for name, value in parameters.items():
        if isinstance(value, (bool, str, bytes)):
            raise ValueError(f"{name} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(number):
            raise ValueError(f"{name} must be a finite number")
        values[name] = number

    rng = np.random.default_rng(seed)
    if family == "uniform":
        if values["low"] >= values["high"]:
            raise ValueError("uniform requires low < high")
        points = rng.uniform(values["low"], values["high"], size=n)
    else:
        if values["gap"] <= 0 or values["std"] <= 0:
            raise ValueError("gap and std must be positive")
        means = (np.arange(k, dtype=float) - (k - 1) / 2) * values["gap"]
        weights = np.ones(k)
        deviations = np.full(k, values["std"])
        outlier_count = 0

        if family == "unequal_sizes":
            if values["size_ratio"] < 1:
                raise ValueError("size_ratio must be at least 1")
            weights = np.exp(np.linspace(0.0, -math.log(values["size_ratio"]), k))
        elif family == "unequal_spreads":
            if values["spread_ratio"] < 1:
                raise ValueError("spread_ratio must be at least 1")
            relative = np.exp(np.linspace(-math.log(values["spread_ratio"]), 0.0, k))
            # Keep the RMS calculation scaled even when k=1 and the requested
            # ratio is huge (a single component still has deviation std).
            relative /= relative.max()
            deviations = values["std"] * relative / np.sqrt(np.mean(relative**2))
        elif family == "outliers":
            fraction = values["outlier_fraction"]
            if not 0 <= fraction < 1:
                raise ValueError("outlier_fraction must lie in [0, 1)")
            if values["outlier_distance"] <= 0:
                raise ValueError("outlier_distance must be positive")
            outlier_count = math.floor(n * fraction + 0.5)
            if fraction > 0 and outlier_count == 0:
                raise ValueError("outlier_fraction allocates zero points; increase n or fraction")
            if n - outlier_count < k:
                raise ValueError("outlier budget leaves fewer than k core points")

        counts = _component_counts(n - outlier_count, weights)
        components = [
            rng.normal(mean, deviation, int(count))
            for mean, deviation, count in zip(means, deviations, counts)
        ]
        if outlier_count:
            sides = np.where(np.arange(outlier_count) % 2 == 0, -1.0, 1.0)
            locations = np.where(sides < 0, means[0], means[-1])
            locations += sides * values["outlier_distance"] * values["std"]
            components.append(rng.normal(locations, values["std"]))
        points = np.concatenate(components)

    if not np.isfinite(points).all():
        raise ValueError("parameters produce non-finite observations")
    return rng.permutation(points)
