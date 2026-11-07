# How much does careful seeding help K-means?

An experimental study comparing random-point and K-means++ initialisation,
followed by the same Lloyd iterations, on six types of synthetic one-dimensional
instances. An exact dynamic programme provides the optimal clustering cost so
we can compare solution quality, consistency, and computational cost.

## Directory overview

| Directory | Contents |
|---|---|
| `src/kmeans_study/` | Clustering algorithms, synthetic data generation, experiment execution, and analysis |
| `configs/` | Settings for the main experiments, timing pilot, and quick checks |
| `tests/` | Correctness checks for the algorithms, generators, experiments, and analysis |

## Key commands

Run these from the repository root using Python 3.10 or newer.

```sh
# Install the package and dependencies
python -m pip install -e .

# Run a small end-to-end check
python study.py run --config configs/smoke.json --output results/smoke --analyse

# Run the main experiments and generate summaries and plots
python study.py run --config configs/main.json --output results/main --analyse

# Regenerate summaries and plots from saved results
python study.py analyse --output results/main

# Run the tests
python -m unittest discover -s tests -v
```

The `--output` directory holds the generated datasets, raw results, summaries,
and plots. Repeating a run resumes missing attempts.
