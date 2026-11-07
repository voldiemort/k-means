"""Command-line entry points for running and summarising a study."""

import argparse
from pathlib import Path

from .experiments import load_config, run_experiment


def main() -> None:
    """Execute a JSON configuration or regenerate analysis from saved raw data."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run or resume an experiment")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--initializations", type=int, help="Override attempts, e.g. top up 50 to 100")
    run.add_argument("--datasets", type=int, help="Override independent dataset count")
    run.add_argument("--analyse", action="store_true", help="Also generate summaries and figures")
    analyse = subparsers.add_parser("analyse", help="Summarise existing results")
    analyse.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.command == "run":
            config = load_config(arguments.config)
            for argument, key in ((arguments.initializations, "initializations"),
                                  (arguments.datasets, "datasets_per_setting")):
                if argument is not None:
                    if argument < 1:
                        parser.error(f"{key} must be positive")
                    config[key] = argument
            run_experiment(config, arguments.output)
            if not arguments.analyse:
                return
        from .analysis import analyse_results
        analyse_results(arguments.output)
        print(f"Analysis written to {arguments.output.resolve()}")
    except (ValueError, OSError, ArithmeticError) as error:
        parser.exit(1, f"Study failed: {error}\n")
