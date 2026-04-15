from __future__ import annotations

import argparse
import csv
from pathlib import Path

from serverless_bandits.config import EnvironmentConfig, ExperimentConfig
from serverless_bandits.policies import build_default_policies
from serverless_bandits.runner import aggregate_records, format_summary, run_experiments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serverless prewarming simulator with contextual bandit policies."
    )
    parser.add_argument("--horizon", type=int, default=240, help="Number of time windows per run.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 11, 19],
        help="Random seeds for repeated experiments.",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["steady_low", "steady_medium", "bursty", "periodic_spikes", "abrupt_shift"],
        help="Workload names to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "summary.csv",
        help="Where to write aggregated results as CSV.",
    )
    return parser.parse_args()


def write_summary_csv(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for workload, policy_metrics in summary.items():
        for policy, metrics in policy_metrics.items():
            row = {"workload": workload, "policy": policy}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    experiment_config = ExperimentConfig(
        horizon=args.horizon,
        seeds=tuple(args.seeds),
        workload_names=tuple(args.workloads),
    )
    environment_config = EnvironmentConfig()
    policies = build_default_policies(environment_config.action_space)

    records = run_experiments(policies, experiment_config, environment_config)
    summary = aggregate_records(records)

    print(format_summary(summary))
    write_summary_csv(summary, args.output)
    print(f"\nSaved aggregated summary to {args.output}")


if __name__ == "__main__":
    main()
