from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICY_LABELS = {
    "always_0": "Always 0",
    "always_1": "Always 1",
    "always_2": "Always 2",
    "threshold": "Threshold",
    "linucb": "LinUCB",
    "linear_thompson_sampling": "Linear Thompson Sampling",
    "epsilon_greedy_linear": "Epsilon-Greedy Linear",
}

POLICY_ORDER = [
    "always_0",
    "always_1",
    "always_2",
    "threshold",
    "linucb",
    "linear_thompson_sampling",
    "epsilon_greedy_linear",
]

POLICY_COLORS = {
    "always_0": "#7a7a7a",
    "always_1": "#9db4c0",
    "always_2": "#6f8fa3",
    "threshold": "#d98c3f",
    "linucb": "#1f77b4",
    "linear_thompson_sampling": "#2ca02c",
    "epsilon_greedy_linear": "#d62728",
}

FIGURES = [
    ("cumulative_reward", "Cumulative Reward", True, "figure_1_cumulative_reward.png"),
    ("avg_latency_ms", "Average Latency (ms)", False, "figure_2_avg_latency.png"),
    ("total_cost", "Total Cost", False, "figure_3_total_cost.png"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the 3 proposal-ready figures.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "summary.csv",
        help="Path to aggregated summary CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "proposal_figures",
        help="Directory for the exported figure PNGs.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {
                "workload": row["workload"],
                "policy": row["policy"],
            }
            for key, value in row.items():
                if key in {"workload", "policy"}:
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def make_metric_figure(
    rows: List[Dict[str, object]],
    metric_key: str,
    metric_label: str,
    higher_is_better: bool,
    output_path: Path,
) -> None:
    workloads = list(dict.fromkeys(str(row["workload"]) for row in rows))
    grouped: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        workload = str(row["workload"])
        policy = str(row["policy"])
        grouped[workload][policy] = {
            "value": float(row[metric_key]),
            "std": float(row.get(f"{metric_key}_std", 0.0)),
        }

    fig, axes = plt.subplots(1, len(workloads), figsize=(4.8 * len(workloads), 5.5), constrained_layout=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        policy_data = grouped[workload]
        policies = [policy for policy in POLICY_ORDER if policy in policy_data]
        values = [policy_data[policy]["value"] for policy in policies]
        errors = [policy_data[policy]["std"] for policy in policies]
        colors = [POLICY_COLORS[policy] for policy in policies]

        bars = ax.bar(
            range(len(policies)),
            values,
            yerr=errors,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=3,
        )
        ax.set_title(workload.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels([POLICY_LABELS[policy] for policy in policies], rotation=35, ha="right")
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.35)

        best_value = max(values) if higher_is_better else min(values)
        for bar, value in zip(bars, values):
            if abs(value - best_value) < 1e-9:
                bar.set_linewidth(1.6)
                bar.set_edgecolor("#111111")

    fig.suptitle(f"{metric_label} Across Workloads", fontsize=16, fontweight="bold")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.input)

    for metric_key, metric_label, higher_is_better, filename in FIGURES:
        make_metric_figure(
            rows=rows,
            metric_key=metric_key,
            metric_label=metric_label,
            higher_is_better=higher_is_better,
            output_path=args.output_dir / filename,
        )

    print(f"Saved proposal figures to {args.output_dir}")


if __name__ == "__main__":
    main()
