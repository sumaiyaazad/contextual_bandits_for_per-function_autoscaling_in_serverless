from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


AI_POLICIES = {
    "linucb": "LinUCB",
    "linear_thompson_sampling": "Linear Thompson Sampling",
    "epsilon_greedy_linear": "Epsilon-Greedy Linear",
}

BASELINE_POLICIES = {
    "always_0": "Always 0",
    "always_1": "Always 1",
    "always_2": "Always 2",
    "threshold": "Threshold",
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

METRICS = [
    ("cumulative_reward", "Cumulative Reward", True),
    ("avg_latency_ms", "Average Latency (ms)", False),
    ("sla_violation_rate", "SLA Violation Rate", False),
    ("cold_starts", "Cold Starts", False),
    ("total_cost", "Total Cost", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comparison charts from summary.csv.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results") / "summary.csv",
        help="Aggregated experiment CSV input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "charts",
        help="Directory where charts will be written.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> List[Dict[str, object]]:
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


def policy_label(policy: str) -> str:
    if policy in AI_POLICIES:
        return AI_POLICIES[policy]
    return BASELINE_POLICIES.get(policy, policy)


def save_metric_chart(rows: List[Dict[str, object]], metric_key: str, metric_label: str, higher_is_better: bool, output_dir: Path) -> None:
    workloads = list(dict.fromkeys(str(row["workload"]) for row in rows))
    grouped: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for row in rows:
        workload = str(row["workload"])
        policy = str(row["policy"])
        grouped[workload][policy] = {
            "value": float(row[metric_key]),
            "std": float(row.get(f"{metric_key}_std", 0.0)),
        }

    fig, axes = plt.subplots(len(workloads), 1, figsize=(12, 3.6 * len(workloads)), constrained_layout=True)
    if len(workloads) == 1:
        axes = [axes]

    for ax, workload in zip(axes, workloads):
        policy_data = grouped[workload]
        policies = [p for p in POLICY_ORDER if p in policy_data]
        values = [policy_data[p]["value"] for p in policies]
        errors = [policy_data[p]["std"] for p in policies]
        colors = [POLICY_COLORS.get(p, "#333333") for p in policies]

        bars = ax.bar(
            [policy_label(p) for p in policies],
            values,
            yerr=errors,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            capsize=4,
        )
        ax.set_title(workload.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_ylabel(metric_label)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", rotation=20)

        best_value = max(values) if higher_is_better else min(values)
        for bar, value in zip(bars, values):
            is_best = abs(value - best_value) < 1e-9
            if is_best:
                bar.set_linewidth(1.5)
                bar.set_edgecolor("#111111")

    fig.suptitle(f"{metric_label} by Workload and Policy", fontsize=16, fontweight="bold")
    filename = metric_key.replace("_", "-") + ".png"
    fig.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_ai_vs_baseline_chart(rows: List[Dict[str, object]], output_dir: Path) -> None:
    best_ai: Dict[str, float] = {}
    best_baseline: Dict[str, float] = {}
    for workload in dict.fromkeys(str(row["workload"]) for row in rows):
        workload_rows = [row for row in rows if row["workload"] == workload]
        best_ai[workload] = max(
            float(row["cumulative_reward"])
            for row in workload_rows
            if str(row["policy"]) in AI_POLICIES
        )
        best_baseline[workload] = max(
            float(row["cumulative_reward"])
            for row in workload_rows
            if str(row["policy"]) in BASELINE_POLICIES
        )

    workloads = list(best_ai.keys())
    x = range(len(workloads))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    ax.bar(
        [i - width / 2 for i in x],
        [best_baseline[w] for w in workloads],
        width=width,
        color="#d98c3f",
        label="Best Baseline",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar(
        [i + width / 2 for i in x],
        [best_ai[w] for w in workloads],
        width=width,
        color="#2ca02c",
        label="Best AI Policy",
        edgecolor="black",
        linewidth=0.6,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([w.replace("_", " ").title() for w in workloads], rotation=15)
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Best AI Policy vs Best Baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.savefig(output_dir / "best-ai-vs-best-baseline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary(args.input)

    for metric_key, metric_label, higher_is_better in METRICS:
        save_metric_chart(rows, metric_key, metric_label, higher_is_better, args.output_dir)
    save_ai_vs_baseline_chart(rows, args.output_dir)

    print(f"Saved charts to {args.output_dir}")


if __name__ == "__main__":
    main()
