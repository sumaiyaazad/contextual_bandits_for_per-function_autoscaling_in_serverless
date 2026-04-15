from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Dict, Iterable, List

from serverless_bandits.config import EnvironmentConfig, ExperimentConfig
from serverless_bandits.environment import ServerlessEnvironment, summarize_results
from serverless_bandits.policies import Policy
from serverless_bandits.workloads import build_workload


@dataclass
class RunRecord:
    workload: str
    seed: int
    policy: str
    metrics: Dict[str, float]


def run_single_episode(
    policy: Policy,
    workload_name: str,
    seed: int,
    experiment_config: ExperimentConfig,
    environment_config: EnvironmentConfig,
) -> RunRecord:
    workload = build_workload(workload_name, experiment_config.horizon, seed)
    environment = ServerlessEnvironment(environment_config, experiment_config.horizon)
    policy.reset()

    results = []
    for t, demand in enumerate(workload.requests):
        context = environment.context()
        action = policy.select_action(context, t=t)
        step = environment.step(int(demand), int(action))
        policy.update(context, int(action), step.reward)
        results.append(step)

    return RunRecord(
        workload=workload_name,
        seed=seed,
        policy=policy.name,
        metrics=summarize_results(results),
    )


def run_experiments(
    policies: Iterable[Policy],
    experiment_config: ExperimentConfig,
    environment_config: EnvironmentConfig,
) -> List[RunRecord]:
    records: List[RunRecord] = []
    for workload_name in experiment_config.workload_names:
        for seed in experiment_config.seeds:
            for policy in policies:
                records.append(
                    run_single_episode(
                        policy=policy,
                        workload_name=workload_name,
                        seed=seed,
                        experiment_config=experiment_config,
                        environment_config=environment_config,
                    )
                )
    return records


def aggregate_records(records: List[RunRecord]) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for record in records:
        grouped.setdefault(record.workload, {}).setdefault(record.policy, []).append(record.metrics)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for workload, workload_records in grouped.items():
        summary[workload] = {}
        for policy, metrics_list in workload_records.items():
            metric_names = metrics_list[0].keys()
            summary[workload][policy] = {}
            for metric_name in metric_names:
                values = [metrics[metric_name] for metrics in metrics_list]
                summary[workload][policy][metric_name] = mean(values)
                summary[workload][policy][f"{metric_name}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary


def format_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    lines: List[str] = []
    for workload, policy_metrics in summary.items():
        lines.append(f"Workload: {workload}")
        ordered = sorted(
            policy_metrics.items(),
            key=lambda item: item[1]["cumulative_reward"],
            reverse=True,
        )
        for policy, metrics in ordered:
            lines.append(
                "  "
                + f"{policy:<28} reward={metrics['cumulative_reward']:.2f} "
                + f"avg_latency={metrics['avg_latency_ms']:.1f}ms "
                + f"p95={metrics['p95_latency_ms']:.1f}ms "
                + f"viol_rate={metrics['sla_violation_rate']:.2f} "
                + f"cold={metrics['cold_starts']:.1f} "
                + f"cost={metrics['total_cost']:.1f}"
            )
        lines.append("")
    return "\n".join(lines).strip()

