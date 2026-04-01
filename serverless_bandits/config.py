from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RewardWeights:
    latency: float = 1.0
    sla_violation: float = 8.0
    cold_start: float = 4.0
    warm_cost: float = 0.5


@dataclass(frozen=True)
class EnvironmentConfig:
    action_space: Sequence[int] = (0, 1, 2, 4)
    window_minutes: int = 1
    requests_per_instance: int = 40
    base_latency_ms: float = 120.0
    cold_start_latency_ms: float = 900.0
    sla_latency_ms: float = 800.0
    warm_instance_cost: float = 0.2
    queue_penalty_ms: float = 30.0
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


@dataclass(frozen=True)
class ExperimentConfig:
    horizon: int = 240
    seeds: Sequence[int] = (7, 11, 19)
    workload_names: Sequence[str] = (
        "steady_low",
        "steady_medium",
        "bursty",
        "periodic_spikes",
        "abrupt_shift",
    )

