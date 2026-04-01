from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Sequence

import numpy as np

from serverless_bandits.config import EnvironmentConfig


@dataclass
class StepResult:
    reward: float
    mean_latency_ms: float
    p95_latency_ms: float
    sla_violations: int
    cold_starts: int
    warm_instances: int
    cost: float
    demand: int
    capacity: int
    action: int


class ServerlessEnvironment:
    def __init__(self, config: EnvironmentConfig, horizon: int):
        self.config = config
        self.horizon = horizon
        self.reset()

    def reset(self) -> None:
        self.t = 0
        self.prev_demand = 0
        self.prev_mean_latency = self.config.base_latency_ms
        self.prev_cold_starts = 0
        self.prev_sla_violations = 0
        self.history_requests: Deque[int] = deque([0] * 5, maxlen=5)
        self.history_latency: Deque[float] = deque(
            [self.config.base_latency_ms] * 5,
            maxlen=5,
        )
        self.history_cold_starts: Deque[int] = deque([0] * 5, maxlen=5)
        self.last_step: StepResult | None = None

    def context(self) -> np.ndarray:
        requests = np.array(self.history_requests, dtype=float)
        latencies = np.array(self.history_latency, dtype=float)
        cold_starts = np.array(self.history_cold_starts, dtype=float)
        latest = requests[-1]
        moving_avg = requests.mean()
        moving_std = requests.std()
        burstiness = moving_std / (moving_avg + 1.0)
        trend = latest - requests[0]
        time_phase = 2 * np.pi * (self.t % 48) / 48
        return np.array(
            [
                1.0,
                latest,
                moving_avg,
                moving_std,
                burstiness,
                trend,
                latencies[-1],
                latencies.mean(),
                cold_starts[-1],
                cold_starts.mean(),
                self.prev_sla_violations,
                np.sin(time_phase),
                np.cos(time_phase),
            ],
            dtype=float,
        )

    def step(self, demand: int, action: int) -> StepResult:
        capacity = action * self.config.requests_per_instance
        overflow = max(0, demand - capacity)
        warm_served = min(demand, capacity)
        cold_served = overflow

        queue_penalty = (overflow / max(1, self.config.requests_per_instance)) * self.config.queue_penalty_ms
        mean_latency = self.config.base_latency_ms
        if demand > 0:
            mean_latency = (
                (warm_served * self.config.base_latency_ms)
                + (cold_served * (self.config.base_latency_ms + self.config.cold_start_latency_ms))
            ) / demand
            mean_latency += queue_penalty

        if cold_served > 0:
            p95_latency = self.config.base_latency_ms + self.config.cold_start_latency_ms + queue_penalty
        else:
            p95_latency = self.config.base_latency_ms + queue_penalty

        sla_violations = cold_served + int(p95_latency > self.config.sla_latency_ms)
        cost = action * self.config.warm_instance_cost
        weights = self.config.reward_weights
        reward = -(
            weights.latency * (mean_latency / 1000.0)
            + weights.sla_violation * sla_violations
            + weights.cold_start * cold_served
            + weights.warm_cost * cost
        )

        result = StepResult(
            reward=reward,
            mean_latency_ms=mean_latency,
            p95_latency_ms=p95_latency,
            sla_violations=sla_violations,
            cold_starts=cold_served,
            warm_instances=action,
            cost=cost,
            demand=demand,
            capacity=capacity,
            action=action,
        )

        self.prev_demand = demand
        self.prev_mean_latency = mean_latency
        self.prev_cold_starts = cold_served
        self.prev_sla_violations = sla_violations
        self.history_requests.append(demand)
        self.history_latency.append(mean_latency)
        self.history_cold_starts.append(cold_served)
        self.t += 1
        self.last_step = result
        return result


def cumulative_regret(step_rewards: Sequence[float]) -> np.ndarray:
    rewards = np.array(step_rewards, dtype=float)
    if rewards.size == 0:
        return rewards
    best_fixed = rewards.max()
    return np.cumsum(best_fixed - rewards)


def summarize_results(results: List[StepResult]) -> Dict[str, float]:
    rewards = np.array([step.reward for step in results], dtype=float)
    mean_latencies = np.array([step.mean_latency_ms for step in results], dtype=float)
    p95_latencies = np.array([step.p95_latency_ms for step in results], dtype=float)
    sla_violations = np.array([step.sla_violations for step in results], dtype=float)
    cold_starts = np.array([step.cold_starts for step in results], dtype=float)
    costs = np.array([step.cost for step in results], dtype=float)
    regrets = cumulative_regret(rewards.tolist())

    return {
        "avg_latency_ms": float(mean_latencies.mean()),
        "p95_latency_ms": float(np.percentile(p95_latencies, 95)),
        "sla_violation_rate": float(sla_violations.sum() / max(1, len(results))),
        "cold_starts": float(cold_starts.sum()),
        "total_cost": float(costs.sum()),
        "cumulative_reward": float(rewards.sum()),
        "cumulative_regret": float(regrets[-1] if len(regrets) else 0.0),
    }
