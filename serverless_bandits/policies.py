from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


class Policy:
    def __init__(self, action_space: Sequence[int], name: str):
        self.action_space = list(action_space)
        self.name = name

    def reset(self) -> None:
        """Reset any internal state before a new run."""

    def select_action(self, context: np.ndarray, t: int) -> int:
        raise NotImplementedError

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        """Update state after observing reward."""


class FixedPolicy(Policy):
    def __init__(self, action_space: Sequence[int], action: int):
        super().__init__(action_space, name=f"always_{action}")
        self.fixed_action = action

    def select_action(self, context: np.ndarray, t: int) -> int:
        return self.fixed_action


class ThresholdPolicy(Policy):
    def __init__(
        self,
        action_space: Sequence[int],
        low_threshold: float = 20.0,
        medium_threshold: float = 60.0,
        high_threshold: float = 120.0,
    ):
        super().__init__(action_space, name="threshold")
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold

    def select_action(self, context: np.ndarray, t: int) -> int:
        moving_avg = context[2]
        latest = context[1]
        burstiness = context[4]
        signal = moving_avg + latest * 0.25 + burstiness * 10.0
        if signal < self.low_threshold:
            return 0
        if signal < self.medium_threshold:
            return 1
        if signal < self.high_threshold:
            return 2
        return 4


class EpsilonGreedyLinearPolicy(Policy):
    def __init__(self, action_space: Sequence[int], epsilon: float = 0.1):
        super().__init__(action_space, name="epsilon_greedy_linear")
        self.epsilon = epsilon
        self.rng = np.random.default_rng(0)
        self._state: Dict[int, Dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._state = {}

    def _ensure(self, dimension: int) -> None:
        if self._state:
            return
        for action in self.action_space:
            self._state[action] = {
                "A": np.eye(dimension),
                "b": np.zeros(dimension),
            }

    def select_action(self, context: np.ndarray, t: int) -> int:
        self._ensure(context.shape[0])
        if self.rng.random() < self.epsilon:
            return int(self.rng.choice(self.action_space))

        scores = []
        for action in self.action_space:
            inv_a = np.linalg.inv(self._state[action]["A"])
            theta = inv_a @ self._state[action]["b"]
            scores.append(float(theta @ context))
        return self.action_space[int(np.argmax(scores))]

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._ensure(context.shape[0])
        self._state[action]["A"] += np.outer(context, context)
        self._state[action]["b"] += reward * context


class LinUCBPolicy(Policy):
    def __init__(self, action_space: Sequence[int], alpha: float = 0.8):
        super().__init__(action_space, name="linucb")
        self.alpha = alpha
        self._state: Dict[int, Dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._state = {}

    def _ensure(self, dimension: int) -> None:
        if self._state:
            return
        for action in self.action_space:
            self._state[action] = {
                "A": np.eye(dimension),
                "b": np.zeros(dimension),
            }

    def select_action(self, context: np.ndarray, t: int) -> int:
        self._ensure(context.shape[0])
        ucb_scores: List[float] = []
        for action in self.action_space:
            inv_a = np.linalg.inv(self._state[action]["A"])
            theta = inv_a @ self._state[action]["b"]
            mean = float(theta @ context)
            bonus = self.alpha * float(np.sqrt(context @ inv_a @ context))
            ucb_scores.append(mean + bonus)
        return self.action_space[int(np.argmax(ucb_scores))]

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._ensure(context.shape[0])
        self._state[action]["A"] += np.outer(context, context)
        self._state[action]["b"] += reward * context


class LinearThompsonSamplingPolicy(Policy):
    def __init__(self, action_space: Sequence[int], variance: float = 0.3):
        super().__init__(action_space, name="linear_thompson_sampling")
        self.variance = variance
        self.rng = np.random.default_rng(1)
        self._state: Dict[int, Dict[str, np.ndarray]] = {}

    def reset(self) -> None:
        self._state = {}

    def _ensure(self, dimension: int) -> None:
        if self._state:
            return
        for action in self.action_space:
            self._state[action] = {
                "A": np.eye(dimension),
                "b": np.zeros(dimension),
            }

    def select_action(self, context: np.ndarray, t: int) -> int:
        self._ensure(context.shape[0])
        samples: List[float] = []
        for action in self.action_space:
            inv_a = np.linalg.inv(self._state[action]["A"])
            mean = inv_a @ self._state[action]["b"]
            covariance = (self.variance**2) * inv_a
            theta_sample = self.rng.multivariate_normal(mean, covariance)
            samples.append(float(theta_sample @ context))
        return self.action_space[int(np.argmax(samples))]

    def update(self, context: np.ndarray, action: int, reward: float) -> None:
        self._ensure(context.shape[0])
        self._state[action]["A"] += np.outer(context, context)
        self._state[action]["b"] += reward * context


def build_default_policies(action_space: Sequence[int]) -> Iterable[Policy]:
    return [
        FixedPolicy(action_space, 0),
        FixedPolicy(action_space, 1),
        FixedPolicy(action_space, 2),
        ThresholdPolicy(action_space),
        LinUCBPolicy(action_space),
        LinearThompsonSamplingPolicy(action_space),
        EpsilonGreedyLinearPolicy(action_space),
    ]

