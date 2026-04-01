from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np


@dataclass(frozen=True)
class WorkloadTrace:
    name: str
    requests: np.ndarray


def _poisson(rng: np.random.Generator, lam: float, horizon: int) -> np.ndarray:
    return rng.poisson(lam=lam, size=horizon).astype(int)


def steady_low(horizon: int, seed: int) -> WorkloadTrace:
    rng = np.random.default_rng(seed)
    return WorkloadTrace("steady_low", _poisson(rng, lam=18, horizon=horizon))


def steady_medium(horizon: int, seed: int) -> WorkloadTrace:
    rng = np.random.default_rng(seed)
    return WorkloadTrace("steady_medium", _poisson(rng, lam=48, horizon=horizon))


def bursty(horizon: int, seed: int) -> WorkloadTrace:
    rng = np.random.default_rng(seed)
    base = _poisson(rng, lam=20, horizon=horizon)
    burst_flags = rng.binomial(1, p=0.14, size=horizon)
    burst_sizes = rng.integers(40, 120, size=horizon)
    requests = base + burst_flags * burst_sizes
    return WorkloadTrace("bursty", requests.astype(int))


def periodic_spikes(horizon: int, seed: int) -> WorkloadTrace:
    rng = np.random.default_rng(seed)
    t = np.arange(horizon)
    periodic = 22 + 18 * (1 + np.sin(2 * np.pi * t / 48))
    noise = rng.poisson(4, size=horizon)
    spikes = ((t % 60) < 6).astype(int) * rng.integers(30, 90, size=horizon)
    requests = np.maximum(0, periodic.astype(int) + noise + spikes)
    return WorkloadTrace("periodic_spikes", requests.astype(int))


def abrupt_shift(horizon: int, seed: int) -> WorkloadTrace:
    rng = np.random.default_rng(seed)
    split = horizon // 2
    before = rng.poisson(16, size=split)
    after = rng.poisson(72, size=horizon - split)
    requests = np.concatenate([before, after])
    shock_indices = rng.choice(horizon, size=max(1, horizon // 15), replace=False)
    requests[shock_indices] += rng.integers(25, 80, size=shock_indices.shape[0])
    return WorkloadTrace("abrupt_shift", requests.astype(int))


WORKLOAD_BUILDERS: Dict[str, Callable[[int, int], WorkloadTrace]] = {
    "steady_low": steady_low,
    "steady_medium": steady_medium,
    "bursty": bursty,
    "periodic_spikes": periodic_spikes,
    "abrupt_shift": abrupt_shift,
}


def build_workload(name: str, horizon: int, seed: int) -> WorkloadTrace:
    if name not in WORKLOAD_BUILDERS:
        raise ValueError(f"Unknown workload '{name}'. Available: {sorted(WORKLOAD_BUILDERS)}")
    return WORKLOAD_BUILDERS[name](horizon=horizon, seed=seed)
