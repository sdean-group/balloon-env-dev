"""Small dependency-free statistical helpers for paired architecture comparisons."""
from __future__ import annotations

import numpy as np


def bootstrap_summary(values: list[float], *, seed: int = 20260726) -> dict:
    """Summarize paired deltas with a deterministic percentile bootstrap interval."""
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        raise ValueError("cannot summarize an empty paired sample")
    if len(array) == 1:
        low = high = float(array[0])
    else:
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(array), size=(20_000, len(array)))
        means = array[indices].mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n_seeds": int(len(array)),
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "per_seed": array.tolist(),
    }
