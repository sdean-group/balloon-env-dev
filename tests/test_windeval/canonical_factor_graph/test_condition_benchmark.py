from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = (
    Path(__file__).resolve().parents[3]
    / "src/eval/windeval/generators/canonical_factor_graph"
)
sys.path.insert(0, str(MODULE_DIR))

from paired_statistics import bootstrap_summary  # noqa: E402


def test_bootstrap_summary_preserves_paired_deltas() -> None:
    summary = bootstrap_summary([-1.0, -2.0, -3.0, -4.0, -5.0])

    assert summary["n_seeds"] == 5
    assert summary["mean"] == -3.0
    assert summary["sample_std"] == np.std([-1, -2, -3, -4, -5], ddof=1)
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    assert summary["ci95_high"] < 0.0


def test_single_value_bootstrap_is_degenerate() -> None:
    summary = bootstrap_summary([1.25])

    assert summary["ci95_low"] == 1.25
    assert summary["ci95_high"] == 1.25
