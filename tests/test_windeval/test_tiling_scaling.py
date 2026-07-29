"""Unit tests for the fixed-domain tile-scaling protocol."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODULE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion/tiling_scaling"
)
sys.path.insert(0, str(MODULE_DIR))

from protocol import PROFILES, boundary_coherence, coherence_length_km  # noqa: E402


def test_profiles_define_requested_core_counts_and_half_overlap() -> None:
    assert set(PROFILES) == {4, 16, 64}
    for count, profile in PROFILES.items():
        assert profile.tiles_per_axis**2 == count
        assert profile.window == 2 * profile.stride
        assert 64 // profile.stride == profile.tiles_per_axis


def test_boundary_coherence_is_one_for_uniform_linear_jump() -> None:
    x = np.arange(64, dtype=np.float64)
    u = np.broadcast_to(x, (2, 3, 64, 64)).copy()
    v = np.ones_like(u)
    metrics = boundary_coherence(u, v, stride=16)
    assert np.isclose(metrics["boundary jump ratio"], 1.0)
    assert np.isclose(metrics["boundary squared-jump ratio"], 1.0)
    assert abs(metrics["boundary direction gap"]) < 5e-3


def test_boundary_coherence_detects_inserted_seam() -> None:
    rng = np.random.default_rng(4)
    u = rng.normal(size=(2, 3, 64, 64))
    v = rng.normal(size=(2, 3, 64, 64))
    u[..., 32:] += 20.0
    metrics = boundary_coherence(u, v, stride=32)
    assert metrics["boundary jump ratio"] > 3.0
    assert metrics["boundary squared-jump ratio"] > metrics["boundary jump ratio"]


def test_correlation_length_distinguishes_smooth_and_white_fields() -> None:
    rng = np.random.default_rng(9)
    white_u = rng.normal(size=(4, 2, 64, 64))
    white_v = rng.normal(size=(4, 2, 64, 64))
    coarse_u = rng.normal(size=(4, 2, 8, 8))
    coarse_v = rng.normal(size=(4, 2, 8, 8))
    smooth_u = np.repeat(np.repeat(coarse_u, 8, axis=-2), 8, axis=-1)
    smooth_v = np.repeat(np.repeat(coarse_v, 8, axis=-2), 8, axis=-1)
    white_length, _ = coherence_length_km(white_u, white_v)
    smooth_length, _ = coherence_length_km(smooth_u, smooth_v)
    assert smooth_length > white_length
