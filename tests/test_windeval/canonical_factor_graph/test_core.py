"""Mechanical guarantees for Canonical Factor-Graph Diffusion."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

GENERATOR_DIR = Path(__file__).resolve().parents[3] / "src/eval/windeval/generators"
sys.path.insert(0, str(GENERATOR_DIR))

from canonical_factor_graph.core import (  # noqa: E402
    CanonicalFactorGraphField,
    ChartConfig,
    _coordinate_noise,
)


class _IdentityStats:
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ToyModel(torch.nn.Module):
    def forward(self, x, sigma, cond=None, tfeat=None):  # noqa: ARG002
        scale = 1.0 / (1.0 + sigma.reshape(-1, 1, 1, 1, 1))
        return scale * x


class _ToySampler:
    device = torch.device("cpu")
    n_channels = 2
    n_levels = 1
    tau = 2
    num_steps = 2
    conditional = True
    model = _ToyModel()
    stats = _IdentityStats()

    def sigma_schedule(self, *, device=None, dtype=torch.float64):
        return torch.tensor([1.0, 0.5, 0.0], device=device or self.device, dtype=dtype)

    def _condition(self, hw, lat, lon, times):  # noqa: ARG002
        h, w = hw
        return torch.zeros(1, 2, h, w), torch.zeros(1, self.tau, 6)


def _field(seed: int = 5, max_cached_charts: int = 16):
    return CanonicalFactorGraphField(
        _ToySampler(),
        config=ChartConfig(
            core_time=2,
            core_size=4,
            halo_time=0,
            halo_size=2,
            window_size=4,
            window_stride=2,
            time_stride=1,
            window_batch_size=3,
        ),
        seed=seed,
        max_cached_charts=max_cached_charts,
    )


def test_coordinate_noise_is_shared_in_overlap() -> None:
    left = _coordinate_noise(2, 2, 4, 4, t0=0, y0=0, x0=0, seed=11,
                             device=torch.device("cpu"), dtype=torch.float32)
    right = _coordinate_noise(2, 2, 4, 4, t0=0, y0=0, x0=2, seed=11,
                              device=torch.device("cpu"), dtype=torch.float32)
    assert torch.equal(left[..., 2:], right[..., :2])


def test_shape_exact_repeat_and_cache() -> None:
    field = _field()
    first = field.materialize(0, 2, 1, 5, 1, 5)
    generated = field.charts_generated
    second = field.materialize(0, 2, 1, 5, 1, 5)
    assert first.shape == (2, 2, 4, 4)
    assert torch.equal(first, second)
    assert field.charts_generated == generated
    assert field.chart_cache_hits > 0


def test_query_order_and_subquery_consistency() -> None:
    expected = _field(17).materialize(0, 2, 1, 5, 1, 5)

    reordered = _field(17)
    _ = reordered.materialize(0, 2, 20, 24, -12, -8)
    larger = reordered.materialize(0, 2, 0, 8, 0, 8)
    assert torch.equal(expected, larger[:, :, 1:5, 1:5])


def test_seed_changes_field_and_eviction_recomputes_exactly() -> None:
    field = _field(23, max_cached_charts=1)
    first = field.materialize(0, 2, 1, 3, 1, 3)
    _ = field.materialize(0, 2, 20, 22, 20, 22)
    recomputed = field.materialize(0, 2, 1, 3, 1, 3)
    other = _field(24).materialize(0, 2, 1, 3, 1, 3)
    assert torch.equal(first, recomputed)
    assert not torch.equal(first, other)


def test_model_evaluation_count_matches_fixed_factor_graph() -> None:
    field = _field()
    keys = field.chart_keys_for_query(0, 2, 1, 3, 1, 3)
    _ = field.materialize(0, 2, 1, 3, 1, 3)
    factors_per_chart = 9
    heun_evaluations = 2 * field.sampler.num_steps - 1
    assert field.model_window_evaluations == len(keys) * factors_per_chart * heun_evaluations
    assert np.isfinite(field.materialize(0, 2, 1, 3, 1, 3).numpy()).all()


def test_heldout_condition_geometry_has_expected_bounded_work() -> None:
    class ConditionSampler(_ToySampler):
        tau = 4

        def sigma_schedule(self, *, device=None, dtype=torch.float64):
            return torch.tensor([1.0, 0.5, 0.0], device=device or self.device, dtype=dtype)

        def _condition(self, hw, lat, lon, times):  # noqa: ARG002
            h, w = hw
            return torch.zeros(1, 2, h, w), torch.zeros(1, self.tau, 6)

    field = CanonicalFactorGraphField(
        ConditionSampler(),
        config=ChartConfig(
            core_time=2,
            core_size=80,
            halo_time=1,
            halo_size=8,
            window_size=64,
            window_stride=32,
            time_stride=2,
        ),
    )
    keys = field.chart_keys_for_query(1, 5, 32, 48, 32, 48)
    _ = field.materialize(1, 5, 32, 48, 32, 48)

    assert keys == [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
    assert len(field._factor_offsets) == 4
    assert field.model_window_evaluations == 3 * 4 * (2 * field.sampler.num_steps - 1)
