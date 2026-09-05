"""Mechanical tests for canonical synchronized-diffusion strategies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

GENERATOR_DIR = Path(__file__).resolve().parents[3] / "src/eval/windeval/generators"
sys.path.insert(0, str(GENERATOR_DIR))

from canonical_factor_graph.core import ChartConfig  # noqa: E402
from synchronized_diffusion.core import SynchronizedChartField  # noqa: E402


class _IdentityStats:
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ConditionedToyModel(torch.nn.Module):
    def forward(self, x, sigma, cond=None, tfeat=None):
        scale = 1.0 / (1.0 + sigma.reshape(-1, 1, 1, 1, 1))
        bias = 0.0 if tfeat is None else 0.01 * tfeat[:, :1, :1, None, None]
        return scale * x + bias


class _ToySampler:
    device = torch.device("cpu")
    n_channels = 2
    n_levels = 1
    tau = 2
    num_steps = 2
    conditional = True
    model = _ConditionedToyModel()
    stats = _IdentityStats()

    def sigma_schedule(self, *, device=None, dtype=torch.float64):
        return torch.tensor(
            [1.0, 0.5, 0.0],
            device=device or self.device,
            dtype=dtype,
        )

    def _condition(self, hw, lat, lon, times):
        h, w = hw
        location = float(lat[0] + lon[0])
        return (
            torch.zeros(1, 2, h, w),
            torch.full((1, self.tau, 6), location),
        )


def _field(strategy: str) -> SynchronizedChartField:
    return SynchronizedChartField(
        _ToySampler(),
        strategy=strategy,
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
        guidance_strength=0.1,
        consensus_rounds=2,
        seed=13,
        max_cached_charts=4,
    )


@pytest.mark.parametrize(
    "strategy",
    ["sync_tweedies", "overlap_guided", "consensus_equilibrium"],
)
def test_strategies_are_finite_deterministic_and_crop_consistent(strategy: str) -> None:
    field = _field(strategy)
    larger = field.materialize(0, 2, 0, 6, 0, 6)
    repeated = field.materialize(0, 2, 1, 5, 1, 5)
    fresh = _field(strategy).materialize(0, 2, 1, 5, 1, 5)

    assert larger.shape == (2, 2, 6, 6)
    assert np.isfinite(larger.numpy()).all()
    assert torch.equal(repeated, larger[:, :, 1:5, 1:5])
    assert torch.equal(repeated, fresh)


@pytest.mark.parametrize(
    ("strategy", "evaluation_multiplier"),
    [
        ("sync_tweedies", 1),
        ("overlap_guided", 2),
        ("consensus_equilibrium", 2),
    ],
)
def test_strategy_model_work_is_fixed(
    strategy: str,
    evaluation_multiplier: int,
) -> None:
    field = _field(strategy)
    keys = field.chart_keys_for_query(0, 2, 1, 3, 1, 3)
    field.materialize(0, 2, 1, 3, 1, 3)

    factors = 9
    heun_evaluations = 2 * field.sampler.num_steps - 1
    assert field.model_window_evaluations == (
        len(keys) * factors * heun_evaluations * evaluation_multiplier
    )
