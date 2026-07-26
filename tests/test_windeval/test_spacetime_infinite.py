"""Mechanical guarantees for the four-dimensional InfiniteDiffusion wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MODULE_DIR = Path(__file__).resolve().parents[2] / "src/eval/windeval/generators/infinite_diffusion"
sys.path.insert(0, str(MODULE_DIR))

from spacetime_infinite import (  # noqa: E402
    InfiniteSpaceTimeDiffusion,
    SpaceTimeGrid,
    _coordinate_noise,
)
from spacetime import SpaceTimeSampler  # noqa: E402


class _IdentityStats:
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _IdentitySampler:
    conditional = True
    device = torch.device("cpu")
    n_channels = 2
    n_levels = 1
    tau = 2
    num_steps = 4
    stats = _IdentityStats()

    def _condition(self, hw, lat, lon, times):  # noqa: ARG002
        h, w = hw
        return torch.zeros(1, 2, h, w), torch.zeros(1, self.tau, 6)

    def _heun_block(self, x, cond=None, tfeat=None):  # noqa: ARG002
        return x

    def _heun_segment(self, x, *, start_step, end_step, unit_noise=False,
                      cond=None, tfeat=None):  # noqa: ARG002
        return x


class _ToyDenoiser:
    def __call__(self, x, sigma, cond=None, tfeat=None):  # noqa: ARG002
        scale = 1.0 / (1.0 + sigma.reshape(-1, 1, 1, 1, 1))
        return scale * x

def _field(
    seed: int = 11,
    outer_depth: int = 1,
    split_steps: tuple[int, ...] | None = None,
) -> InfiniteSpaceTimeDiffusion:
    return InfiniteSpaceTimeDiffusion(
        _IdentitySampler(),
        grid=SpaceTimeGrid(),
        window=4,
        stride=2,
        time_stride=1,
        seed=seed,
        outer_depth=outer_depth,
        split_step=2 if outer_depth == 2 and split_steps is None else None,
        split_steps=split_steps,
        cache_bytes=8 * 1024 * 1024,
    )


def test_coordinate_noise_is_shared_in_overlap() -> None:
    kwargs = {"t0": -1, "y0": 3, "seed": 19, "device": torch.device("cpu"),
              "dtype": torch.float32}
    left = _coordinate_noise(2, 2, 4, 4, x0=5, **kwargs)
    right = _coordinate_noise(2, 2, 4, 4, x0=7, **kwargs)

    assert torch.equal(left[..., 2:], right[..., :2])


def test_spacetime_shape_and_cached_repeat() -> None:
    field = _field()
    first = field.materialize(1, 3, 2, 6, 2, 6)
    calls = field.model_window_calls
    second = field.materialize(1, 3, 2, 6, 2, 6)
    assert first.shape == (2, 2, 4, 4)
    assert torch.equal(first, second)
    assert field.model_window_calls == calls


def test_spacetime_query_order_and_seed_consistency() -> None:
    direct = _field(17).materialize(1, 3, 2, 6, 2, 6)

    reordered = _field(17)
    _ = reordered.materialize(20, 22, -8, -4, 10, 14)
    larger = reordered.materialize(0, 4, 0, 8, 0, 8)
    assert torch.allclose(direct, larger[:, 1:3, 2:6, 2:6], atol=1e-6)

    same_seed = _field(17).materialize(1, 3, 2, 6, 2, 6)
    other_seed = _field(18).materialize(1, 3, 2, 6, 2, 6)
    assert torch.equal(direct, same_seed)
    assert not torch.allclose(direct, other_seed)


def test_field_uv_layout() -> None:
    u, v = _field().field_uv(0, 2, 0, 4, 0, 4)
    assert u.shape == v.shape == (2, 1, 4, 4)
    assert np.isfinite(u).all() and np.isfinite(v).all()


def test_t2_spacetime_shape_cache_and_query_order() -> None:
    field = _field(outer_depth=2)
    first = field.materialize(1, 3, 2, 6, 2, 6)
    calls = field.model_window_calls
    second = field.materialize(1, 3, 2, 6, 2, 6)

    assert first.shape == (2, 2, 4, 4)
    assert torch.equal(first, second)
    assert field.model_window_calls == calls
    assert field.phase_window_calls["initial"] > 0
    assert field.phase_window_calls["continuation"] > 0
    assert field.model_forward_evaluations > 0
    assert sum(field.phase_forward_evaluations.values()) == field.model_forward_evaluations

    reordered = _field(outer_depth=2)
    _ = reordered.materialize(20, 22, -8, -4, 10, 14)
    larger = reordered.materialize(0, 4, 0, 8, 0, 8)
    assert torch.allclose(first, larger[:, 1:3, 2:6, 2:6], atol=1e-6)


def test_t3_spacetime_shape_cache_query_order_and_phase_accounting() -> None:
    field = _field(outer_depth=3, split_steps=(1, 3))
    first = field.materialize(1, 3, 2, 6, 2, 6)
    calls = field.model_window_calls
    second = field.materialize(1, 3, 2, 6, 2, 6)

    assert first.shape == (2, 2, 4, 4)
    assert torch.equal(first, second)
    assert field.model_window_calls == calls
    assert field.split_steps == (1, 3)
    assert len(field.phases) == 3
    assert all(field.phase_window_calls[f"continuation_{index}"] > 0 for index in (1, 2))
    assert sum(field.phase_forward_evaluations.values()) == field.model_forward_evaluations

    reordered = _field(outer_depth=3, split_steps=(1, 3))
    _ = reordered.materialize(20, 22, -8, -4, 10, 14)
    larger = reordered.materialize(0, 4, 0, 8, 0, 8)
    assert torch.allclose(first, larger[:, 1:3, 2:6, 2:6], atol=1e-6)


def test_split_schedule_validation_and_even_defaults() -> None:
    assert _field(outer_depth=3).split_steps == (1, 3)
    assert _field(outer_depth=4).split_steps == (1, 2, 3)

    for split_steps in ((2,), (2, 2), (0, 3), (1, 4)):
        try:
            _field(outer_depth=3, split_steps=split_steps)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid split schedule accepted: {split_steps}")


def test_segmented_heun_matches_unsplit_trajectory_without_blending() -> None:
    sampler = object.__new__(SpaceTimeSampler)
    sampler.num_steps = 6
    sampler.sigma_min = 0.002
    sampler.sigma_max = 80.0
    sampler.model = _ToyDenoiser()
    unit_noise = torch.randn(1, 2, 2, 4, 4)

    full = sampler._heun_block(unit_noise)
    middle = sampler._heun_segment(
        unit_noise, start_step=0, end_step=3, unit_noise=True
    )
    segmented = sampler._heun_segment(
        middle, start_step=3, end_step=6, unit_noise=False
    )

    assert torch.equal(full, segmented)
