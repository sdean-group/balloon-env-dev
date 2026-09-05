from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

MODULE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion"
)
sys.path.insert(0, str(MODULE_DIR))

from generate_direct_base_condition_set import sample_direct_block  # noqa: E402


class _IdentityStats:
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _IdentitySampler:
    device = torch.device("cpu")
    n_channels = 2
    n_levels = 1
    tau = 2
    stats = _IdentityStats()

    def _condition(self, hw, lat, lon, times):  # noqa: ARG002
        height, width = hw
        return torch.zeros(1, 2, height, width), torch.zeros(1, self.tau, 6)

    def _heun_block(self, x, cond=None, tfeat=None):  # noqa: ARG002
        return x


def test_direct_base_block_uses_coordinate_keyed_noise() -> None:
    sampler = _IdentitySampler()
    kwargs = {
        "sampler": sampler,
        "lat": np.arange(4, dtype=float),
        "lon": np.arange(4, dtype=float),
        "times": np.arange(2),
        "global_t0": 2,
        "global_y0": 32,
        "global_x0": 32,
    }

    u1, v1 = sample_direct_block(seed=7, **kwargs)
    u2, v2 = sample_direct_block(seed=7, **kwargs)
    u3, v3 = sample_direct_block(seed=8, **kwargs)

    assert u1.shape == v1.shape == (2, 1, 4, 4)
    assert np.array_equal(u1, u2)
    assert np.array_equal(v1, v2)
    assert not np.array_equal(u1, u3)
    assert not np.array_equal(v1, v3)
