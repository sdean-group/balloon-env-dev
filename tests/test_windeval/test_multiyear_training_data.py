"""Lazy multi-year training data must match the original eager data path exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

MODULE_DIR = Path(__file__).resolve().parents[2] / "src/eval/windeval/generators/infinite_diffusion"
sys.path.insert(0, str(MODULE_DIR))

from data import (  # noqa: E402
    WindCondSpaceTimeDataset,
    compute_stats,
    compute_zarr_stats,
)


def _store(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    u = rng.normal(3.0, 2.0, size=(8, 2, 6, 6)).astype(np.float32)
    v = rng.normal(-1.0, 4.0, size=(8, 2, 6, 6)).astype(np.float32)
    ds = xr.Dataset(
        {"u": (("time", "level", "y", "x"), u),
         "v": (("time", "level", "y", "x"), v)},
        coords={
            "time": np.arange("2020-01-01T00", "2020-01-01T08", dtype="datetime64[h]"),
            "level": [49, 50],
            "lat": ("y", np.linspace(25, 26.25, 6)),
            "lon": ("x", np.linspace(225, 226.25, 6)),
        },
    )
    path = tmp_path / "tiny.zarr"
    ds.to_zarr(path, mode="w", consolidated=False)
    return path, u, v


def test_streaming_stats_match_eager_stats(tmp_path: Path) -> None:
    path, u, v = _store(tmp_path)
    eager = compute_stats(u, v, np.array([49, 50]))
    streamed = compute_zarr_stats(path, levels=(49, 50), time_chunk=3)
    np.testing.assert_allclose(streamed.mean_u, eager.mean_u, rtol=1e-6)
    np.testing.assert_allclose(streamed.std_u, eager.std_u, rtol=1e-6)
    np.testing.assert_allclose(streamed.mean_v, eager.mean_v, rtol=1e-6)
    np.testing.assert_allclose(streamed.std_v, eager.std_v, rtol=1e-6)


def test_lazy_conditional_sample_matches_eager_sample(tmp_path: Path) -> None:
    path, u, v = _store(tmp_path)
    stats = compute_stats(u, v, np.array([49, 50]))
    common = dict(crop=4, levels=(49, 50), n_frames=3, frame_stride=1,
                  stats=stats, length=4, seed=19)
    eager = WindCondSpaceTimeDataset(path, lazy=False, **common)
    lazy = WindCondSpaceTimeDataset(path, lazy=True, **common)

    eager_x, eager_coords, eager_time = eager[2]
    lazy_x, lazy_coords, lazy_time = lazy[2]
    np.testing.assert_allclose(lazy_x.numpy(), eager_x.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(lazy_coords.numpy(), eager_coords.numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(lazy_time.numpy(), eager_time.numpy(), rtol=0, atol=0)
