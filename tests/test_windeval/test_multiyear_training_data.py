"""Lazy multi-year training data must match the original eager data path exactly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

MODULE_DIR = Path(__file__).resolve().parents[2] / "src/eval/windeval/generators/infinite_diffusion"
sys.path.insert(0, str(MODULE_DIR))

from data import (  # noqa: E402
    WindCondSpaceTimeDataset,
    compute_stats,
    compute_zarr_stats,
)
from validate_era5_multiyear import (  # noqa: E402
    expected_hourly_times,
    validate_hourly_times,
)
from prepare_era5_multiyear import _month_specs  # noqa: E402
from train import TrainConfig, save_ckpt  # noqa: E402


CONFIG_DIR = MODULE_DIR / "configs"


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
    # Pin the v2 layout explicitly: data._open_zarr reads zarr_format=2, and zarr>=3 writes
    # v3 by default, which makes this fixture unreadable on a zarr 3 environment. Older
    # xarray releases do not accept the kwarg, hence the fallback (same as _open_zarr).
    try:
        ds.to_zarr(path, mode="w", consolidated=False, zarr_format=2)
    except TypeError:
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


def test_streaming_stats_resume_matches_uninterrupted_scan(tmp_path: Path) -> None:
    path, u, v = _store(tmp_path)
    eager = compute_stats(u, v, np.array([49, 50]))
    progress = tmp_path / "stats.progress.npz"
    paused = compute_zarr_stats(
        path,
        levels=(49, 50),
        time_chunk=3,
        progress_path=progress,
        stop_requested=lambda: True,
    )
    assert paused is None
    assert progress.exists()

    resumed = compute_zarr_stats(
        path,
        levels=(49, 50),
        time_chunk=3,
        progress_path=progress,
    )
    assert resumed is not None
    np.testing.assert_allclose(resumed.mean_u, eager.mean_u, rtol=1e-6)
    np.testing.assert_allclose(resumed.std_u, eager.std_u, rtol=1e-6)
    np.testing.assert_allclose(resumed.mean_v, eager.mean_v, rtol=1e-6)
    np.testing.assert_allclose(resumed.std_v, eager.std_v, rtol=1e-6)
    assert not progress.exists()


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


def test_2018_2021_config_preserves_original_training_method() -> None:
    original = yaml.safe_load((CONFIG_DIR / "era5_2023_m2cond.yaml").read_text())
    scaled = yaml.safe_load((CONFIG_DIR / "era5_2018_2021_m2cond.yaml").read_text())

    for key in ("spacetime", "conditional", "n_frames", "frame_stride", "temporal_kernel"):
        assert scaled[key] == original[key]
    assert scaled["model"] == original["model"]
    for key in (
        "batch_size", "lr", "ema_decay", "n_steps", "warmup_steps",
        "num_workers", "ckpt_every", "log_every", "device", "seed",
    ):
        assert scaled["train"][key] == original["train"][key]
    assert scaled["data"]["crop"] == original["data"]["crop"]
    assert scaled["data"]["levels"] == original["data"]["levels"]


def test_expected_hourly_timeline_includes_leap_year_and_rejects_gaps() -> None:
    times = expected_hourly_times(2019, 2020)
    assert len(times) == 365 * 24 + 366 * 24
    validate_hourly_times(times, first_year=2019, last_year=2020)

    with pytest.raises(ValueError, match="incomplete or duplicated"):
        validate_hourly_times(
            np.delete(times, 10_000),
            first_year=2019,
            last_year=2020,
        )


def test_2018_2021_download_schedule_excludes_validation_and_benchmark_years() -> None:
    years = list(range(2018, 2022))
    specs = list(_month_specs(years))
    assert len(specs) == 4 * 12
    assert specs[0][:2] == (2018, 1)
    assert specs[-1][:2] == (2021, 12)
    assert all(year not in (2022, 2023) for year, _, _ in specs)


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters())
    stats = compute_stats(
        np.ones((2, 1, 2, 2), dtype=np.float32),
        np.ones((2, 1, 2, 2), dtype=np.float32),
        np.array([49]),
    )

    class _Ema:
        shadow = model

    destination = tmp_path / "latest.pt"
    save_ckpt(
        destination,
        model=model,
        ema=_Ema(),
        opt=optimizer,
        step=17,
        stats=stats,
        cfg=TrainConfig(),
    )
    payload = torch.load(destination, map_location="cpu", weights_only=False)
    assert payload["step"] == 17
    assert not destination.with_suffix(".pt.tmp").exists()
