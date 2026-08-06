from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from windeval import artifact
from windeval.benchmark_temporal_all_methods import (
    _mean_adjacent_change,
    _seam_jump_ratio,
    _write_report,
)
from windeval.download_arco_temporal_reference import _times
from windeval.metrics.temporal import contiguous_segments


def _field_with_four_episodes() -> object:
    episode_times = []
    for month in (1, 4, 7, 10):
        start = np.datetime64(f"2023-{month:02d}-08T00", "h")
        episode_times.extend(start + np.arange(24) * np.timedelta64(1, "h"))
    times = np.asarray(episode_times)
    signal = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    u = np.broadcast_to(signal, (len(times), 2, 4, 4)).copy()
    v = np.zeros_like(u)
    return artifact.make_field(
        u,
        v,
        level=np.array([60.0, 70.0]),
        lat=np.linspace(35.0, 36.5, 4),
        lon=np.linspace(235.0, 236.5, 4),
        time=times,
    )


def test_temporal_reference_times_are_hourly() -> None:
    times = _times(1, 8)
    assert len(times) == 24
    assert np.all(np.diff(times) == np.timedelta64(1, "h"))


def test_temporal_diagnostics_preserve_episode_boundaries() -> None:
    field = _field_with_four_episodes()
    assert len(contiguous_segments(field)) == 4
    assert _mean_adjacent_change(field) == 1.0
    assert _seam_jump_ratio(field, 4) == 1.0


def test_temporal_report_writes_markdown_and_json(tmp_path: Path) -> None:
    output = tmp_path / "temporal.md"
    rows = {
        "ERA5 self-split floor": {
            "SR_time": 0.1,
            "disp log-MSD RMSE": 0.2,
            "final spread ratio": 1.0,
            "mean adjacent change (m/s)": 2.0,
            "time seam jump ratio": np.nan,
        },
        "BLE-VAE": {
            "SR_time": np.nan,
            "disp log-MSD RMSE": np.nan,
            "final spread ratio": np.nan,
            "mean adjacent change (m/s)": np.nan,
            "time seam jump ratio": np.nan,
        },
    }
    _write_report(rows, output)
    assert "BLE-VAE" in output.read_text()
    assert "N/A" in output.read_text()
    assert json.loads(output.with_suffix(".json").read_text())[
        "ERA5 self-split floor"
    ]["SR_time"] == 0.1
