from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODULE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion/tiling_scaling"
)
sys.path.insert(0, str(MODULE_ROOT))

from eye_test import (  # noqa: E402
    _boundary_atlas,
    _multilevel_atlas,
    _seasonal_atlas,
)


def _field(offset: float) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.mgrid[:64, :64]
    levels = np.arange(3, dtype=np.float32)[:, None, None]
    u = 4.0 + offset + 0.04 * x[None] + 0.2 * levels
    v = -1.0 + 0.03 * y[None] - 0.1 * levels
    return u.astype(np.float32), v.astype(np.float32)


def test_eye_test_figures_render(tmp_path: Path) -> None:
    keys = [(month, 12, 12, 0) for month in (1, 4, 7, 10)]
    records = {}
    for count, offset in ((4, 0.1), (16, 0.2), (64, 0.3)):
        records[count] = {}
        for key in keys:
            u, v = _field(offset + key[0] / 100.0)
            records[count][key] = {
                "u": u,
                "v": v,
                "levels": np.array([49, 58, 66]),
            }
    reference = {}
    for month, day, hour, _ in keys:
        reference[(month, day, hour)] = _field(month / 100.0)

    _seasonal_atlas(records, reference, keys, level_index=1, output=tmp_path)
    _multilevel_atlas(
        records, reference, keys[0], level_indices=(0, 1, 2), output=tmp_path
    )
    _boundary_atlas(records, reference, keys[0], level_index=1, output=tmp_path)

    for name in (
        "eye_test_seasons.png",
        "eye_test_levels.png",
        "eye_test_boundaries.png",
    ):
        assert (tmp_path / name).stat().st_size > 10_000
