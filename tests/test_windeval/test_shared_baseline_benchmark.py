from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVAL_SRC = Path(__file__).resolve().parents[2] / "src/eval"
sys.path.insert(0, str(EVAL_SRC))

from windeval import artifact  # noqa: E402
from windeval.benchmark_shared_baselines import (  # noqa: E402
    _ensure_contains_grid,
    _vertical_interpolate,
)


def _field(levels: np.ndarray):
    values = np.broadcast_to(
        levels.reshape(1, -1, 1, 1),
        (2, len(levels), 4, 4),
    ).astype(np.float32)
    return artifact.make_field(
        values,
        -2.0 * values,
        level=levels,
        lat=np.arange(4.0),
        lon=np.arange(10.0, 14.0),
        time=np.arange(2),
    )


def test_vertical_interpolation_uses_pressure_coordinates() -> None:
    source = np.array([50.0, 100.0, 150.0])
    result = _vertical_interpolate(
        _field(source),
        source,
        np.array([75.0, 125.0]),
    )

    np.testing.assert_allclose(result["u"].values[:, :, 0, 0], [[75.0, 125.0]] * 2)
    np.testing.assert_allclose(result["v"].values[:, :, 0, 0], [[-150.0, -250.0]] * 2)
    np.testing.assert_array_equal(result["level"].values, [75.0, 125.0])


def test_shared_grid_must_be_inside_source() -> None:
    ds = _field(np.array([50.0, 100.0]))
    _ensure_contains_grid(ds, np.array([0.5, 2.5]), np.array([10.5, 12.5]))

    try:
        _ensure_contains_grid(ds, np.array([-0.5, 2.5]), np.array([10.5, 12.5]))
    except ValueError as error:
        assert "complete shared grid" in str(error)
    else:
        raise AssertionError("out-of-domain target grid should fail")
