"""Tests for offline ERA5 pressure-axis construction."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).parents[2]
    / "experiments"
    / "field_estimation"
    / "scripts"
    / "fetch_era5.py"
)
SPEC = importlib.util.spec_from_file_location("fetch_era5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pressure_axis_increases_in_altitude():
    levels = MODULE.pressure_levels_for_grid([50, 100, 200, 300], 4)

    np.testing.assert_array_equal(levels, [300.0, 200.0, 100.0, 50.0])
    assert np.all(np.diff(levels) < 0)


def test_pressure_axis_resamples_in_log_pressure():
    levels = MODULE.pressure_levels_for_grid([50, 300], 5)

    assert levels[0] == pytest.approx(300.0)
    assert levels[-1] == pytest.approx(50.0)
    np.testing.assert_allclose(np.diff(np.log(levels)), np.diff(np.log(levels))[0])
