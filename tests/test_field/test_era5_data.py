"""Loader / data-integrity tests for the ERA5 cache (criterion 9).

Contract assumed (see plan section 4b):
    load_era5(path) -> Era5Bundle  with attributes ``.winds`` (np.ndarray) and ``.meta`` (dict).
The loader validates that the data is finite and that the trailing component axis is
consistent with the spatial ndim (1 for 2D, 2 for 3D); it raises ValueError otherwise.
"""

import numpy as np
import pytest

from src.env.field.era5_data import load_era5


def test_load_returns_array_and_meta(affine_npz_2d):
    bundle = load_era5(affine_npz_2d["path"])
    assert bundle.winds.shape == (
        affine_npz_2d["T"], affine_npz_2d["n_x"], affine_npz_2d["n_y"], 1
    )
    assert bundle.meta["units"] == "m/s"


def test_2d_has_one_component(affine_npz_2d):
    bundle = load_era5(affine_npz_2d["path"])
    assert bundle.winds.shape[-1] == 1


def test_3d_has_two_components(affine_npz_3d):
    bundle = load_era5(affine_npz_3d["path"])
    assert bundle.winds.shape == (
        affine_npz_3d["T"], affine_npz_3d["n_x"],
        affine_npz_3d["n_y"], affine_npz_3d["n_z"], 2,
    )
    assert bundle.winds.shape[-1] == 2


def test_nan_in_data_rejected(nan_npz):
    with pytest.raises(ValueError):
        load_era5(nan_npz)


def test_all_values_finite(affine_npz_3d):
    bundle = load_era5(affine_npz_3d["path"])
    assert np.all(np.isfinite(bundle.winds))
