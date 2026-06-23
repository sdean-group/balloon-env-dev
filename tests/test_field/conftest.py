"""Shared fixtures for ReanalysisFlowField (real-wind linear-interpolation) tests.

The keystone trick: bilinear / trilinear interpolation reproduces an AFFINE field
(u = a*x + b*y [+ c*z] + d) *exactly*. So a fixture whose data is an affine function
gives a closed-form ground truth -- we can assert ``velocity_at`` to float tolerance
instead of eyeballing. No real ERA5 download is needed for the hermetic suite.

Each fixture writes a tiny ``.npz`` matching the cached-data contract from the design doc:
    2D: winds shape (T, n_x, n_y, 1)        component u only
    3D: winds shape (T, n_x, n_y, n_z, 2)   components (u, v)
and returns a dict with the path plus analytic truth callables for assertions.
"""

import numpy as np
import pytest


@pytest.fixture
def affine_npz_2d(tmp_path):
    """2D affine field u = 2*x + 3*y + 1, with a +10 offset per time slice.

    Linear interpolation reproduces this exactly, so ``truth(x, y, t)`` is the
    ground-truth value at any fractional (x, y) for slice t.
    """
    n_x, n_y, T = 12, 8, 4
    xs = np.arange(1, n_x + 1)[:, None]      # 1-indexed continuous domain
    ys = np.arange(1, n_y + 1)[None, :]
    base = 2.0 * xs + 3.0 * ys + 1.0         # (n_x, n_y)
    winds = np.stack([base + 10.0 * t for t in range(T)])  # (T, n_x, n_y)
    winds = winds[..., None].astype(np.float64)            # (T, n_x, n_y, 1)

    path = tmp_path / "affine2d.npz"
    np.savez(path, winds=winds, meta=dict(units="m/s", n=(n_x, n_y, None)))
    return {
        "path": str(path),
        "n_x": n_x, "n_y": n_y, "n_z": None, "T": T,
        "truth": lambda x, y, t=0: 2.0 * x + 3.0 * y + 1.0 + 10.0 * t,
    }


@pytest.fixture
def affine_npz_3d(tmp_path):
    """3D affine field u = x + 2y + 3z, v = 4x - y + 2z, with +t offset per slice."""
    n_x, n_y, n_z, T = 6, 5, 4, 3
    X, Y, Z = np.meshgrid(
        np.arange(1, n_x + 1), np.arange(1, n_y + 1), np.arange(1, n_z + 1),
        indexing="ij",
    )
    u = (X + 2.0 * Y + 3.0 * Z).astype(np.float64)
    v = (4.0 * X - Y + 2.0 * Z).astype(np.float64)
    winds = np.stack([np.stack([u + t, v + t], axis=-1) for t in range(T)])

    path = tmp_path / "affine3d.npz"
    np.savez(path, winds=winds, meta=dict(units="m/s", n=(n_x, n_y, n_z)))
    return {
        "path": str(path),
        "n_x": n_x, "n_y": n_y, "n_z": n_z, "T": T,
        "u": lambda x, y, z, t=0: x + 2.0 * y + 3.0 * z + t,
        "v": lambda x, y, z, t=0: 4.0 * x - y + 2.0 * z + t,
    }


@pytest.fixture
def nan_npz(tmp_path):
    """A 2D file with a NaN (ERA5 fill value) -- the loader must reject it."""
    winds = np.ones((2, 4, 3, 1), dtype=np.float64)
    winds[0, 1, 1, 0] = np.nan
    path = tmp_path / "nan2d.npz"
    np.savez(path, winds=winds, meta=dict(units="m/s", n=(4, 3, None)))
    return str(path)
