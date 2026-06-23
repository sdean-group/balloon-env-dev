"""Real-data plausibility & regression tests (criteria 11, 12).

These need an actual regridded ERA5 cache and are therefore OPT-IN: they are skipped unless
the path in the ERA5_CACHE environment variable exists. They are NOT run in CI.

    ERA5_CACHE=data/era5_sf_80x80x10.npz pixi run pytest tests/test_field/ -m era5 -v

Register the marker in pyproject.toml:
    [tool.pytest.ini_options]
    markers = ["era5: requires a real ERA5 cache; skipped in CI"]
"""

import os

import jax
import numpy as np
import pytest

from src.env.field import ReanalysisFlowField
from src.env.field.era5_data import load_era5
from src.env.utils.types import GridConfig, GridPosition

pytestmark = pytest.mark.era5

CACHE = os.environ.get("ERA5_CACHE", "")
_have_cache = bool(CACHE) and os.path.exists(CACHE)
skip_no_cache = pytest.mark.skipif(not _have_cache, reason="set ERA5_CACHE to a real .npz")

# Raw ERA5 winds are m/s. Strong winter jet-stream events can exceed 80 m/s,
# so this is a corruption guard rather than a climatological upper bound.
MAX_PLAUSIBLE_MS = 100.0
# After scale -> cells/step, drift should be comparable to a GP run with sigma~3.
MAX_PLAUSIBLE_CELLS_PER_STEP = 10.0


def _config_from_cache(bundle):
    n = bundle.winds.shape[1:-1]              # (n_x, n_y[, n_z])
    return GridConfig.create(*n)


@skip_no_cache
def test_raw_magnitude_plausible():
    bundle = load_era5(CACHE)
    speed = np.linalg.norm(bundle.winds, axis=-1)
    assert np.nanpercentile(speed, 99.9) < MAX_PLAUSIBLE_MS
    assert np.all(np.isfinite(bundle.winds))


@skip_no_cache
def test_scaled_magnitude_comparable_to_gp():
    bundle = load_era5(CACHE)
    config = _config_from_cache(bundle)
    field = ReanalysisFlowField(config, CACHE, scale=float(os.environ.get("ERA5_SCALE", 0.06)))
    field.reset(jax.random.PRNGKey(0))
    g = field.velocity_field()
    speed = np.linalg.norm(g, axis=-1)
    assert np.percentile(speed, 95) < MAX_PLAUSIBLE_CELLS_PER_STEP


@skip_no_cache
def test_spatially_smooth():
    """Reanalysis is smooth at these scales: adjacent-cell differences stay small."""
    bundle = load_era5(CACHE)
    u = bundle.winds[0, ..., 0]
    grad = np.abs(np.diff(u, axis=0)).mean()
    typical = np.abs(u).mean() + 1e-9
    assert grad < typical            # neighbour change << overall magnitude


@skip_no_cache
def test_consecutive_slices_correlated():
    """Weather evolves continuously: slice t and t+1 should be positively correlated."""
    bundle = load_era5(CACHE)
    if bundle.winds.shape[0] < 2:
        pytest.skip("need >= 2 time slices")
    a = bundle.winds[0].ravel()
    b = bundle.winds[1].ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.5


@skip_no_cache
def test_divergence_bounded():
    """Trilinear interp is NOT divergence-free (unlike the streamfunction GP); just bound it."""
    bundle = load_era5(CACHE)
    if bundle.winds.shape[-1] < 2:
        pytest.skip("divergence needs a (u, v) field (3D)")
    u, v = bundle.winds[0, ..., 0], bundle.winds[0, ..., 1]
    div = np.gradient(u, axis=0) + np.gradient(v, axis=1)
    typical = np.abs(bundle.winds[0]).mean() + 1e-9
    assert np.abs(div).mean() < typical


@skip_no_cache
def test_golden_snapshot():
    """Pin one (slice, position) -> velocity so refactors can't silently change numbers.

    On first run, print the observed value and paste it into GOLDEN below.
    """
    bundle = load_era5(CACHE)
    config = _config_from_cache(bundle)
    field = ReanalysisFlowField(config, CACHE, scale=1.0, slice_mode="fixed")
    field.reset(jax.random.PRNGKey(0))
    probe = GridPosition(2.5, 3.5) if config.ndim == 2 else GridPosition(2.5, 3.5, 1.5)
    got = field.velocity_at(probe)

    GOLDEN = os.environ.get("ERA5_GOLDEN")  # e.g. "1.234,5.678"
    if not GOLDEN:
        pytest.skip(f"set ERA5_GOLDEN to pin this snapshot; observed = {got}")
    expected = tuple(float(x) for x in GOLDEN.split(","))
    got_t = (got[0],) if got[1] is None else (got[0], got[1])
    np.testing.assert_allclose(got_t, expected, rtol=1e-6)
