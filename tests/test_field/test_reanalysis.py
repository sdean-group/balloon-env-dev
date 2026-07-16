"""Core contract tests for ReanalysisFlowField (criteria 1-9).

A FlowField is a pure spatial source: ``reset(key)`` selects a realization (here, a real
weather time-slice) and ``velocity_at(p)`` reports the deterministic interpolated velocity
there. These tests lean on the affine-ground-truth fixtures: bilinear / trilinear
interpolation reproduces an affine field exactly, giving a closed-form expected value.
"""

import jax
import numpy as np
import pytest

from src.env.field import ReanalysisFlowField
from src.env.utils.types import GridConfig, GridPosition

KEY = jax.random.PRNGKey(0)


# --------------------------------------------------------------------------- helpers
def _field_2d(case, *, scale=1.0, slice_mode="fixed"):
    config = GridConfig.create(case["n_x"], case["n_y"])
    return ReanalysisFlowField(config, case["path"], scale=scale, slice_mode=slice_mode)


def _field_3d(case, *, scale=1.0, slice_mode="fixed"):
    config = GridConfig.create(case["n_x"], case["n_y"], case["n_z"])
    return ReanalysisFlowField(config, case["path"], scale=scale, slice_mode=slice_mode)


# --------------------------------------------------------------- 1. interface contract
def test_2d_returns_u_and_none(affine_npz_2d):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    u, v = f.velocity_at(GridPosition(2.0, 3.0))
    assert isinstance(u, float) and v is None


def test_3d_returns_u_and_v(affine_npz_3d):
    f = _field_3d(affine_npz_3d); f.reset(KEY)
    u, v = f.velocity_at(GridPosition(2.0, 2.0, 1.5))
    assert isinstance(u, float) and isinstance(v, float)


def test_velocity_field_shape_2d(affine_npz_2d):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    assert f.velocity_field().shape == (affine_npz_2d["n_x"], affine_npz_2d["n_y"], 1)


def test_velocity_field_shape_3d(affine_npz_3d):
    f = _field_3d(affine_npz_3d); f.reset(KEY)
    assert f.velocity_field().shape == (
        affine_npz_3d["n_x"], affine_npz_3d["n_y"], affine_npz_3d["n_z"], 2
    )


# ---------------------------------------- 2. interpolation is EXACTLY linear (keystone)
@pytest.mark.parametrize("seed", range(20))
def test_affine_reproduced_exactly_2d(affine_npz_2d, seed):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, affine_npz_2d["n_x"])
    y = rng.uniform(1, affine_npz_2d["n_y"])
    u, _ = f.velocity_at(GridPosition(x, y))
    assert u == pytest.approx(affine_npz_2d["truth"](x, y, 0), abs=1e-5)


@pytest.mark.parametrize("seed", range(20))
def test_affine_reproduced_exactly_3d(affine_npz_3d, seed):
    f = _field_3d(affine_npz_3d); f.reset(KEY)
    rng = np.random.default_rng(seed)
    x = rng.uniform(1, affine_npz_3d["n_x"])
    y = rng.uniform(1, affine_npz_3d["n_y"])
    z = rng.uniform(1, affine_npz_3d["n_z"])
    u, v = f.velocity_at(GridPosition(x, y, z))
    assert u == pytest.approx(affine_npz_3d["u"](x, y, z, 0), abs=1e-5)
    assert v == pytest.approx(affine_npz_3d["v"](x, y, z, 0), abs=1e-5)


# ------------------------------------------------- 3. node-exactness + 4. paths agree
def test_node_values_exact_and_paths_agree(affine_npz_2d):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    grid = f.velocity_field()
    for i in range(1, affine_npz_2d["n_x"] + 1):
        for j in range(1, affine_npz_2d["n_y"] + 1):
            u, _ = f.velocity_at(GridPosition(float(i), float(j)))
            assert u == pytest.approx(affine_npz_2d["truth"](i, j, 0), abs=1e-6)
            assert u == pytest.approx(grid[i - 1, j - 1, 0], abs=1e-6)


def test_midpoint_is_neighbor_average(affine_npz_2d):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    g = f.velocity_field()
    u, _ = f.velocity_at(GridPosition(2.5, 3.0))   # midpoint of nodes (2,3),(3,3)
    assert u == pytest.approx(0.5 * (g[1, 2, 0] + g[2, 2, 0]), abs=1e-6)


# ------------------------------------------------------------ 5. determinism / reset
def test_same_key_same_field(affine_npz_2d):
    f = _field_2d(affine_npz_2d, slice_mode="random")
    f.reset(KEY); first = f.velocity_field().copy()
    f.reset(KEY)
    assert np.array_equal(first, f.velocity_field())


def test_fixed_mode_always_slice_zero(affine_npz_2d):
    f = _field_2d(affine_npz_2d, slice_mode="fixed")
    for k in range(5):
        f.reset(jax.random.PRNGKey(k))
        u, _ = f.velocity_at(GridPosition(1.0, 1.0))
        assert u == pytest.approx(affine_npz_2d["truth"](1, 1, 0))   # t=0 always


def test_fixed_mode_can_select_explicit_slice(affine_npz_2d):
    case = affine_npz_2d
    config = GridConfig.create(case["n_x"], case["n_y"])
    field = ReanalysisFlowField(
        config,
        case["path"],
        slice_mode="fixed",
        fixed_index=1,
    )

    field.reset(KEY)

    assert field.current_time_index == 1
    assert field.velocity_at(GridPosition(1.0, 1.0))[0] == pytest.approx(
        case["truth"](1.0, 1.0, 1)
    )


# ----------------------------------------------------------- 6. realization diversity
def test_random_mode_visits_multiple_slices(affine_npz_2d):
    f = _field_2d(affine_npz_2d, slice_mode="random")
    seen = set()
    for k in range(50):
        f.reset(jax.random.PRNGKey(k))
        u, _ = f.velocity_at(GridPosition(1.0, 1.0))
        # slices differ by the +10*t offset -> recover t from the corner value
        seen.add(round((u - affine_npz_2d["truth"](1, 1, 0)) / 10.0))
    assert len(seen) >= 2
    assert all(0 <= t < affine_npz_2d["T"] for t in seen)


def test_velocity_before_reset_raises(affine_npz_2d):
    f = _field_2d(affine_npz_2d)
    with pytest.raises(RuntimeError):
        f.velocity_at(GridPosition(2.0, 2.0))


# -------------------------------------------------------------- 7. unit scaling linear
def test_scale_is_linear(affine_npz_2d):
    f1 = _field_2d(affine_npz_2d, scale=1.0); f1.reset(KEY)
    f2 = _field_2d(affine_npz_2d, scale=2.0); f2.reset(KEY)
    p = GridPosition(2.3, 3.1)
    assert f2.velocity_at(p)[0] == pytest.approx(2.0 * f1.velocity_at(p)[0])


# ----------------------------------------------------------- 8. boundary / no NaN-Inf
def test_corners_finite(affine_npz_2d):
    f = _field_2d(affine_npz_2d); f.reset(KEY)
    corners = [
        GridPosition(1.0, 1.0),
        GridPosition(float(affine_npz_2d["n_x"]), float(affine_npz_2d["n_y"])),
        GridPosition(1.0, float(affine_npz_2d["n_y"])),
        GridPosition(float(affine_npz_2d["n_x"]), 1.0),
    ]
    for p in corners:
        assert np.isfinite(f.velocity_at(p)[0])


@pytest.mark.parametrize("seed", range(200))
def test_no_nan_anywhere_in_domain_3d(affine_npz_3d, seed):
    f = _field_3d(affine_npz_3d); f.reset(KEY)
    rng = np.random.default_rng(seed)
    p = GridPosition(
        rng.uniform(1, affine_npz_3d["n_x"]),
        rng.uniform(1, affine_npz_3d["n_y"]),
        rng.uniform(1, affine_npz_3d["n_z"]),
    )
    u, v = f.velocity_at(p)
    assert np.isfinite(u) and np.isfinite(v)


# ----------------------------------------------------- 9. construct-time validation
def test_shape_mismatch_raises(affine_npz_2d):
    bad = GridConfig.create(affine_npz_2d["n_x"] + 1, affine_npz_2d["n_y"])
    with pytest.raises(ValueError):
        ReanalysisFlowField(bad, affine_npz_2d["path"])


def test_ndim_mismatch_raises(affine_npz_3d):
    bad = GridConfig.create(affine_npz_3d["n_x"], affine_npz_3d["n_y"])  # 2D vs 3D file
    with pytest.raises(ValueError):
        ReanalysisFlowField(bad, affine_npz_3d["path"])


# --------------------------------------------------- 10. temporal mode (steps_per_slice)
# The affine fixtures add a per-slice offset (2D: +10*slice; 3D: +1*slice). Linear
# interpolation in BOTH space and slice-index is exact for affine data, so the temporal
# field at episode time t (start slice t0, cadence S) has a closed-form value at
# fractional slice s = t0 + t/S.

def _temporal_2d(case, *, steps_per_slice, slice_mode="fixed"):
    config = GridConfig.create(case["n_x"], case["n_y"])
    return ReanalysisFlowField(
        config, case["path"], slice_mode=slice_mode, steps_per_slice=steps_per_slice
    )


def test_steps_per_slice_validated(affine_npz_2d):
    config = GridConfig.create(affine_npz_2d["n_x"], affine_npz_2d["n_y"])
    with pytest.raises(ValueError, match="steps_per_slice"):
        ReanalysisFlowField(config, affine_npz_2d["path"], steps_per_slice=0.0)


def test_static_default_is_time_invariant(affine_npz_2d):
    """Without steps_per_slice the field ignores t and reports time_varying=False."""
    f = _field_2d(affine_npz_2d, slice_mode="fixed")
    f.reset(KEY)
    assert f.time_varying is False
    p = GridPosition(2.5, 3.5, None)
    assert f.velocity_at(p, t=0.0) == f.velocity_at(p, t=50.0)


def test_temporal_t0_matches_frozen_start_slice(affine_npz_2d):
    """At t=0 the temporal field equals the static frozen slice exactly."""
    static = _field_2d(affine_npz_2d, slice_mode="fixed"); static.reset(KEY)
    temporal = _temporal_2d(affine_npz_2d, steps_per_slice=4.0); temporal.reset(KEY)
    assert temporal.time_varying is True
    p = GridPosition(2.5, 3.5, None)
    assert temporal.velocity_at(p, t=0.0)[0] == pytest.approx(static.velocity_at(p)[0])


def test_temporal_interpolates_between_slices_2d(affine_npz_2d):
    """velocity_at(p, t) matches the affine ground truth at fractional slice s = t/S."""
    case = affine_npz_2d
    S = 4.0
    f = _temporal_2d(case, steps_per_slice=S); f.reset(KEY)  # fixed -> t0=0
    x, y = 2.5, 3.5
    p = GridPosition(x, y, None)
    for t in [0.0, 2.0, 4.0, 6.0, 10.0]:
        s = min(t / S, case["T"] - 1)              # clamp at last slice
        expected = case["truth"](x, y, s)          # 2x+3y+1 + 10*s
        assert f.velocity_at(p, t=t)[0] == pytest.approx(expected, abs=1e-9)


def test_temporal_clamps_past_window_end(affine_npz_2d):
    """Past the last slice the field clamps (no extrapolation in time)."""
    case = affine_npz_2d
    f = _temporal_2d(case, steps_per_slice=1.0); f.reset(KEY)  # 1 step per slice
    p = GridPosition(2.5, 3.5, None)
    last = case["truth"](2.5, 3.5, case["T"] - 1)
    # Way past the end of T slices -> pinned at the final slice.
    assert f.velocity_at(p, t=999.0)[0] == pytest.approx(last, abs=1e-9)


def test_temporal_interpolates_between_slices_3d(affine_npz_3d):
    case = affine_npz_3d
    S = 2.0
    config = GridConfig.create(case["n_x"], case["n_y"], case["n_z"])
    f = ReanalysisFlowField(config, case["path"], slice_mode="fixed", steps_per_slice=S)
    f.reset(KEY)
    x, y, z = 2.5, 2.5, 2.5
    p = GridPosition(x, y, z)
    for t in [0.0, 1.0, 2.0, 5.0]:
        s = min(t / S, case["T"] - 1)
        u, v = f.velocity_at(p, t=t)
        assert u == pytest.approx(case["u"](x, y, z, s), abs=1e-9)
        assert v == pytest.approx(case["v"](x, y, z, s), abs=1e-9)


def test_temporal_velocity_field_blends_slices(affine_npz_2d):
    """velocity_field(t) returns the linear blend of bracketing slices (affine -> exact)."""
    case = affine_npz_2d
    S = 4.0
    f = _temporal_2d(case, steps_per_slice=S); f.reset(KEY)
    grid_mid = f.velocity_field(t=2.0)            # s = 0.5
    half = 0.5 * (f._winds[0] + f._winds[1])
    np.testing.assert_allclose(grid_mid, half, atol=1e-9)
