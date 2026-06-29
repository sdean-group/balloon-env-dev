"""Contract tests for SyntheticFlowField (RFF GP wind source).

A FlowField is a pure spatial source: ``reset(key)`` draws a realization and
``velocity_at(p)`` reports the deterministic velocity there. No noise, clipping,
or PMFs live on the field anymore (those moved to the arena / were deleted).
"""

import jax
import numpy as np
import pytest

from src.env.field import SyntheticFlowField
from src.env.utils.types import GridConfig, GridPosition

RNG = np.random.default_rng(2026)


def _sample_case(ndim: int, seed: int) -> dict:
    """Build one randomized-yet-safe initialization case."""
    if ndim == 2:
        n_x = int(RNG.integers(1, 128))
        n_y = int(RNG.integers(1, 128))
        n_z = None
    else:
        n_x = int(RNG.integers(1, 128))
        n_y = int(RNG.integers(1, 128))
        n_z = int(RNG.integers(1, 64))

    return {
        "ndim": ndim,
        "seed": seed,
        "n_x": n_x,
        "n_y": n_y,
        "n_z": n_z,
        "sigma": float(10 ** RNG.uniform(-1, 1)),
        "lengthscale": float(10 ** RNG.uniform(-2, 2)),
        "nu": float(RNG.choice([0.5, 1.5, 2.5, 4.5])),
        "num_features": int(RNG.integers(32, 160)),
    }


RANDOM_CASES = (
    [_sample_case(2, seed) for seed in range(4)]
    + [_sample_case(3, 100 + seed) for seed in range(4)]
)


def _build_field(case: dict) -> SyntheticFlowField:
    config = GridConfig.create(case["n_x"], case["n_y"], case["n_z"])
    return SyntheticFlowField(
        config=config,
        sigma=case["sigma"],
        lengthscale=case["lengthscale"],
        nu=case["nu"],
        num_features=case["num_features"],
    )


def _probe_positions(case: dict) -> list[GridPosition]:
    center_i = (case["n_x"] + 1) // 2
    center_j = (case["n_y"] + 1) // 2
    if case["ndim"] == 2:
        return [
            GridPosition(1, 1, None),
            GridPosition(center_i, center_j, None),
            GridPosition(case["n_x"], case["n_y"], None),
        ]
    center_k = (case["n_z"] + 1) // 2
    return [
        GridPosition(1, 1, 1),
        GridPosition(center_i, center_j, center_k),
        GridPosition(case["n_x"], case["n_y"], case["n_z"]),
    ]


@pytest.mark.parametrize("case", RANDOM_CASES, ids=lambda c: f'{c["ndim"]}d-seed{c["seed"]}')
def test_randomized_field_contracts(case: dict):
    """init + reset + velocity_at + velocity_field contracts."""
    field = _build_field(case)
    spatial_dim = 2 if case["ndim"] == 2 else 3

    # Initialization contract
    assert field.ndim == case["ndim"]
    assert field._precomputed_u is None
    if case["ndim"] == 3:
        assert field._precomputed_v is None

    # Reset and shape contract
    field.reset(jax.random.PRNGKey(case["seed"]))
    assert field._omegas.shape == (case["num_features"], spatial_dim)
    assert field._phases.shape == (case["num_features"],)
    assert field._weights.shape == (case["num_features"],)

    vel_field = field.velocity_field()
    if case["ndim"] == 2:
        assert vel_field.shape == (case["n_x"], case["n_y"], 1)
    else:
        assert vel_field.shape == (case["n_x"], case["n_y"], case["n_z"], 2)

    for pos in _probe_positions(case):
        u, v = field.velocity_at(pos)

        # float32: matmul-based precompute vs sum-based pointwise eval differ slightly
        if case["ndim"] == 2:
            assert v is None
            assert isinstance(u, float)
            i_idx, j_idx = pos.i - 1, pos.j - 1
            assert u == pytest.approx(float(vel_field[i_idx, j_idx, 0]), rel=1e-4, abs=1e-4)
        else:
            assert isinstance(u, float)
            assert isinstance(v, float)
            i_idx, j_idx, k_idx = pos.i - 1, pos.j - 1, pos.k - 1
            assert u == pytest.approx(float(vel_field[i_idx, j_idx, k_idx, 0]), rel=1e-4, abs=1e-4)
            assert v == pytest.approx(float(vel_field[i_idx, j_idx, k_idx, 1]), rel=1e-4, abs=1e-4)

        # velocity_at_point (differentiable core) matches velocity_at at grid points
        if case["ndim"] == 2:
            up, vp = field.velocity_at_point(float(pos.i), float(pos.j))
            assert vp is None
            assert float(up) == pytest.approx(u, rel=1e-4, abs=1e-6)
        else:
            up, vp = field.velocity_at_point(float(pos.i), float(pos.j), float(pos.k))
            assert float(up) == pytest.approx(u, rel=1e-4, abs=1e-6)
            assert float(vp) == pytest.approx(v, rel=1e-4, abs=1e-6)


@pytest.mark.parametrize("case", RANDOM_CASES[:3] + RANDOM_CASES[4:7], ids=lambda c: f'{c["ndim"]}d-seed{c["seed"]}')
def test_reset_reproducibility_and_seed_change(case: dict):
    """Same key reproduces exactly, different key changes sampled field."""
    field = _build_field(case)
    key = jax.random.PRNGKey(case["seed"])

    field.reset(key)
    vel_1 = field.velocity_field().copy()

    field.reset(key)
    vel_2 = field.velocity_field().copy()

    np.testing.assert_array_equal(vel_1, vel_2)

    field.reset(jax.random.PRNGKey(case["seed"] + 10_000))
    vel_3 = field.velocity_field().copy()
    assert not np.allclose(vel_1, vel_3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma": 0.0}, "sigma must be positive"),
        ({"sigma": -1.0}, "sigma must be positive"),
        ({"lengthscale": 0.0}, "lengthscale must be positive"),
        ({"lengthscale": -1.0}, "lengthscale must be positive"),
        ({"nu": 0.0}, "nu must be positive"),
        ({"nu": -0.5}, "nu must be positive"),
        ({"num_features": 0}, "num_features must be positive"),
        ({"num_features": -16}, "num_features must be positive"),
    ],
)
def test_positive_hyperparameters_validated(kwargs: dict, match: str):
    """Constructor enforces positivity for GP/RFF hyperparameters."""
    config = GridConfig.create(n_x=8, n_y=7)
    with pytest.raises(ValueError, match=match):
        SyntheticFlowField(config=config, **kwargs)


@pytest.mark.parametrize("case", RANDOM_CASES[4:7], ids=lambda c: f'3d-seed{c["seed"]}')
def test_divergence_free_streamfunction(case: dict):
    """3D streamfunction parameterization is divergence-free: du/dx + dv/dy ~= 0."""
    field = _build_field(case)
    field.reset(jax.random.PRNGKey(case["seed"]))

    def u_fn(x, y, z):
        u, _ = field.velocity_at_point(x, y, z)
        return u

    def v_fn(x, y, z):
        _, v = field.velocity_at_point(x, y, z)
        return v

    points = [
        (1.25, 1.75, 1.5),
        (case["n_x"] * 0.6, case["n_y"] * 0.4, case["n_z"] * 0.7),
    ]
    for x, y, z in points:
        du_dx = float(jax.grad(u_fn, argnums=0)(x, y, z))
        dv_dy = float(jax.grad(v_fn, argnums=1)(x, y, z))
        divergence = du_dx + dv_dy
        # Divergence is analytically exact zero; the residual is float32 rounding,
        # which scales with the (potentially large) second-derivative magnitudes.
        scale = abs(du_dx) + abs(dv_dy) + 1.0
        assert abs(divergence) < 1e-3 * scale


# ---------------------------------------------------------------------------
# Temporal axis (lengthscale_t): the field evolves WITHIN an episode.
# ---------------------------------------------------------------------------

def _temporal_field(ndim: int = 2, lengthscale_t=5.0, seed: int = 7) -> SyntheticFlowField:
    n_z = None if ndim == 2 else 6
    config = GridConfig.create(10, 9, n_z)
    field = SyntheticFlowField(
        config=config, sigma=2.0, lengthscale=4.0, nu=2.5,
        num_features=128, lengthscale_t=lengthscale_t,
    )
    field.reset(jax.random.PRNGKey(seed))
    return field


def test_lengthscale_t_validated():
    config = GridConfig.create(8, 7)
    with pytest.raises(ValueError, match="lengthscale_t"):
        SyntheticFlowField(config=config, lengthscale_t=0.0)


def test_static_field_is_time_invariant():
    """Without lengthscale_t the field ignores t (back-compat) and reports time_varying=False."""
    config = GridConfig.create(10, 9)
    field = SyntheticFlowField(config=config, sigma=2.0, lengthscale=4.0, num_features=128)
    field.reset(jax.random.PRNGKey(3))
    assert field.time_varying is False
    p = GridPosition(4.3, 5.1, None)
    u0, _ = field.velocity_at(p, t=0.0)
    u9, _ = field.velocity_at(p, t=9.0)
    assert u0 == u9


@pytest.mark.parametrize("ndim", [2, 3])
def test_temporal_field_changes_over_time(ndim: int):
    """A field with lengthscale_t actually evolves: velocity at a fixed point moves with t."""
    field = _temporal_field(ndim=ndim)
    assert field.time_varying is True
    p = GridPosition(4.3, 5.1, None if ndim == 2 else 3.2)
    u0 = field.velocity_at(p, t=0.0)[0]
    u_late = field.velocity_at(p, t=40.0)[0]
    assert abs(u_late - u0) > 1e-3


@pytest.mark.parametrize("ndim", [2, 3])
def test_temporal_field_continuous_in_time(ndim: int):
    """Small steps in t produce small changes (the temporal GP is smooth)."""
    field = _temporal_field(ndim=ndim)
    p = GridPosition(4.3, 5.1, None if ndim == 2 else 3.2)
    a = field.velocity_at(p, t=10.0)[0]
    b = field.velocity_at(p, t=10.001)[0]
    assert abs(b - a) < 1e-2


def test_temporal_field_deterministic_in_key_and_t():
    """Same key + same t => identical field; t=0 matches the cached reset grid."""
    f1 = _temporal_field(seed=11)
    f2 = _temporal_field(seed=11)
    p = GridPosition(4.3, 5.1, None)
    assert f1.velocity_at(p, t=17.0) == f2.velocity_at(p, t=17.0)
    # velocity_field(t=0) reuses the cached grid; recomputing at 0 must agree.
    np.testing.assert_allclose(f1.velocity_field(0.0), f1._grid_velocity(0.0)[0][:, :, None])


def test_temporal_field_velocity_field_evolves():
    field = _temporal_field(ndim=2)
    grid0 = field.velocity_field(0.0)
    grid_late = field.velocity_field(30.0)
    assert grid0.shape == grid_late.shape
    assert not np.allclose(grid0, grid_late)


@pytest.mark.parametrize("seed", [101, 102, 103])
def test_temporal_3d_divergence_free_at_nonzero_t(seed: int):
    """The 3D streamfunction stays divergence-free at t>0 (time only rides in theta)."""
    field = _temporal_field(ndim=3, seed=seed)
    t = 23.0

    def u_fn(x, y, z):
        return field.velocity_at_point(x, y, z, t=t)[0]

    def v_fn(x, y, z):
        return field.velocity_at_point(x, y, z, t=t)[1]

    for x, y, z in [(2.25, 3.75, 2.5), (6.0, 4.0, 4.5)]:
        du_dx = float(jax.grad(u_fn, argnums=0)(x, y, z))
        dv_dy = float(jax.grad(v_fn, argnums=1)(x, y, z))
        scale = abs(du_dx) + abs(dv_dy) + 1.0
        assert abs(du_dx + dv_dy) < 1e-3 * scale
