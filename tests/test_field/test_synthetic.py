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
