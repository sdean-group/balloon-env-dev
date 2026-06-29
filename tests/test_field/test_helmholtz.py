import jax
import numpy as np
import pytest

from src.env.field import HelmholtzDataDrivenFlowField, HelmholtzSyntheticFlowField
from src.env.utils.types import GridConfig, GridPosition


def test_helmholtz_synthetic_contract_and_reproducibility():
    config = GridConfig.create(9, 8, 4)
    field = HelmholtzSyntheticFlowField(
        config,
        sigma=2.0,
        lengthscale=5.0,
        num_features=64,
    )

    key = jax.random.PRNGKey(3)
    field.reset(key)
    first = field.velocity_field().copy()
    assert first.shape == (9, 8, 4, 2)

    u, v = field.velocity_at(GridPosition(4.5, 3.25, 2.0))
    assert isinstance(u, float)
    assert isinstance(v, float)
    assert np.isfinite(u)
    assert np.isfinite(v)

    field.reset(key)
    np.testing.assert_array_equal(first, field.velocity_field())

    field.reset(jax.random.PRNGKey(4))
    assert not np.allclose(first, field.velocity_field())


def test_helmholtz_synthetic_curl_only_is_divergence_free():
    config = GridConfig.create(8, 7, 3)
    field = HelmholtzSyntheticFlowField(
        config,
        num_features=64,
        divergence_weight=0.0,
        curl_weight=1.0,
    )
    field.reset(jax.random.PRNGKey(9))

    def u_fn(x, y, z):
        u, _ = field.velocity_at_point(x, y, z)
        return u

    def v_fn(x, y, z):
        _, v = field.velocity_at_point(x, y, z)
        return v

    x, y, z = 3.2, 4.1, 2.0
    divergence = float(jax.grad(u_fn, argnums=0)(x, y, z) + jax.grad(v_fn, argnums=1)(x, y, z))
    assert abs(divergence) < 1e-5


def test_helmholtz_data_driven_fits_constant_vector_field():
    config = GridConfig.create(6, 5, 3)
    axes = [np.arange(1, n + 1, dtype=np.float64) for n in config.shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    positions = np.stack(mesh, axis=-1).reshape(-1, 3)
    velocities = np.repeat([[1.5, -0.75]], repeats=positions.shape[0], axis=0)

    field = HelmholtzDataDrivenFlowField(
        config,
        positions,
        velocities,
        num_features=128,
        lengthscale=8.0,
        noise_std=0.05,
        feature_seed=2,
    )
    field.reset(jax.random.PRNGKey(0))
    u, v = field.velocity_at(GridPosition(3.5, 2.5, 2.0))
    assert u == pytest.approx(1.5, abs=0.25)
    assert v == pytest.approx(-0.75, abs=0.25)
    assert field.velocity_field().shape == (6, 5, 3, 2)

