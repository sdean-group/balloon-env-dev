"""Tests for the RFF GP fitted from wind observations."""

import jax
import numpy as np
import pytest

from src.env.field import DataDrivenFlowField
from src.env.utils.types import GridConfig, GridPosition


def _grid_training_data(config):
    axes = [np.arange(1, n + 1, dtype=float) for n in config.shape]
    mesh = np.meshgrid(*axes, indexing="ij")
    positions = np.stack(mesh, axis=-1).reshape(-1, config.ndim)
    if config.ndim == 2:
        values = (
            1.2
            + np.sin(positions[:, 0] / 3.0)
            + 0.5 * np.cos(positions[:, 1] / 2.5)
        )[:, None]
    else:
        u = 2.0 + 0.1 * positions[:, 0] - 0.05 * positions[:, 2]
        v = -1.0 + 0.08 * positions[:, 1] + 0.03 * positions[:, 2]
        values = np.column_stack([u, v])
    return positions, values


def test_fitted_gp_reconstructs_smooth_2d_field():
    config = GridConfig.create(9, 8)
    positions, velocities = _grid_training_data(config)
    field = DataDrivenFlowField(
        config,
        positions,
        velocities,
        num_features=128,
        lengthscale=4.0,
        noise_std=0.03,
        feature_seed=4,
    )
    field.reset(jax.random.PRNGKey(0))

    assert field.velocity_field().shape == (9, 8, 1)
    assert field.training_rmse < 0.08
    u, v = field.velocity_at(GridPosition(4.25, 5.5, None))
    expected = 1.2 + np.sin(4.25 / 3.0) + 0.5 * np.cos(5.5 / 2.5)
    assert u == pytest.approx(expected, abs=0.12)
    assert v is None


def test_3d_gp_returns_two_components_and_grid():
    config = GridConfig.create(5, 4, 3)
    positions, velocities = _grid_training_data(config)
    field = DataDrivenFlowField(
        config,
        positions,
        velocities,
        num_features=96,
        lengthscale=5.0,
        noise_std=0.02,
        feature_seed=2,
    )
    field.reset(jax.random.PRNGKey(0))
    u, v = field.velocity_at(GridPosition(2.5, 2.25, 1.5))

    assert field.velocity_field().shape == (5, 4, 3, 2)
    assert np.isfinite(u)
    assert np.isfinite(v)
    assert u == pytest.approx(2.175, abs=0.12)
    assert v == pytest.approx(-0.775, abs=0.12)


def test_posterior_sampling_is_seeded_and_coherent():
    config = GridConfig.create(6, 5)
    positions, velocities = _grid_training_data(config)
    field = DataDrivenFlowField(
        config,
        positions,
        velocities,
        num_features=64,
        lengthscale=3.0,
        noise_std=0.2,
        feature_seed=1,
        sample_posterior=True,
    )

    key = jax.random.PRNGKey(12)
    field.reset(key)
    first = field.velocity_field().copy()
    field.reset(key)
    np.testing.assert_array_equal(field.velocity_field(), first)
    field.reset(jax.random.PRNGKey(13))

    assert not np.array_equal(field.velocity_field(), first)


def test_fit_from_era5_cache_selects_slice_and_stride(tmp_path):
    config = GridConfig.create(6, 4)
    _, first = _grid_training_data(config)
    first_grid = first.reshape(6, 4, 1)
    winds = np.stack([first_grid, first_grid + 4.0], axis=0)
    path = tmp_path / "winds.npz"
    np.savez_compressed(path, winds=winds, meta={"source": "test"})

    field = DataDrivenFlowField.from_era5_cache(
        config,
        str(path),
        time_index=1,
        scale=0.5,
        training_stride=2,
        num_features=64,
        lengthscale=4.0,
        noise_std=0.03,
        feature_seed=3,
    )
    field.reset(jax.random.PRNGKey(0))

    assert field.time_index == 1
    assert field.training_points == 6
    assert field.data_metadata["source"] == "test"
    assert field.velocity_field().shape == (6, 4, 1)


@pytest.mark.parametrize(
    "positions, velocities, match",
    [
        (np.ones((3, 3)), np.ones((3, 1)), "positions"),
        (np.ones((3, 2)), np.ones((3, 2)), "velocities"),
        (np.array([[0.0, 1.0]]), np.ones((1, 1)), "domain"),
    ],
)
def test_training_data_validation(positions, velocities, match):
    with pytest.raises(ValueError, match=match):
        DataDrivenFlowField(GridConfig.create(4, 4), positions, velocities)
