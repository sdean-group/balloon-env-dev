"""Helmholtz-style RFF vector wind fields.

These fields model horizontal wind as a sum of curl-free and divergence-free
components. This is a more physical vector-field prior than fitting independent
scalar GPs to each component.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .era5_data import load_era5
from .flow_field import FlowField
from ..utils.types import GridConfig, GridPosition


class HelmholtzSyntheticFlowField(FlowField):
    """Synthetic horizontal vector field from Helmholtz RFF components.

    In 3D, the field varies with ``(x, y, z)`` but returns horizontal velocity
    ``(u, v)``. The horizontal vector field is built from two scalar potentials:

    - curl-free component: gradient of a potential
    - divergence-free component: rotated gradient of a streamfunction

    The ``divergence_weight`` and ``curl_weight`` names follow the vector-field
    behavior, not the scalar potentials.
    """

    def __init__(
        self,
        config: GridConfig,
        *,
        sigma: float = 1.0,
        lengthscale: float = 8.0,
        num_features: int = 256,
        divergence_weight: float = 1.0,
        curl_weight: float = 1.0,
    ):
        super().__init__(config)
        if self.ndim != 3:
            raise ValueError("HelmholtzSyntheticFlowField currently requires 3D config")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if num_features <= 0 or num_features != int(num_features):
            raise ValueError("num_features must be a positive integer")
        if divergence_weight < 0 or curl_weight < 0:
            raise ValueError("component weights must be non-negative")
        if divergence_weight == 0 and curl_weight == 0:
            raise ValueError("at least one component weight must be positive")

        self.sigma = float(sigma)
        self.lengthscale = float(lengthscale)
        self.num_features = int(num_features)
        self.divergence_weight = float(divergence_weight)
        self.curl_weight = float(curl_weight)

        self._omegas: Optional[jnp.ndarray] = None
        self._phases: Optional[jnp.ndarray] = None
        self._div_weights: Optional[jnp.ndarray] = None
        self._curl_weights: Optional[jnp.ndarray] = None
        self._precomputed: Optional[np.ndarray] = None

    def reset(self, rng_key: jnp.ndarray) -> None:
        key_omega, key_phase, key_div, key_curl = jax.random.split(rng_key, 4)
        self._omegas = jax.random.normal(
            key_omega,
            shape=(self.num_features, self.ndim),
        ) / self.lengthscale
        self._phases = jax.random.uniform(
            key_phase,
            shape=(self.num_features,),
            minval=0,
            maxval=2 * jnp.pi,
        )
        self._div_weights = jax.random.normal(key_div, shape=(self.num_features,))
        self._curl_weights = jax.random.normal(key_curl, shape=(self.num_features,))
        self._precompute_field()

    def _velocity_at_array(self, coords: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        theta = coords @ self._omegas.T + self._phases
        sin_theta = jnp.sin(theta)
        scale = jnp.sqrt(2.0 * self.sigma**2 / self.num_features)
        omega_x = self._omegas[:, 0]
        omega_y = self._omegas[:, 1]

        # grad(phi): (-omega_x sin, -omega_y sin)
        div_u = -scale * jnp.sum(self._div_weights * omega_x * sin_theta)
        div_v = -scale * jnp.sum(self._div_weights * omega_y * sin_theta)

        # rotated grad(psi): (omega_y sin, -omega_x sin)
        curl_u = scale * jnp.sum(self._curl_weights * omega_y * sin_theta)
        curl_v = -scale * jnp.sum(self._curl_weights * omega_x * sin_theta)

        return (
            self.divergence_weight * div_u + self.curl_weight * curl_u,
            self.divergence_weight * div_v + self.curl_weight * curl_v,
        )

    def velocity_at_point(self, x: float, y: float, z: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if self._omegas is None:
            raise RuntimeError("HelmholtzSyntheticFlowField.reset() must be called first")
        return self._velocity_at_array(jnp.array([x, y, z], dtype=jnp.float32))

    def velocity_at(self, position: GridPosition) -> Tuple[float, Optional[float]]:
        u, v = self.velocity_at_point(position.i, position.j, position.k)
        return float(u), float(v)

    def _precompute_field(self) -> None:
        xs = jnp.arange(1, self.config.n_x + 1)
        ys = jnp.arange(1, self.config.n_y + 1)
        zs = jnp.arange(1, self.config.n_z + 1)
        gx, gy, gz = jnp.meshgrid(xs, ys, zs, indexing="ij")
        coords = jnp.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
        theta = coords @ self._omegas.T + self._phases
        sin_theta = jnp.sin(theta)
        scale = jnp.sqrt(2.0 * self.sigma**2 / self.num_features)
        omega_x = self._omegas[:, 0]
        omega_y = self._omegas[:, 1]
        div_u = -scale * (sin_theta @ (self._div_weights * omega_x))
        div_v = -scale * (sin_theta @ (self._div_weights * omega_y))
        curl_u = scale * (sin_theta @ (self._curl_weights * omega_y))
        curl_v = -scale * (sin_theta @ (self._curl_weights * omega_x))
        u = self.divergence_weight * div_u + self.curl_weight * curl_u
        v = self.divergence_weight * div_v + self.curl_weight * curl_v
        self._precomputed = np.asarray(
            jnp.stack([u, v], axis=-1).reshape(
                self.config.n_x, self.config.n_y, self.config.n_z, 2
            )
        )

    def velocity_field(self) -> np.ndarray:
        if self._precomputed is None:
            raise RuntimeError("HelmholtzSyntheticFlowField.reset() must be called first")
        return np.asarray(self._precomputed)


class HelmholtzDataDrivenFlowField(FlowField):
    """Ridge-fitted Helmholtz RFF vector field for ERA5-style samples.

    This is a mean-field baseline, not a full exact GP posterior. The important
    difference from ``DataDrivenFlowField`` is the vector-valued feature map:
    every feature contributes through Helmholtz curl-free/divergence-free vector
    bases instead of fitting u and v as unrelated scalar outputs.
    """

    def __init__(
        self,
        config: GridConfig,
        positions: np.ndarray,
        velocities: np.ndarray,
        *,
        num_features: int = 256,
        lengthscale: float = 8.0,
        noise_std: float = 0.1,
        feature_seed: int = 0,
    ):
        super().__init__(config)
        if self.ndim != 3:
            raise ValueError("HelmholtzDataDrivenFlowField currently requires 3D config")
        if num_features <= 0 or num_features != int(num_features):
            raise ValueError("num_features must be a positive integer")
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")

        x = np.asarray(positions, dtype=np.float64)
        y = np.asarray(velocities, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(f"positions must have shape (n, {self.ndim})")
        if y.ndim != 2 or y.shape != (x.shape[0], 2):
            raise ValueError(f"velocities must have shape ({x.shape[0]}, 2)")
        if x.shape[0] == 0:
            raise ValueError("at least one training observation is required")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must contain only finite values")

        lower = np.ones(self.ndim)
        upper = np.asarray(self.config.shape, dtype=np.float64)
        if np.any(x < lower) or np.any(x > upper):
            raise ValueError("training positions must lie inside the environment domain")

        self.num_features = int(num_features)
        self.lengthscale = float(lengthscale)
        self.noise_std = float(noise_std)
        self.feature_seed = int(feature_seed)

        rng = np.random.default_rng(self.feature_seed)
        self._omegas = rng.normal(
            scale=1.0 / self.lengthscale,
            size=(self.num_features, self.ndim),
        )
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=self.num_features)
        self._target_mean = y.mean(axis=0)
        self._target_std = y.std(axis=0)
        self._target_std = np.where(self._target_std < 1e-8, 1.0, self._target_std)
        y_standardized = (y - self._target_mean) / self._target_std

        design = self._design_matrix(x)
        target = y_standardized.reshape(-1)
        precision = design.T @ design + self.noise_std**2 * np.eye(2 * self.num_features + 2)
        rhs = design.T @ target
        chol = np.linalg.cholesky(precision)
        intermediate = np.linalg.solve(chol, rhs)
        self._weights = np.linalg.solve(chol.T, intermediate)

        fitted = self._unstandardize(self._predict(x))
        self.training_rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        self.training_points = int(x.shape[0])
        self._precomputed: Optional[np.ndarray] = None

    @classmethod
    def from_era5_cache(
        cls,
        config: GridConfig,
        data_path: str,
        *,
        time_index: int = 0,
        scale: float = 1.0,
        training_stride: int = 1,
        max_training_points: Optional[int] = None,
        **kwargs,
    ) -> "HelmholtzDataDrivenFlowField":
        bundle = load_era5(data_path)
        winds = bundle.winds
        if winds.ndim != config.ndim + 2:
            raise ValueError("cache rank does not match config")
        if tuple(winds.shape[1 : 1 + config.ndim]) != tuple(config.shape):
            raise ValueError("cache grid shape does not match config")
        if not 0 <= time_index < winds.shape[0]:
            raise ValueError("time_index is outside cache range")
        if training_stride <= 0:
            raise ValueError("training_stride must be positive")

        axes = [np.arange(1, n + 1, dtype=np.float64) for n in config.shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        positions_grid = np.stack(mesh, axis=-1)
        slices = tuple(slice(None, None, training_stride) for _ in config.shape)
        positions = positions_grid[slices].reshape(-1, config.ndim)
        velocities = (winds[time_index][slices] * float(scale)).reshape(
            positions.shape[0], -1
        )

        if max_training_points is not None and positions.shape[0] > max_training_points:
            rng = np.random.default_rng(int(kwargs.get("feature_seed", 0)))
            chosen = rng.choice(positions.shape[0], size=max_training_points, replace=False)
            positions = positions[chosen]
            velocities = velocities[chosen]

        model = cls(config, positions, velocities, **kwargs)
        model.data_metadata = bundle.meta
        model.time_index = int(time_index)
        model.training_points = int(positions.shape[0])
        return model

    def _helmholtz_features(self, positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        phase = np.asarray(positions, dtype=np.float64) @ self._omegas.T + self._phases
        sin_phase = np.sqrt(2.0 / self.num_features) * np.sin(phase)
        omega_x = self._omegas[:, 0]
        omega_y = self._omegas[:, 1]

        div_u = -sin_phase * omega_x
        div_v = -sin_phase * omega_y
        curl_u = sin_phase * omega_y
        curl_v = -sin_phase * omega_x
        u_features = np.concatenate([div_u, curl_u], axis=1)
        v_features = np.concatenate([div_v, curl_v], axis=1)
        return u_features, v_features

    def _design_matrix(self, positions: np.ndarray) -> np.ndarray:
        u_features, v_features = self._helmholtz_features(positions)
        n = positions.shape[0]
        design = np.zeros((2 * n, 2 * self.num_features + 2), dtype=np.float64)
        design[0::2, :-2] = u_features
        design[1::2, :-2] = v_features
        design[0::2, -2] = 1.0
        design[1::2, -1] = 1.0
        return design

    def _unstandardize(self, values: np.ndarray) -> np.ndarray:
        return values * self._target_std + self._target_mean

    def _predict(self, positions: np.ndarray) -> np.ndarray:
        u_features, v_features = self._helmholtz_features(positions)
        return np.column_stack([
            u_features @ self._weights[:-2] + self._weights[-2],
            v_features @ self._weights[:-2] + self._weights[-1],
        ])

    def reset(self, rng_key: jnp.ndarray) -> None:
        axes = [np.arange(1, n + 1, dtype=np.float64) for n in self.config.shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        positions = np.stack(mesh, axis=-1).reshape(-1, self.ndim)
        values = self._unstandardize(self._predict(positions))
        self._precomputed = values.reshape(*self.config.shape, 2)

    def velocity_at(self, position: GridPosition) -> Tuple[float, Optional[float]]:
        coords: Sequence[float] = (position.i, position.j, position.k)
        value = self._unstandardize(self._predict(np.asarray([coords], dtype=np.float64)))[0]
        return float(value[0]), float(value[1])

    def velocity_field(self) -> np.ndarray:
        if self._precomputed is None:
            raise RuntimeError("HelmholtzDataDrivenFlowField.reset() must be called first")
        return np.asarray(self._precomputed)
