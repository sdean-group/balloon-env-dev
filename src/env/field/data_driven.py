"""Data-driven wind source fitted with random Fourier feature GP regression."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .era5_data import load_era5
from .flow_field import FlowField
from ..utils.types import GridConfig, GridPosition


class DataDrivenFlowField(FlowField):
    """Continuous GP approximation fitted to measured or reanalysis winds.

    The model uses an RBF random Fourier feature basis and Bayesian linear
    regression. One independent output model is fitted per horizontal wind
    component while sharing the same spatial features. ``reset`` selects either
    the posterior mean or one coherent posterior weight sample for the episode.
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
        sample_posterior: bool = False,
    ):
        super().__init__(config)
        if num_features <= 0 or num_features != int(num_features):
            raise ValueError("num_features must be a positive integer")
        if lengthscale <= 0:
            raise ValueError("lengthscale must be positive")
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")

        expected_components = 1 if self.ndim == 2 else 2
        x = np.asarray(positions, dtype=np.float64)
        y = np.asarray(velocities, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(
                f"positions must have shape (n, {self.ndim}), got {x.shape}"
            )
        if y.ndim != 2 or y.shape != (x.shape[0], expected_components):
            raise ValueError(
                f"velocities must have shape ({x.shape[0]}, {expected_components}), "
                f"got {y.shape}"
            )
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
        self.sample_posterior = bool(sample_posterior)

        rng = np.random.default_rng(self.feature_seed)
        self._omegas = rng.normal(
            scale=1.0 / self.lengthscale,
            size=(self.num_features, self.ndim),
        )
        self._phases = rng.uniform(0.0, 2.0 * np.pi, size=self.num_features)

        # Standardizing each output makes one noise_std meaningful across u/v and
        # avoids poorly conditioned solves when raw winds have a large mean.
        self._target_mean = y.mean(axis=0)
        self._target_std = y.std(axis=0)
        self._target_std = np.where(self._target_std < 1e-8, 1.0, self._target_std)
        y_standardized = (y - self._target_mean) / self._target_std

        phi = self._features(x)
        regularizer = self.noise_std**2
        precision = phi.T @ phi + regularizer * np.eye(self.num_features)
        try:
            self._precision_cholesky = np.linalg.cholesky(precision)
        except np.linalg.LinAlgError as exc:
            raise ValueError("could not factor GP posterior precision") from exc

        rhs = phi.T @ y_standardized
        intermediate = np.linalg.solve(self._precision_cholesky, rhs)
        self._posterior_mean = np.linalg.solve(
            self._precision_cholesky.T, intermediate
        )

        fitted = self._unstandardize(phi @ self._posterior_mean)
        self.training_rmse = float(np.sqrt(np.mean((fitted - y) ** 2)))
        self._weights: Optional[np.ndarray] = None
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
    ) -> "DataDrivenFlowField":
        """Fit a GP to one time slice from a cached ERA5/reanalysis field."""
        bundle = load_era5(data_path)
        winds = bundle.winds
        expected_rank = config.ndim + 2
        if winds.ndim != expected_rank:
            raise ValueError(
                f"cache rank {winds.ndim} does not match {config.ndim}D config"
            )
        if tuple(winds.shape[1 : 1 + config.ndim]) != tuple(config.shape):
            raise ValueError("cache grid shape does not match config")
        if not 0 <= time_index < winds.shape[0]:
            raise ValueError(
                f"time_index must be in [0, {winds.shape[0] - 1}], got {time_index}"
            )
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

        if max_training_points is not None:
            if max_training_points <= 0:
                raise ValueError("max_training_points must be positive")
            if positions.shape[0] > max_training_points:
                rng = np.random.default_rng(int(kwargs.get("feature_seed", 0)))
                chosen = rng.choice(
                    positions.shape[0], size=max_training_points, replace=False
                )
                positions = positions[chosen]
                velocities = velocities[chosen]

        model = cls(config, positions, velocities, **kwargs)
        model.data_metadata = bundle.meta
        model.time_index = int(time_index)
        model.training_points = int(positions.shape[0])
        return model

    def _features(self, positions: np.ndarray) -> np.ndarray:
        positions = np.asarray(positions, dtype=np.float64)
        phase = positions @ self._omegas.T + self._phases
        return np.sqrt(2.0 / self.num_features) * np.cos(phase)

    def _unstandardize(self, values: np.ndarray) -> np.ndarray:
        return values * self._target_std + self._target_mean

    def reset(self, rng_key: jnp.ndarray) -> None:
        """Select the posterior mean or draw one coherent posterior field."""
        if self.sample_posterior:
            z = np.asarray(
                jax.random.normal(
                    rng_key,
                    shape=self._posterior_mean.shape,
                    dtype=jnp.float64 if jax.config.x64_enabled else jnp.float32,
                ),
                dtype=np.float64,
            )
            perturbation = self.noise_std * np.linalg.solve(
                self._precision_cholesky.T, z
            )
            self._weights = self._posterior_mean + perturbation
        else:
            self._weights = self._posterior_mean.copy()

        axes = [np.arange(1, n + 1, dtype=np.float64) for n in self.config.shape]
        mesh = np.meshgrid(*axes, indexing="ij")
        positions = np.stack(mesh, axis=-1).reshape(-1, self.ndim)
        values = self._predict(positions)
        components = 1 if self.ndim == 2 else 2
        self._precomputed = values.reshape(*self.config.shape, components)

    def _predict(self, positions: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError(
                "DataDrivenFlowField.reset() must be called before prediction"
            )
        return self._unstandardize(self._features(positions) @ self._weights)

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        # v1: the fitted GP is static in time; t is accepted but ignored.
        # Temporal features (a t-frequency, mirroring SyntheticFlowField) are a TODO.
        coords: Sequence[float]
        if self.ndim == 2:
            coords = (position.i, position.j)
        else:
            coords = (position.i, position.j, position.k)
        value = self._predict(np.asarray([coords], dtype=np.float64))[0]
        if self.ndim == 2:
            return (float(value[0]), None)
        return (float(value[0]), float(value[1]))

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        if self._precomputed is None:
            raise RuntimeError(
                "DataDrivenFlowField.reset() must be called before velocity_field()"
            )
        return np.asarray(self._precomputed)
