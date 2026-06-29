"""Simple constant / uniform-drift flow fields.

These replace the old ``SimpleField``, which sampled a fresh uniform displacement on
*every* call -- that was per-step noise, not a field, and violated "deterministic
after reset()". Both fields here are proper :class:`FlowField` sources: spatially
uniform and fixed after ``reset()``.

- ``ConstantDriftField`` -- a fixed drift vector supplied at construction.
- ``UniformDriftField`` -- one random drift vector drawn (uniformly) at ``reset()``.
"""

from typing import Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .flow_field import FlowField
from ..utils.types import GridPosition, GridConfig


class ConstantDriftField(FlowField):
    """Spatially uniform, constant velocity field.

    Returns the same drift vector everywhere, fixed for all time. ``reset`` is a
    no-op (the drift is deterministic).
    """

    def __init__(self, config: GridConfig, drift: Sequence[float]):
        """Initialize the constant-drift field.

        Args:
            config: Grid configuration.
            drift: Velocity vector. Length 1 for 2D ((u,)) or 2 for 3D ((u, v)).
        """
        super().__init__(config)
        expected = 1 if self.ndim == 2 else 2
        if len(drift) != expected:
            raise ValueError(
                f"drift must have length {expected} for {self.ndim}D, got {len(drift)}"
            )
        self._u = float(drift[0])
        self._v = float(drift[1]) if self.ndim == 3 else None

    def reset(self, rng_key: jnp.ndarray) -> None:
        pass

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        return (self._u, self._v)  # time-invariant: t ignored

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        if self.ndim == 3:
            field = np.empty((self.config.n_x, self.config.n_y, self.config.n_z, 2))
            field[..., 0] = self._u
            field[..., 1] = self._v
            return field
        field = np.empty((self.config.n_x, self.config.n_y, 1))
        field[..., 0] = self._u
        return field


class UniformDriftField(FlowField):
    """Spatially uniform field whose drift is drawn uniformly once per episode.

    At ``reset()`` a single drift vector is sampled from ``[-max_drift, max_drift]``
    per component and held fixed for the episode.
    """

    def __init__(self, config: GridConfig, max_drift: float = 1.0):
        """Initialize the uniform-drift field.

        Args:
            config: Grid configuration.
            max_drift: Magnitude bound; each drift component ~ U[-max_drift, max_drift].
        """
        super().__init__(config)
        if max_drift < 0.0:
            raise ValueError(f"max_drift must be non-negative, got {max_drift}")
        self.max_drift = float(max_drift)
        self._u: Optional[float] = None
        self._v: Optional[float] = None

    def reset(self, rng_key: jnp.ndarray) -> None:
        if self.ndim == 3:
            key_u, key_v = jax.random.split(rng_key)
            self._u = float(
                jax.random.uniform(key_u, minval=-self.max_drift, maxval=self.max_drift)
            )
            self._v = float(
                jax.random.uniform(key_v, minval=-self.max_drift, maxval=self.max_drift)
            )
        else:
            self._u = float(
                jax.random.uniform(rng_key, minval=-self.max_drift, maxval=self.max_drift)
            )
            self._v = None

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        if self._u is None:
            raise RuntimeError("UniformDriftField.reset() must be called before velocity_at()")
        return (self._u, self._v)  # time-invariant: t ignored

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        if self._u is None:
            raise RuntimeError("UniformDriftField.reset() must be called before velocity_field()")
        if self.ndim == 3:
            field = np.empty((self.config.n_x, self.config.n_y, self.config.n_z, 2))
            field[..., 0] = self._u
            field[..., 1] = self._v
            return field
        field = np.empty((self.config.n_x, self.config.n_y, 1))
        field[..., 0] = self._u
        return field
