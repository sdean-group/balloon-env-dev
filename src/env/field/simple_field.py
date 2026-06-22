"""Simple field implementation with uniform random displacements on ambient axes."""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Optional, Tuple

from .abstract_field import AbstractField
from ..utils.types import GridPosition, DisplacementObservation, GridConfig


class SimpleField(AbstractField):
    """Simple field with uniform random displacements (continuous).

    Supports both 2D and 3D settings:
    - 3D: Samples (u, v) uniformly from [-d_max, +d_max]^2
    - 2D: Samples (u,) uniformly from [-d_max, +d_max]

    This is a stateless field - displacements are sampled independently
    on each call, so reset() is a no-op.
    """

    def __init__(self, config: GridConfig, d_max: float, *, disp_levels=None):
        """Initialize simple field.

        Args:
            config: Grid configuration.
            d_max: Maximum displacement magnitude on ambient axes (continuous).
            disp_levels: Integer displacement resolution for analytical PMFs.
        """
        super().__init__(config, d_max, disp_levels=disp_levels)
    
    def reset(self, rng_key: jnp.ndarray) -> None:
        """Reset field (no-op for stateless simple field).
        
        Args:
            rng_key: RNG key (unused - displacements sampled fresh each time).
        """
        # No state to reset - each sample_displacement call is independent
        pass
    
    def sample_displacement(
        self, position: GridPosition, rng_key: jnp.ndarray
    ) -> DisplacementObservation:
        """Sample uniform random displacement on ambient axes (continuous).

        Args:
            position: Current grid position (unused in this simple field).
            rng_key: JAX PRNG key for sampling.

        Returns:
            Displacement observation:
            - 3D: (u, v) both sampled uniformly from [-d_max, d_max]
            - 2D: (u, None) only first ambient axis
        """
        d_max = self.d_max

        if self.ndim == 3:
            # 3D: sample both u and v
            key_u, key_v = jax.random.split(rng_key)
            u = jax.random.uniform(key_u, shape=(), minval=-d_max, maxval=d_max)
            v = jax.random.uniform(key_v, shape=(), minval=-d_max, maxval=d_max)
            return DisplacementObservation(float(u), float(v))
        else:
            # 2D: sample only u
            u = jax.random.uniform(rng_key, shape=(), minval=-d_max, maxval=d_max)
            return DisplacementObservation(float(u), None)

    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Return uniform PMF over the discretized displacement space.

        Returns: (L = disp_levels)
            - 3D: Array of shape (2*L+1, 2*L+1) with uniform probabilities
            - 2D: Array of shape (2*L+1,) with uniform probabilities
        """
        size = 2 * self.disp_levels + 1

        if self.ndim == 3:
            return np.ones((size, size), dtype=np.float32) / (size ** 2)
        else:
            return np.ones((size,), dtype=np.float32) / size

    def get_displacement_pmf_grid(self) -> jnp.ndarray:
        """Uniform PMF broadcast over the entire grid."""
        size = 2 * self.disp_levels + 1
        if self.ndim == 2:
            return jnp.full((*self.config.shape, size), 1.0 / size)
        else:
            return jnp.full((*self.config.shape, size, size), 1.0 / (size * size))
    
    def get_mean_displacement(self, position: GridPosition) -> Tuple[float, ...]:
        """Return mean displacement (zero for uniform distribution).
        
        For uniform distribution over {-d_max, ..., +d_max}, mean is 0.
        
        Args:
            position: Grid position (unused - field is spatially uniform).
            
        Returns:
            - 3D: (0.0, 0.0) tuple
            - 2D: (0.0,) tuple
        """
        if self.ndim == 3:
            return (0.0, 0.0)
        else:
            return (0.0,)