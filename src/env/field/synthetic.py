"""Synthetic wind source: a Random Fourier Features Gaussian Process field.

Implements a GP velocity field using the RFF approximation for O(L) complexity per
test point.
- 2D: Scalar GP for the single ambient velocity component u.
- 3D: Streamfunction GP for a divergence-free (u, v) field.

This is a pure :class:`FlowField` -- it only draws a realization (``reset``) and
reports the deterministic velocity at a point (``velocity_at``). All dynamics
(noise, clipping, displacement) live in the arena.
"""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .flow_field import FlowField
from ..utils.types import GridPosition, GridConfig


class SyntheticFlowField(FlowField):
    """GP velocity field using Random Fourier Features approximation.

    Samples from a zero-mean GP with a Matern-nu covariance kernel using RFF.
    All internal computations use JAX arrays for autodiff compatibility.

    2D Mode:
        Scalar GP U(x, y) defines the velocity on the single ambient axis;
        ``velocity_at`` returns (u, None).

    3D Mode (streamfunction method):
        Scalar GP psi(x, y, z) is the streamfunction. The velocity field is
        u = -dpsi/dy, v = dpsi/dx (divergence-free by construction); RFF gives
        analytical derivatives via sin() terms.
    """

    def __init__(
        self,
        config: GridConfig,
        sigma: float = 1.0,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        num_features: int = 500,
        lengthscale_t: Optional[float] = None,
    ):
        """Initialize the synthetic RFF GP field.

        Args:
            config: Grid configuration specifying dimensions.
            sigma: GP marginal standard deviation (amplitude).
            lengthscale: Correlation length of the GP.
            nu: Matern smoothness parameter (commonly 0.5, 1.5, 2.5, or infinity for RBF).
            num_features: Number of random Fourier features (L). Higher = better approximation.
            lengthscale_t: Temporal correlation length (in elapsed-step units). ``None``
                (default) makes the field frozen-per-episode -- identical to before. A
                positive value adds a temporal frequency so the field *evolves within an
                episode*; larger = slower drift. Uses the same Matern-nu smoothness as space.
        """
        super().__init__(config)

        if sigma <= 0.0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if lengthscale <= 0.0:
            raise ValueError(f"lengthscale must be positive, got {lengthscale}")
        if nu <= 0.0:
            raise ValueError(f"nu must be positive, got {nu}")
        if num_features != int(num_features):
            raise ValueError(f"num_features must be an integer, got {num_features}")
        if num_features <= 0:
            raise ValueError(f"num_features must be positive, got {num_features}")
        if lengthscale_t is not None and lengthscale_t <= 0.0:
            raise ValueError(f"lengthscale_t must be positive, got {lengthscale_t}")

        self.sigma = sigma
        self.lengthscale = lengthscale
        self.nu = nu
        self.num_features = int(num_features)
        self.lengthscale_t = float(lengthscale_t) if lengthscale_t is not None else None

        # Spatial dimension for frequency sampling
        self._spatial_dim = 2 if self.ndim == 2 else 3

        # RFF components as JAX arrays (initialized in reset)
        self._omegas: Optional[jnp.ndarray] = None  # (L, d) spatial frequencies
        self._phases: Optional[jnp.ndarray] = None  # (L,) phase shifts
        self._weights: Optional[jnp.ndarray] = None  # (L,) Gaussian weights
        self._omega_t: Optional[jnp.ndarray] = None  # (L,) temporal frequencies (if time-varying)

        # Precomputed grid locations and field values (JAX arrays)
        self._grid_locations: Optional[jnp.ndarray] = None
        self._precomputed_u: Optional[jnp.ndarray] = None  # Mean u field
        self._precomputed_v: Optional[jnp.ndarray] = None  # Mean v field (3D only)

        # For 3D: store omega components for velocity computation
        self._omega_x: Optional[jnp.ndarray] = None
        self._omega_y: Optional[jnp.ndarray] = None

        # Build grid locations
        self._build_grid_locations()

    def _build_grid_locations(self) -> None:
        """Create JAX array of grid point coordinates."""
        if self.ndim == 2:
            i_coords = jnp.arange(1, self.config.n_x + 1)
            j_coords = jnp.arange(1, self.config.n_y + 1)
            I, J = jnp.meshgrid(i_coords, j_coords, indexing="ij")
            self._grid_locations = jnp.column_stack([I.ravel(), J.ravel()])
        else:
            i_coords = jnp.arange(1, self.config.n_x + 1)
            j_coords = jnp.arange(1, self.config.n_y + 1)
            k_coords = jnp.arange(1, self.config.n_z + 1)
            I, J, K = jnp.meshgrid(i_coords, j_coords, k_coords, indexing="ij")
            self._grid_locations = jnp.column_stack([I.ravel(), J.ravel(), K.ravel()])

    def _sample_matern_frequencies(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Sample frequencies from the Matern spectral density (Student's t form).

        The spectral density of a Matern-nu kernel is a multivariate Student's t:
            omega ~ t_d(0, (1/ell^2)*I, 2*nu)
        Sampling: omega = (1/ell) * sqrt(2*nu / U) * Z, with Z ~ N(0, I_d),
        U ~ chi^2_{2*nu}, and chi^2_k = 2 * Gamma(k/2, 1).
        """
        L = self.num_features
        d = self._spatial_dim

        key_z, key_u = jax.random.split(rng_key)

        Z = jax.random.normal(key_z, shape=(L, d))
        U = 2.0 * jax.random.gamma(key_u, a=self.nu, shape=(L,))

        scale = 1.0 / self.lengthscale
        omegas = scale * jnp.sqrt(2 * self.nu / U[:, None]) * Z
        return omegas

    def _sample_temporal_frequencies(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Sample (L,) temporal frequencies from a 1-D Matern-nu spectral density.

        Same scale-mixture form as :meth:`_sample_matern_frequencies` but 1-D and using
        ``lengthscale_t``: omega_t = (1/ell_t) * sqrt(2*nu / U) * Z, Z ~ N(0, 1).
        """
        L = self.num_features
        key_z, key_u = jax.random.split(rng_key)
        Z = jax.random.normal(key_z, shape=(L,))
        U = 2.0 * jax.random.gamma(key_u, a=self.nu, shape=(L,))
        return (1.0 / self.lengthscale_t) * jnp.sqrt(2 * self.nu / U) * Z

    def reset(self, rng_key: jnp.ndarray) -> None:
        """Draw a new realization by sampling RFF weights and recomputing the field."""
        L = self.num_features

        # Keep the static (lengthscale_t is None) RNG split identical to preserve the
        # exact realizations existing golden tests assert against.
        if self.lengthscale_t is None:
            key_omega, key_phase, key_weights = jax.random.split(rng_key, 3)
            self._omega_t = None
        else:
            key_omega, key_phase, key_weights, key_omega_t = jax.random.split(rng_key, 4)
            self._omega_t = self._sample_temporal_frequencies(key_omega_t)

        self._omegas = self._sample_matern_frequencies(key_omega)
        self._phases = jax.random.uniform(
            key_phase, shape=(L,), minval=0, maxval=2 * jnp.pi
        )
        self._weights = jax.random.normal(key_weights, shape=(L,))

        if self.ndim == 3:
            self._omega_x = self._omegas[:, 0]
            self._omega_y = self._omegas[:, 1]

        self._precompute_field()

    @property
    def time_varying(self) -> bool:
        return self.lengthscale_t is not None

    def _grid_velocity(self, t: float = 0.0):
        """GP field values at all grid points and episode time ``t`` (JAX arrays).

        Returns ``(u_grid, v_grid)`` reshaped to the grid; ``v_grid`` is None in 2D.
        The temporal frequency enters ``theta`` additively (``+ t * omega_t``), so the
        spatial derivatives used for the 3D curl are unchanged -- the field stays
        divergence-free at every fixed ``t``.
        """
        theta = self._grid_locations @ self._omegas.T + self._phases[None, :]
        if self._omega_t is not None:
            theta = theta + t * self._omega_t[None, :]

        scale = jnp.sqrt(2 * self.sigma**2 / self.num_features)

        if self.ndim == 2:
            psi = scale * (jnp.cos(theta) @ self._weights)
            return psi.reshape(self.config.n_x, self.config.n_y), None

        # Streamfunction method for divergence-free field:
        # u = -dpsi/dy = scale * sum_l w_l * omega_y,l * sin(theta_l)
        # v =  dpsi/dx = -scale * sum_l w_l * omega_x,l * sin(theta_l)
        sin_theta = jnp.sin(theta)
        u = scale * (sin_theta @ (self._weights * self._omega_y))
        v = -scale * (sin_theta @ (self._weights * self._omega_x))
        shape = (self.config.n_x, self.config.n_y, self.config.n_z)
        return u.reshape(shape), v.reshape(shape)

    def _precompute_field(self) -> None:
        """Cache the t=0 grid (cheap default for velocity_field() and static fields)."""
        self._precomputed_u, self._precomputed_v = self._grid_velocity(0.0)

    def velocity_at_point(
        self, x: float, y: float, z: Optional[float] = None, t: float = 0.0
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Compute velocity (u, v) at a continuous point and time -- JAX differentiable.

        Recomputes the field at arbitrary continuous coordinates, enabling autodiff
        (e.g. for divergence verification). For 3D returns the streamfunction-derived
        velocity; for 2D returns (u, None). Returns JAX scalars.
        """
        scale = jnp.sqrt(2 * self.sigma**2 / self.num_features)

        if self.ndim == 2:
            r = jnp.array([x, y])
            theta = self._omegas @ r + self._phases
            if self._omega_t is not None:
                theta = theta + t * self._omega_t
            u = scale * jnp.sum(self._weights * jnp.cos(theta))
            return (u, None)
        else:
            r = jnp.array([x, y, z])
            theta = self._omegas @ r + self._phases
            if self._omega_t is not None:
                theta = theta + t * self._omega_t
            sin_theta = jnp.sin(theta)
            u = scale * jnp.sum(self._weights * self._omega_y * sin_theta)
            v = -scale * jnp.sum(self._weights * self._omega_x * sin_theta)
            return (u, v)

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        """Deterministic (u, v) at a continuous grid position. v is None in 2D.

        Evaluates the GP directly at the continuous coordinate so fractional
        positions are supported; at integer positions this matches the precomputed
        grid values exactly. (Temporal axis added in a later step; ``t`` ignored here.)
        """
        if self.ndim == 2:
            u, _ = self.velocity_at_point(position.i, position.j, t=t)
            return (float(u), None)
        else:
            u, v = self.velocity_at_point(position.i, position.j, position.k, t=t)
            return (float(u), float(v))

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        """Velocity field over the grid at episode time ``t``.

        ``t == 0`` reuses the cached grid from reset; other times recompute on demand
        (a time-varying field can't precompute the whole episode).

        Returns:
            - 2D: shape (n_x, n_y, 1) with (u,) at each point
            - 3D: shape (n_x, n_y, n_z, 2) with (u, v) at each point
        """
        if t == 0.0:
            u_grid, v_grid = self._precomputed_u, self._precomputed_v
        else:
            u_grid, v_grid = self._grid_velocity(t)

        if self.ndim == 2:
            return np.asarray(u_grid[:, :, jnp.newaxis])
        return np.asarray(jnp.stack([u_grid, v_grid], axis=-1))
