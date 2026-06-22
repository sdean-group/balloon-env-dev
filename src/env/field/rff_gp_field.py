"""Random Fourier Features Gaussian Process field implementation.

Implements GP-based environmental fields using RFF approximation for O(L) complexity per test point.
- 2D: Scalar GP for single ambient displacement u
- 3D: Streamfunction GP for divergence-free (u, v) field

"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
from typing import Optional, Tuple

from .abstract_field import AbstractField
from ..utils.types import GridPosition, DisplacementObservation, GridConfig


def _compute_1d_pmf_grid(
    means: jnp.ndarray, levels: int, noise_std: float
) -> jnp.ndarray:
    """Vectorized 1D clipped-normal PMF over an array of means.

    Args:
        means: shape ``(*grid_shape)`` — mean displacement at each cell.
        levels: Integer displacement resolution; PMF spans offsets {-levels..levels}.
        noise_std: Standard deviation of observation noise.

    Returns:
        shape ``(*grid_shape, 2*levels+1)`` PMF array.
    """
    n_bins = 2 * levels + 1

    if levels == 0:
        return jnp.ones((*means.shape, 1), dtype=jnp.float32)

    if noise_std == 0.0:
        rounded = jnp.round(jnp.clip(means, -levels, levels)).astype(jnp.int32)
        return jax.nn.one_hot(rounded + levels, n_bins)

    k = jnp.arange(-levels, levels + 1, dtype=jnp.float32)  # (n_bins,)
    mu = means[..., None]  # (*grid, 1)

    edges_upper = (k + 0.5 - mu) / noise_std  # (*grid, n_bins)
    edges_lower = (k - 0.5 - mu) / noise_std  # (*grid, n_bins)

    cdf_upper = jax_norm.cdf(edges_upper)
    cdf_lower = jax_norm.cdf(edges_lower)

    pmf = cdf_upper - cdf_lower
    # Left boundary: accumulate all mass below -d_max + 0.5
    pmf = pmf.at[..., 0].set(cdf_upper[..., 0])
    # Right boundary: accumulate all mass above d_max - 0.5
    pmf = pmf.at[..., -1].set(1.0 - cdf_lower[..., -1])

    return pmf


class RFFGPField(AbstractField):
    """GP field using Random Fourier Features approximation.

    Samples from a zero-mean GP with Matern-nu covariance kernel using RFF.
    All internal computations use JAX arrays for autodiff compatibility.

    2D Mode:
        - Scalar GP U(x, y) defines displacement on single ambient axis
        - sample_displacement returns (u, None) where u = U(pos) + noise

    3D Mode (streamfunction method):
        - Scalar GP psi(x, y, z) is the streamfunction
        - Velocity field: u = -dpsi/dy, v = dpsi/dx (divergence-free by construction)
        - RFF gives analytical derivatives via sin() terms

    Displacement flow:
        1. GP sample gives mean displacement at each point (fixed per episode)
        2. sample_displacement adds Gaussian noise to mean
        3. Continuous value is clipped to [-d_max, d_max]
        4. Arena applies the continuous (u, v) displacement directly (no rounding)
    """

    def __init__(
        self,
        config: GridConfig,
        d_max: float,
        sigma: float = 1.0,
        lengthscale: float = 1.0,
        nu: float = 2.5,
        num_features: int = 500,
        noise_std: float = 0.1,
        disp_levels: Optional[int] = None,
    ):
        """Initialize RFF GP field.

        Args:
            config: Grid configuration specifying dimensions.
            d_max: Maximum displacement magnitude (displacements clipped to [-d_max, d_max]).
            sigma: GP marginal standard deviation (amplitude).
            lengthscale: Correlation length of the GP.
            nu: Matern smoothness parameter (commonly 0.5, 1.5, 2.5, or infinity for RBF).
            num_features: Number of random Fourier features (L). Higher = better approximation.
            noise_std: Standard deviation of observation noise added to GP samples.
            disp_levels: Integer displacement resolution for analytical PMFs
                (DP oracle). Defaults to ceil(d_max).
        """
        super().__init__(config, d_max, disp_levels=disp_levels)

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
        if noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}")

        self.sigma = sigma
        self.lengthscale = lengthscale
        self.nu = nu
        self.num_features = int(num_features)
        self.noise_std = noise_std

        # Spatial dimension for frequency sampling
        self._spatial_dim = 2 if self.ndim == 2 else 3

        # RFF components as JAX arrays (initialized in reset)
        self._omegas: Optional[jnp.ndarray] = None  # (L, d) frequencies
        self._phases: Optional[jnp.ndarray] = None  # (L,) phase shifts
        self._weights: Optional[jnp.ndarray] = None  # (L,) Gaussian weights

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
            # 2D: grid over (x, y) = (i, j) coordinates
            i_coords = jnp.arange(1, self.config.n_x + 1)
            j_coords = jnp.arange(1, self.config.n_y + 1)
            I, J = jnp.meshgrid(i_coords, j_coords, indexing="ij")
            self._grid_locations = jnp.column_stack([I.ravel(), J.ravel()])
        else:
            # 3D: grid over (x, y, z) = (i, j, k) coordinates
            i_coords = jnp.arange(1, self.config.n_x + 1)
            j_coords = jnp.arange(1, self.config.n_y + 1)
            k_coords = jnp.arange(1, self.config.n_z + 1)
            I, J, K = jnp.meshgrid(i_coords, j_coords, k_coords, indexing="ij")
            self._grid_locations = jnp.column_stack([I.ravel(), J.ravel(), K.ravel()])

    def _sample_matern_frequencies(self, rng_key: jnp.ndarray) -> jnp.ndarray:
        """Sample frequencies from Matern spectral density using Student's t representation.

        The spectral density of Matern-nu kernel is a multivariate Student's t:
            omega ~ t_d(0, (1/ell^2)*I, 2*nu)

        Sampling: omega = (1/ell) * sqrt(2*nu / U) * Z
        where Z ~ N(0, I_d), U ~ chi^2_{2*nu}

        chi^2_k can be sampled as 2 * Gamma(k/2, 1).

        Args:
            rng_key: JAX PRNG key.

        Returns:
            (L, d) JAX array of frequency samples.
        """
        L = self.num_features
        d = self._spatial_dim

        key_z, key_u = jax.random.split(rng_key)

        # Sample standard normals: Z ~ N(0, I_d)
        Z = jax.random.normal(key_z, shape=(L, d))

        # Sample chi-squared: chi^2_{2*nu} = 2 * Gamma(nu, 1)
        # JAX gamma uses shape (alpha) and rate parameterization isn't directly available,
        # so we use: chi^2_k = 2 * Gamma(k/2, 1) where Gamma is shape-parameterized TODO: check this!
        U = 2.0 * jax.random.gamma(key_u, a=self.nu, shape=(L,))

        # Compute frequencies
        scale = 1.0 / self.lengthscale
        omegas = scale * jnp.sqrt(2 * self.nu / U[:, None]) * Z

        return omegas

    def reset(self, rng_key: jnp.ndarray) -> None:
        """Reset field by sampling new RFF weights and recomputing field values.

        Args:
            rng_key: JAX PRNG key for reproducibility.
        """
        L = self.num_features

        # Split keys for different random components
        key_omega, key_phase, key_weights = jax.random.split(rng_key, 3)

        # Sample frequencies from Matern spectral density
        self._omegas = self._sample_matern_frequencies(key_omega)

        # Sample uniform phases in [0, 2*pi)
        self._phases = jax.random.uniform(
            key_phase, shape=(L,), minval=0, maxval=2 * jnp.pi
        )

        # Sample Gaussian weights
        self._weights = jax.random.normal(key_weights, shape=(L,))

        # Store omega components for velocity computation (3D)
        if self.ndim == 3:
            self._omega_x = self._omegas[:, 0]
            self._omega_y = self._omegas[:, 1]

        # Precompute field values over entire grid
        self._precompute_field()

    def _precompute_field(self) -> None:
        """Precompute GP field values at all grid points using JAX."""
        # Compute theta = omega . r + phase for all locations
        # theta shape: (n_points, L)
        theta = self._grid_locations @ self._omegas.T + self._phases[None, :]

        cos_theta = jnp.cos(theta)
        sin_theta = jnp.sin(theta)

        # Scale factor for RFF
        scale = jnp.sqrt(2 * self.sigma**2 / self.num_features)

        if self.ndim == 2:
            # 2D: scalar GP for u displacement
            # psi = scale * sum_l w_l * cos(theta_l)
            psi = scale * (cos_theta @ self._weights)
            self._precomputed_u = psi.reshape(self.config.n_x, self.config.n_y)
            self._precomputed_v = None
        else:
            # 3D: streamfunction method for divergence-free field
            # psi = scale * sum_l w_l * cos(theta_l)
            # u = -dpsi/dy = scale * sum_l w_l * omega_y,l * sin(theta_l)
            # v =  dpsi/dx = -scale * sum_l w_l * omega_x,l * sin(theta_l)

            u = scale * (sin_theta @ (self._weights * self._omega_y))
            v = -scale * (sin_theta @ (self._weights * self._omega_x))

            self._precomputed_u = u.reshape(
                self.config.n_x, self.config.n_y, self.config.n_z
            )
            self._precomputed_v = v.reshape(
                self.config.n_x, self.config.n_y, self.config.n_z
            )

    def _get_mean_at_position(
        self, position: GridPosition
    ) -> Tuple[float, Optional[float]]:
        """Get GP mean displacement at a (continuous) grid position.

        Evaluates the GP directly at the continuous coordinate via
        ``velocity_at_point`` so that fractional positions are supported. At
        integer positions this matches the precomputed grid values exactly
        (both apply the same RFF formula).

        Args:
            position: Grid position (1-indexed, continuous).

        Returns:
            (u_mean, v_mean) where v_mean is None for 2D.
        """
        if self.ndim == 2:
            u_mean, _ = self.velocity_at_point(position.i, position.j)
            return (float(u_mean), None)
        else:
            u_mean, v_mean = self.velocity_at_point(position.i, position.j, position.k)
            return (float(u_mean), float(v_mean))

    def velocity_at_point(
        self, x: float, y: float, z: Optional[float] = None
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """Compute velocity (u, v) at a continuous point - JAX differentiable.

        This method recomputes the field at arbitrary continuous coordinates,
        enabling autodiff for divergence verification. For 3D, returns the
        streamfunction-derived velocity; for 2D, returns (u, None).

        Args:
            x: x-coordinate (can be non-integer).
            y: y-coordinate (can be non-integer).
            z: z-coordinate for 3D (can be non-integer), ignored for 2D.

        Returns:
            (u, v) tuple of JAX scalars. v is None for 2D.
        """
        scale = jnp.sqrt(2 * self.sigma**2 / self.num_features)

        if self.ndim == 2:
            r = jnp.array([x, y])
            theta = self._omegas @ r + self._phases
            u = scale * jnp.sum(self._weights * jnp.cos(theta))
            return (u, None)
        else:
            r = jnp.array([x, y, z])
            theta = self._omegas @ r + self._phases
            sin_theta = jnp.sin(theta)
            u = scale * jnp.sum(self._weights * self._omega_y * sin_theta)
            v = -scale * jnp.sum(self._weights * self._omega_x * sin_theta)
            return (u, v)

    def sample_displacement(
        self, position: GridPosition, rng_key: jnp.ndarray
    ) -> DisplacementObservation:
        """Sample displacement at position by adding noise to GP mean.

        Args:
            position: Current grid position.
            rng_key: JAX PRNG key for sampling noise.

        Returns:
            DisplacementObservation with values clipped to [-d_max, d_max].
        """
        u_mean, v_mean = self._get_mean_at_position(position)

        if self.ndim == 2:
            # Sample noise and add to mean
            noise = float(jax.random.normal(rng_key) * self.noise_std)
            u = u_mean + noise
            return self._clip_displacement(u, None)
        else:
            # Sample noise for both components
            key_u, key_v = jax.random.split(rng_key)
            noise_u = float(jax.random.normal(key_u) * self.noise_std)
            noise_v = float(jax.random.normal(key_v) * self.noise_std)
            u = u_mean + noise_u
            v = v_mean + noise_v
            return self._clip_displacement(u, v)

    def get_mean_displacement(self, position: GridPosition) -> Tuple[float, ...]:
        """Get GP mean displacement at position (no noise).

        Args:
            position: Grid position to query.

        Returns:
            Mean displacement tuple:
            - 3D: (u_mean, v_mean)
            - 2D: (u_mean,)
        """
        u_mean, v_mean = self._get_mean_at_position(position)
        if self.ndim == 2:
            return (u_mean,)
        else:
            return (u_mean, v_mean)

    def get_mean_displacement_field(self) -> np.ndarray:
        """Get precomputed mean displacement field over entire grid.

        Returns:
            NumPy array with displacement values at each grid point:
            - 3D: shape (n_x, n_y, n_z, 2) with (u, v) at each point
            - 2D: shape (n_x, n_y, 1) with (u,) at each point
        """
        if self.ndim == 2:
            return np.asarray(self._precomputed_u[:, :, jnp.newaxis])
        else:
            return np.asarray(
                jnp.stack([self._precomputed_u, self._precomputed_v], axis=-1)
            )

    def get_displacement_pmf(self, position: GridPosition) -> np.ndarray:
        """Compute clipped PMF of discretized displacement at position.

        This matches runtime sampling:
            1) sample U_obs = mu + epsilon, epsilon ~ N(0, noise_std^2)
            2) clip U_obs to [-d_max, d_max]
            3) use round(clipped value) for state transitions

        Therefore, probability mass outside [-d_max, d_max] is accumulated at
        the boundary values ±d_max

        Args:
            position: Grid position to query.

        Returns:
            NumPy PMF array:
            - 2D: shape (2*d_max+1,), entry [i] = P(u=i-d_max)
            - 3D: shape (2*d_max+1, 2*d_max+1), entry [i,j] = P(u=i-d_max, v=j-d_max)
        """
        u_mean, v_mean = self._get_mean_at_position(position)
        L = self.disp_levels

        def compute_1d_pmf(mu: float) -> jnp.ndarray:
            """Compute 1D clipped PMF for a single component using JAX."""
            if L == 0:
                return jnp.ones(1, dtype=jnp.float32)

            k_values = jnp.arange(-L, L + 1)
            sigma_noise = self.noise_std

            if sigma_noise == 0.0:
                clipped = float(np.clip(mu, -L, L))
                rounded = int(round(clipped))
                one_hot = jnp.zeros_like(k_values, dtype=jnp.float32)
                return one_hot.at[rounded + L].set(1.0)

            pmf = jnp.zeros_like(k_values, dtype=jnp.float32)

            # Left boundary: P(U_obs < -L + 0.5)
            left_boundary = jax_norm.cdf((-L + 0.5 - mu) / sigma_noise)
            pmf = pmf.at[0].set(left_boundary)

            # Interior bins: P(k-0.5 <= U_obs < k+0.5)
            if L > 0:
                interior_k = jnp.arange(-L + 1, L)
                upper = (interior_k + 0.5 - mu) / sigma_noise
                lower = (interior_k - 0.5 - mu) / sigma_noise
                interior_pmf = jax_norm.cdf(upper) - jax_norm.cdf(lower)
                pmf = pmf.at[1:-1].set(interior_pmf.astype(jnp.float32))

            # Right boundary: P(U_obs >= L - 0.5)
            right_boundary = 1.0 - jax_norm.cdf((L - 0.5 - mu) / sigma_noise)
            pmf = pmf.at[-1].set(right_boundary)
            return pmf

        if self.ndim == 2:
            return np.asarray(compute_1d_pmf(u_mean), dtype=np.float32)
        else:
            # 3D: joint PMF assuming independence of u and v given the field
            pmf_u = compute_1d_pmf(u_mean)
            pmf_v = compute_1d_pmf(v_mean)
            # Outer product for joint PMF
            joint_pmf = jnp.outer(pmf_u, pmf_v)
            return np.asarray(joint_pmf, dtype=np.float32)

    def get_displacement_pmf_grid(self) -> jnp.ndarray:
        """Vectorized PMF computation over the entire grid.

        Returns:
            - 2D: shape (n_x, n_y, 2*d_max+1)
            - 3D: shape (n_x, n_y, n_z, 2*d_max+1, 2*d_max+1)
        """
        if self.ndim == 2:
            return _compute_1d_pmf_grid(self._precomputed_u, self.disp_levels, self.noise_std)
        else:
            pmf_u = _compute_1d_pmf_grid(
                self._precomputed_u, self.disp_levels, self.noise_std
            )
            pmf_v = _compute_1d_pmf_grid(
                self._precomputed_v, self.disp_levels, self.noise_std
            )
            return pmf_u[..., :, None] * pmf_v[..., None, :]
