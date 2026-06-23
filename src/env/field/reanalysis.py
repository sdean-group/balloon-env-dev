"""Real wind source: ERA5 reanalysis via linear interpolation.

A :class:`ReanalysisFlowField` answers ``velocity_at(p)`` by (bi/tri)linear interpolation of
real ERA5 winds that were pre-resampled onto the env grid offline. Like every
:class:`FlowField` it is a pure spatial source: ``reset(key)`` selects which historical time
slice is "this episode" and ``velocity_at`` reports the deterministic interpolated velocity.
All dynamics (noise, clipping, displacement) live in the arena.

Design parallels :class:`SyntheticFlowField`:
- 2D: the wind varies over BOTH grid axes ``(x, y)`` but has a single component ``u``;
  ``velocity_at`` returns ``(u, None)``.
- 3D: ambient ``(x, y)`` plus controllable ``z`` (vertical level); returns ``(u, v)``.

Unlike the streamfunction GP, trilinear interpolation is NOT divergence-free -- a real but
usually-acceptable tradeoff for using measured winds.
"""

from typing import Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import RegularGridInterpolator

from .flow_field import FlowField
from .era5_data import load_era5
from ..utils.types import GridPosition, GridConfig


class ReanalysisFlowField(FlowField):
    """ERA5 wind field interpolated linearly onto continuous grid positions.

    The cached data is fixed; "drawing a realization" at :meth:`reset` means selecting which
    historical time slice to use this episode -- the data-backed analogue of the GP drawing
    new weights. ``slice_mode="random"`` samples a slice from the PRNG key (training
    diversity); ``"fixed"`` always uses slice 0 (reproducible evaluation).
    """

    def __init__(
        self,
        config: GridConfig,
        data_path: str,
        *,
        scale: float = 1.0,
        slice_mode: str = "random",
        fixed_index: int = 0,
    ):
        """Initialize the reanalysis field.

        Args:
            config: Grid configuration; its shape must match the cached data.
            data_path: Path to the ``.npz`` ERA5 cache (see :mod:`era5_data`).
            scale: Multiplier converting native units (m/s) to grid cells/step.
            slice_mode: ``"random"`` (sample a time slice per reset) or ``"fixed"`` (slice 0).
            fixed_index: Cache time index selected when ``slice_mode="fixed"``.

        Raises:
            ValueError: if ``slice_mode`` is invalid, or if the data rank/shape does not
                match ``config`` (dimension or grid-size mismatch).
        """
        super().__init__(config)

        if slice_mode not in ("fixed", "random"):
            raise ValueError(
                f"slice_mode must be 'fixed' or 'random', got {slice_mode!r}"
            )

        bundle = load_era5(data_path)
        winds = bundle.winds  # (T, n_x, n_y[, n_z], C)

        data_spatial_ndim = winds.ndim - 2
        if data_spatial_ndim != self.ndim:
            raise ValueError(
                f"data is {data_spatial_ndim}D but config is {self.ndim}D"
            )
        data_grid_shape = tuple(winds.shape[1 : 1 + self.ndim])
        if data_grid_shape != tuple(self.config.shape):
            raise ValueError(
                f"data grid {data_grid_shape} does not match config grid "
                f"{tuple(self.config.shape)}"
            )

        self._scale = float(scale)
        self._slice_mode = slice_mode
        self._meta = bundle.meta

        # Apply unit scaling once, up front, so velocity_at / velocity_field are scale-free.
        self._winds = winds * self._scale
        self._T = self._winds.shape[0]
        if not 0 <= fixed_index < self._T:
            raise ValueError(
                f"fixed_index must be in [0, {self._T - 1}], got {fixed_index}"
            )
        self._fixed_index = int(fixed_index)
        self.current_time_index: Optional[int] = None

        # Interpolation axes over the 1-indexed continuous domain [1, n] on each grid axis,
        # matching GridPosition's convention.
        self._axes = tuple(
            np.arange(1, n + 1, dtype=np.float64) for n in self.config.shape
        )

        # Built at reset() once a slice is chosen.
        self._interp_u: Optional[RegularGridInterpolator] = None
        self._interp_v: Optional[RegularGridInterpolator] = None  # 3D only
        self._current_slice: Optional[np.ndarray] = None

    def reset(self, rng_key: jnp.ndarray) -> None:
        """Select this episode's time slice and (re)build the interpolators."""
        if self._slice_mode == "fixed":
            t = self._fixed_index
        else:
            t = int(jax.random.randint(rng_key, (), 0, self._T))

        self.current_time_index = t
        sl = self._winds[t]  # (n_x, n_y[, n_z], C)
        self._current_slice = sl
        # bounds_error=False, fill_value=None -> linear extrapolation at the edges, so a
        # position sitting exactly on n_x (or a hair beyond from clipping) stays finite.
        self._interp_u = RegularGridInterpolator(
            self._axes, sl[..., 0], method="linear", bounds_error=False, fill_value=None
        )
        if self.ndim == 3:
            self._interp_v = RegularGridInterpolator(
                self._axes, sl[..., 1], method="linear", bounds_error=False, fill_value=None
            )

    def velocity_at(self, position: GridPosition) -> Tuple[float, Optional[float]]:
        """Linearly interpolated (u, v) at a continuous position. v is None in 2D.

        The 2D field varies over both grid axes (x, y) but has a single component u, mirroring
        :class:`SyntheticFlowField`; the 3D field interpolates over (x, y, z).
        """
        if self._interp_u is None:
            raise RuntimeError(
                "ReanalysisFlowField.reset() must be called before velocity_at()"
            )

        if self.ndim == 2:
            pt = np.array([[position.i, position.j]], dtype=np.float64)
            return (float(self._interp_u(pt)[0]), None)

        pt = np.array([[position.i, position.j, position.k]], dtype=np.float64)
        return (float(self._interp_u(pt)[0]), float(self._interp_v(pt)[0]))

    def velocity_field(self) -> np.ndarray:
        """Current slice's gridded velocities: ``(n_x,n_y,1)`` (2D) / ``(n_x,n_y,n_z,2)`` (3D)."""
        if self._current_slice is None:
            raise RuntimeError(
                "ReanalysisFlowField.reset() must be called before velocity_field()"
            )
        return np.asarray(self._current_slice)
