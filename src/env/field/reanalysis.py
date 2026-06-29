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
        steps_per_slice: Optional[float] = None,
    ):
        """Initialize the reanalysis field.

        Args:
            config: Grid configuration; its shape must match the cached data.
            data_path: Path to the ``.npz`` ERA5 cache (see :mod:`era5_data`).
            scale: Multiplier converting native units (m/s) to grid cells/step.
            slice_mode: ``"random"`` (sample a time slice per reset) or ``"fixed"`` (slice 0).
            fixed_index: Cache time index selected when ``slice_mode="fixed"``.
            steps_per_slice: How many episode steps span one cached slice interval. ``None``
                (default) freezes the chosen start slice for the whole episode -- identical to
                before. A positive value makes the field *evolve within an episode* by linearly
                interpolating between consecutive slices as ``t`` advances (smaller = faster
                weather). At ``t=0`` the temporal field equals the frozen start slice.

        Raises:
            ValueError: if ``slice_mode``/``steps_per_slice`` is invalid, or if the data
                rank/shape does not match ``config`` (dimension or grid-size mismatch).
        """
        super().__init__(config)

        if slice_mode not in ("fixed", "random"):
            raise ValueError(
                f"slice_mode must be 'fixed' or 'random', got {slice_mode!r}"
            )
        if steps_per_slice is not None and steps_per_slice <= 0:
            raise ValueError(
                f"steps_per_slice must be positive, got {steps_per_slice}"
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
        self.steps_per_slice = float(steps_per_slice) if steps_per_slice is not None else None

        # Apply unit scaling once, up front, so velocity_at / velocity_field are scale-free.
        self._winds = winds * self._scale
        self._T = self._winds.shape[0]
        if not 0 <= fixed_index < self._T:
            raise ValueError(
                f"fixed_index must be in [0, {self._T - 1}], got {fixed_index}"
            )
        self._fixed_index = int(fixed_index)
        self.current_time_index: Optional[int] = None
        self._t0: int = 0  # episode start slice, set at reset (temporal mode anchor)

        # Interpolation axes over the 1-indexed continuous domain [1, n] on each grid axis,
        # matching GridPosition's convention. Temporal mode prepends a slice-index axis.
        self._axes = tuple(
            np.arange(1, n + 1, dtype=np.float64) for n in self.config.shape
        )
        self._time_axis = np.arange(self._T, dtype=np.float64)

        # Built at reset() once a slice is chosen. In temporal mode these interpolate over
        # (time, *space); in static mode over space only.
        self._interp_u: Optional[RegularGridInterpolator] = None
        self._interp_v: Optional[RegularGridInterpolator] = None  # 3D only
        self._current_slice: Optional[np.ndarray] = None

    @property
    def time_varying(self) -> bool:
        # Evolves within an episode only if a temporal cadence is set AND there is more
        # than one slice to move between.
        return self.steps_per_slice is not None and self._T > 1

    def reset(self, rng_key: jnp.ndarray) -> None:
        """Select this episode's start slice and (re)build the interpolators.

        Static mode (``steps_per_slice is None``) builds spatial-only interpolators on the
        chosen slice, frozen for the episode. Temporal mode builds interpolators over
        ``(time, *space)`` so ``velocity_at(p, t)`` can move between slices as ``t`` grows.
        """
        if self._slice_mode == "fixed":
            t0 = self._fixed_index
        else:
            t0 = int(jax.random.randint(rng_key, (), 0, self._T))

        self.current_time_index = t0
        self._t0 = t0
        self._current_slice = self._winds[t0]  # (n_x, n_y[, n_z], C)

        # bounds_error=False, fill_value=None -> linear extrapolation at the edges, so a
        # position sitting exactly on n_x (or a hair beyond from clipping) stays finite.
        if self.steps_per_slice is None:
            grid, source = self._axes, self._current_slice  # space only
        else:
            grid, source = (self._time_axis, *self._axes), self._winds  # (time, *space)

        self._interp_u = RegularGridInterpolator(
            grid, source[..., 0], method="linear", bounds_error=False, fill_value=None
        )
        if self.ndim == 3:
            self._interp_v = RegularGridInterpolator(
                grid, source[..., 1], method="linear", bounds_error=False, fill_value=None
            )

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        """Linearly interpolated (u, v) at a continuous position and time. v is None in 2D.

        The 2D field varies over both grid axes (x, y) but has a single component u, mirroring
        :class:`SyntheticFlowField`; the 3D field interpolates over (x, y, z). In temporal mode
        (``steps_per_slice`` set) it additionally interpolates between cached slices using ``t``.
        """
        if self._interp_u is None:
            raise RuntimeError(
                "ReanalysisFlowField.reset() must be called before velocity_at()"
            )

        space = [position.i, position.j] + ([position.k] if self.ndim == 3 else [])
        if self.steps_per_slice is None:
            coords = space  # static: spatial interpolation on the frozen slice
        else:
            # Map episode time to a fractional slice index, clamped to the cached window
            # (decision: clamp on the last slice past the end). t=0 -> exactly slice t0.
            s = self._t0 + t / self.steps_per_slice
            s = min(max(s, 0.0), self._T - 1)
            coords = [s, *space]  # temporal: (time, *space) interpolation

        pt = np.array([coords], dtype=np.float64)
        if self.ndim == 2:
            return (float(self._interp_u(pt)[0]), None)
        return (float(self._interp_u(pt)[0]), float(self._interp_v(pt)[0]))

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        """Gridded velocities at time ``t``: ``(n_x,n_y,1)`` (2D) / ``(n_x,n_y,n_z,2)`` (3D).

        Static mode (or ``t==0``) returns the frozen start slice; temporal mode at ``t>0``
        returns the linear blend of the two cached slices bracketing the current time.
        """
        if self._current_slice is None:
            raise RuntimeError(
                "ReanalysisFlowField.reset() must be called before velocity_field()"
            )
        if self.steps_per_slice is None or t == 0.0:
            return np.asarray(self._current_slice)

        s = self._t0 + t / self.steps_per_slice
        s = min(max(s, 0.0), self._T - 1)
        lo = int(np.floor(s))
        hi = min(lo + 1, self._T - 1)
        frac = s - lo
        return np.asarray((1.0 - frac) * self._winds[lo] + frac * self._winds[hi])
