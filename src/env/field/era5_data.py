"""Loader for the cached, pre-regridded ERA5 wind data.

This module is the boundary between offline data prep (``fetch_era5.py``, which downloads
ERA5 and resamples it onto the env grid) and the pure :class:`FlowField` that consumes it.
It does no env coupling and no interpolation -- it just loads the ``.npz`` cache, validates
it, and hands back the raw array plus metadata.

Cache contract (see the design doc):
    2D: ``winds`` shape ``(T, n_x, n_y, 1)``        -- component ``u`` only
    3D: ``winds`` shape ``(T, n_x, n_y, n_z, 2)``   -- components ``(u, v)``
where ``T`` is the number of historical time slices (the realizations sampled at reset).
"""

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class Era5Bundle:
    """Loaded ERA5 cache: the wind array and its metadata.

    Attributes:
        winds: float64 array, ``(T, n_x, n_y, 1)`` (2D) or ``(T, n_x, n_y, n_z, 2)`` (3D).
        meta: dict of provenance (region, units, timestamps, the grid it was built for).
    """

    winds: np.ndarray
    meta: Dict[str, Any]


def load_era5(path: str) -> Era5Bundle:
    """Load and validate a cached ERA5 wind file.

    Args:
        path: Path to a ``.npz`` produced by the offline regrid step.

    Returns:
        An :class:`Era5Bundle` with finite winds and parsed metadata.

    Raises:
        ValueError: if the array rank is not 4 (2D) or 5 (3D), if the trailing
            component axis is inconsistent with the spatial rank, or if the data
            contains any non-finite value (e.g. an ERA5 fill value).
    """
    # allow_pickle is required because the offline step stores ``meta`` as a dict.
    # The cache is locally generated data, not an untrusted download.
    data = np.load(path, allow_pickle=True)
    winds = np.asarray(data["winds"], dtype=np.float64)

    if winds.ndim not in (4, 5):
        raise ValueError(
            f"winds must have rank 4 (2D: T,n_x,n_y,1) or 5 (3D: T,n_x,n_y,n_z,2), "
            f"got rank {winds.ndim} with shape {winds.shape}"
        )

    spatial_ndim = winds.ndim - 2  # drop the time axis and the component axis
    expected_components = 1 if spatial_ndim == 2 else 2
    n_components = winds.shape[-1]
    if n_components != expected_components:
        raise ValueError(
            f"{spatial_ndim}D data must have {expected_components} component(s) on the "
            f"last axis, got {n_components} (shape {winds.shape})"
        )

    if not np.all(np.isfinite(winds)):
        raise ValueError(
            "winds contain non-finite values (NaN/Inf) -- ERA5 fill values must be "
            "masked or filled during the offline regrid step"
        )

    meta = data["meta"].item() if "meta" in data.files else {}
    return Era5Bundle(winds=winds, meta=meta)
