"""ERA5 L137 hybrid sigma-pressure coefficients (half-level / interface values).

Per-column pressure floats with surface pressure: half-level n pressure
    ph(n) = a(n) + b(n) * sp        [Pa]
and full (data) level k sits between half-levels k-1 and k:
    p(k) = 0.5 * (ph(k-1) + ph(k))

Source: ECMWF "L137 model level definitions". Only the band bounding our model-level
range (49-66) is stored; full-level 49 needs half-level 48, so we keep 47..67.
"""
from __future__ import annotations

import numpy as np

# n -> (a [Pa], b)   half-level coefficients
_AB = {
    47: (4799.149414, 0.000000),
    48: (5119.895020, 0.000000),
    49: (5452.990723, 0.000000),
    50: (5798.344727, 0.000000),
    51: (6156.074219, 0.000000),
    52: (6526.946777, 0.000000),
    53: (6911.870605, 0.000000),
    54: (7311.869141, 0.000000),
    55: (7727.412109, 0.000007),
    56: (8159.354004, 0.000024),
    57: (8608.525391, 0.000059),
    58: (9076.400391, 0.000112),
    59: (9562.682617, 0.000199),
    60: (10065.978516, 0.000340),
    61: (10584.631836, 0.000562),
    62: (11116.662109, 0.000890),
    63: (11660.067383, 0.001353),
    64: (12211.547852, 0.001992),
    65: (12766.873047, 0.002857),
    66: (13324.668945, 0.003971),
    67: (13881.331055, 0.005378),
}


def half_level_pressure(n: int, sp):
    """ph(n) = a(n) + b(n)*sp, in Pa. sp may be scalar or ndarray."""
    a, b = _AB[n]
    return a + b * np.asarray(sp)


def full_level_pressure(levels, sp):
    """Full-level pressure (Pa) for an array of model-level indices, given sp (Pa).

    Returns array shaped (len(levels), *sp.shape) — pressure per level per column.
    """
    sp = np.asarray(sp, dtype=float)
    out = np.empty((len(levels),) + sp.shape, dtype=float)
    for i, k in enumerate(np.asarray(levels).astype(int)):
        out[i] = 0.5 * (half_level_pressure(k - 1, sp) + half_level_pressure(k, sp))
    return out
