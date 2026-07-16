"""Derived vertical quantities: pressure, altitude, vertical shear.

Stored raw (u,v,T,q,sp + L137 coeffs); everything here is derived per the spec so a
formula fix never forces regeneration.

Vertical shear needs Δaltitude between levels, which is *local* to the band — the
hypsometric (layer-thickness) equation needs only in-band T,q,p, NOT a full
surface-up geopotential integration. So we get true du/dz from just the band + sp.
"""
from __future__ import annotations

import numpy as np

from .l137 import full_level_pressure

R_D = 287.05      # J/kg/K, dry-air gas constant
G = 9.80665       # m/s^2


def full_pressure(ds) -> np.ndarray:
    """Per-column full-level pressure (Pa), shape (time, level, y, x)."""
    levels = ds["level"].values
    sp = ds["sp"].values                       # (time, y, x)
    p = full_level_pressure(levels, sp)        # (level, time, y, x)
    return np.moveaxis(p, 0, 1)                 # (time, level, y, x)


def virtual_temp(ds) -> np.ndarray:
    """Virtual temperature Tv = T(1 + 0.61 q), (time, level, y, x)."""
    return ds["T"].values * (1.0 + 0.61 * ds["q"].values)


def altitude(ds) -> np.ndarray:
    """Relative geometric altitude (m), (time, level, y, x).

    Hypsometric integration within the band, bottom level = 0, increasing upward.
    Level index increases downward (higher pressure), so index 0 is the top.
    """
    p = full_pressure(ds)                       # (t, L, y, x), increases with index
    tv = virtual_temp(ds)
    nt, nl, ny, nx = p.shape
    z = np.zeros_like(p)
    for i in range(nl - 2, -1, -1):             # from second-bottom up to top
        tv_mean = 0.5 * (tv[:, i] + tv[:, i + 1])
        dz = (R_D / G) * tv_mean * np.log(p[:, i + 1] / p[:, i])
        z[:, i] = z[:, i + 1] + dz
    return z


def vector_shear(ds):
    """Magnitude of vertical wind shear |dV/dz| (1/s) between adjacent levels.

    Returns (shear, z_mid):
      shear: (time, level-1, y, x)  — sqrt(du^2+dv^2)/Δz
      z_mid: (time, level-1, y, x)  — mid-layer altitude (m)
    """
    u, v = ds["u"].values, ds["v"].values
    z = altitude(ds)
    du = u[:, :-1] - u[:, 1:]
    dv = v[:, :-1] - v[:, 1:]
    dz = z[:, :-1] - z[:, 1:]                    # positive (upper minus lower)
    shear = np.sqrt(du ** 2 + dv ** 2) / dz
    z_mid = 0.5 * (z[:, :-1] + z[:, 1:])
    return shear, z_mid
