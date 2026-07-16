"""Ingest ERA5 model-level GRIB(s) -> the real-ERA5 anchor artifact.

This is the first 'generator': a passthrough whose materialize() reads ingested
ERA5. Capability flags mark it bounded / not-tiled / random-access / evolving, so
the harness treats it as the null control for the seam + revisit metrics.

Stage 1: u,v,T,q on the band (single or multi timestep).
Stage 2: + lnsp -> surface pressure `sp`, enabling pressure (a+b·sp) and altitude
(hypsometric) derivation in derive.py for the vertical/dynamics metrics.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from . import artifact
from .l137 import _AB

# San Francisco — the locked eval location.
SF_LAT, SF_LON = 37.77, 237.58  # lon in 0-360 convention (= -122.42 W)


def ingest(uvtq_path, out_path, *, lnsp_path=None, season="winter") -> Path:
    src = xr.open_dataset(uvtq_path, engine="cfgrib")

    u, v, T, q = src["u"].values, src["v"].values, src["t"].values, src["q"].values
    # ensure (time, level, lat, lon)
    if u.ndim == 3:  # single timestep -> add time axis
        u, v, T, q = (a[None] for a in (u, v, T, q))

    level = src["hybrid"].values.astype(int)
    lat = src["latitude"].values.astype(float)
    lon = src["longitude"].values.astype(float)
    valid_time = np.atleast_1d(src["valid_time"].values)

    ds = artifact.make_field(u, v, level=level, lat=lat, lon=lon, time=valid_time,
                             extra={"T": T, "q": q})

    # Stage 2: surface pressure from lnsp -> per-column pressure/altitude downstream.
    has_sp = False
    if lnsp_path is not None:
        ln = xr.open_dataset(lnsp_path, engine="cfgrib")["lnsp"].values
        if ln.ndim == 2:
            ln = ln[None]
        sp = np.exp(ln).astype("float32")          # (time, y, x), Pa
        ds["sp"] = (("time", "y", "x"), sp)
        has_sp = True

    band = {int(n): list(_AB[int(n)]) for n in range(int(level.min()) - 1, int(level.max()) + 1)}
    attrs = artifact.default_attrs(
        generator={"name": "era5_real", "config": {"source": str(Path(uvtq_path).name),
                                                   "has_sp": has_sp}},
        capabilities={"extent": "bounded", "tiled": False,
                      "random_access": True, "temporally_evolving": True},
        conditioning={"lat": SF_LAT, "lon": SF_LON, "season": season,
                      "time": str(valid_time[0])},
        model_levels=level,
        coord_to_meters="tangent_plane",
    )
    attrs["level_coeffs"] = band  # {n: [a,b]} half-level coeffs for the band
    return artifact.write(ds, attrs, out_path)
