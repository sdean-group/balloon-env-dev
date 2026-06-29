"""WindArtifact read/write — the frozen contract between generators and metrics.

Implements the locked schema from `artifact-format-spec.md`:
- a dense `field/` (time, level, y, x) with named raw vars (u, v, T, ...)
- provenance + capability flags in attrs (capabilities drive metric selection)

Metrics depend only on this format, never on how a field was produced.
"""
from __future__ import annotations

import json
import datetime as _dt
from pathlib import Path

import numpy as np
import xarray as xr

FORMAT_VERSION = "v1"

# attrs whose values are dicts get JSON-encoded so zarr can store them.
_JSON_ATTRS = ("generator", "conditioning", "capabilities", "hardware", "level_coeffs")


def _encode_attrs(attrs: dict) -> dict:
    out = {}
    for k, v in attrs.items():
        out[k] = json.dumps(v) if k in _JSON_ATTRS and isinstance(v, (dict, list)) else v
    return out


def _decode_attrs(attrs: dict) -> dict:
    out = dict(attrs)
    for k in _JSON_ATTRS:
        if k in out and isinstance(out[k], str):
            try:
                out[k] = json.loads(out[k])
            except (json.JSONDecodeError, TypeError):
                pass
    return out


def make_field(
    u: np.ndarray,
    v: np.ndarray,
    *,
    level: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time: np.ndarray | None = None,
    extra: dict[str, np.ndarray] | None = None,
) -> xr.Dataset:
    """Build a `field/` Dataset with dims (time, level, y, x).

    u, v expected shape (time, level, lat, lon) or (level, lat, lon) (time added).
    """
    if u.ndim == 3:
        u = u[None]
        v = v[None]
        if extra:
            extra = {k: (a[None] if a.ndim == 3 else a) for k, a in extra.items()}
    if time is None:
        time = np.arange(u.shape[0])

    dims = ("time", "level", "y", "x")
    data_vars = {"u": (dims, u.astype("float32")), "v": (dims, v.astype("float32"))}
    for name, arr in (extra or {}).items():
        data_vars[name] = (dims, np.asarray(arr, dtype="float32"))

    return xr.Dataset(
        data_vars,
        coords={
            "time": ("time", time),
            "level": ("level", level),
            "lat": ("y", lat),
            "lon": ("x", lon),
        },
    )


def default_attrs(
    *,
    generator: dict,
    capabilities: dict,
    conditioning: dict,
    model_levels,
    seed: int | None = None,
    dt_native: str = "1h",
    lon_convention: str = "0-360",
    coord_to_meters: str = "tangent_plane",
    units: str = "u,v:m/s; T:K",
) -> dict:
    return {
        "format_version": FORMAT_VERSION,
        "generator": generator,
        "capabilities": capabilities,
        "conditioning": conditioning,
        "model_levels": list(np.asarray(model_levels).tolist()),
        "seed": -1 if seed is None else int(seed),
        "dt_native": dt_native,
        "units": units,
        "lon_convention": lon_convention,
        "coord_to_meters": coord_to_meters,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }


def write(ds: xr.Dataset, attrs: dict, path: str | Path) -> Path:
    path = Path(path)
    ds = ds.copy()
    ds.attrs = _encode_attrs(attrs)
    if path.exists():
        import shutil
        shutil.rmtree(path)
    ds.to_zarr(path, mode="w", consolidated=False, zarr_format=2)
    return path


def read(path: str | Path) -> xr.Dataset:
    ds = xr.open_zarr(path, consolidated=False, zarr_format=2)
    ds.attrs = _decode_attrs(ds.attrs)
    return ds


def grid_spacing_m(ds: xr.Dataset) -> tuple[float, float]:
    """(dx, dy) in metres for the field grid (tangent-plane approximation)."""
    lat = np.asarray(ds["lat"].values, dtype=float)
    lon = np.asarray(ds["lon"].values, dtype=float)
    deg_m = 111_320.0
    dy = abs(np.mean(np.diff(lat))) * deg_m
    dx = abs(np.mean(np.diff(lon))) * deg_m * np.cos(np.deg2rad(lat.mean()))
    return float(dx), float(dy)
