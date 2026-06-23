"""Offline: build the regridded wind cache that ``ReanalysisFlowField`` loads.

This is the only network-touching part of the real-wind pipeline; it lives outside ``src/``.
It produces a ``.npz`` matching the loader contract in ``src/env/field/era5_data.py``:

    2D: winds (T, n_x, n_y, 1)        component u
    3D: winds (T, n_x, n_y, n_z, 2)   components (u, v)
    + meta: dict(units='m/s', source, region, levels, timestamps, dx, dt, ...)

Two sources:

  --source demo   (default)  Realistic, spatially+temporally smooth synthetic winds with a
                             mean westerly. NO credentials or network needed -- exercises the
                             full interpolation + agent pipeline immediately.

  --source cds               Real ERA5 via the Copernicus CDS API. Requires ``cdsapi`` +
                             ``xarray`` (+ a netCDF engine) installed and a ``~/.cdsapirc``
                             with your CDS key. Downloads u/v over a lat/lon box, then
                             RegularGridInterpolator-resamples each timestep onto the env grid.

Examples
--------
    # demo 2D cache (default 80x80), then run experiments/viz_real_wind_drift.py
    pixi run python experiments/field_estimation/scripts/fetch_era5.py \
        --source demo --grid 80 80 --out data/era5_demo_2d.npz

    # real ERA5, 3D, San Francisco box (needs CDS credentials)
    pixi run python experiments/field_estimation/scripts/fetch_era5.py \
        --source cds --grid 40 40 8 --out data/era5_sf_3d.npz \
        --region -125 36 -120 40 --levels 50 70 100 150 200 250 300 --date 2023-01-01
"""

import argparse
import os
import tempfile
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- demo source
def make_demo_cache(
    grid_shape: Sequence[int],
    *,
    T: int = 24,
    seed: int = 0,
    mean_wind: float = 12.0,
    std_wind: float = 6.0,
    time_sigma: float = 2.0,
) -> Tuple[np.ndarray, Dict]:
    """Generate realistic, smooth synthetic winds on the env grid (no external data).

    The field is a low-pass-filtered Gaussian random field -- smooth in space AND time, so it
    looks like coherent, slowly-evolving weather rather than noise -- plus a constant mean
    westerly on the ``u`` component so a passive agent visibly drifts. Magnitudes are in m/s,
    matching the unit convention real ERA5 data uses.

    Args:
        grid_shape: ``(n_x, n_y)`` for 2D or ``(n_x, n_y, n_z)`` for 3D.
        T: number of time slices (the realizations sampled at ``reset``).
        seed: RNG seed for reproducibility.
        mean_wind: mean westerly speed (m/s) added to the ``u`` component.
        std_wind: std of the spatial/temporal fluctuations (m/s).
        time_sigma: Gaussian smoothing width along the time axis (larger = slower evolution).

    Returns:
        ``(winds, meta)`` where winds is ``(T, *grid_shape, C)`` float64, C=1 (2D) / 2 (3D).
    """
    from scipy.ndimage import gaussian_filter

    grid_shape = tuple(int(n) for n in grid_shape)
    ndim = len(grid_shape)
    if ndim not in (2, 3):
        raise ValueError(f"grid_shape must have 2 or 3 entries, got {grid_shape}")
    n_components = 1 if ndim == 2 else 2

    rng = np.random.default_rng(seed)
    space_sigma = tuple(max(1.0, 0.15 * n) for n in grid_shape)  # ~15% of each axis

    def smooth_field() -> np.ndarray:
        raw = rng.normal(size=(T, *grid_shape))
        sm = gaussian_filter(raw, sigma=(time_sigma, *space_sigma), mode="wrap")
        return sm / (sm.std() + 1e-9)  # unit variance after smoothing

    u = mean_wind + std_wind * smooth_field()
    comps = [u]
    if n_components == 2:
        comps.append(std_wind * smooth_field())  # zero-mean meridional v

    winds = np.stack(comps, axis=-1).astype(np.float64)  # (T, *grid, C)
    meta = dict(
        units="m/s",
        source="demo",
        region="synthetic (SF-like, periodic)",
        mean_wind=float(mean_wind),
        std_wind=float(std_wind),
        seed=int(seed),
        n=grid_shape + (None,) * (3 - ndim),
        note="stand-in for real ERA5; identical .npz contract -> same interpolation path",
    )
    return winds, meta


# ---------------------------------------------------------------------------- cds source
def fetch_from_cds(
    grid_shape: Sequence[int],
    region: Tuple[float, float, float, float],  # (west, south, east, north) in degrees
    *,
    levels: Optional[Sequence[int]] = None,     # pressure levels (hPa); 3D only
    date: str = "2023-01-01",
    times: Sequence[str] = ("00:00", "06:00", "12:00", "18:00"),
) -> Tuple[np.ndarray, Dict]:
    """Download ERA5 u/v over ``region`` and resample onto the env grid (linear).

    Requires ``cdsapi`` + ``xarray`` and CDS credentials in ``~/.cdsapirc``. The env grid axes
    map to physical coordinates as: ambient (x, y) -> (lon, lat); controllable z -> pressure
    level (3D). Returns raw m/s so the field's ``scale`` can be retuned without re-downloading.
    """
    try:
        import cdsapi
        import xarray as xr
    except ImportError as exc:  # pragma: no cover - exercised only with real creds
        raise RuntimeError(
            "Real ERA5 fetch needs 'cdsapi' and 'xarray' (+ a netCDF engine). Install them and "
            "put your key in ~/.cdsapirc, or use --source demo to test the pipeline now."
        ) from exc

    grid_shape = tuple(int(n) for n in grid_shape)
    ndim = len(grid_shape)
    west, south, east, north = region
    dataset = "reanalysis-era5-pressure-levels"
    if ndim == 3 and not levels:
        levels = [300, 250, 200, 150, 100, 70, 50]
    if ndim == 2:
        levels = [int(levels[0])] if levels else [200]  # a single level for the 2D slice
    levels = sorted((int(level) for level in levels), reverse=True)

    request = {
        "product_type": "reanalysis",
        "variable": ["u_component_of_wind", "v_component_of_wind"],
        "pressure_level": [str(p) for p in levels],
        "year": date[:4],
        "month": date[5:7],
        "day": date[8:10],
        "time": list(times),
        # CDS area order is North, West, South, East.
        "area": [north, west, south, east],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
        tmp = handle.name
    try:
        client_key = os.environ.get("CDSAPI_KEY")
        if client_key:
            client = cdsapi.Client(
                url=os.environ.get(
                    "CDSAPI_URL", "https://cds.climate.copernicus.eu/api"
                ),
                key=client_key,
            )
        else:
            client = cdsapi.Client()
        client.retrieve(dataset, request, tmp)
        ds = xr.open_dataset(tmp)
        rename = {}
        if "pressure_level" in ds.dims:
            rename["pressure_level"] = "level"
        if "valid_time" in ds.dims:
            rename["valid_time"] = "time"
        if rename:
            ds = ds.rename(rename)

        # ERA5 latitude is descending; interpolation axes must be ascending.
        ds = ds.sortby("latitude").sortby("longitude").sortby("level")
        src_lat = ds["latitude"].values
        src_lon = ds["longitude"].values
        src_lvl = ds["level"].values

        # Target env-grid coordinates at cell centres.
        tgt_lon = np.linspace(west, east, grid_shape[0])
        tgt_lat = np.linspace(south, north, grid_shape[1])
        times_arr = ds["time"].values
        T = len(times_arr)

        winds, pressure_by_k = _resample_to_grid(
            ds, grid_shape, src_lvl, src_lat, src_lon,
            tgt_lon, tgt_lat, levels, T, ndim,
        )
        ds.close()
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    meta = dict(
        units="m/s", source="era5-cds", region=dict(west=west, south=south, east=east, north=north),
        levels=[int(p) for p in levels], date=date, times=list(times),
        vertical_coordinate="pressure_hpa",
        pressure_hpa_by_k=pressure_by_k,
        vertical_order="k increases with altitude (pressure decreases)",
        timestamps=[str(t) for t in times_arr], n=grid_shape + (None,) * (3 - ndim),
    )
    return winds.astype(np.float64), meta


def _resample_to_grid(ds, grid_shape, src_lvl, src_lat, src_lon,
                      tgt_lon, tgt_lat, levels, T, ndim):
    """Resample ERA5, using log-pressure with increasing ``k`` = increasing altitude."""
    from scipy.interpolate import RegularGridInterpolator

    n_components = 1 if ndim == 2 else 2
    winds = np.empty((T, *grid_shape, n_components), dtype=np.float64)
    comp_vars = ["u", "v"][:n_components]

    if ndim == 2:
        XX, YY = np.meshgrid(tgt_lon, tgt_lat, indexing="ij")
        pts = np.column_stack([YY.ravel(), XX.ravel()])  # (lat, lon) order
        for t in range(T):
            field = ds["u"].isel(time=t, level=0).values  # (lat, lon)
            rgi = RegularGridInterpolator((src_lat, src_lon), field,
                                          method="linear", bounds_error=False, fill_value=None)
            winds[t, ..., 0] = rgi(pts).reshape(grid_shape)
        pressure_by_k = [int(levels[0])]
    else:
        # Pressure falls as altitude rises. k=1 is the highest pressure (lowest
        # altitude), and k=n_z is the lowest pressure (highest altitude).
        tgt_lvl = pressure_levels_for_grid(levels, grid_shape[2])
        src_log_pressure = np.log(np.asarray(src_lvl, dtype=np.float64))
        tgt_log_pressure = np.log(tgt_lvl)
        ZZ, XX, YY = np.meshgrid(tgt_log_pressure, tgt_lon, tgt_lat, indexing="ij")
        # interpolation order matches source axes (log-pressure, lat, lon)
        pts = np.column_stack([ZZ.ravel(), YY.ravel(), XX.ravel()])
        for t in range(T):
            for c, var in enumerate(comp_vars):
                field = ds[var].isel(time=t).values  # (level, lat, lon)
                rgi = RegularGridInterpolator((src_log_pressure, src_lat, src_lon), field,
                                              method="linear", bounds_error=False, fill_value=None)
                # reshape from (n_z, n_x, n_y) back to (n_x, n_y, n_z)
                vals = rgi(pts).reshape(grid_shape[2], grid_shape[0], grid_shape[1])
                winds[t, ..., c] = np.transpose(vals, (1, 2, 0))
        pressure_by_k = [float(level) for level in tgt_lvl]
    return winds, pressure_by_k


def pressure_levels_for_grid(levels: Sequence[int], n_z: int) -> np.ndarray:
    """Map env ``k`` to pressure, ordered from low to high physical altitude."""
    requested = np.asarray(sorted(levels, reverse=True), dtype=np.float64)
    if requested.size == 0 or np.any(requested <= 0):
        raise ValueError("pressure levels must be positive")
    if n_z <= 0:
        raise ValueError("n_z must be positive")
    if n_z == requested.size:
        return requested
    return np.geomspace(requested.max(), requested.min(), n_z)


# -------------------------------------------------------------------- open-meteo source
def fetch_from_openmeteo(
    grid_shape: Sequence[int],
    region: Tuple[float, float, float, float],  # (west, south, east, north)
    *,
    date: str = "2023-01-01",
    end_date: Optional[str] = None,
    src_grid: int = 6,
) -> Tuple[np.ndarray, Dict]:
    """Real ERA5-derived **surface (10m)** winds via the keyless Open-Meteo archive.

    No Copernicus account needed. Queries a coarse ``src_grid`` x ``src_grid`` lattice of
    lat/lon points over ``region``, pulls hourly 10m wind speed+direction (m/s), converts to
    (u, v), then linearly resamples onto the env grid. Because real ``u`` changes sign, the
    agent will drift both east and west -- a clear visual contrast with the demo source.

    Note: this is a single (surface) level. For a 2D env it stores ``u`` only; for a 3D env it
    broadcasts the horizontal (u, v) across all altitude levels (no real vertical structure --
    that needs pressure-level ERA5 via ``--source cds``).
    """
    import json
    import urllib.request

    grid_shape = tuple(int(n) for n in grid_shape)
    ndim = len(grid_shape)
    west, south, east, north = region
    end_date = end_date or date

    # Coarse source lattice we actually download, then interpolate up to the env grid.
    src_lon = np.linspace(west, east, src_grid)
    src_lat = np.linspace(south, north, src_grid)
    LON, LAT = np.meshgrid(src_lon, src_lat, indexing="ij")  # (src_grid, src_grid)
    flat_lat = LAT.ravel()
    flat_lon = LON.ravel()

    # Batch the points into requests (Open-Meteo accepts comma-separated coordinate lists).
    speeds, dirs, n_hours = [], [], None
    for start in range(0, flat_lat.size, 100):
        sl = slice(start, start + 100)
        lat_csv = ",".join(f"{v:.4f}" for v in flat_lat[sl])
        lon_csv = ",".join(f"{v:.4f}" for v in flat_lon[sl])
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat_csv}&longitude={lon_csv}"
            f"&start_date={date}&end_date={end_date}"
            "&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=ms&timezone=UTC"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            payload = json.load(r)
        locs = payload if isinstance(payload, list) else [payload]
        for loc in locs:
            h = loc["hourly"]
            speeds.append(np.asarray(h["wind_speed_10m"], dtype=np.float64))
            dirs.append(np.asarray(h["wind_direction_10m"], dtype=np.float64))
            n_hours = len(h["time"])

    speed = np.array(speeds).reshape(src_grid, src_grid, n_hours)      # (lon, lat, T)
    direction = np.array(dirs).reshape(src_grid, src_grid, n_hours)
    speed = np.nan_to_num(speed)
    # Meteorological convention: direction is where wind comes FROM.
    rad = np.deg2rad(direction)
    u_src = -speed * np.sin(rad)   # eastward
    v_src = -speed * np.cos(rad)   # northward

    winds = _resample_openmeteo(u_src, v_src, src_lon, src_lat, grid_shape, n_hours, ndim)
    meta = dict(
        units="m/s", source="openmeteo-era5-10m",
        region=dict(west=west, south=south, east=east, north=north),
        level="10m surface", date=date, end_date=end_date, src_grid=src_grid,
        n=grid_shape + (None,) * (3 - ndim),
        note="real ERA5-derived SURFACE winds; for winds aloft use --source cds",
    )
    return winds.astype(np.float64), meta


def _resample_openmeteo(u_src, v_src, src_lon, src_lat, grid_shape, n_hours, ndim):
    """Linear-resample the coarse (u, v) lattice onto the env grid for every hour."""
    from scipy.interpolate import RegularGridInterpolator

    tgt_lon = np.linspace(src_lon.min(), src_lon.max(), grid_shape[0])
    tgt_lat = np.linspace(src_lat.min(), src_lat.max(), grid_shape[1])
    XX, YY = np.meshgrid(tgt_lon, tgt_lat, indexing="ij")
    pts = np.column_stack([XX.ravel(), YY.ravel()])  # (lon, lat) order matches axes below

    n_components = 1 if ndim == 2 else 2
    winds = np.empty((n_hours, *grid_shape, n_components), dtype=np.float64)
    for t in range(n_hours):
        u_t = RegularGridInterpolator((src_lon, src_lat), u_src[..., t],
                                      method="linear", bounds_error=False, fill_value=None)
        u_grid = u_t(pts).reshape(grid_shape[0], grid_shape[1])
        v_t = RegularGridInterpolator((src_lon, src_lat), v_src[..., t],
                                      method="linear", bounds_error=False, fill_value=None)
        v_grid = v_t(pts).reshape(grid_shape[0], grid_shape[1])
        if ndim == 2:
            winds[t, ..., 0] = u_grid                      # 2D env: eastward component only
        else:
            for k in range(grid_shape[2]):                 # broadcast across altitude levels
                winds[t, ..., k, 0] = u_grid
                winds[t, ..., k, 1] = v_grid
    return winds


# ------------------------------------------------------------------------------- io / cli
def save_cache(path: str, winds: np.ndarray, meta: Dict) -> None:
    """Write the wind array + meta to a compressed ``.npz`` (loader uses allow_pickle)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, winds=winds, meta=meta)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the ERA5 wind cache for ReanalysisFlowField.")
    p.add_argument("--source", choices=["demo", "openmeteo", "cds"], default="demo",
                   help="demo=synthetic; openmeteo=real ERA5 10m winds (keyless); cds=real ERA5 aloft")
    p.add_argument("--grid", type=int, nargs="+", default=[80, 80],
                   help="n_x n_y [n_z] -- 2 entries for 2D, 3 for 3D")
    p.add_argument("--out", default="data/era5_demo_2d.npz")
    p.add_argument("--slices", type=int, default=24, help="T time slices (demo)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mean-wind", type=float, default=12.0)
    p.add_argument("--std-wind", type=float, default=6.0)
    # cds-only
    p.add_argument("--region", type=float, nargs=4, metavar=("W", "S", "E", "N"),
                   default=[-125.0, 36.0, -120.0, 40.0])
    p.add_argument("--levels", type=int, nargs="+", default=None)
    p.add_argument("--date", default="2023-01-01")
    p.add_argument("--end-date", default=None, help="openmeteo: last day (inclusive); more T")
    args = p.parse_args()

    if len(args.grid) not in (2, 3):
        p.error("--grid needs 2 (2D) or 3 (3D) integers")

    if args.source == "demo":
        winds, meta = make_demo_cache(
            args.grid, T=args.slices, seed=args.seed,
            mean_wind=args.mean_wind, std_wind=args.std_wind,
        )
    elif args.source == "openmeteo":
        winds, meta = fetch_from_openmeteo(
            args.grid, tuple(args.region), date=args.date, end_date=args.end_date,
        )
    else:
        winds, meta = fetch_from_cds(
            args.grid, tuple(args.region), levels=args.levels, date=args.date,
        )

    save_cache(args.out, winds, meta)
    print(f"wrote {args.out}")
    print(f"  winds shape : {winds.shape}  (T, *grid, components)")
    print(f"  source      : {meta['source']}")
    print(f"  |wind| range: {np.linalg.norm(winds, axis=-1).min():.1f} .. "
          f"{np.linalg.norm(winds, axis=-1).max():.1f} m/s")


if __name__ == "__main__":
    main()
