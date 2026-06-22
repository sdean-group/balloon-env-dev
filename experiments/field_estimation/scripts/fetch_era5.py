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
        levels = [50, 70, 100, 150, 200, 250, 300]
    if ndim == 2:
        levels = [int(levels[0])] if levels else [200]  # a single level for the 2D slice

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
        "format": "netcdf",
    }

    tmp = ".era5_download.nc"
    cdsapi.Client().retrieve(dataset, request, tmp)
    ds = xr.open_dataset(tmp)

    # ERA5 latitude is descending; RegularGridInterpolator needs ascending source axes.
    ds = ds.sortby("latitude").sortby("longitude").sortby("level")
    src_lat = ds["latitude"].values
    src_lon = ds["longitude"].values
    src_lvl = ds["level"].values

    # Target env-grid coordinates at cell centres.
    tgt_lon = np.linspace(west, east, grid_shape[0])
    tgt_lat = np.linspace(south, north, grid_shape[1])
    times_arr = ds["valid_time"].values if "valid_time" in ds else ds["time"].values
    T = len(times_arr)

    winds = _resample_to_grid(ds, grid_shape, src_lvl, src_lat, src_lon,
                              tgt_lon, tgt_lat, levels, T, ndim)
    ds.close()
    os.remove(tmp)

    meta = dict(
        units="m/s", source="era5-cds", region=dict(west=west, south=south, east=east, north=north),
        levels=[int(p) for p in levels], date=date, times=list(times),
        timestamps=[str(t) for t in times_arr], n=grid_shape + (None,) * (3 - ndim),
    )
    return winds.astype(np.float64), meta


def _resample_to_grid(ds, grid_shape, src_lvl, src_lat, src_lon,
                      tgt_lon, tgt_lat, levels, T, ndim):
    """Linear-resample each timestep of the ERA5 dataset onto the env grid."""
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
    else:
        tgt_lvl = np.linspace(src_lvl.min(), src_lvl.max(), grid_shape[2])
        ZZ, XX, YY = np.meshgrid(tgt_lvl, tgt_lon, tgt_lat, indexing="ij")
        # interpolation order matches source axes (level, lat, lon)
        pts = np.column_stack([ZZ.ravel(), YY.ravel(), XX.ravel()])
        for t in range(T):
            for c, var in enumerate(comp_vars):
                field = ds[var].isel(time=t).values  # (level, lat, lon)
                rgi = RegularGridInterpolator((src_lvl, src_lat, src_lon), field,
                                              method="linear", bounds_error=False, fill_value=None)
                # reshape from (n_z, n_x, n_y) back to (n_x, n_y, n_z)
                vals = rgi(pts).reshape(grid_shape[2], grid_shape[0], grid_shape[1])
                winds[t, ..., c] = np.transpose(vals, (1, 2, 0))
    return winds


# ------------------------------------------------------------------------------- io / cli
def save_cache(path: str, winds: np.ndarray, meta: Dict) -> None:
    """Write the wind array + meta to a compressed ``.npz`` (loader uses allow_pickle)."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    np.savez_compressed(path, winds=winds, meta=meta)


def main() -> None:
    p = argparse.ArgumentParser(description="Build the ERA5 wind cache for ReanalysisFlowField.")
    p.add_argument("--source", choices=["demo", "cds"], default="demo")
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
    args = p.parse_args()

    if len(args.grid) not in (2, 3):
        p.error("--grid needs 2 (2D) or 3 (3D) integers")

    if args.source == "demo":
        winds, meta = make_demo_cache(
            args.grid, T=args.slices, seed=args.seed,
            mean_wind=args.mean_wind, std_wind=args.std_wind,
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
