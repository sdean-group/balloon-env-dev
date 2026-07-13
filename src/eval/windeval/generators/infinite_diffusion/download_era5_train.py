"""Download + ingest a larger ERA5 model-level set for Phase-2 training.

The M0 slice (24 hourly steps over a tiny SF box) is enough to smoke-test the pipeline but
far too little to train a diffusion model on random crops. This script pulls a bigger
region x longer time span of model-level u, v on the frozen band and ingests it into a
training zarr (dims time, level, y, x; vars u, v) ready for ``data.WindCropDataset``.

YOU run this — it needs your CDS credentials (``~/.cdsapirc``) and the request is slow.
The dataset is ``reanalysis-era5-complete`` (model levels), which requires CDS licence
acceptance for that dataset on your account.

Example (a ~30deg box around SF, one full year, every 6h, band ML 49-66):

    python -m src.eval.windeval.generators.infinite_diffusion.download_era5_train \
        --year 2022 --step 6 --area 55 220 25 255 \
        --grib src/eval/windeval/data/era5_train.grib \
        --out  src/eval/windeval/data/era5_train.zarr

Notes
-----
- More spatial extent matters more than fine time resolution for random-crop training
  (translation invariance comes from area). A larger box + 6-hourly is a good tradeoff.
- Keep the level band identical to training/eval (default 49-66) so channels line up.
- Re-run with different years/areas and pass multiple gribs to ``--grib`` to concatenate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from ...ingest_era5 import SF_LAT, SF_LON  # noqa: F401  (kept for provenance/conditioning)
from ...l137 import _AB
from ... import artifact


def _mlevels(lo: int, hi: int) -> str:
    return "/".join(str(n) for n in range(lo, hi + 1))


def download(grib_path: str, *, dates: str, time: str,
             area: list[float], levels: tuple[int, int]) -> Path:
    """Issue ONE CDS model-level request for u, v over a MARS date/time spec.

    ``dates`` is a raw MARS date spec, e.g. ``"2023-01-15/to/2023-01-21"`` (contiguous,
    tape-efficient) or a slash list ``"2023-01-01/2023-04-01"``. ``time`` is a MARS time
    spec, e.g. ``"00/to/23/by/3"``. Requires cdsapi + credentials.
    """
    import cdsapi

    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-complete",
        {
            "class": "ea",
            "type": "an",
            "stream": "oper",
            "expver": "1",
            "levtype": "ml",
            "levelist": _mlevels(*levels),
            "param": "131/132",                 # u, v
            "date": dates,
            "time": time,
            "area": area,                        # [N, W, S, E]
            "grid": "0.25/0.25",
            "format": "grib",
        },
        grib_path,
    )
    return Path(grib_path)


def ingest(grib_paths: list[str], out_path: str, *, levels: tuple[int, int]) -> Path:
    """Read one or more model-level u,v gribs and write a training zarr (time,level,y,x)."""
    us, vs, times = [], [], []
    lat = lon = level = None
    for gp in grib_paths:
        src = xr.open_dataset(gp, engine="cfgrib")
        u, v = src["u"].values, src["v"].values
        if u.ndim == 3:                          # (level,lat,lon) -> add time
            u, v = u[None], v[None]
        us.append(u.astype("float32"))
        vs.append(v.astype("float32"))
        times.append(np.atleast_1d(src["valid_time"].values))
        level = src["hybrid"].values.astype(int)
        lat = src["latitude"].values.astype(float)
        lon = src["longitude"].values.astype(float)

    u = np.concatenate(us, axis=0)
    v = np.concatenate(vs, axis=0)
    time = np.concatenate(times, axis=0)
    order = np.argsort(time)            # chunks arrive disjoint (e.g. seasonal) -> sort by time
    u, v, time = u[order], v[order], time[order]
    print(f"[ingest] {u.shape[0]} steps x {u.shape[1]} levels x {u.shape[2]}x{u.shape[3]}")

    ds = artifact.make_field(u, v, level=level, lat=lat, lon=lon, time=time)
    band = {int(n): list(_AB[int(n)]) for n in range(levels[0] - 1, levels[1] + 1)}
    attrs = artifact.default_attrs(
        generator={"name": "era5_train", "config": {"sources": [Path(p).name for p in grib_paths]}},
        capabilities={"extent": "bounded", "tiled": False,
                      "random_access": True, "temporally_evolving": True},
        conditioning={"region": "multi", "purpose": "phase2_training"},
        model_levels=level,
        coord_to_meters="tangent_plane",
    )
    attrs["level_coeffs"] = band
    return artifact.write(ds, attrs, out_path)


def ingest_stream(grib_paths: list[str], out_path: str, *, levels: tuple[int, int]) -> Path:
    """Constant-memory ingest: append each grib chunk to the zarr in the order given.

    The in-RAM :func:`ingest` concatenates the whole download (a full hourly year is
    ~28 GB peak — dies on a laptop); this one holds a single ~7-day chunk (~350 MB) at a
    time. Chunks MUST be passed in ascending time order (checked; no global sort).
    """
    last_time = None
    out = Path(out_path)
    total = 0
    for i, gp in enumerate(grib_paths):
        src = xr.open_dataset(gp, engine="cfgrib")
        u, v = src["u"].values, src["v"].values
        if u.ndim == 3:
            u, v = u[None], v[None]
        time = np.atleast_1d(src["valid_time"].values)
        order = np.argsort(time)
        u, v, time = u[order].astype("float32"), v[order].astype("float32"), time[order]
        if last_time is not None and time[0] <= last_time:
            raise ValueError(f"{gp}: starts at {time[0]} <= previous chunk end {last_time} "
                             "(pass --dates in ascending order when using --stream)")
        last_time = time[-1]
        level = src["hybrid"].values.astype(int)
        lat = src["latitude"].values.astype(float)
        lon = src["longitude"].values.astype(float)
        ds = artifact.make_field(u, v, level=level, lat=lat, lon=lon, time=time)
        if i == 0:
            band = {int(n): list(_AB[int(n)]) for n in range(levels[0] - 1, levels[1] + 1)}
            attrs = artifact.default_attrs(
                generator={"name": "era5_train",
                           "config": {"sources": [Path(p).name for p in grib_paths]}},
                capabilities={"extent": "bounded", "tiled": False,
                              "random_access": True, "temporally_evolving": True},
                conditioning={"region": "multi", "purpose": "phase2_training"},
                model_levels=level,
                coord_to_meters="tangent_plane",
            )
            attrs["level_coeffs"] = band
            artifact.write(ds, attrs, out)
        else:
            ds.to_zarr(out, mode="a", append_dim="time", consolidated=False, zarr_format=2)
        total += len(time)
        print(f"[ingest --stream] {Path(gp).name}: +{len(time)} steps (total {total})",
              flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Download + ingest ERA5 model-level u,v for training.")
    ap.add_argument("--out", required=True, help="output training zarr path")
    ap.add_argument("--grib-dir", default="src/eval/windeval/data",
                    help="dir for the per-chunk grib downloads")
    ap.add_argument("--prefix", default="era5_train", help="grib filename prefix")
    ap.add_argument("--dates", nargs="+", required=True,
                    help="one MARS date spec per chunk, e.g. 2023-01-15/to/2023-01-21")
    ap.add_argument("--time", default="00/to/23/by/3", help="MARS time spec")
    ap.add_argument("--area", default="55/-135/25/-105",
                    help="N/W/S/E (slash-separated; negative lon OK, e.g. 55/-135/25/-105)")
    ap.add_argument("--levels", type=int, nargs=2, default=[49, 66])
    ap.add_argument("--skip-download", action="store_true", help="ingest existing gribs only")
    ap.add_argument("--stream", action="store_true",
                    help="constant-memory ingest (append per chunk; --dates must be ascending)")
    args = ap.parse_args(argv)

    levels = (args.levels[0], args.levels[1])
    area = [float(x) for x in args.area.split("/")]
    grib_dir = Path(args.grib_dir)
    grib_dir.mkdir(parents=True, exist_ok=True)
    gribs = [str(grib_dir / f"{args.prefix}_{i}.grib") for i in range(len(args.dates))]

    if not args.skip_download:
        for gp, dspec in zip(gribs, args.dates):
            if Path(gp).exists():                      # resumable: re-run skips finished chunks
                print(f"[download] {dspec_summary(dspec)} -> {gp} (exists, skipping)")
                continue
            print(f"[download] {dspec_summary(dspec)} -> {gp}")
            download(gp + ".part", dates=dspec, time=args.time, area=area, levels=levels)
            Path(gp + ".part").rename(gp)
    ingest_fn = ingest_stream if args.stream else ingest
    out = ingest_fn(gribs, args.out, levels=levels)
    print(f"[done] -> {out}")


def dspec_summary(dspec: str) -> str:
    return dspec.replace("/to/", "…").replace("/", " ")


if __name__ == "__main__":
    main()
