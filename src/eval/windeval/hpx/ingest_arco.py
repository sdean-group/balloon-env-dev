"""ARCO-ERA5 -> HEALPix training stores for the stratospheric wind cascade.

Streams hourly global ERA5 model-level ``u, v`` (levels 49-66, ~50-134 hPa) straight from
the public ARCO-ERA5 zarr on Google Cloud, regrids every field bilinearly from the 0.25°
lat/lon grid onto HEALPix ``nside=256`` (NEST order), and writes two zarr stores:

- ``fine/uv``   ``(T, 36, 12*256*256)`` float16  - nside 256, ~0.23° (~25 km) pixels
- ``coarse/uv`` ``(T, 36, 12*32*32)``   float32  - nside 32, ~1.83° (~200 km) pixels,
  the **exact** 8x8 nested block mean of the fine field (NEST order makes the 64 children
  of a coarse pixel contiguous, so the pooling is a reshape-and-mean)

Channels interleave ``(u_l, v_l)`` per level, ``2*l = u``, ``2*l+1 = v``, as every existing
dataset in this repo does. Days 8-14 of every month are excluded (benchmark contract).
This is the data path cBottle uses (bilinear ERA5 -> HPX256, average-pool to the coarse
grid) without its Redis/Celery machinery: one process pool, resumable at chunk granularity.

Why NEST and not lat/lon or the earth2grid XY face layout: NEST is healpy-native (spherical
harmonics, ``pix2ang``), the nested hierarchy is the coarse/fine contract, and the 12-face
XY layout convolutions want is a fixed permutation applied at load time.

Cost: one hour of global u+v on our levels is four ARCO chunks (~220 MB compressed); a
compute node fetches ~50 MB/s per stream, ~270 MB/s with 8. Regridding 36 fields is ~0.3 s.

Usage (on a compute node with local scratch)::

    python -m src.eval.windeval.hpx.ingest_arco --out /scratch/sps252/era5_hpx \
        --years 2022 2023 --workers 8              # fine + coarse
    python -m src.eval.windeval.hpx.ingest_arco --out ~/data/era5_hpx_coarse \
        --years 2020 2021 2022 2023 --workers 8 --no-fine   # coarse only (~48 GB float32)
    python -m src.eval.windeval.hpx.ingest_arco --check 2023-01-11T00   # regrid self-test
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ARCO = "https://storage.googleapis.com/gcp-public-data-arco-era5/ar/model-level-1h-0p25deg.zarr-v1"
LEVELS = np.arange(49, 67)              # ERA5 hybrid model levels 49..66 (18 levels)
LEVEL_CHUNK = 18                        # ARCO chunks the 137 levels in blocks of 18
NLAT, NLON = 721, 1440                  # 0.25° grid, lat 90..-90, lon 0..359.75
NSIDE_FINE, NSIDE_COARSE = 256, 32
FACTOR = NSIDE_FINE // NSIDE_COARSE     # 8 -> 64 children per coarse pixel
NPIX_FINE = 12 * NSIDE_FINE ** 2
NPIX_COARSE = 12 * NSIDE_COARSE ** 2
EPOCH = dt.datetime(1900, 1, 1)         # ARCO time axis: hours since 1900-01-01
EXCLUDED_DAYS = range(8, 15)            # benchmark reference days, never trained on


# --------------------------------------------------------------------------- time axis
def hour_index(t: dt.datetime) -> int:
    return int((t - EPOCH).total_seconds() // 3600)


def training_hours(years: list[int]) -> np.ndarray:
    """Hourly timestamps (as ARCO hour indices) for the given years, days 8-14 excluded."""
    out = []
    for y in years:
        d = dt.datetime(y, 1, 1)
        while d.year == y:
            if d.day not in EXCLUDED_DAYS:
                base = hour_index(d)
                out.extend(range(base, base + 24))
            d += dt.timedelta(days=1)
    return np.asarray(out, dtype=np.int64)


# --------------------------------------------------------------------------- ARCO reads
def _level_slices() -> list[tuple[int, slice]]:
    """(level-chunk index, row slice) pairs covering LEVELS. Level value v is index v-1."""
    idx = LEVELS - 1
    out = []
    for c in np.unique(idx // LEVEL_CHUNK):
        rows = idx[idx // LEVEL_CHUNK == c] - c * LEVEL_CHUNK
        out.append((int(c), slice(int(rows[0]), int(rows[-1]) + 1)))
    return out


def fetch_chunk(var: str, t: int, level_chunk: int, retries: int = 6) -> np.ndarray:
    """One ARCO chunk (1 h, 18 levels, global) decoded to float32 (18, 721, 1440).

    curl rather than urllib: it has the system CA bundle everywhere we run, resumes, and
    retries; the payload is blosc-compressed float32 in C order.
    """
    import numcodecs
    url = f"{ARCO}/{var}/{t}.{level_chunk}.0.0"
    err = None
    for attempt in range(retries):
        r = subprocess.run(["curl", "-sS", "--fail", "--max-time", "300", "--retry", "3", url],
                           capture_output=True)
        if r.returncode == 0 and len(r.stdout) > 0:
            raw = numcodecs.Blosc().decode(r.stdout)
            return np.frombuffer(raw, dtype="<f4").reshape(LEVEL_CHUNK, NLAT, NLON)
        err = r.stderr.decode(errors="replace")[:200]
        time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed to fetch {url}: {err}")


def fetch_hour(t: int) -> np.ndarray:
    """(36, 721, 1440) float32: channels interleave (u_l, v_l) over LEVELS."""
    parts = {var: [] for var in ("u_component_of_wind", "v_component_of_wind")}
    for var in parts:
        for c, rows in _level_slices():
            parts[var].append(fetch_chunk(var, t, c)[rows])
    u = np.concatenate(parts["u_component_of_wind"], axis=0)     # (18, 721, 1440)
    v = np.concatenate(parts["v_component_of_wind"], axis=0)
    return np.stack([u, v], axis=1).reshape(2 * len(LEVELS), NLAT, NLON)


# --------------------------------------------------------------------------- regridding
class Bilinear:
    """Bilinear lat/lon -> HEALPix(NEST) regridder as four gathers + weights.

    Precomputed once per process (~50 MB). Longitude is periodic; latitude is clamped at
    the poles (the last row is the pole itself, so no extrapolation happens in practice).
    """

    def __init__(self, nside: int) -> None:
        import healpy as hp
        npix = 12 * nside ** 2
        lon, lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
        fi = (90.0 - lat) / 0.25                 # fractional row, 0 at the north pole
        fj = (lon % 360.0) / 0.25                # fractional column, periodic
        i0 = np.clip(np.floor(fi).astype(np.int64), 0, NLAT - 2)
        j0 = np.floor(fj).astype(np.int64) % NLON
        wi = np.clip(fi - i0, 0.0, 1.0)
        wj = fj - np.floor(fj)
        i1, j1 = i0 + 1, (j0 + 1) % NLON
        self.k = np.stack([i0 * NLON + j0, i0 * NLON + j1, i1 * NLON + j0, i1 * NLON + j1])
        self.w = np.stack([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj]
                          ).astype(np.float32)

    def __call__(self, fields: np.ndarray) -> np.ndarray:
        """(C, 721, 1440) -> (C, npix) float32."""
        flat = fields.reshape(fields.shape[0], -1)
        out = np.zeros((fields.shape[0], self.k.shape[1]), dtype=np.float32)
        for k, w in zip(self.k, self.w):
            out += flat[:, k] * w
        return out


def pool_nested(fine: np.ndarray, factor: int = FACTOR) -> np.ndarray:
    """Exact block mean on the nested hierarchy: (C, npix) -> (C, npix / factor**2)."""
    C, n = fine.shape
    return fine.reshape(C, n // factor ** 2, factor ** 2).mean(axis=2)


# --------------------------------------------------------------------------- store
def open_store(out: Path, T: int, *, fine: bool):
    """Create (or open) the zarr stores. Chunks are one hour so workers never collide."""
    import zarr
    out.mkdir(parents=True, exist_ok=True)
    C = 2 * len(LEVELS)
    root = zarr.open_group(str(out), mode="a")
    if "coarse" not in root:
        g = root.create_group("coarse")
        g.create_array("uv", shape=(T, C, NPIX_COARSE), chunks=(24, C, NPIX_COARSE),
                       dtype="float32", fill_value=np.nan)
    if fine and "fine" not in root:
        g = root.create_group("fine")
        g.create_array("uv", shape=(T, C, NPIX_FINE), chunks=(1, C, NPIX_FINE),
                       dtype="float16", fill_value=np.nan)
    return root


def _write_attrs(root, hours: np.ndarray, years: list[int]) -> None:
    import zarr  # noqa: F401
    root.attrs.update({
        "source": ARCO, "levels": LEVELS.tolist(), "channels": "interleaved (u_l, v_l) over levels",
        "order": "nest", "nside_fine": NSIDE_FINE, "nside_coarse": NSIDE_COARSE,
        "regrid": "bilinear from 0.25deg lat/lon to HEALPix pixel centres",
        "coarse": f"exact nested {FACTOR}x{FACTOR} block mean of fine",
        "time_units": "hours since 1900-01-01 00:00 (ARCO index)", "years": years,
        "excluded_days": list(EXCLUDED_DAYS),
    })
    if "time" not in root:
        root.create_array("time", shape=hours.shape, dtype="int64")
    root["time"][:] = hours


# --------------------------------------------------------------------------- workers
_RG = None


def _worker_init(nside: int) -> None:
    global _RG
    _RG = Bilinear(nside)


def _process(args) -> tuple[int, float]:
    """Fetch, regrid, pool, and write one hour. Returns (index, seconds)."""
    out, i, t, fine = args
    import zarr
    t0 = time.time()
    fields = fetch_hour(t)
    hpx = _RG(fields)                                       # (36, NPIX_FINE) float32
    root = zarr.open_group(str(out), mode="r+")
    root["coarse/uv"][i] = pool_nested(hpx)
    if fine:
        root["fine/uv"][i] = hpx.astype(np.float16)
    return i, time.time() - t0


def _done_mask(out: Path, T: int, fine: bool) -> np.ndarray:
    """Which hours are already written (resume): a NaN-free coarse row means done."""
    import zarr
    root = zarr.open_group(str(out), mode="r")
    coarse = root["coarse/uv"]
    done = np.zeros(T, dtype=bool)
    for a in range(0, T, 24):
        block = coarse[a:a + 24, 0, :8]
        done[a:a + 24][:block.shape[0]] = ~np.isnan(block).any(axis=1)
    if fine:
        # fine chunks are files; a written coarse row without its fine chunk is a torn hour
        f = root["fine/uv"]
        for i in np.where(done)[0]:
            if np.isnan(f[i, 0, :8]).any():
                done[i] = False
    return done


def ingest(out: Path, years: list[int], *, workers: int, fine: bool, limit: int | None) -> None:
    hours = training_hours(years)
    if limit:
        hours = hours[:limit]
    T = len(hours)
    root = open_store(out, T, fine=fine)
    _write_attrs(root, hours, years)
    done = _done_mask(out, T, fine)
    todo = [(out, int(i), int(t), fine) for i, t in enumerate(hours) if not done[i]]
    print(f"[ingest] {T} hours over {years}; {int(done.sum())} done, {len(todo)} to do; "
          f"fine={'yes' if fine else 'no'}; {workers} workers -> {out}", flush=True)
    t0 = time.time()
    n = 0
    with Pool(workers, initializer=_worker_init, initargs=(NSIDE_FINE,)) as pool:
        for i, sec in pool.imap_unordered(_process, todo, chunksize=1):
            n += 1
            if n % 50 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"[ingest] {n}/{len(todo)} hours  {rate * 3600:.0f} h/h  "
                      f"last {sec:.1f}s  eta {(len(todo) - n) / max(rate, 1e-9) / 3600:.1f} h",
                      flush=True)
    print(f"[ingest] done in {(time.time() - t0) / 3600:.2f} h", flush=True)


# --------------------------------------------------------------------------- self-test
def check(stamp: str) -> None:
    """Regrid self-test on one hour: native vs regridded zonal means, coarse == pooled fine."""
    import healpy as hp
    t = hour_index(dt.datetime.fromisoformat(stamp))
    t0 = time.time(); f = fetch_hour(t); t_fetch = time.time() - t0
    t0 = time.time(); rg = Bilinear(NSIDE_FINE); t_init = time.time() - t0
    t0 = time.time(); hpx = rg(f); t_rg = time.time() - t0
    coarse = pool_nested(hpx)
    lat = 90.0 - 0.25 * np.arange(NLAT)
    w = np.cos(np.radians(lat))[:, None]
    lon_p, lat_p = hp.pix2ang(NSIDE_FINE, np.arange(NPIX_FINE), nest=True, lonlat=True)
    u0 = f[0]                                                    # u at level 49
    print(f"fetch {t_fetch:.1f}s (4 chunks), regridder init {t_init:.1f}s, regrid 36 fields {t_rg:.2f}s")
    print("band     native   fine   coarse   (zonal-mean u, level 49)")
    lon_c, lat_c = hp.pix2ang(NSIDE_COARSE, np.arange(NPIX_COARSE), nest=True, lonlat=True)
    for a in range(-90, 90, 30):
        m = (lat >= a) & (lat < a + 30)
        nat = float((u0[m] * w[m]).sum() / (w[m].sum() * NLON))
        fin = float(hpx[0][(lat_p >= a) & (lat_p < a + 30)].mean())
        crs = float(coarse[0][(lat_c >= a) & (lat_c < a + 30)].mean())
        print(f"{a:4d}..{a+30:4d} {nat:8.2f} {fin:7.2f} {crs:8.2f}")
    # pointwise: regridded value at a pixel centre vs the native cell that contains it
    k = np.random.default_rng(0).integers(0, NPIX_FINE, 5000)
    i = np.clip(np.round((90 - lat_p[k]) / 0.25).astype(int), 0, NLAT - 1)
    j = np.round(lon_p[k] / 0.25).astype(int) % NLON
    print(f"nearest-cell agreement (5000 px): RMS {np.sqrt(((hpx[0][k] - u0[i, j]) ** 2).mean()):.3f} m/s "
          f"vs field std {u0.std():.2f}")
    print(f"coarse == pooled fine: {np.allclose(coarse, pool_nested(hpx))};  "
          f"fine store row = {hpx.astype(np.float16).nbytes / 1e6:.1f} MB, coarse row = {coarse.nbytes / 1e6:.2f} MB")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, help="zarr store directory")
    ap.add_argument("--years", type=int, nargs="*", default=[2022, 2023])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-fine", action="store_true", help="coarse store only")
    ap.add_argument("--limit", type=int, default=None, help="first N hours only (testing)")
    ap.add_argument("--check", metavar="ISO_TIME", help="regrid self-test on one hour and exit")
    a = ap.parse_args(argv)
    if a.check:
        check(a.check)
        return
    if a.out is None:
        ap.error("--out is required")
    ingest(a.out, a.years, workers=a.workers, fine=not a.no_fine, limit=a.limit)


if __name__ == "__main__":
    main()
