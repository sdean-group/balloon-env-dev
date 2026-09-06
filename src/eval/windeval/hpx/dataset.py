"""Training data for the coarse global stage: time blocks of whole-sphere HEALPix wind.

Reads the ``coarse/uv`` group of one or more stores written by :mod:`ingest_arco`
(``(T, C, 12*nside**2)`` NEST order, hourly, days 8-14 of every month absent) and yields

    x      (τ, C, 12, nside, nside)   normalised wind block, face (XY) layout, float32
    coords (3, 12, nside, nside)      lat/90, sin(lon), cos(lon)   - constant across τ
    tfeat  (τ, 6)                     annual / semiannual / diurnal harmonics per frame
    slow   (τ, 1)                     optional QBO-style slow-state index (see below), else zeros

Design notes (Stage 1 register, 2026-09-05):
- **Whole sphere per item**: no spatial cropping. Stage 1 is the planetary weather state;
  coherence across the globe is the point. Augmentation comes from time, not crops.
- **Blocks of ``n_frames`` at ``stride_hours``** (default 8 x 6 h = 48 h; the synoptic
  e-folding time at 2° is ~30 h, so a block spans >1 decorrelation time). Frames never
  straddle a gap in the store (the excluded days), and the phase within the 6-hourly
  cadence is random per item so every hour of the day is seen.
- **Normalisation** is per channel over the whole store (mean/std), stored alongside.
- **Slow-state index** (optional, ``slow_index=True``): the tropical (|lat| < 10°) mean of
  channel 0 (u at the top level, ~53 hPa) over the ``lookback_hours`` before the block.
  Cyclic time harmonics cannot know the QBO phase (measured on cBottle, 2026-09-05); this
  scalar can. It is computed from the store itself, so it is available for every block
  that has a full lookback, and is zero with ``slow_mask=0`` otherwise.
- **RAM**: the coarse store is small (1.8 MB per hour); four years float16 is ~24 GB and
  is loaded fully (the node has 1 TB). Each item then costs two gathers, no I/O.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .layout import FaceLayout, coord_channels
except ImportError:  # standalone script path on the cluster
    from layout import FaceLayout, coord_channels  # type: ignore


def time_features(hours_since_1900: np.ndarray) -> np.ndarray:
    """Cyclic harmonics for ARCO hour indices -> ``(n, 6)`` float32. Same six columns as the
    regional model (annual, semiannual, diurnal; UTC), so checkpoints stay comparable."""
    t = (np.datetime64("1900-01-01T00", "h") + hours_since_1900.astype("timedelta64[h]")).astype("datetime64[s]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    hod = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.float64) / 3600.0
    a = 2.0 * np.pi * doy / 365.25
    d = 2.0 * np.pi * hod / 24.0
    return np.stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a),
                     np.sin(d), np.cos(d)], axis=-1).astype(np.float32)


def _open_coarse(path: str | Path):
    import zarr
    root = zarr.open_group(str(path), mode="r")
    return root["coarse/uv"], np.asarray(root["time"][:], dtype=np.int64), root


def valid_rows(path: str | Path, *, refresh: bool = False) -> np.ndarray:
    """Which hours are fully written (no NaN). Cached next to the store as ``valid.npy``."""
    p = Path(path) / "valid.npy"
    arr, _, _ = _open_coarse(path)
    if p.exists() and not refresh:
        v = np.load(p)
        if v.shape[0] == arr.shape[0]:
            return v
    T = arr.shape[0]
    v = np.zeros(T, dtype=bool)
    step = arr.chunks[0]
    for a in range(0, T, step):
        v[a:a + step] = ~np.isnan(arr[a:a + step, 0, :8]).any(axis=1)
    np.save(p, v)
    return v


def compute_stats(paths: list[str | Path], *, max_hours: int = 4000, seed: int = 0) -> dict:
    """Per-channel mean/std over a random subset of valid hours across the stores."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in paths:
        arr, _, _ = _open_coarse(p)
        v = np.where(valid_rows(p))[0]
        pick = np.sort(rng.choice(v, size=min(len(v), max_hours // len(paths)), replace=False))
        for a in range(0, len(pick), 64):           # contiguous-ish reads
            rows.append(arr[pick[a:a + 64]].astype(np.float64))
    x = np.concatenate(rows, axis=0)                 # (n, C, npix)
    mean = x.mean(axis=(0, 2)); std = x.std(axis=(0, 2)) + 1e-6
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "n_hours": int(x.shape[0])}


def _runs(hours: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous hourly runs as [start, stop) index ranges (gaps = excluded days / store edges)."""
    br = np.where(np.diff(hours) != 1)[0] + 1
    b = [0, *br.tolist(), len(hours)]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


class HpxCoarseBlocks(Dataset):
    def __init__(self, stores: list[str | Path], layout_dir: str | Path, *,
                 n_frames: int = 8, stride_hours: int = 6, stats: dict | None = None,
                 length: int = 10_000, seed: int = 0, storage_dtype: str = "float16",
                 slow_index: bool = False, lookback_hours: int = 720) -> None:
        arrs, times, valids = [], [], []
        for p in stores:
            arr, t, _ = _open_coarse(p)
            arrs.append(arr); times.append(t); valids.append(valid_rows(p))
        C, npix = arrs[0].shape[1:]
        nside = int(round((npix / 12) ** 0.5))
        for a in arrs[1:]:
            if a.shape[1:] != (C, npix):
                raise ValueError("stores differ in channels or nside")
        self.C, self.nside, self.npix = int(C), nside, int(npix)
        self.layout = FaceLayout.load(nside, layout_dir)
        self.coords = torch.from_numpy(coord_channels(nside, self.layout.perm))   # (3,12,n,n)

        # load everything into RAM (float16 by default), in store order, dropping invalid rows
        chunks, hrs = [], []
        dtype = np.dtype(storage_dtype)
        for arr, t, v in zip(arrs, times, valids):
            keep = np.where(v)[0]
            for a in range(0, len(keep), 256):
                idx = keep[a:a + 256]
                chunks.append(arr[idx].astype(dtype))
            hrs.append(t[keep])
        self.uv = np.concatenate(chunks, axis=0)                 # (T, C, npix)
        self.hours = np.concatenate(hrs)
        order = np.argsort(self.hours, kind="stable")
        self.uv, self.hours = self.uv[order], self.hours[order]
        if np.any(np.diff(self.hours) <= 0):
            raise ValueError("stores overlap in time; pass disjoint stores")
        self.T = len(self.hours)

        self.n_frames, self.stride = int(n_frames), int(stride_hours)
        span = (self.n_frames - 1) * self.stride
        self.runs = _runs(self.hours)
        self.block_starts = np.asarray([s for (a, b) in self.runs for s in range(a, b - span)], dtype=np.int64)
        if len(self.block_starts) == 0:
            raise ValueError("no valid blocks: stores too short for the requested block span")
        self.length, self.seed = int(length), int(seed)

        st = stats or compute_stats(stores)
        self.mean = torch.from_numpy(np.asarray(st["mean"], dtype=np.float32))[:, None]
        self.std = torch.from_numpy(np.asarray(st["std"], dtype=np.float32))[:, None]
        self.stats = st

        self.slow_index, self.lookback = bool(slow_index), int(lookback_hours)
        if self.slow_index:
            # tropical mean of channel 0 per hour, then a causal running mean over the lookback
            import healpy as hp
            lon, lat = hp.pix2ang(nside, np.arange(npix), nest=True, lonlat=True)
            trop = np.abs(lat) < 10.0
            u_trop = self.uv[:, 0, trop].astype(np.float32).mean(axis=1)             # (T,)
            # causal mean over the previous `lookback` hours of CALENDAR time, tolerating the
            # gaps (the excluded week each month means no contiguous run is 30 days long);
            # present when at least half of the window's hours exist in the store
            cs = np.concatenate([[0.0], np.cumsum(u_trop, dtype=np.float64)])
            hi = np.arange(self.T)                                            # rows before i
            lo = np.searchsorted(self.hours, self.hours - self.lookback, side="left")
            n = hi - lo
            mean = (cs[hi] - cs[lo]) / np.maximum(n, 1)
            self._slow = np.where(n >= 0.5 * self.lookback, mean, np.nan).astype(np.float32)
            s = self._slow[~np.isnan(self._slow)]
            self.slow_mean, self.slow_std = float(s.mean()), float(s.std() + 1e-6)

    def __len__(self) -> int:
        return self.length

    def frames(self, ts: np.ndarray) -> torch.Tensor:
        """Normalised face-layout block for explicit row indices: ``(len(ts), C, 12, n, n)``."""
        x = torch.from_numpy(self.uv[ts].astype(np.float32))              # (τ, C, npix)
        x = (x - self.mean) / self.std
        return self.layout.to_faces(x)

    def __getitem__(self, idx: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t0 = int(self.block_starts[rng.integers(len(self.block_starts))])
        ts = t0 + self.stride * np.arange(self.n_frames)
        x = self.frames(ts)
        tfeat = torch.from_numpy(time_features(self.hours[ts]))
        slow = torch.zeros(self.n_frames, 2, dtype=torch.float32)          # (value, present)
        if self.slow_index:
            v = self._slow[ts]
            ok = ~np.isnan(v)
            slow[:, 0] = torch.from_numpy(np.where(ok, (v - self.slow_mean) / self.slow_std, 0.0).astype(np.float32))
            slow[:, 1] = torch.from_numpy(ok.astype(np.float32))
        return x, self.coords, tfeat, slow
