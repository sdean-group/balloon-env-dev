"""Training data: random (u,v) crops from an ERA5 zarr, per-(level,variable) normalised.

Phase-2 design decisions baked in here (see the progress tracker):
- **per-(level, variable) normalisation** — wind variance swings ~10x with altitude, so
  one global scale would give the diffusion model wildly uneven SNR across channels. We
  standardise each (level, u|v) channel to ~unit variance, which also makes the EDM
  ``sigma_data ≈ 1`` assumption hold. Stats are saved with the checkpoint so samples can
  be mapped back to m/s.
- **random crops** — the balloon roams, so the model must be location-agnostic; we never
  show it absolute coordinates, only fixed-size windows sampled uniformly in space/time.
- **level = channels** (interleaved ``2*l`` = u, ``2*l+1`` = v) — matches the WindowDenoiser
  contract and treats the vertical as feature channels, NOT a 3rd isotropic conv axis
  (horizontal grid ~28 km vs vertical ~380 m — anisotropic by ~70x).

Augmentation is reflection only, with the correct sign flips on the wind components
(mirror in x ⇒ u→−u; mirror in y ⇒ v→−v). Rotations are deferred (they couple u,v per
level and aren't needed for a baseline).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import xarray as xr
except ModuleNotFoundError:  # Sampling from a checkpoint does not require ERA5 I/O.
    xr = None


def _open_zarr(path: str | Path) -> xr.Dataset:
    """Open a v2 store across both pre-2025 and current xarray releases."""
    if xr is None:
        raise ModuleNotFoundError(
            "xarray is required to read ERA5 datasets; install xarray and zarr"
        )
    try:
        return xr.open_zarr(path, consolidated=False, zarr_format=2)
    except TypeError:  # xarray versions before the zarr v3 transition
        return xr.open_zarr(path, consolidated=False)


@dataclass
class NormStats:
    """Per-(level) mean/std for u and v (each shape ``(n_levels,)``)."""

    mean_u: np.ndarray
    std_u: np.ndarray
    mean_v: np.ndarray
    std_v: np.ndarray
    levels: np.ndarray

    @property
    def n_levels(self) -> int:
        return int(len(self.levels))

    def to_torch(self, device="cpu", dtype=torch.float32) -> dict:
        t = lambda a: torch.as_tensor(a, dtype=dtype, device=device)[:, None, None]
        return {"mu_u": t(self.mean_u), "sd_u": t(self.std_u),
                "mu_v": t(self.mean_v), "sd_v": t(self.std_v)}

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """(C,H,W) or (B,C,H,W) raw m/s -> standardised. C = 2*n_levels interleaved."""
        return self._apply(x, invert=False)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Standardised -> m/s."""
        return self._apply(x, invert=True)

    def _apply(self, x: torch.Tensor, *, invert: bool) -> torch.Tensor:
        s = self.to_torch(device=x.device, dtype=x.dtype)
        squeeze = x.dim() == 3
        if squeeze:
            x = x[None]
        B, C, H, W = x.shape
        f = x.reshape(B, self.n_levels, 2, H, W).clone()
        mu_u, sd_u = s["mu_u"], s["sd_u"]
        mu_v, sd_v = s["mu_v"], s["sd_v"]
        if invert:
            f[:, :, 0] = f[:, :, 0] * sd_u + mu_u
            f[:, :, 1] = f[:, :, 1] * sd_v + mu_v
        else:
            f[:, :, 0] = (f[:, :, 0] - mu_u) / sd_u
            f[:, :, 1] = (f[:, :, 1] - mu_v) / sd_v
        out = f.reshape(B, C, H, W)
        return out[0] if squeeze else out

    def save(self, path: str | Path) -> None:
        np.savez(path, mean_u=self.mean_u, std_u=self.std_u,
                 mean_v=self.mean_v, std_v=self.std_v, levels=self.levels)

    @classmethod
    def load(cls, path: str | Path) -> "NormStats":
        d = np.load(path)
        return cls(d["mean_u"], d["std_u"], d["mean_v"], d["std_v"], d["levels"])


def _select_levels(ds: xr.Dataset, levels: tuple[int, int] | None) -> xr.Dataset:
    if levels is None:
        return ds
    lo, hi = levels
    lv = ds["level"].values
    keep = (lv >= lo) & (lv <= hi)
    return ds.isel(level=np.where(keep)[0])


def _moments_chunked(a: np.ndarray, budget_bytes: int = 1 << 30) -> tuple:
    """Exact per-level (mean, std) over (time, y, x) with BOUNDED memory.

    Why this exists (2026-08-02, OOM on the 4-year run): the obvious
    ``a.std(axis, dtype=np.float64)`` is not memory-free — numpy materializes
    ``a - mean`` as a full-size array in the accumulation dtype. On 4 years that
    temporary is 26976x18x121x121x8 B = **56.9 GB**, on top of the 28.4 GB of resident
    float16 arrays; the job was OOM-killed at --mem=48G before printing a single line.
    At 1 year the same temporary is only 14.2 GB, which is why every earlier run was fine
    and the bug stayed latent.

    Two passes (mean, then squared deviations) rather than sum/sumsq in one: it is exact
    and immune to the cancellation in ``E[x^2] - E[x]^2``, and the second pass over an
    in-RAM array is cheap. Peak extra memory is one time-chunk cast to float64, held to
    ``budget_bytes``.
    """
    T, L, Y, X = a.shape
    per_step = L * Y * X * 8                      # bytes for one float64 timestep
    chunk = max(1, int(budget_bytes // max(per_step, 1)))
    n = T * Y * X                                  # samples per level
    s = np.zeros(L, dtype=np.float64)
    for t in range(0, T, chunk):
        s += a[t:t + chunk].astype(np.float64).sum(axis=(0, 2, 3))
    mean = s / n
    m = mean.reshape(1, L, 1, 1)
    ss = np.zeros(L, dtype=np.float64)
    for t in range(0, T, chunk):
        d = a[t:t + chunk].astype(np.float64)
        d -= m
        d *= d
        ss += d.sum(axis=(0, 2, 3))
    return mean, np.sqrt(ss / n)


def compute_stats(u: np.ndarray, v: np.ndarray, levels: np.ndarray, *, eps: float = 1e-6
                  ) -> NormStats:
    """Per-level mean/std over (time, y, x). u, v shape (T, L, Y, X).

    Accumulates in float64: the arrays may be stored float16 (multi-year RAM budget),
    whose native accumulation would be garbage; float64 also tightens the float32 path
    (shift vs the old float32 accumulation is ~1e-6 relative — below training noise).
    Computed chunk-wise (:func:`_moments_chunked`) so peak memory does not scale with the
    number of years — see that function for the OOM this fixed.
    """
    mu, su = _moments_chunked(u)
    mv, sv = _moments_chunked(v)
    return NormStats(
        mean_u=mu.astype(np.float32), std_u=su.astype(np.float32) + eps,
        mean_v=mv.astype(np.float32), std_v=sv.astype(np.float32) + eps,
        levels=np.asarray(levels),
    )


def _open_parts(zarr_path, levels) -> list[xr.Dataset]:
    """Open one path or a list of paths as level-selected datasets (time-concat parts)."""
    paths = [zarr_path] if isinstance(zarr_path, (str, Path)) else list(zarr_path)
    return [_select_levels(_open_zarr(p), levels) for p in paths]


def _concat_var(parts: list[xr.Dataset], var: str, dtype: np.dtype,
                budget_bytes: int = 1 << 30) -> np.ndarray:
    """Time-concatenate ``var`` across parts into one preallocated array.

    Reads in bounded TIME SLICES via ``isel``, never as a whole variable. This is the
    fix for the 2026-08-02 four-year OOM, and the reason is subtle enough to record:

    These zarrs open WITHOUT dask (``chunks=None``, dask-backed = False), so the old
    ``np.asarray(p[var].data, dtype=...)`` fallback did two expensive things at once —
    it materialised the whole variable in its native **float32** (26.5 GiB at 4 years),
    and, because touching ``.data`` on a lazily-backed xarray Variable populates its
    ``MemoryCachedArray``, that float32 copy then stayed resident inside ``parts`` for
    the life of the dataset. Measured peak was 13.25 (float16 out) + 26.5 (cached
    float32) per variable, i.e. **80 GiB** for u+v — against a 64 GB cgroup limit.

    Slicing with ``isel`` reads only the requested window from the store and never
    populates the whole-variable cache, so peak extra memory is one slice
    (``budget_bytes``) regardless of how many years are concatenated. Works whether or
    not the source is dask-backed.
    """
    T = sum(p.sizes["time"] for p in parts)
    L, Y, X = (parts[0].sizes[d] for d in ("level", "y", "x"))
    out = np.empty((T, L, Y, X), dtype=dtype)
    step = max(1, int(budget_bytes // max(L * Y * X * 4, 1)))   # float32 source
    t0 = 0
    for p in parts:
        n = p.sizes["time"]
        for t in range(0, n, step):
            k = min(step, n - t)
            out[t0 + t:t0 + t + k] = p[var].isel(time=slice(t, t + k)).values.astype(dtype)
        t0 += n
    return out


def compute_zarr_stats(
    zarr_path: str | Path,
    *,
    levels: tuple[int, int] | None = (49, 66),
    time_chunk: int = 168,
    eps: float = 1e-6,
    progress_path: str | Path | None = None,
    stop_requested=None,
) -> NormStats | None:
    """Compute training-only normalization statistics without loading the store into RAM.

    The scan uses float64 accumulators and reads at most ``time_chunk`` timestamps at once.
    This is intended for multi-year stores where the eager :func:`compute_stats` path is
    impossible. Non-finite values are ignored independently for each variable and level.
    """
    if time_chunk <= 0:
        raise ValueError("time_chunk must be positive")
    ds = _open_zarr(zarr_path)
    ds = _select_levels(ds, levels)
    level_vals = np.asarray(ds["level"].values)
    n_levels = len(level_vals)
    sums = {name: np.zeros(n_levels, dtype=np.float64) for name in ("u", "v")}
    sums_sq = {name: np.zeros(n_levels, dtype=np.float64) for name in ("u", "v")}
    counts = {name: np.zeros(n_levels, dtype=np.int64) for name in ("u", "v")}

    n_time = int(ds.sizes["time"])
    progress = Path(progress_path) if progress_path is not None else None
    next_start = 0
    if progress is not None and progress.exists():
        saved = np.load(progress)
        if int(saved["n_time"]) != n_time or not np.array_equal(saved["levels"], level_vals):
            raise ValueError(f"{progress}: progress metadata does not match {zarr_path}")
        next_start = int(saved["next_start"])
        for name in ("u", "v"):
            sums[name] = saved[f"{name}_sum"]
            sums_sq[name] = saved[f"{name}_sum_sq"]
            counts[name] = saved[f"{name}_count"]

    def save_progress(stop: int) -> None:
        if progress is None:
            return
        temporary = progress.with_suffix(progress.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                n_time=n_time,
                levels=level_vals,
                next_start=stop,
                u_sum=sums["u"],
                u_sum_sq=sums_sq["u"],
                u_count=counts["u"],
                v_sum=sums["v"],
                v_sum_sq=sums_sq["v"],
                v_count=counts["v"],
            )
        temporary.replace(progress)

    for start in range(next_start, n_time, time_chunk):
        stop = min(n_time, start + time_chunk)
        for name in ("u", "v"):
            values = np.asarray(ds[name].isel(time=slice(start, stop)).values, dtype=np.float64)
            finite = np.isfinite(values)
            axes = (0, 2, 3)
            sums[name] += np.where(finite, values, 0.0).sum(axis=axes)
            sums_sq[name] += np.where(finite, values * values, 0.0).sum(axis=axes)
            counts[name] += finite.sum(axis=axes)
        save_progress(stop)
        if stop_requested is not None and stop_requested() and stop < n_time:
            ds.close()
            return None
    ds.close()

    if np.any(counts["u"] == 0) or np.any(counts["v"] == 0):
        raise ValueError(f"no finite samples for at least one level in {zarr_path}")

    def moments(name: str) -> tuple[np.ndarray, np.ndarray]:
        mean = sums[name] / counts[name]
        variance = np.maximum(sums_sq[name] / counts[name] - mean * mean, 0.0)
        return mean, np.sqrt(variance) + eps

    mean_u, std_u = moments("u")
    mean_v, std_v = moments("v")
    result = NormStats(mean_u, std_u, mean_v, std_v, level_vals)
    if progress is not None and progress.exists():
        progress.unlink()
    return result


class WindCropDataset(Dataset):
    """Random fixed-size (u,v) crops from an ERA5 zarr, normalised + reflection-augmented.

    Args:
        zarr_path: ERA5 artifact / training zarr with dims (time, level, y, x), vars u, v.
        crop: square crop side in pixels. Must be <= grid (y, x) extent.
        levels: inclusive (lo, hi) model-level band to keep, or None for all.
        stats: precomputed NormStats; if None they are computed from this dataset.
        length: virtual epoch length (number of random crops per pass).
        augment: enable reflection augmentation (with wind sign flips).
        seed: base RNG seed (per-item streams derive from it for reproducibility).
    """

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        crop: int = 64,
        levels: tuple[int, int] | None = (49, 66),
        stats: NormStats | None = None,
        length: int = 10_000,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        ds = _open_zarr(zarr_path)
        ds = _select_levels(ds, levels)
        self.u = np.ascontiguousarray(ds["u"].values, dtype=np.float32)  # (T,L,Y,X)
        self.v = np.ascontiguousarray(ds["v"].values, dtype=np.float32)
        self.level_vals = np.asarray(ds["level"].values)
        self.T, self.L, self.Y, self.X = self.u.shape
        if crop > min(self.Y, self.X):
            raise ValueError(f"crop {crop} > grid {(self.Y, self.X)}")
        self.crop = int(crop)
        self.length = int(length)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.n_channels = 2 * self.L
        self.stats = stats or compute_stats(self.u, self.v, self.level_vals)

    def __len__(self) -> int:
        return self.length

    def _raw_crop(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        t = int(rng.integers(self.T))
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        c = self.crop
        u = self.u[t, :, y0:y0 + c, x0:x0 + c]
        v = self.v[t, :, y0:y0 + c, x0:x0 + c]
        return u, v  # each (L, c, c)

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        u, v = self._raw_crop(rng)
        f = np.stack([u, v], axis=1).reshape(self.n_channels, self.crop, self.crop)
        x = self.stats.normalize(torch.from_numpy(f))
        if self.augment:
            # Reflect in *anomaly* space (post-normalisation): the climatological mean jet
            # is NOT reflection-symmetric, but the zero-mean anomaly is. Mirror in x negates
            # the (normalised) u anomaly; mirror in y negates v. This keeps each channel
            # zero-mean/unit-var, unlike mirroring the raw field.
            g = x.reshape(self.L, 2, self.crop, self.crop)
            if rng.random() < 0.5:                    # mirror in x: reverse cols, u_anom -> -u_anom
                g = torch.flip(g, dims=(3,))
                g[:, 0] = -g[:, 0]
            if rng.random() < 0.5:                    # mirror in y: reverse rows, v_anom -> -v_anom
                g = torch.flip(g, dims=(2,))
                g[:, 1] = -g[:, 1]
            x = g.reshape(self.n_channels, self.crop, self.crop).contiguous()
        return x


def _augment_pair(xt: torch.Tensor, xtp1: torch.Tensor, L: int, c: int,
                  rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Reflection augmentation shared by a (frame_t, frame_{t+k}) pair.

    Both frames MUST get the *same* flip so the learned transition stays coherent — a
    different mirror per frame would teach a spurious dynamic. Same anomaly-space sign-flip
    rule as ``WindCropDataset`` (mirror x ⇒ u→−u; mirror y ⇒ v→−v).
    """
    gt = xt.reshape(L, 2, c, c)
    gp = xtp1.reshape(L, 2, c, c)
    if rng.random() < 0.5:
        gt, gp = torch.flip(gt, dims=(3,)), torch.flip(gp, dims=(3,))
        gt[:, 0], gp[:, 0] = -gt[:, 0], -gp[:, 0]
    if rng.random() < 0.5:
        gt, gp = torch.flip(gt, dims=(2,)), torch.flip(gp, dims=(2,))
        gt[:, 1], gp[:, 1] = -gt[:, 1], -gp[:, 1]
    n = 2 * L
    return gt.reshape(n, c, c).contiguous(), gp.reshape(n, c, c).contiguous()


# --------------------------------------------------------------- conditioning features
@dataclass
class CoordNorm:
    """Training-domain coordinate normalization: (coord - center) / half_width -> ~[-1, 1].

    Stored in the checkpoint so inference normalizes lat/lon identically. ``wrap_lon``
    maps an inference longitude onto the training convention (0–360 vs ±180) before
    normalizing — the two conventions differ by a silent 360° branch.
    """

    lat0: float
    lat_half: float
    lon0: float
    lon_half: float

    @classmethod
    def from_grid(cls, lat: np.ndarray, lon: np.ndarray) -> "CoordNorm":
        return cls(
            lat0=float((lat.max() + lat.min()) / 2), lat_half=float((lat.max() - lat.min()) / 2),
            lon0=float((lon.max() + lon.min()) / 2), lon_half=float((lon.max() - lon.min()) / 2),
        )

    def wrap_lon(self, lon: np.ndarray) -> np.ndarray:
        lon = np.asarray(lon, dtype=np.float64)
        lon = np.where(lon - self.lon0 > 180.0, lon - 360.0, lon)
        lon = np.where(lon - self.lon0 < -180.0, lon + 360.0, lon)
        return lon

    def channels(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Per-pixel coord channels for a window: lat (H,), lon (W,) -> (2, H, W) float32."""
        la = (np.asarray(lat, dtype=np.float64) - self.lat0) / self.lat_half
        lo = (self.wrap_lon(lon) - self.lon0) / self.lon_half
        H, W = len(la), len(lo)
        out = np.empty((2, H, W), dtype=np.float32)
        out[0] = la[:, None]
        out[1] = lo[None, :]
        return out

    def to_dict(self) -> dict:
        return {"lat0": self.lat0, "lat_half": self.lat_half,
                "lon0": self.lon0, "lon_half": self.lon_half}


N_TIME_FEATURES = 6


def time_features(times: np.ndarray) -> np.ndarray:
    """Cyclic time harmonics for datetime64 stamps -> (len(times), 6) float32.

    Columns: [sin, cos] annual phase, [sin, cos] semiannual phase, [sin, cos] diurnal
    phase (UTC hour — local solar time is a learnable lon/15 offset given the coordinate
    channels). Years are exchangeable by construction: only the phase within the
    year/day enters. Low-order harmonics keep the encoding smooth in date, so a single
    training year cannot be memorized day-by-day.
    """
    t = np.asarray(times).astype("datetime64[s]")
    doy = (t - t.astype("datetime64[Y]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    hod = (t - t.astype("datetime64[D]")).astype("timedelta64[s]").astype(np.float64) / 3600.0
    a = 2.0 * np.pi * doy / 365.25
    d = 2.0 * np.pi * hod / 24.0
    return np.stack([np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a),
                     np.sin(d), np.cos(d)], axis=-1).astype(np.float32)


def _time_blocks(times: np.ndarray, *, step_tol: float = 1.5) -> list[tuple[int, int]]:
    """Split a time axis into contiguous blocks at gaps larger than the median step.

    Returns inclusive-exclusive ``(start, stop)`` index ranges. A "gap" is a step more than
    ``step_tol``x the median spacing — this is how the seasonal-block boundaries in
    ``era5_temporal.zarr`` (and the 3 blocks in ``era5_train.zarr``) get detected so the
    pair dataset never pairs *across* a discontinuity.
    """
    n = len(times)
    if n < 2:
        return [(0, n)]
    if np.issubdtype(np.asarray(times).dtype, np.datetime64):
        d = np.diff(times).astype("timedelta64[s]").astype(np.float64)
    else:
        d = np.diff(np.asarray(times, dtype=np.float64))
    med = float(np.median(d))
    breaks = np.where(d > step_tol * med)[0] + 1 if med > 0 else np.array([], dtype=int)
    bounds = [0, *breaks.tolist(), n]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


class WindPairDataset(Dataset):
    """Co-located (frame_t, frame_{t+k}) crop pairs for autoregressive (M3) temporal training.

    Yields ``(x_t, x_tp1)``, each a normalised ``(2L, crop, crop)`` tensor sampled from the
    SAME spatial window at times ``t`` and ``t + frame_stride``. The model learns
    ``p(frame_{t+k} | frame_t)``; ``frame_stride`` is the temporal-step knob (1 = native
    cadence, larger = coarser steps / more dynamics per step, GenCast-style).

    Pairs that would straddle a time discontinuity are excluded (see :func:`_time_blocks`),
    so a pair is always two genuinely-consecutive frames of the same contiguous block.

    Args mirror :class:`WindCropDataset`; ``frame_stride`` is the extra temporal knob.
    """

    def __init__(
        self,
        zarr_path: str | Path,
        *,
        crop: int = 64,
        levels: tuple[int, int] | None = (49, 66),
        frame_stride: int = 1,
        stats: NormStats | None = None,
        length: int = 10_000,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        ds = _open_zarr(zarr_path)
        ds = _select_levels(ds, levels)
        self.u = np.ascontiguousarray(ds["u"].values, dtype=np.float32)  # (T,L,Y,X)
        self.v = np.ascontiguousarray(ds["v"].values, dtype=np.float32)
        self.level_vals = np.asarray(ds["level"].values)
        self.T, self.L, self.Y, self.X = self.u.shape
        if crop > min(self.Y, self.X):
            raise ValueError(f"crop {crop} > grid {(self.Y, self.X)}")
        self.crop = int(crop)
        self.frame_stride = int(frame_stride)
        self.length = int(length)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.n_channels = 2 * self.L
        self.stats = stats or compute_stats(self.u, self.v, self.level_vals)

        # valid start indices t such that (t, t+frame_stride) lie in the same contiguous block
        blocks = _time_blocks(np.asarray(ds["time"].values))
        starts = [t for (a, b) in blocks for t in range(a, b - self.frame_stride)]
        self.pair_starts = np.asarray(starts, dtype=np.int64)
        if len(self.pair_starts) == 0:
            raise ValueError(f"no valid frame pairs at frame_stride={self.frame_stride} "
                             f"(blocks={blocks})")
        self.blocks = blocks

    def __len__(self) -> int:
        return self.length

    def _frame(self, t: int, y0: int, x0: int) -> np.ndarray:
        c = self.crop
        u = self.u[t, :, y0:y0 + c, x0:x0 + c]
        v = self.v[t, :, y0:y0 + c, x0:x0 + c]
        return np.stack([u, v], axis=1).reshape(self.n_channels, c, c)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t = int(self.pair_starts[rng.integers(len(self.pair_starts))])
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        ft = self.stats.normalize(torch.from_numpy(self._frame(t, y0, x0)))
        fp = self.stats.normalize(torch.from_numpy(self._frame(t + self.frame_stride, y0, x0)))
        if self.augment:
            ft, fp = _augment_pair(ft, fp, self.L, self.crop, rng)
        return ft, fp


class WindSpaceTimeDataset(Dataset):
    """Contiguous H×W×τ blocks of consecutive frames for the joint-spacetime (M2) denoiser.

    Yields a single normalised ``(τ, 2L, crop, crop)`` block (``τ`` = ``n_frames``) sampled
    from one co-located window across ``frame_stride``-spaced consecutive times, never
    straddling a time discontinuity (see :func:`_time_blocks`). Reflection augmentation applies
    the SAME flip to every frame in the block.

    Args mirror :class:`WindCropDataset`; ``n_frames``/``frame_stride`` are the block knobs.
    """

    def __init__(
        self,
        zarr_path: str | Path | list,
        *,
        crop: int = 64,
        levels: tuple[int, int] | None = (49, 66),
        n_frames: int = 4,
        frame_stride: int = 1,
        stats: NormStats | None = None,
        length: int = 10_000,
        augment: bool = True,
        seed: int = 0,
        storage_dtype: str = "float32",
        lazy: bool = False,
    ) -> None:
        # zarr_path may be a LIST of per-year zarrs (time-concatenated in the given order;
        # grids must match). storage_dtype="float16" halves in-RAM size for multi-year
        # sets (quantization ~0.01 m/s vs ~8 m/s data std); items are cast back to
        # float32 before normalization, so the model always sees float32.
        # lazy=True keeps NOTHING resident and reads each block from the store(s) on
        # demand — for multi-year stores that do not fit in RAM. It requires precomputed
        # NormStats (see compute_zarr_stats), since a full pass is exactly what we avoid.
        paths = [zarr_path] if isinstance(zarr_path, (str, Path)) else list(zarr_path)
        self.zarr_paths = [str(p) for p in paths]
        self.levels = levels
        self.lazy = bool(lazy)
        self._lazy_parts = None
        parts = _open_parts(paths, levels)
        p0 = parts[0]
        for p in parts[1:]:
            if not (np.array_equal(p["level"].values, p0["level"].values)
                    and np.allclose(p["lat"].values, p0["lat"].values)
                    and np.allclose(p["lon"].values, p0["lon"].values)):
                raise ValueError("multi-zarr training set: level/lat/lon grids differ")
        self.level_vals = np.asarray(p0["level"].values)
        self.lat_vals = np.asarray(p0["lat"].values, dtype=np.float64)
        self.lon_vals = np.asarray(p0["lon"].values, dtype=np.float64)
        times = np.concatenate([np.asarray(p["time"].values) for p in parts])
        if np.any(np.diff(times.astype("datetime64[ns]").astype(np.int64)) <= 0):
            raise ValueError("multi-zarr training set: time must be strictly increasing "
                             "(pass paths in ascending year order)")
        self.times = times
        # global time index -> (part, local index) for the lazy reader
        self._part_offsets = np.cumsum([0] + [int(p.sizes["time"]) for p in parts])
        self.T = int(len(times))
        self.L, self.Y, self.X = (int(p0.sizes[d]) for d in ("level", "y", "x"))
        if self.lazy:
            self.u = self.v = None
        else:
            dtype = np.dtype(storage_dtype)
            self.u = _concat_var(parts, "u", dtype)                      # (T,L,Y,X)
            self.v = _concat_var(parts, "v", dtype)
        if crop > min(self.Y, self.X):
            raise ValueError(f"crop {crop} > grid {(self.Y, self.X)}")
        self.crop = int(crop)
        self.n_frames = int(n_frames)
        self.frame_stride = int(frame_stride)
        self.length = int(length)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.n_channels = 2 * self.L
        if self.lazy and stats is None:
            raise ValueError("lazy spacetime loading requires precomputed NormStats")
        self.stats = stats or compute_stats(self.u, self.v, self.level_vals)
        if not np.array_equal(np.asarray(self.stats.levels), self.level_vals):
            raise ValueError(
                f"normalization levels {self.stats.levels} do not match data levels "
                f"{self.level_vals}"
            )

        # a block needs (n_frames-1)*frame_stride frames after t, all in one contiguous block
        span = (self.n_frames - 1) * self.frame_stride
        blocks = _time_blocks(self.times)
        starts = [t for (a, b) in blocks for t in range(a, b - span)]
        self.block_starts = np.asarray(starts, dtype=np.int64)
        if len(self.block_starts) == 0:
            raise ValueError(f"no valid {self.n_frames}-frame blocks at frame_stride="
                             f"{self.frame_stride} (blocks={blocks})")
        self.blocks = blocks
        for p in parts:
            p.close()

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_lazy_parts"] = None      # never pickle open stores into workers
        return state

    def _read_block(self, ts: list[int], y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        """(u, v) for frames ``ts`` in the window: from RAM, or lazily from the store(s)."""
        c = self.crop
        if not self.lazy:
            return (self.u[ts][:, :, y0:y0 + c, x0:x0 + c],
                    self.v[ts][:, :, y0:y0 + c, x0:x0 + c])
        if self._lazy_parts is None:               # opened per worker process (see __getstate__)
            self._lazy_parts = _open_parts(self.zarr_paths, self.levels)
        ts = np.asarray(ts, dtype=np.int64)
        part = np.searchsorted(self._part_offsets, ts, side="right") - 1
        u = np.empty((len(ts), self.L, c, c), dtype=np.float32)
        v = np.empty_like(u)
        for k in np.unique(part):                  # a block may straddle a year boundary
            sel = np.where(part == k)[0]
            local = (ts[sel] - self._part_offsets[k]).tolist()
            ds = self._lazy_parts[k]
            idx = {"time": local, "y": slice(y0, y0 + c), "x": slice(x0, x0 + c)}
            u[sel] = np.asarray(ds["u"].isel(**idx).values, dtype=np.float32)
            v[sel] = np.asarray(ds["v"].isel(**idx).values, dtype=np.float32)
        return u, v

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t0 = int(self.block_starts[rng.integers(len(self.block_starts))])
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        c = self.crop
        ts = [t0 + k * self.frame_stride for k in range(self.n_frames)]
        u, v = self._read_block(ts, y0, x0)               # (τ,L,c,c)
        f = (np.stack([u, v], axis=2).reshape(self.n_frames, self.n_channels, c, c)
             .astype(np.float32, copy=False))           # no-op unless storage is float16
        x = self.stats.normalize(torch.from_numpy(f))   # (τ,C,c,c), normalize handles 4D
        if self.augment:
            g = x.reshape(self.n_frames, self.L, 2, c, c)
            if rng.random() < 0.5:                        # mirror x: u_anom -> -u_anom (all frames)
                g = torch.flip(g, dims=(4,))
                g[:, :, 0] = -g[:, :, 0]
            if rng.random() < 0.5:                        # mirror y: v_anom -> -v_anom (all frames)
                g = torch.flip(g, dims=(3,))
                g[:, :, 1] = -g[:, :, 1]
            x = g.reshape(self.n_frames, self.n_channels, c, c).contiguous()
        return x


def coarsen(x: torch.Tensor, factor: int) -> torch.Tensor:
    """Block-mean a ``(τ, C, H, W)`` block down by ``factor`` → ``(τ, C, H/f, W/f)``.

    Area average (``avg_pool2d``), NOT a blur-and-subsample: the cell mean of the fine
    field IS the coarse value, which is how a coarse-resolution model represents a grid
    cell and makes "what did the diffusion model add inside the cell" a well-posed
    question. Block-mean commutes with the per-channel affine normalisation, so
    coarsening the already-normalised block equals normalising the coarsened raw field.
    """
    if x.shape[-1] % factor or x.shape[-2] % factor:
        raise ValueError(f"crop {tuple(x.shape[-2:])} not divisible by coarse factor {factor}")
    return torch.nn.functional.avg_pool2d(x, int(factor))


def measure_residual_scale(ds, n_blocks: int = 256) -> float:
    """RMS of ``x - bilinear_upsample(blockmean(x))`` over a deterministic block sample.

    This is the unit that makes the residual diffusion well-scaled (see
    :class:`spacetime.EDMPrecondSpaceTime`). Measured on the TRAINING set at run start
    rather than hard-coded, so it stays correct if the coarse factor, the level set, or
    the year changes. ``ds[i]`` is deterministic (seeded per index), so the value is
    reproducible for a given dataset and ``n_blocks``.

    The residual is zero-mean by construction, so RMS == std to within the sample error.
    """
    tot, cnt = 0.0, 0
    for i in range(int(n_blocks)):
        x, _, _, c = ds[i]
        base = torch.nn.functional.interpolate(c, size=x.shape[-2:], mode="bilinear",
                                               align_corners=False)
        r = x - base
        tot += float((r.double() ** 2).sum())
        cnt += r.numel()
    return (tot / max(cnt, 1)) ** 0.5


class WindCondSpaceTimeDataset(WindSpaceTimeDataset):
    """Conditional H×W×τ blocks: (block, coord channels, per-frame time features).

    The Phase-5 conditional dataset. Yields
    ``(x, coords, tfeat)`` = (normalised ``(τ, 2L, crop, crop)`` block,
    ``(2, crop, crop)`` per-pixel lat/lon channels via :class:`CoordNorm`,
    ``(τ, 6)`` cyclic time harmonics via :func:`time_features`).

    Reflection augmentation is DISABLED regardless of the flag: mirroring the field while
    keeping coordinates teaches false geography, and mirroring both trains on a mirrored
    Earth that never occurs at inference (phase-5 decision — real 2023 data replaces it).
    """

    def __init__(self, zarr_path: str | Path | list, **kw) -> None:
        kw["augment"] = False
        super().__init__(zarr_path, **kw)
        self.lat = self.lat_vals            # parent stores grid + times (multi-zarr aware)
        self.lon = self.lon_vals
        self.coord_norm = CoordNorm.from_grid(self.lat, self.lon)
        self.n_cond_channels = 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t0 = int(self.block_starts[rng.integers(len(self.block_starts))])
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        c = self.crop
        ts = [t0 + k * self.frame_stride for k in range(self.n_frames)]
        u, v = self._read_block(ts, y0, x0)               # (τ,L,c,c)
        f = (np.stack([u, v], axis=2).reshape(self.n_frames, self.n_channels, c, c)
             .astype(np.float32, copy=False))           # no-op unless storage is float16
        x = self.stats.normalize(torch.from_numpy(f))   # (τ,C,c,c)
        coords = torch.from_numpy(
            self.coord_norm.channels(self.lat[y0:y0 + c], self.lon[x0:x0 + c]))
        tfeat = torch.from_numpy(time_features(self.times[ts]))
        return x, coords, tfeat


class WindCoarseCondSpaceTimeDataset(WindCondSpaceTimeDataset):
    """Coarse-conditioned blocks: ``(x, coords, tfeat, coarse)`` — the downscaling setup.

    Adds a horizontally block-meaned copy of the SAME block, ``(τ, 2L, crop/f, crop/f)``,
    as a fourth item. Everything else (coords, cyclic time harmonics, no augmentation,
    the block sampler) is inherited unchanged, so a run with this dataset differs from
    the plain conditional run in exactly one respect.

    Why the vertical stack is kept at full resolution while only the horizontal is
    coarsened: the measured defect (2026-07-30) is that the model invents almost no
    vertical decoupling — 1-3% opposing-wind columns against ERA5's 20% — so the coarse
    field deliberately SUPPLIES the vertical structure and asks the model only for
    horizontal detail. That is a real fix for the simulator (the sim's job is "given a
    forecast, produce a plausible realised field") but it is a WEAKER generative claim
    than inventing the structure, and the evaluation must say so: the coarse row is not
    comparable head-to-head with the unconditional rows, and it must additionally beat
    plain upsampling of its own conditioning field to have added anything at all.

    ``coarse_factor`` = 8 by default: a 64² crop at 0.25° → 8×8 cells of 2° ≈ 200 km,
    which is coarser than the 56 km L_eff target (so real work is left to the model) and
    comparable to operational coarse products.
    """

    def __init__(self, zarr_path: str | Path | list, *, coarse_factor: int = 8, **kw) -> None:
        super().__init__(zarr_path, **kw)
        if self.crop % int(coarse_factor):
            raise ValueError(f"crop {self.crop} not divisible by coarse_factor {coarse_factor}")
        self.coarse_factor = int(coarse_factor)
        self.n_coarse_channels = self.n_channels          # all 2L levels are kept

    def __getitem__(self, idx: int):
        x, coords, tfeat = super().__getitem__(idx)
        return x, coords, tfeat, coarsen(x, self.coarse_factor)
