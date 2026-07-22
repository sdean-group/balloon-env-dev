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
import xarray as xr
from torch.utils.data import Dataset


def _open_zarr(path: str | Path) -> xr.Dataset:
    """Open a v2 store across both pre-2025 and current xarray releases."""
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


def compute_stats(u: np.ndarray, v: np.ndarray, levels: np.ndarray, *, eps: float = 1e-6
                  ) -> NormStats:
    """Per-level mean/std over (time, y, x). u, v shape (T, L, Y, X)."""
    ax = (0, 2, 3)
    return NormStats(
        mean_u=u.mean(ax), std_u=u.std(ax) + eps,
        mean_v=v.mean(ax), std_v=v.std(ax) + eps,
        levels=np.asarray(levels),
    )


def compute_zarr_stats(
    zarr_path: str | Path,
    *,
    levels: tuple[int, int] | None = (49, 66),
    time_chunk: int = 168,
    eps: float = 1e-6,
) -> NormStats:
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
    for start in range(0, n_time, time_chunk):
        stop = min(n_time, start + time_chunk)
        for name in ("u", "v"):
            values = np.asarray(ds[name].isel(time=slice(start, stop)).values, dtype=np.float64)
            finite = np.isfinite(values)
            axes = (0, 2, 3)
            sums[name] += np.where(finite, values, 0.0).sum(axis=axes)
            sums_sq[name] += np.where(finite, values * values, 0.0).sum(axis=axes)
            counts[name] += finite.sum(axis=axes)
    ds.close()

    if np.any(counts["u"] == 0) or np.any(counts["v"] == 0):
        raise ValueError(f"no finite samples for at least one level in {zarr_path}")

    def moments(name: str) -> tuple[np.ndarray, np.ndarray]:
        mean = sums[name] / counts[name]
        variance = np.maximum(sums_sq[name] / counts[name] - mean * mean, 0.0)
        return mean, np.sqrt(variance) + eps

    mean_u, std_u = moments("u")
    mean_v, std_v = moments("v")
    return NormStats(mean_u, std_u, mean_v, std_v, level_vals)


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
        zarr_path: str | Path,
        *,
        crop: int = 64,
        levels: tuple[int, int] | None = (49, 66),
        n_frames: int = 4,
        frame_stride: int = 1,
        stats: NormStats | None = None,
        length: int = 10_000,
        augment: bool = True,
        seed: int = 0,
        lazy: bool = False,
    ) -> None:
        ds = _open_zarr(zarr_path)
        ds = _select_levels(ds, levels)
        self.zarr_path = str(zarr_path)
        self.lazy = bool(lazy)
        self._lazy_ds = None
        self.u = None if self.lazy else np.ascontiguousarray(ds["u"].values, dtype=np.float32)
        self.v = None if self.lazy else np.ascontiguousarray(ds["v"].values, dtype=np.float32)
        self.level_vals = np.asarray(ds["level"].values)
        self.T, self.L, self.Y, self.X = map(int, ds["u"].shape)
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
        blocks = _time_blocks(np.asarray(ds["time"].values))
        starts = [t for (a, b) in blocks for t in range(a, b - span)]
        self.block_starts = np.asarray(starts, dtype=np.int64)
        if len(self.block_starts) == 0:
            raise ValueError(f"no valid {self.n_frames}-frame blocks at frame_stride="
                             f"{self.frame_stride} (blocks={blocks})")
        self.blocks = blocks
        ds.close()

    def __len__(self) -> int:
        return self.length

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_lazy_ds"] = None
        return state

    def _read_block(self, ts: list[int], y0: int, x0: int) -> tuple[np.ndarray, np.ndarray]:
        c = self.crop
        if not self.lazy:
            return (self.u[ts][:, :, y0:y0 + c, x0:x0 + c],
                    self.v[ts][:, :, y0:y0 + c, x0:x0 + c])
        if self._lazy_ds is None:
            self._lazy_ds = _select_levels(
                _open_zarr(self.zarr_path),
                tuple((int(self.level_vals.min()), int(self.level_vals.max()))),
            )
        indexers = {"time": ts, "y": slice(y0, y0 + c), "x": slice(x0, x0 + c)}
        u = np.asarray(self._lazy_ds["u"].isel(**indexers).values, dtype=np.float32)
        v = np.asarray(self._lazy_ds["v"].isel(**indexers).values, dtype=np.float32)
        return u, v

    def __getitem__(self, idx: int) -> torch.Tensor:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t0 = int(self.block_starts[rng.integers(len(self.block_starts))])
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        c = self.crop
        ts = [t0 + k * self.frame_stride for k in range(self.n_frames)]
        u, v = self._read_block(ts, y0, x0)               # (τ,L,c,c)
        f = np.stack([u, v], axis=2).reshape(self.n_frames, self.n_channels, c, c)
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

    def __init__(self, zarr_path: str | Path, **kw) -> None:
        kw["augment"] = False
        super().__init__(zarr_path, **kw)
        ds = _open_zarr(zarr_path)
        self.lat = np.asarray(ds["lat"].values, dtype=np.float64)
        self.lon = np.asarray(ds["lon"].values, dtype=np.float64)
        self.times = np.asarray(ds["time"].values)
        self.coord_norm = CoordNorm.from_grid(self.lat, self.lon)
        self.n_cond_channels = 2
        ds.close()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        t0 = int(self.block_starts[rng.integers(len(self.block_starts))])
        y0 = int(rng.integers(self.Y - self.crop + 1))
        x0 = int(rng.integers(self.X - self.crop + 1))
        c = self.crop
        ts = [t0 + k * self.frame_stride for k in range(self.n_frames)]
        u, v = self._read_block(ts, y0, x0)               # (τ,L,c,c)
        f = np.stack([u, v], axis=2).reshape(self.n_frames, self.n_channels, c, c)
        x = self.stats.normalize(torch.from_numpy(f))   # (τ,C,c,c)
        coords = torch.from_numpy(
            self.coord_norm.channels(self.lat[y0:y0 + c], self.lon[x0:x0 + c]))
        tfeat = torch.from_numpy(time_features(self.times[ts]))
        return x, coords, tfeat
