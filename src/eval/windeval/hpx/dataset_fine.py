"""Stage 2 training data: residual patches on the nside-256 faces with the coarse conditioner.

One item = one contiguous run of ``n_frames`` hourly frames (13 = hours 0..12) and ``k``
random coarse-aligned patches cut from it (cBottle trains on all patches of one snapshot; we
take k of them so one item is one batch). Per patch and frame t, with coarse frames C0, C6,
C12 taken from the true block means (the coarse group of the same store):

    q = min(t // 6, 1); w = (t - 6q) / 6;  prev = C[q]; next = C[q+1];  lin = (1-w) prev + w next
    target   r = (x_t - lift(lin)) / scale              (per-channel scale = RMS of the numerator)
    cond     [lift(lin), lift(prev), lift(next)]        (3C channels, fine resolution)
    tfeat    [annual, semiannual, diurnal harmonics, w] (7)
    coords   (lat/90, sin lon, cos lon) per pixel        (3)

Everything is in the Stage 1 normalised units (per-channel mean/std of the coarse stats).
``lift`` is the nearest lift = gather by the fine pixel's coarse parent, so block_mean(lift(c)) = c
exactly and the projection at hours 0/6/12 is well defined. Patch offsets are coarse-aligned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import _runs, time_features, valid_rows  # noqa: E402
from layout import FaceLayout, coord_channels  # noqa: E402
from patches import PatchGeometry  # noqa: E402


def _open(path: str | Path):
    import zarr
    root = zarr.open_group(str(Path(path).expanduser()), mode="r")
    return root["fine/uv"], root["coarse/uv"], np.asarray(root["time"][:], dtype=np.int64)


def block_starts(path: str | Path, n_frames: int) -> np.ndarray:
    """Row indices t0 such that rows t0..t0+n_frames-1 are consecutive valid hours."""
    _, _, hours = _open(path)
    v = valid_rows(path)
    idx = np.where(v)[0]
    starts = []
    for a, b in _runs(hours[idx]):
        rows = idx[a:b]
        if len(rows) >= n_frames:
            starts.append(rows[: len(rows) - n_frames + 1])
    return np.concatenate(starts) if starts else np.zeros(0, dtype=np.int64)


def split_starts(starts: np.ndarray, hours: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole contiguous runs (by run id of the start hour) so validation blocks never overlap training ones."""
    run_id = np.cumsum(np.r_[0, np.diff(hours[starts]) != 1])
    ids = np.unique(run_id); rng = np.random.default_rng(seed)
    val = set(rng.choice(ids, size=max(1, int(round(fraction * len(ids)))), replace=False).tolist())
    m = np.array([r in val for r in run_id])
    return starts[~m], starts[m]


class HpxFineRuns(Dataset):
    def __init__(self, store: str | Path, layout_dir: str | Path, *, stats: dict, starts: np.ndarray,
                 n_frames: int = 13, coarse_stride: int = 6, patch: int = 64, pad: int = 32,
                 patches_per_item: int = 16, scale: np.ndarray | None = None, length: int = 10_000,
                 seed: int = 0) -> None:
        self.store = str(Path(store).expanduser())
        self.fine, self.coarse, self.hours = _open(self.store)
        self.C, npix = self.fine.shape[1:]
        self.nside = int(round((npix / 12) ** 0.5)); self.nside_c = int(round((self.coarse.shape[-1] / 12) ** 0.5))
        self.geom = PatchGeometry(self.nside, self.nside_c, patch=patch, pad=pad, cache_dir=layout_dir)
        layout = FaceLayout.load(self.nside, layout_dir)
        self.coords = layout.from_faces(coord_channels(self.nside, layout.perm)).astype(np.float32)   # (3, npix) NEST
        self.mean = np.asarray(stats["mean"], np.float32)[:, None, None]; self.std = np.asarray(stats["std"], np.float32)[:, None, None]
        self.scale = np.ones((self.C, 1, 1), np.float32) if scale is None else np.asarray(scale, np.float32).reshape(self.C, 1, 1)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.tau, self.stride, self.k, self.length, self.seed = int(n_frames), int(coarse_stride), int(patches_per_item), int(length), int(seed)
        assert (self.tau - 1) % self.stride == 0, "block must end on a coarse frame"
        self.n_int = (self.tau - 1) // self.stride

    def __len__(self) -> int:
        return self.length

    def _frame_weights(self):
        t = np.arange(self.tau); q = np.minimum(t // self.stride, self.n_int - 1); w = (t - q * self.stride) / self.stride
        return q, w.astype(np.float32)

    def read_run(self, t0: int):
        fine = np.asarray(self.fine[t0:t0 + self.tau])                                   # (τ, C, npix) float16
        cidx = t0 + self.stride * np.arange(self.n_int + 1)
        coarse = np.stack([np.asarray(self.coarse[int(t)]) for t in cidx]).astype(np.float32)  # (n_int+1, C, npix_c)
        return fine, coarse

    def cut(self, fine: np.ndarray, coarse: np.ndarray, face: int, i: int, j: int, *, scaled: bool = True):
        fi = self.geom.fine_index(face, i, j); ci = fi // (self.geom.ratio ** 2)
        x = (fine[:, :, fi].astype(np.float32) - self.mean) / self.std                    # (τ, C, P, P)
        cc = (coarse[:, :, ci] - self.mean) / self.std                                    # (n_int+1, C, P, P) lifted
        q, w = self._frame_weights()
        prev, nxt = cc[q], cc[q + 1]                                                      # (τ, C, P, P)
        lin = (1.0 - w)[:, None, None, None] * prev + w[:, None, None, None] * nxt
        r = x - lin
        if scaled:
            r = r / self.scale
        cond = np.concatenate([lin, prev, nxt], axis=1)                                   # (τ, 3C, P, P)
        return r, cond, self.coords[:, fi], w

    def __getitem__(self, idx: int):
        rng = np.random.default_rng((self.seed, idx))
        t0 = int(self.starts[rng.integers(len(self.starts))])
        fine, coarse = self.read_run(t0)
        rs, conds, coords = [], [], []
        for face, i, j in self.geom.random_patches(rng, self.k):
            r, cond, co, w = self.cut(fine, coarse, face, i, j)
            rs.append(r); conds.append(cond); coords.append(co)
        tf = np.concatenate([time_features(self.hours[t0] + np.arange(self.tau)), w[:, None]], axis=1)  # (τ, 7)
        tfeat = np.broadcast_to(tf, (self.k, self.tau, 7)).copy()
        return (torch.from_numpy(np.stack(rs)).to(torch.float16), torch.from_numpy(np.stack(conds)).to(torch.float16),
                torch.from_numpy(np.stack(coords)), torch.from_numpy(tfeat))


def collate_runs(items):
    """Items already hold k patches each; concatenate them into one batch."""
    return tuple(torch.cat([it[n] for it in items], dim=0) for n in range(4))


def estimate_scale(ds: HpxFineRuns, *, n_items: int = 24, seed: int = 0) -> np.ndarray:
    """Per-channel RMS of the unscaled residual (x - lift(lin)) in normalised units."""
    rng = np.random.default_rng(seed); acc = np.zeros(ds.C); n = 0
    for _ in range(n_items):
        t0 = int(ds.starts[rng.integers(len(ds.starts))]); fine, coarse = ds.read_run(t0)
        for face, i, j in ds.geom.random_patches(rng, 4):
            r, _, _, _ = ds.cut(fine, coarse, face, i, j, scaled=False)
            acc += (r.astype(np.float64) ** 2).mean(axis=(0, 2, 3)); n += 1
    return np.sqrt(acc / n).astype(np.float32)
