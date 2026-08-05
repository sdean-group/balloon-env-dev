"""Structure & vertical-realism metrics — the suite's answer to "fitted noise wins".

Every other metric in this suite is a POPULATION statistic: 1-point marginals (W1, tails,
shear) and the 2-point covariance (the PSD rows). A field matched on mean + covariance
reproduces all of them essentially by construction, which is exactly why the fitted
simplex/GP baselines sit at the floor almost everywhere. The metrics here are the ones
noise has to earn: they look at the JOINT structure across levels and at the geometry of
coherent features, neither of which is fixed by a matched spectrum.

1. **Opposing-wind (vertical decoupling)** — the fraction of vertical columns containing
   at least one pair of levels whose wind vectors differ in direction by more than
   `thr_deg`, counting only levels where both winds exceed `smin` (slow winds have a
   meaningless bearing). This is RL-HAB's "Forecast Score" idea (Schuler et al., NRL,
   arXiv 2502.05014): opposing winds at different altitudes are the physical basis of
   balloon station-keeping, and NRL rejected BLE-style simplex noise precisely because
   smooth vertical correlation washes the decoupling out. Note `shear.py` CANNOT see
   this: it pools ADJACENT-level differences into one histogram per component, so it
   never forms an angle between wind vectors and never spans a large altitude gap — a
   field can match the shear distribution exactly while never reversing direction.

2. **Wind speed / jet intensity** — per-level W1 of |V| and the error at its upper
   quantiles. u and v are moment-matched by the noise baselines; |V| and especially its
   upper tail (jet cores) are not automatically.

3. **Jet geometry** — thresholding each (time, level) slice at ITS OWN per-level 95th
   speed percentile (so exceedance coverage is 5% for every row by construction, which
   isolates shape from intensity), then comparing the connected components' mean area
   and mean elongation to the reference's. Real jets are long, oriented streaks;
   isotropic noise makes round blobs of the same total area.
   **Reliability, measured 2026-07-29 (read before using these two):** splitting the
   reference against ITSELF within one weather period gives elongation 0.97 (range
   0.93-1.02) but area 1.07 (range 0.80-1.42) — and ACROSS the two held-out periods the
   floors are elongation 1.17, area 1.42, because days 11-14 carry ~2× as many, smaller
   components as days 8-10 (1093 vs 2065). So `jet elong ratio` is a tight estimator
   against a ~±0.17 synoptic floor, while **`jet area ratio` is PROVISIONAL — its floor
   is genuine weather variability, not noise, and a median/p75 statistic does not fix
   it.** The fix is to make the geometry condition-matched the way W1_cond is (compare
   only like months/hours), which removes the synoptic confound; deferred.

Window: structural comparisons are only meaningful on a COMMON domain — connected
component area is bounded by the domain, so scoring a 64² generation against a 121²
reference biases the geometry (and, as measured on 2026-07-29, flips the decoupling
verdict). `common_window` crops both sides to the same centered `STRUCT_WINDOW` box.
Components touching the border are kept (truncation then hits every row identically).

Applicability: these require the reference's own vertical stack — a row with a different
level count (ble_vae: 10 pressure levels on a 0.45° grid) gets NaN, the suite's standard
"not applicable", not a failure.
"""
from __future__ import annotations

import numpy as np

from .distributions import wasserstein1

STRUCT_WINDOW = 64          # common centered box for every structural comparison
OPP_THR_DEG = 90.0          # "opposing" = bearings more than this far apart
OPP_SMIN = 5.0              # m/s; both levels must exceed it to have a meaningful bearing
MAX_COLUMNS = 150_000       # subsample cap for the column statistic
JET_Q = 0.95                # per-level speed percentile defining the exceedance set
JET_MIN_AREA = 8            # px; drop speckle before measuring shape
JET_MAX_SLICES = 96         # time subsample for the (slower) connected-component pass
SPEED_TAILS = (0.99, 0.999)


def _center(ds, n: int):
    """Centered n×n crop (no-op when the field is already n or smaller)."""
    y0 = max((ds.sizes["y"] - n) // 2, 0)
    x0 = max((ds.sizes["x"] - n) // 2, 0)
    return ds.isel(y=slice(y0, y0 + n), x=slice(x0, x0 + n))


def common_window(pred_ds, ref_ds, n: int = STRUCT_WINDOW):
    """Crop both sides to the same centered box — required for any geometry comparison."""
    n = min(n, pred_ds.sizes["y"], pred_ds.sizes["x"], ref_ds.sizes["y"], ref_ds.sizes["x"])
    return _center(pred_ds, n), _center(ref_ds, n)


def applicable(pred_ds, ref_ds) -> bool:
    """Structural metrics need the reference's own vertical stack (see module doc)."""
    return pred_ds.sizes["level"] == ref_ds.sizes["level"]


# ---------- 1. vertical decoupling ----------

def _columns(ds, max_columns: int = MAX_COLUMNS, seed: int = 0) -> np.ndarray:
    """(N, L, 2) sampled vertical wind columns."""
    u, v = ds["u"].values, ds["v"].values           # (t, L, y, x)
    L = u.shape[1]
    U = u.transpose(0, 2, 3, 1).reshape(-1, L)
    V = v.transpose(0, 2, 3, 1).reshape(-1, L)
    if U.shape[0] > max_columns:
        idx = np.random.default_rng(seed).choice(U.shape[0], max_columns, replace=False)
        U, V = U[idx], V[idx]
    return np.stack([U, V], axis=-1)


def opposing_wind_fraction(ds, thr_deg: float = OPP_THR_DEG, smin: float = OPP_SMIN,
                           seed: int = 0) -> float:
    """Fraction of columns holding an opposing level pair (see module doc §1)."""
    c = _columns(ds, seed=seed)
    speed = np.linalg.norm(c, axis=-1)                       # (N, L)
    bearing = np.arctan2(c[..., 1], c[..., 0])
    d = np.abs(bearing[:, :, None] - bearing[:, None, :])
    d = np.minimum(d, 2 * np.pi - d)                         # (N, L, L) angle between
    ok = (speed[:, :, None] >= smin) & (speed[:, None, :] >= smin)
    iu = np.triu_indices(c.shape[1], k=1)                    # unordered level pairs
    d, ok = d[:, iu[0], iu[1]], ok[:, iu[0], iu[1]]
    usable = ok.any(axis=1)                                  # column has ≥1 scorable pair
    if not usable.any():
        return float("nan")
    has = ((d > np.radians(thr_deg)) & ok).any(axis=1)
    return float(has[usable].mean())


# ---------- 2. speed / jet intensity ----------

def _speed_per_level(ds) -> list[np.ndarray]:
    s = np.hypot(ds["u"].values, ds["v"].values)             # (t, L, y, x)
    return [s[:, l].ravel() for l in range(s.shape[1])]


def speed_metrics(pred_ds, ref_ds) -> dict:
    """Per-level W1 of |V| plus upper-tail (jet intensity) errors, in m/s."""
    p, r = _speed_per_level(pred_ds), _speed_per_level(ref_ds)
    out = {"W1 speed (m/s)": float(np.mean([wasserstein1(a, b) for a, b in zip(p, r)]))}
    for q in SPEED_TAILS:
        errs = [abs(np.quantile(a, q) - np.quantile(b, q)) for a, b in zip(p, r)]
        out[f"jet speed err {q:.1%} (m/s)".replace(".0%", "%")] = float(np.mean(errs))
    return out


# ---------- 3. jet geometry ----------

def _geometry(ds, q: float = JET_Q, min_area: int = JET_MIN_AREA,
              max_slices: int = JET_MAX_SLICES) -> tuple[float, float]:
    """(mean component area px, mean elongation) of the own-q95 speed exceedance set."""
    from scipy import ndimage

    s = np.hypot(ds["u"].values, ds["v"].values)             # (t, L, y, x)
    nt = s.shape[0]
    if nt > max_slices:                                       # even time subsample
        s = s[np.linspace(0, nt - 1, max_slices).astype(int)]
    thr = np.quantile(s, q, axis=(0, 2, 3))                   # per level, own distribution
    areas, elongs = [], []
    for li in range(s.shape[1]):
        for ti in range(s.shape[0]):
            lab, n = ndimage.label(s[ti, li] > thr[li])
            if n == 0:
                continue
            for k, sl in enumerate(ndimage.find_objects(lab), start=1):
                if sl is None:
                    continue
                sub = lab[sl] == k
                if sub.sum() < min_area:
                    continue
                ys, xs = np.nonzero(sub)
                areas.append(float(ys.size))
                # elongation = sqrt(major/minor) of the pixel-coordinate covariance.
                # +1/12 = the variance of a unit-width pixel: without it a 1-px-wide
                # component has zero minor variance and elongation blows up (a 1×8 line
                # scored ~1250 before this correction).
                ev = np.linalg.eigvalsh(np.cov(np.stack([ys, xs]).astype(float)))
                lo, hi = float(ev[0]) + 1 / 12, float(ev[1]) + 1 / 12
                elongs.append(float(np.sqrt(hi / lo)))
    if not areas:
        return float("nan"), float("nan")
    return float(np.mean(areas)), float(np.mean(elongs))


def jet_geometry(pred_ds, ref_ds) -> dict:
    """Mean-area and mean-elongation RATIOS (pred/ref) of the exceedance components."""
    pa, pe = _geometry(pred_ds)
    ra, re = _geometry(ref_ds)
    return {
        "jet area ratio": float(pa / ra) if ra and np.isfinite(ra) else float("nan"),
        "jet elong ratio": float(pe / re) if re and np.isfinite(re) else float("nan"),
    }


# ---------- suite entry ----------

def structure_suite(pred_ds, ref_ds) -> dict:
    """All structural metrics for one (pred, ref) pair, on the common window."""
    nan = {"opp-wind frac": np.nan, "opp-wind err": np.nan,
           "W1 speed (m/s)": np.nan, "jet speed err 99% (m/s)": np.nan,
           "jet speed err 99.9% (m/s)": np.nan,
           "jet area ratio": np.nan, "jet elong ratio": np.nan}
    if not applicable(pred_ds, ref_ds):
        return nan
    p, r = common_window(pred_ds, ref_ds)
    fp = opposing_wind_fraction(p)
    fr = opposing_wind_fraction(r, seed=1)
    out = {"opp-wind frac": fp, "opp-wind err": abs(fp - fr)}
    out.update(speed_metrics(p, r))
    out.update(jet_geometry(p, r))
    return out
