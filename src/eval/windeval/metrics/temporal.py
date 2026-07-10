"""Temporal metrics — Physical Consistency §Temporal of the benchmark spec.

  1. Temporal power spectrum: fix grid points, take the 1D FFT of their u/v time series,
     and compare the (log-averaged) power spectrum against ERA5's with the same
     spectral-residual formula used in space → SR_time (0 = same persistence structure).
  2. Trajectory dispersion: advect passive tracers in 2D through the evolving winds, per
     level, and compare mean-square displacement vs lead time and the position spread at
     the final common lead time against ERA5 → "disp log-MSD RMSE" and
     "final spread ratio" (1 = parity).

Both need a real (datetime64) time axis with at least MIN_FRAMES contiguous frames;
static artifacts get NaN (= N/A).

SEGMENTS: a reference like the held-out ERA5 set is several disjoint seasonal blocks —
FFTing or advecting across a block boundary would be nonsense. Every metric here first
splits the time axis into contiguous segments (dt jumps end a segment), computes per
segment, and averages: log-PSDs are averaged on a common frequency grid, MSD curves are
averaged over the common lead-time range. A single contiguous episode is just the
one-segment case.

Frequencies are in cycles/hour; dispersion works in meters on the tangent plane with
edge-clamped bilinear velocity sampling and Heun (RK2) steps — the identical protocol on
both sides is what makes the comparison fair.
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m

MIN_FRAMES = 16
PSD_MAX_POINTS = 8          # sample an ≤8×8 lattice of grid points for the temporal PSD
DISP_N_SIDE = 8             # 8×8 tracer lattice per level


def contiguous_segments(ds) -> list[slice]:
    """Time-index slices of contiguous runs (a >1.5×median-dt jump ends a segment)."""
    t = np.asarray(ds["time"].values)
    if not np.issubdtype(t.dtype, np.datetime64) or len(t) < 2:
        return []
    dt = np.diff(t) / np.timedelta64(1, "h")
    med = np.median(dt)
    breaks = np.where(dt > 1.5 * med)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(t)]])
    return [slice(a, b) for a, b in zip(starts, ends) if b - a >= MIN_FRAMES]


def dt_hours(ds) -> float:
    """Hours per timestep within a contiguous run; NaN if the axis isn't real time."""
    t = np.asarray(ds["time"].values)
    if not np.issubdtype(t.dtype, np.datetime64) or len(t) < 2:
        return float("nan")
    return float(np.median(np.diff(t) / np.timedelta64(1, "h")))


def has_time(ds) -> bool:
    return len(contiguous_segments(ds)) > 0


# ---------- temporal power spectrum ----------

def _segment_psd(u, v, dt):
    """Log-averaged temporal periodogram over sampled points/levels/components."""
    nt = u.shape[0]
    sy = max(1, u.shape[2] // PSD_MAX_POINTS)
    sx = max(1, u.shape[3] // PSD_MAX_POINTS)
    w = np.hanning(nt)
    w = w / np.sqrt(np.mean(w ** 2))
    f = np.fft.rfftfreq(nt, d=dt)[1:]              # cycles/hour, drop DC
    logs = []
    for a in (u, v):
        series = a[:, :, ::sy, ::sx].reshape(nt, -1)
        series = (series - series.mean(axis=0)) * w[:, None]
        P = np.abs(np.fft.rfft(series, axis=0)[1:]) ** 2 * dt / nt
        logs.append(np.log(P + 1e-30))
    return f, np.concatenate(logs, axis=1).mean(axis=1)


def temporal_psd(ds) -> dict:
    """{"f": cycles/hour, "P": density}, segment-wise periodograms averaged in log space
    on the longest segment's frequency grid."""
    segs = contiguous_segments(ds)
    if not segs:
        return {"f": np.array([]), "P": np.array([])}
    dt = dt_hours(ds)
    u, v = ds["u"].values, ds["v"].values          # (t, L, y, x)
    per_seg = [_segment_psd(u[s], v[s], dt) for s in segs]
    f0 = max((f for f, _ in per_seg), key=len)
    logs = [np.interp(np.log(f0), np.log(f), lp) for f, lp in per_seg]
    return {"f": f0, "P": np.exp(np.mean(logs, axis=0))}


def temporal_spectral_residual(pred_ds, ref_ds) -> tuple[dict, dict]:
    """SR_time: RMSE of log temporal PSD over the common frequency range."""
    pp, rr = temporal_psd(pred_ds), temporal_psd(ref_ds)
    if not len(pp["f"]) or not len(rr["f"]):
        return {"SR_time": np.nan}, {"pred": pp, "ref": rr}
    lp = np.interp(np.log(rr["f"]), np.log(pp["f"]), np.log(pp["P"] + 1e-30),
                   left=np.nan, right=np.nan)
    lr = np.log(rr["P"] + 1e-30)
    m = np.isfinite(lp)
    sr = float(np.sqrt(np.mean((lp[m] - lr[m]) ** 2))) if m.any() else np.nan
    return {"SR_time": sr}, {"pred": pp, "ref": rr}


# ---------- trajectory dispersion ----------

def _bilinear(field: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Edge-clamped bilinear sample of field (y, x) at fractional pixel coords."""
    ny, nx = field.shape
    xp = np.clip(xp, 0, nx - 1.001)
    yp = np.clip(yp, 0, ny - 1.001)
    x0, y0 = xp.astype(int), yp.astype(int)
    fx, fy = xp - x0, yp - y0
    return ((1 - fy) * ((1 - fx) * field[y0, x0] + fx * field[y0, x0 + 1])
            + fy * ((1 - fx) * field[y0 + 1, x0] + fx * field[y0 + 1, x0 + 1]))


def _segment_dispersion(u, v, dt_s, dx, dy, n_side):
    """One contiguous run: MSD(t) (m², per level) + final positions (m)."""
    nt, nl, ny, nx = u.shape
    gx = np.linspace(0.1 * (nx - 1), 0.9 * (nx - 1), n_side)
    gy = np.linspace(0.1 * (ny - 1), 0.9 * (ny - 1), n_side)
    X0, Y0 = np.meshgrid(gx * dx, gy * dy)
    x = np.tile(X0.ravel(), (nl, 1))               # (L, P)
    y = np.tile(Y0.ravel(), (nl, 1))
    x0, y0 = x.copy(), y.copy()

    msd = np.zeros((nt, nl))
    for t in range(nt - 1):
        for l in range(nl):
            k1u = _bilinear(u[t, l], x[l] / dx, y[l] / dy)
            k1v = _bilinear(v[t, l], x[l] / dx, y[l] / dy)
            xm, ym = x[l] + k1u * dt_s, y[l] + k1v * dt_s
            k2u = _bilinear(u[t + 1, l], xm / dx, ym / dy)
            k2v = _bilinear(v[t + 1, l], xm / dx, ym / dy)
            x[l] += 0.5 * (k1u + k2u) * dt_s
            y[l] += 0.5 * (k1v + k2v) * dt_s
        msd[t + 1] = ((x - x0) ** 2 + (y - y0) ** 2).mean(axis=1)

    spread = np.sqrt((x - x.mean(axis=1, keepdims=True)) ** 2
                     + (y - y.mean(axis=1, keepdims=True)) ** 2).mean(axis=1)
    return msd, spread


def trajectory_dispersion(ds, n_side: int = DISP_N_SIDE) -> dict:
    """Segment-averaged MSD(lead time) per level + spread at the common final lead time."""
    segs = contiguous_segments(ds)
    if not segs:
        return {"hours": np.array([]), "msd": np.array([]), "final spread (m)": np.array([])}
    dt = dt_hours(ds)
    dx, dy = grid_spacing_m(ds)
    u, v = ds["u"].values, ds["v"].values
    nlead = min(s.stop - s.start for s in segs)    # common lead-time range
    msds, spreads = [], []
    for s in segs:
        m, sp = _segment_dispersion(u[s][:nlead], v[s][:nlead], dt * 3600.0, dx, dy, n_side)
        msds.append(m)
        spreads.append(sp)
    return {"hours": np.arange(nlead) * dt,
            "msd": np.mean(msds, axis=0),                    # (nlead, L)
            "final spread (m)": np.mean(spreads, axis=0)}    # (L,)


def dispersion_compare(pred_ds, ref_ds) -> tuple[dict, dict]:
    """Scalars: RMSE of log MSD over common lead times (levels averaged) and the
    pred/ref final-spread ratio averaged over levels (1 = parity)."""
    dp, dr = trajectory_dispersion(pred_ds), trajectory_dispersion(ref_ds)
    if not len(dp["hours"]) or not len(dr["hours"]):
        return {"disp log-MSD RMSE": np.nan, "final spread ratio": np.nan}, {"pred": dp, "ref": dr}
    lp = np.interp(dr["hours"][1:], dp["hours"][1:],
                   np.log(dp["msd"][1:].mean(axis=1) + 1e-30), left=np.nan, right=np.nan)
    lr = np.log(dr["msd"][1:].mean(axis=1) + 1e-30)
    m = np.isfinite(lp)
    rmse = float(np.sqrt(np.mean((lp[m] - lr[m]) ** 2))) if m.any() else np.nan
    ratio = float(np.mean(dp["final spread (m)"] / (dr["final spread (m)"] + 1e-30)))
    return ({"disp log-MSD RMSE": rmse, "final spread ratio": ratio},
            {"pred": dp, "ref": dr})
