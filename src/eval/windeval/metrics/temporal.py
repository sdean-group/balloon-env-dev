"""Temporal / dynamics metrics — is the field evolving like real weather?

The reference-free discriminator is TEMPORAL PERSISTENCE: real wind evolves slowly
(consecutive hours strongly correlated, small tendency); the per-timestep anchors
have independent frames, so persistence collapses and tendency explodes.

DRIFT (quality vs lead time) is the metric that matters for *generators* over long
rollouts; on real data it should be ~flat. We compute the machinery here and report
the slope — on real ERA5 it validates "no drift"; it bites when a generator rolls out.

(Future dynamics test for generators: structure-advection matching — do features move
at the flow speed — which separates 'truly dynamical' from 'temporally smooth but
undynamical'. Not needed to separate the incoherent anchors.)
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m


def temporal_persistence(ds) -> float:
    """Mean correlation between consecutive timesteps (u and v)."""
    u, v = ds["u"].values, ds["v"].values   # (t, L, y, x)
    nt, nl = u.shape[0], u.shape[1]
    if nt < 2:
        return np.nan
    cs = []
    for t in range(nt - 1):
        for k in range(nl):
            for a in (u, v):
                cs.append(np.corrcoef(a[t, k].ravel(), a[t + 1, k].ravel())[0, 1])
    return float(np.nanmean(cs))


def tendency_mag(ds) -> float:
    """Mean |dV/dt| in m/s per step (hour) between consecutive frames."""
    u, v = ds["u"].values, ds["v"].values
    if u.shape[0] < 2:
        return np.nan
    du, dv = u[1:] - u[:-1], v[1:] - v[:-1]
    return float(np.sqrt(du ** 2 + dv ** 2).mean())


def _dt_seconds(ds) -> float:
    """Seconds per timestep from the (datetime64) time axis; NaN if not real time."""
    t = ds["time"].values
    if not np.issubdtype(np.asarray(t).dtype, np.datetime64) or len(t) < 2:
        return np.nan
    return float((t[1] - t[0]) / np.timedelta64(1, "s"))


def _advect_corr(sp0: np.ndarray, sp1: np.ndarray, sx: float, sy: float) -> float:
    """Correlation between sp0 advected by (sx, sy) px and sp1, on the wrap-safe interior."""
    isx, isy = int(round(sx)), int(round(sy))
    rolled = np.roll(np.roll(sp0, isy, axis=0), isx, axis=1)
    m = max(abs(isy), abs(isx)) + 1
    if 2 * m >= min(sp0.shape):
        a, b = rolled.ravel(), sp1.ravel()
    else:
        a, b = rolled[m:-m, m:-m].ravel(), sp1[m:-m, m:-m].ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def structure_advection(ds) -> float:
    """Do features MOVE WITH THE FLOW? — the truly-dynamical discriminator.

    Advective predictability: shift frame ``t`` by the per-level mean-wind displacement over
    one timestep and correlate with frame ``t+1``. If the flow carries the structure (real
    weather, or our advective baseline) this is high; for time-incoherent frames it collapses;
    for a smooth-but-static field that fades in place, advecting by a nonzero wind *misaligns*
    the features, so it also drops — separating "truly dynamical" from "temporally smooth but
    undynamical". Wind-magnitude weighted (the hypothesis is only testable where there's flow).
    Needs a real (datetime64) time axis to convert wind → pixel displacement; else NaN.
    """
    dt = _dt_seconds(ds)
    if not np.isfinite(dt):
        return np.nan
    dx, dy = grid_spacing_m(ds)
    u, v = ds["u"].values, ds["v"].values            # (t, L, y, x)
    nt, nl = u.shape[0], u.shape[1]
    if nt < 2:
        return np.nan
    num = den = 0.0
    for t in range(nt - 1):
        for k in range(nl):
            wu = float(u[t, k].mean()); wv = float(v[t, k].mean())
            sx = wu * dt / dx; sy = wv * dt / dy      # downwind pixel displacement
            if not (np.isfinite(sx) and np.isfinite(sy)):  # NaN winds (e.g. a noise anchor) -> skip
                continue
            c = _advect_corr(np.hypot(u[t, k], v[t, k]), np.hypot(u[t + 1, k], v[t + 1, k]), sx, sy)
            if not np.isfinite(c):
                continue
            w = np.hypot(wu, wv)                       # weight by flow strength
            num += w * np.clip(c, 0.0, 1.0); den += w
    return float(num / den) if den > 0 else np.nan


def drift(ds, spatial_score_fn) -> dict:
    """Per-timestep spatial composite over time -> mean, std, slope (drift rate)."""
    nt = ds.sizes["time"]
    comps = np.array([spatial_score_fn(ds.isel(time=[t]))["COMPOSITE"] for t in range(nt)])
    slope = float(np.polyfit(np.arange(nt), comps, 1)[0]) if nt > 1 else 0.0
    return {"drift mean": float(comps.mean()), "drift std": float(comps.std()),
            "drift slope/step": slope}


def temporal_scores(ds, *, ref_persistence: float | None = None,
                     ref_tendency: float | None = None) -> dict:
    """Temporal metrics. ``score: temporal`` (legacy) just rewards persistence — it catches
    *incoherent* frames (shuffle/noise) but would also reward a *frozen* field, so it is not a
    realism score on its own. When ERA5 peer stats are supplied (``ref_persistence``,
    ``ref_tendency``) we add **peer-matched** realism scores: real evolution lives in a band
    (too-frozen and too-chaotic are both wrong), so we score *closeness* to ERA5. ``tendency``
    is the signal that catches a too-frozen generator (low |dV/dt|); persistence catches
    incoherent ones. ``structure advection`` is a diagnostic (tracer-like vs wave-like), not a
    realism score — real wind patterns are quasi-stationary, not mean-wind-advected.
    """
    persist = temporal_persistence(ds)
    tend = tendency_mag(ds)
    out = {
        "temporal persistence": persist,
        "tendency (m/s/h)": tend,
        "structure advection (diag)": structure_advection(ds),
        "score: temporal": float(np.clip(persist, 0, 1)) if np.isfinite(persist) else np.nan,
    }
    if ref_persistence is not None and ref_tendency is not None:
        s_persist = 1.0 - float(np.clip(abs(persist - ref_persistence), 0, 1)) \
            if np.isfinite(persist) else np.nan
        if np.isfinite(tend) and tend > 0 and ref_tendency > 0:
            r = tend / ref_tendency
            s_tend = float(min(r, 1.0 / r))
        else:
            s_tend = np.nan
        out["score: persistence match"] = s_persist
        out["score: tendency match"] = s_tend
        out["score: temporal realism"] = float(np.nanmean([s_persist, s_tend]))
    return out
