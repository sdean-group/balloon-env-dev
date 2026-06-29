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


def drift(ds, spatial_score_fn) -> dict:
    """Per-timestep spatial composite over time -> mean, std, slope (drift rate)."""
    nt = ds.sizes["time"]
    comps = np.array([spatial_score_fn(ds.isel(time=[t]))["COMPOSITE"] for t in range(nt)])
    slope = float(np.polyfit(np.arange(nt), comps, 1)[0]) if nt > 1 else 0.0
    return {"drift mean": float(comps.mean()), "drift std": float(comps.std()),
            "drift slope/step": slope}


def temporal_scores(ds) -> dict:
    persist = temporal_persistence(ds)
    return {
        "temporal persistence": persist,
        "tendency (m/s/h)": tendency_mag(ds),
        "score: temporal": float(np.clip(persist, 0, 1)) if np.isfinite(persist) else np.nan,
    }
