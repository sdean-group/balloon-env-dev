"""Vertical-structure metrics — the axis the balloon actually controls.

Station-keeping works by changing altitude to catch winds blowing different ways, so
the realism of *vertical structure* is the most control-relevant property.

The discriminating, reference-free signal is VERTICAL COHERENCE: real wind has smooth
profiles (adjacent levels strongly correlated) with structured shear; the per-level
anchors (phase-shuffle, white-noise) randomize each level independently, so adjacent
levels decorrelate and the shear becomes incoherent noise.

(Note for later: a *generator* could fail the opposite way — too coherent / over-smooth,
killing the shear the balloon needs. Coherence discriminates the anchors; an
over-smoothing check is added when we evaluate generators.)
"""
from __future__ import annotations

import numpy as np

from .. import derive


def vertical_coherence(ds) -> float:
    """Mean correlation between adjacent model levels (u and v)."""
    u, v = ds["u"].values, ds["v"].values   # (t, L, y, x)
    nt, nl = u.shape[0], u.shape[1]
    cs = []
    for t in range(nt):
        for k in range(nl - 1):
            for a in (u, v):
                cs.append(np.corrcoef(a[t, k].ravel(), a[t, k + 1].ravel())[0, 1])
    return float(np.nanmean(cs))


def vertical_scores(ds) -> dict:
    coh = vertical_coherence(ds)
    out = {"vertical coherence": coh, "score: vertical": float(np.clip(coh, 0, 1))}
    if "sp" in ds:  # physical shear needs surface pressure (Stage 2)
        shear, _ = derive.vector_shear(ds)
        out["shear mean (m/s/km)"] = float(shear.mean() * 1000)
    return out
