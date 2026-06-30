"""Axis-2 (generation procedure) metrics — does the *method* hold up?

These test claims that only the lazy / unbounded class of generators makes; bounded
generators are N/A (an empty cell, not a 0). InfiniteDiffusion is the first generator that
exercises them. As with Axis-1, each metric is a reference-free *objective* score, and each
has its own calibration anchor (see `tests/test_windeval/test_procedure.py`):

  - seam discontinuity : a non-tiled field scores ≈ ideal; a deliberately tile-mismatched
    field spikes. Measured in DIVERGENCE — a seam in the horizontal wind injects a spurious
    ∂u/∂x (or ∂v/∂y) exactly at the stitch, so divergence is far more sensitive than u,v.
  - revisit determinism: re-querying the same point must return the same wind (seed-consistency).
  - budget / O(1)      : query cost is bounded and independent of location (near ≈ far).
  - extent drift       : field quality must not decay as the canvas grows (only `unbounded`).
    The unbounded analog of temporal drift; consumes a *family* of growing crops.

Metric selection is driven by `attrs.capabilities`: `procedure_scores` runs each sub-metric
only where its capability is declared, and returns NaN (= N/A) otherwise.
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m
from .realism import field_scores


# ---------- seam discontinuity (field/, capability: tiled) ----------

def _divergence(u2d, v2d, dx, dy):
    dudy, dudx = np.gradient(u2d, dy, dx)
    dvdy, dvdx = np.gradient(v2d, dy, dx)
    return dudx + dvdy


def seam_discontinuity(ds) -> dict:
    """Excess of |divergence| (and of the value step) at seam stitches vs the interior.

    Ideal = 1.0 (the stitch is statistically indistinguishable from the interior). A naive
    tiler spikes well above 1. Uses `attrs.seam_boundaries` = {'y': rows, 'x': cols} (local
    array indices). N/A (NaN) if no seams are declared.
    """
    seams = ds.attrs.get("seam_boundaries") or {}
    sx = [int(s) for s in seams.get("x", [])]
    sy = [int(s) for s in seams.get("y", [])]
    if not sx and not sy:
        return {"seam div excess": np.nan, "seam step ratio": np.nan, "score: seam": np.nan}

    dx, dy = grid_spacing_m(ds)
    u, v = ds["u"].values, ds["v"].values  # (t, L, y, x)
    nt, nl, ny, nx = u.shape

    # seam-adjacent cell mask (columns s-1,s for vertical seams; rows s-1,s for horizontal)
    col_mask = np.zeros(nx, bool)
    for s in sx:
        col_mask[max(0, s - 1):min(nx, s + 1)] = True
    row_mask = np.zeros(ny, bool)
    for s in sy:
        row_mask[max(0, s - 1):min(ny, s + 1)] = True
    cell_mask = row_mask[:, None] | col_mask[None, :]
    if cell_mask.all() or not cell_mask.any():
        return {"seam div excess": np.nan, "seam step ratio": np.nan, "score: seam": np.nan}

    seam_div, int_div, seam_step, int_step = [], [], [], []
    for t in range(nt):
        for k in range(nl):
            d = np.abs(_divergence(u[t, k], v[t, k], dx, dy))
            seam_div.append(d[cell_mask].mean())
            int_div.append(d[~cell_mask].mean())
            # value step across the stitch vs typical adjacent step (u and v)
            for f in (u[t, k], v[t, k]):
                step_x = np.abs(np.diff(f, axis=1))  # (y, x-1): step between col j,j+1
                step_y = np.abs(np.diff(f, axis=0))
                seam_cols = [s - 1 for s in sx if 0 < s <= nx - 1]
                seam_rows = [s - 1 for s in sy if 0 < s <= ny - 1]
                sv = np.concatenate([step_x[:, seam_cols].ravel() if seam_cols else np.array([]),
                                     step_y[seam_rows, :].ravel() if seam_rows else np.array([])])
                iv = np.concatenate([np.delete(step_x, seam_cols, axis=1).ravel(),
                                     np.delete(step_y, seam_rows, axis=0).ravel()])
                if sv.size:
                    seam_step.append(sv.mean()); int_step.append(iv.mean())

    excess = float(np.mean(seam_div) / (np.mean(int_div) + 1e-30))
    step_ratio = float(np.mean(seam_step) / (np.mean(int_step) + 1e-30)) if seam_step else np.nan
    return {
        "seam div excess": excess,
        "seam step ratio": step_ratio,
        # ideal excess = 1; excess >= 2 (seam twice as divergent as interior) scores 0.
        "score: seam": float(np.clip(2.0 - excess, 0.0, 1.0)),
    }


# ---------- revisit determinism (querylog/, capability: random_access) ----------

def revisit_determinism(qds) -> dict:
    """Repeated (x,y,level,t,seed) queries must return the same wind (within tolerance)."""
    tol = float(qds.attrs.get("revisit_tolerance", 1e-5))
    cols = [qds[k].values for k in ("x", "y", "level", "t", "seed")]
    u, v = qds["u"].values, qds["v"].values
    keys = list(zip(*[c.tolist() for c in cols]))
    seen, diffs = {}, []
    for i, k in enumerate(keys):
        if k in seen:
            j = seen[k]
            diffs.append(max(abs(u[i] - u[j]), abs(v[i] - v[j])))
        else:
            seen[k] = i
    if not diffs:
        return {"revisit count": 0, "revisit max|Δ|": np.nan, "score: revisit": np.nan}
    maxd = float(max(diffs))
    score = 1.0 if maxd <= tol else float(np.clip(tol / (maxd + 1e-30), 0.0, 1.0))
    return {"revisit count": len(diffs), "revisit max|Δ|": maxd,
            "revisit mean|Δ|": float(np.mean(diffs)), "score: revisit": score}


# ---------- budget / O(1) random access (querylog/, capability: random_access) ----------

def budget(qds) -> dict:
    """Query cost bounded and independent of location: far-from-origin ≈ near."""
    lat = np.asarray(qds["latency_s"].values, float)
    dist = np.hypot(np.asarray(qds["x"].values, float), np.asarray(qds["y"].values, float))
    if lat.size < 4:
        return {"latency p50 (ms)": np.nan, "budget far/near": np.nan, "score: budget": np.nan}
    med = np.median(dist)
    near, far = lat[dist <= med], lat[dist > med]
    if near.size and far.size:
        ratio = float(np.median(far) / (np.median(near) + 1e-30))
    else:
        ratio = np.nan
    # O(1) => ratio ~ 1; score penalises growth OR speedup symmetrically (factor of 2 -> 0).
    score = np.nan if not np.isfinite(ratio) else float(np.clip(2.0 - max(ratio, 1.0 / ratio), 0, 1))
    return {
        "latency p50 (ms)": float(np.median(lat) * 1e3),
        "latency p95 (ms)": float(np.percentile(lat, 95) * 1e3),
        "budget far/near": ratio,
        "score: budget": score,
    }


# ---------- extent drift (family of growing crops, capability: unbounded) ----------

def extent_drift(datasets, spatial_score_fn=field_scores) -> dict:
    """Field-quality COMPOSITE vs canvas size over a family of growing crops (same seed).

    No drift => flat (slope ~ 0 per octave of side length). `datasets` is an ordered iterable
    of WindArtifact `field/` datasets at increasing crop size.
    """
    ds_list = list(datasets)
    if len(ds_list) < 2:
        return {"extent drift slope/oct": np.nan, "extent COMPOSITE mean": np.nan,
                "score: extent": np.nan}
    sides = np.array([float(d.sizes["x"]) for d in ds_list])
    comps = np.array([spatial_score_fn(d)["COMPOSITE"] for d in ds_list])
    slope = float(np.polyfit(np.log2(sides), comps, 1)[0])  # ΔCOMPOSITE per doubling
    return {
        "extent COMPOSITE mean": float(comps.mean()),
        "extent COMPOSITE std": float(comps.std()),
        "extent drift slope/oct": slope,
        # composite is in [0,1]; |slope| of 0.2 per octave is a strong drift -> 0.
        "score: extent": float(np.clip(1.0 - abs(slope) / 0.2, 0.0, 1.0)),
    }


# ---------- orchestration ----------

def procedure_scores(ds, querylog=None, extent_family=None) -> dict:
    """Run the applicable Axis-2 metrics for `ds`, gated by `attrs.capabilities`.

    Returns a flat dict; sub-metrics that don't apply contribute NaN (= N/A, not failure).
    `querylog` is the `querylog/` Dataset; `extent_family` an ordered list of growing-crop
    field datasets (incl. or excl. `ds`).
    """
    caps = ds.attrs.get("capabilities", {})
    out: dict = {}

    if caps.get("tiled"):
        out.update(seam_discontinuity(ds))
    if caps.get("random_access") and querylog is not None:
        out.update(revisit_determinism(querylog))
        out.update(budget(querylog))
    if caps.get("extent") == "unbounded" and extent_family is not None:
        out.update(extent_drift(extent_family))

    score_keys = [k for k in out if k.startswith("score: ")]
    applicable = [out[k] for k in score_keys if np.isfinite(out[k])]
    out["PROC COMPOSITE"] = float(np.mean(applicable)) if applicable else np.nan
    return out
