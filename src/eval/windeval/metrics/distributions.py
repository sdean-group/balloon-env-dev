"""Distributional metrics — Data Distribution §1–2 of the benchmark spec.

  - Marginal W1 per pressure level: Wasserstein-1 between the pooled u (and v) values of
    the generated field and ERA5, per level, averaged across levels → W1_u / W1_v in m/s.
  - Extreme-quantile error: W1 is dominated by the bulk, so tails are reported separately
    as |q_pred − q_ref| at 0.1/1/99/99.9%, averaged over levels and both components.
  - Conditional W1: fix a condition, draw N seeds from the generator, pool their values
    per level and compare to the ERA5 realizations matching that condition; the scalar is
    the average over conditions. For unconditional generators there is exactly one
    condition (= the training climate) and this degenerates to an N-seed pooled marginal
    W1. For the conditional model a condition is (location, month, hour-of-day) — Shaurya
    2026-07-14: the reference pool for a condition is that hour on each of the held-out
    days 8–14 (the harmonics cannot resolve individual days, so day-of-week variability
    IS the within-condition distribution), and the model pool is seeds sampled at those
    same timestamps. `conditional_w1_grouped` averages over such condition groups.

All W1s are computed on quantile grids (the exact 1D W1 is the integral of the quantile
difference; 1024 points is plenty for these sample sizes) and carry the units of the
variable (m/s here).
"""
from __future__ import annotations

import numpy as np

N_QUANTILES = 1024
TAIL_PAIRS = {"1%": (0.01, 0.99), "0.1%": (0.001, 0.999)}


def wasserstein1(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two samples via the quantile-difference integral (units of the data)."""
    qs = np.linspace(0, 1, N_QUANTILES)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _per_level_values(ds, var: str) -> list[np.ndarray]:
    a = ds[var].values                       # (t, L, y, x)
    return [a[:, l].ravel() for l in range(a.shape[1])]


def marginal_w1(pred_ds, ref_ds) -> tuple[dict, dict]:
    """Per-level marginal W1 for u and v. Scalars are the mean over levels."""
    scalars, detail = {}, {}
    for var in ("u", "v"):
        w1 = [wasserstein1(p, r) for p, r in
              zip(_per_level_values(pred_ds, var), _per_level_values(ref_ds, var))]
        detail[f"W1 {var} per level"] = np.array(w1)
        scalars[f"W1 {var} (m/s)"] = float(np.mean(w1))
    return scalars, detail


def extreme_quantile_error(pred_ds, ref_ds) -> dict:
    """Mean |q_pred − q_ref| at the tail quantiles, across levels and components (m/s)."""
    out = {}
    for name, (qlo, qhi) in TAIL_PAIRS.items():
        errs = []
        for var in ("u", "v"):
            for p, r in zip(_per_level_values(pred_ds, var), _per_level_values(ref_ds, var)):
                for q in (qlo, qhi):
                    errs.append(abs(np.quantile(p, q) - np.quantile(r, q)))
        out[f"tail err {name} (m/s)"] = float(np.mean(errs))
    return out


def conditional_w1(seed_datasets, ref_ds) -> dict:
    """Pooled-over-seeds per-level W1 vs the reference for ONE condition (see module doc).

    `seed_datasets`: iterable of field Datasets generated with different seeds under the
    same condition. Averaging over conditions happens in the caller once there are many.
    """
    seed_list = list(seed_datasets)
    if not seed_list:
        return {"W1 cond (m/s)": np.nan}
    w1 = []
    for var in ("u", "v"):
        ref_lv = _per_level_values(ref_ds, var)
        nl = len(ref_lv)
        for l in range(nl):
            pooled = np.concatenate([d[var].values[:, l].ravel() for d in seed_list])
            w1.append(wasserstein1(pooled, ref_lv[l]))
    return {"W1 cond (m/s)": float(np.mean(w1))}


def conditional_w1_grouped(groups) -> dict:
    """Mean of `conditional_w1` over condition groups (see module doc for the protocol).

    `groups`: iterable of (seed_datasets, ref_ds_at_condition) pairs, one per condition.
    """
    vals = [conditional_w1(seeds, ref)["W1 cond (m/s)"] for seeds, ref in groups]
    return {"W1 cond (m/s)": float(np.mean(vals)) if vals else np.nan}
