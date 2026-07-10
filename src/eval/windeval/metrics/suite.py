"""Suite orchestration — run every applicable metric for one (pred, ref) pair.

The suite is a flat dict of RAW values (no 0–1 normalization, no composite): spectral
residuals and L_eff from `spectra`, shear/marginal/tail/conditional W1s from `shear` and
`distributions`, and — when both artifacts have a real time axis — SR_time and the
trajectory-dispersion scalars from `temporal`. Metrics that don't apply are NaN (= N/A in
reports, not a failure).

`tiling_penalty` implements the spec's procedural check: run the suite on a single-tile
generation and on a multi-tile generation of the same generator, subtract → how much each
metric degrades because of tiling (0 = seamless; sign follows the metric's raw units).

`METRIC_INFO` is the single place that says how to read each value (units + direction);
reports are built from it so table labels can't drift from the code.
"""
from __future__ import annotations

import numpy as np

from .spectra import spatial_spectral_suite
from .shear import shear_w1
from .distributions import marginal_w1, extreme_quantile_error, conditional_w1
from .temporal import has_time, temporal_spectral_residual, dispersion_compare

# name -> (better, unit/target note)
METRIC_INFO = {
    "SR_E":                    ("lower", "log-PSD RMSE of kinetic energy vs ref; 0 = identical"),
    "SR_div":                  ("lower", "log-PSD RMSE of divergence vs ref"),
    "SR_vort":                 ("lower", "log-PSD RMSE of vorticity vs ref"),
    "L_eff (km)":              ("lower", "finest trusted wavelength (E_pred/E_ref<0.5 rule)"),
    "W1 shear u ((m/s)/km)":   ("lower", "Wasserstein-1 of ∂u/∂z distribution vs ref"),
    "W1 shear v ((m/s)/km)":   ("lower", "Wasserstein-1 of ∂v/∂z distribution vs ref"),
    "W1 u (m/s)":              ("lower", "per-level marginal W1 of u, level-averaged"),
    "W1 v (m/s)":              ("lower", "per-level marginal W1 of v, level-averaged"),
    "tail err 1% (m/s)":       ("lower", "|q_pred−q_ref| at 1/99%, levels+components avg"),
    "tail err 0.1% (m/s)":     ("lower", "|q_pred−q_ref| at 0.1/99.9%"),
    "W1 cond (m/s)":           ("lower", "N-seed pooled per-level W1 (one condition for now)"),
    "SR_time":                 ("lower", "log temporal-PSD RMSE vs ref"),
    "disp log-MSD RMSE":       ("lower", "tracer mean-square-displacement curve vs ref"),
    "final spread ratio":      ("≈1",    "tracer final position spread, pred/ref"),
}


def run_suite(pred_ds, ref_ds, *, dz=None, seed_datasets=None,
              ref_temporal=None) -> tuple[dict, dict]:
    """All raw metrics for pred vs ref. Returns (scalars, detail-for-figures).

    `ref_temporal`: optionally a different (e.g. denser-in-time) reference for the
    temporal metrics; spatial/distribution metrics always use `ref_ds`.
    """
    detail: dict = {}

    scalars, detail["spectra"] = spatial_spectral_suite(pred_ds, ref_ds)

    # shear needs the Δz climatology to match BOTH level stacks (guards 10-level artifacts)
    if (dz is not None and pred_ds.sizes["level"] == len(dz) + 1
            and ref_ds.sizes["level"] == len(dz) + 1):
        scalars.update(shear_w1(pred_ds, ref_ds, dz))

    m, det = marginal_w1(pred_ds, ref_ds)
    scalars.update(m)
    detail["marginals"] = det
    scalars.update(extreme_quantile_error(pred_ds, ref_ds))

    if seed_datasets is not None:
        scalars.update(conditional_w1(seed_datasets, ref_ds))

    tref = ref_temporal if ref_temporal is not None else ref_ds
    if has_time(pred_ds) and has_time(tref):
        sr, detail["temporal_psd"] = temporal_spectral_residual(pred_ds, tref)
        scalars.update(sr)
        disp, detail["dispersion"] = dispersion_compare(pred_ds, tref)
        scalars.update(disp)

    # every metric appears in every row (NaN = N/A) so report tables are rectangular
    for k in METRIC_INFO:
        scalars.setdefault(k, np.nan)
    return scalars, detail


def tiling_penalty(single_scalars: dict, multi_scalars: dict) -> dict:
    """multi − single per shared metric: what tiling itself costs (0 = seamless)."""
    out = {}
    for k, info in METRIC_INFO.items():
        s, m = single_scalars.get(k, np.nan), multi_scalars.get(k, np.nan)
        if np.isfinite(s) and np.isfinite(m):
            out[k] = float(m - s)
    return out
