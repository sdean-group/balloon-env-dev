"""Spatial spectral metrics — Physical Consistency §Spatial 1 of the benchmark spec.

Everything here compares a generated field against an ERA5 reference in Fourier space:

  - isotropic power spectra of kinetic energy E_h, horizontal divergence δ_h and
    relative vorticity ζ_h (computed per (time, level) slice, averaged in log space)
  - Spectral Residual  SR = sqrt(mean_k (log E_pred(k) − log E_ref(k))²)  for each of
    E / δ / ζ  → SR_E, SR_div, SR_vort  (dimensionless, in natural-log units; 0 = perfect)
  - Effective Resolution L_eff: the wavelength where R(k) = E_pred/E_ref drops below 0.5
    for 5 consecutive wavenumber shells → "the model is trustworthy down to L_eff".

Grids may differ between pred and ref (e.g. a 192² generated crop vs the 121² ERA5 box):
spectra are computed on each grid's native shells in *physical* wavenumber (cycles/m from
the lat/lon spacing) with a density-normalized periodogram, then the pred spectrum is
log-interpolated onto the ref shells over the overlapping k-range before comparing.

A Kaiser window (β=8) is applied before every FFT (identically to both sides) to suppress
leakage from the non-periodic crop boundary; power is renormalized by the window's mean
square so absolute levels stay comparable.
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m

KAISER_BETA = 8.0
LEFF_RATIO = 0.5          # R(k) threshold for effective resolution
LEFF_CONSECUTIVE = 5      # shells the ratio must stay below the threshold


def _kaiser2d(shape: tuple[int, int]) -> np.ndarray:
    w = np.outer(np.kaiser(shape[0], KAISER_BETA), np.kaiser(shape[1], KAISER_BETA))
    return w / np.sqrt(np.mean(w ** 2))   # unit mean-square -> power preserved


def _shell_average(power: np.ndarray, fx: np.ndarray, fy: np.ndarray):
    """Isotropic shell average of a 2D power array.

    Shell width = the grid's fundamental frequency (of the longer side), so every shell
    is populated. Returns (k_centers, shell_mean_power), k in cycles/m.
    """
    FX, FY = np.meshgrid(fx, fy)
    kr = np.sqrt(FX ** 2 + FY ** 2).ravel()
    p = power.ravel()
    df = min(abs(fx[1] - fx[0]), abs(fy[1] - fy[0]))
    nyq = min(np.abs(fx).max(), np.abs(fy).max())   # isotropic range only
    nshell = int(nyq / df)
    idx = np.floor(kr / df - 0.5).astype(int)       # shell j covers [(j+0.5)df, (j+1.5)df)
    k = (np.arange(nshell) + 1) * df
    out = np.full(nshell, np.nan)
    for j in range(nshell):
        m = idx == j
        if m.any():
            out[j] = p[m].mean()
    return k, out


def field_spectra(u2d: np.ndarray, v2d: np.ndarray, dx: float, dy: float) -> dict:
    """Isotropic PSDs of E_h, δ_h, ζ_h for one (level, time) slice.

    Returns {"k": cycles/m, "E": ..., "div": ..., "vort": ...}; PSDs are periodogram
    *densities* (× dx·dy / N), so values are comparable across grid sizes.
    """
    w = _kaiser2d(u2d.shape)
    U = np.fft.fft2((u2d - u2d.mean()) * w)
    V = np.fft.fft2((v2d - v2d.mean()) * w)
    ny, nx = u2d.shape
    norm = (dx * dy) / (nx * ny)

    fy = np.fft.fftfreq(ny, d=dy)                  # cycles/m
    fx = np.fft.fftfreq(nx, d=dx)
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy)   # angular, for derivatives

    E = 0.5 * (np.abs(U) ** 2 + np.abs(V) ** 2) * norm
    Dhat = 1j * (KX * U + KY * V)                  # divergence spectrum
    Zhat = 1j * (KX * V - KY * U)                  # vorticity spectrum
    Pd = np.abs(Dhat) ** 2 * norm
    Pz = np.abs(Zhat) ** 2 * norm

    k, Ek = _shell_average(E, fx, fy)
    _, Dk = _shell_average(Pd, fx, fy)
    _, Zk = _shell_average(Pz, fx, fy)
    return {"k": k, "E": Ek, "div": Dk, "vort": Zk}


def dataset_spectra(ds) -> dict:
    """Log-space (geometric-mean) average of `field_spectra` over all (time, level) slices."""
    dx, dy = grid_spacing_m(ds)
    u, v = ds["u"].values, ds["v"].values          # (t, L, y, x)
    logs: dict[str, list] = {"E": [], "div": [], "vort": []}
    k = None
    for t in range(u.shape[0]):
        for l in range(u.shape[1]):
            s = field_spectra(u[t, l], v[t, l], dx, dy)
            k = s["k"]
            for name in logs:
                logs[name].append(np.log(s[name] + 1e-30))
    out = {"k": k}
    for name, ls in logs.items():
        out[name] = np.exp(np.nanmean(np.stack(ls), axis=0))
    return out


def _interp_log_onto(k_ref, k_pred, p_pred):
    """Pred log-PSD interpolated onto the ref shells; NaN outside pred's k-range."""
    lp = np.interp(np.log(k_ref), np.log(k_pred), np.log(p_pred + 1e-30),
                   left=np.nan, right=np.nan)
    valid = np.isfinite(lp) & (k_ref >= k_pred.min()) & (k_ref <= k_pred.max())
    lp[~valid] = np.nan
    return lp


def spectral_residual(pred_spec: dict, ref_spec: dict) -> dict:
    """SR_E / SR_div / SR_vort between two `dataset_spectra` results (0 = identical)."""
    out = {}
    for name, key in [("E", "SR_E"), ("div", "SR_div"), ("vort", "SR_vort")]:
        lp = _interp_log_onto(ref_spec["k"], pred_spec["k"], pred_spec[name])
        lr = np.log(ref_spec[name] + 1e-30)
        m = np.isfinite(lp) & np.isfinite(lr)
        out[key] = float(np.sqrt(np.mean((lp[m] - lr[m]) ** 2))) if m.any() else np.nan
    return out


def effective_resolution(pred_spec: dict, ref_spec: dict) -> dict:
    """L_eff (km): wavelength where E_pred/E_ref < 0.5 for 5 consecutive shells.

    If the ratio never fails, the model resolves the whole comparable range and L_eff is
    the finest compared wavelength (flagged with resolved_to_grid=True).
    """
    k = ref_spec["k"]
    lp = _interp_log_onto(k, pred_spec["k"], pred_spec["E"])
    lr = np.log(ref_spec["E"] + 1e-30)
    m = np.isfinite(lp) & np.isfinite(lr)
    k, ratio = k[m], np.exp(lp[m] - lr[m])
    below = ratio < LEFF_RATIO
    for i in range(len(below) - LEFF_CONSECUTIVE + 1):
        if below[i:i + LEFF_CONSECUTIVE].all():
            return {"L_eff (km)": float(1.0 / k[i] / 1e3), "resolved_to_grid": False}
    finest = float(1.0 / k[-1] / 1e3) if len(k) else np.nan
    return {"L_eff (km)": finest, "resolved_to_grid": True}


def spatial_spectral_suite(pred_ds, ref_ds) -> tuple[dict, dict]:
    """All spectral scalars for one pred/ref pair; also returns the spectra for figures."""
    ps, rs = dataset_spectra(pred_ds), dataset_spectra(ref_ds)
    scalars = {**spectral_residual(ps, rs), **effective_resolution(ps, rs)}
    return scalars, {"pred": ps, "ref": rs}
