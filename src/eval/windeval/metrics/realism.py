"""Axis-1 (field-quality) metrics as OBJECTIVE quality scores.

Reframe: we do NOT compute distance-to-ERA5. Each metric scores a field against a
*physically-motivated ideal*, needing no reference. ERA5 is then just a strong peer
row on the same scale — not "the answer". This is the correct framing for conditional
generation (we want plausible samples, not a match to one ERA5 field), and it lets a
generator legitimately score at/above reanalysis-level physical self-consistency.

Minimal elegant set — four orthogonal properties of "wind-likeness":
  1. KE spectrum slope        — energy-vs-scale  (necessary, fool-able by phase-shuffle)
  2. Helmholtz rot. fraction  — rotational/divergent balance  (THE discriminator)
  3. Velocity-increment kurtosis — intermittency / non-Gaussianity (independent catcher)
  4. (relative diagnostic) speed Wasserstein vs an ERA5 peer-distribution

#2 and #3 catch phase-shuffle by two *different* mechanisms (cross-structure vs
Gaussianization), so the benchmark never leans on a single discriminator.

Amplitude (scale, NOT shape)
----------------------------
The four metrics above are deliberately **scale-invariant** — they probe spatial/statistical
*structure*, so a generator can reproduce the structure while being systematically too calm
(or too energetic) and they won't notice. Diffusion samplers in particular tend to
*under-disperse* (a deterministic ODE recovers less than the full marginal variance), which
shows up as low wind amplitude with the right shape. ``amplitude rms`` (= RMS wind speed)
captures this. It is inherently *scale-relative* — there is no reference-free "ideal" wind
speed — so unlike the others it is scored only against a supplied peer RMS (``ref_rms``) and
is kept OUT of the reference-free COMPOSITE.
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m

# headline objective metrics (raw value, ideal direction)
RAW_NAMES = [
    "spectrum slope",        # ideal: steeply negative (~ -3 synoptic)
    "Helmholtz rot. frac",   # ideal: high (rotational-dominated); 0.5 = equipartition
    "vort/div ratio",        # ideal: > 1 (rotational-dominated)
    "increment kurtosis",    # ideal: > 3 (intermittent; 3 = Gaussian)
]
SCORE_NAMES = ["score: spectrum", "score: Helmholtz", "score: intermittency", "COMPOSITE"]


# ---------- field statistics ----------

def _radial_psd(field2d, dx, dy, nbins=16):
    f = field2d - field2d.mean()
    F = np.fft.fft2(f)
    power = np.abs(F) ** 2
    ky = np.fft.fftfreq(f.shape[0], d=dy)
    kx = np.fft.fftfreq(f.shape[1], d=dx)
    KX, KY = np.meshgrid(kx, ky)
    kr = np.sqrt(KX ** 2 + KY ** 2).ravel()
    p = power.ravel()
    kmax = kr[kr > 0].max()
    edges = np.linspace(0, kmax, nbins + 1)
    idx = np.digitize(kr, edges) - 1
    kc = 0.5 * (edges[:-1] + edges[1:])
    pw = np.array([p[idx == b].mean() if np.any(idx == b) else np.nan
                   for b in range(nbins)])
    return kc, pw


def _spectrum_slope(field2d, dx, dy):
    kc, pw = _radial_psd(field2d, dx, dy)
    m = np.isfinite(pw) & (kc > 0) & (pw > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log10(kc[m]), np.log10(pw[m]), 1)[0])


def _tukey1d(n, alpha):
    """Tukey window: cosine-tapered edges, flat center (fraction alpha tapered)."""
    if alpha <= 0:
        return np.ones(n)
    w = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return w
    t = np.arange(edge + 1)
    taper = 0.5 * (1 + np.cos(np.pi * (2.0 * t / (alpha * (n - 1)) - 1)))
    w[: edge + 1] = taper
    w[-(edge + 1):] = taper[::-1]
    return w


def _tukey2d(shape, alpha=0.5):
    return np.outer(_tukey1d(shape[0], alpha), _tukey1d(shape[1], alpha))


def _helmholtz_rot_frac(u2d, v2d, dx, dy, taper=0.5):
    """Fourier Helmholtz-Hodge split -> rotational energy / total energy in [0,1].

    Real (balanced) stratospheric flow is rotational-dominated (high). Independently
    phase-randomized or white-noise u,v have no k-alignment -> ~0.5 equipartition.

    The FFT assumes periodicity; on a bounded box the wrap-around discontinuity injects
    spurious divergence and biases the fraction low. We mitigate with mean-removal + a
    Tukey edge taper so the field decays smoothly to zero before transforming. This is a
    diagnostic-grade fix (validated on analytic vortex/source fields), not a full
    bounded-domain Poisson solve.
    """
    nu, nv = u2d - u2d.mean(), v2d - v2d.mean()
    if taper:
        w = _tukey2d(u2d.shape, taper)
        nu, nv = nu * w, nv * w
    U, V = np.fft.fft2(nu), np.fft.fft2(nv)
    ky = 2 * np.pi * np.fft.fftfreq(u2d.shape[0], d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(u2d.shape[1], d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX ** 2 + KY ** 2
    k2[0, 0] = np.inf  # kill DC (the removed mean)
    coef = (KX * U + KY * V) / k2          # divergent (curl-free) projection coeff
    Ud, Vd = coef * KX, coef * KY
    Ur, Vr = U - Ud, V - Vd                # rotational (divergence-free) remainder
    e_div = np.sum(np.abs(Ud) ** 2 + np.abs(Vd) ** 2)
    e_rot = np.sum(np.abs(Ur) ** 2 + np.abs(Vr) ** 2)
    return float(e_rot / (e_rot + e_div + 1e-30))


def _vort_div_ratio(u2d, v2d, dx, dy):
    du_dy, du_dx = np.gradient(u2d, dy, dx)
    dv_dy, dv_dx = np.gradient(v2d, dy, dx)
    vort = dv_dx - du_dy
    div = du_dx + dv_dy
    return float(vort.std() / (div.std() + 1e-30))


def _increment_kurtosis(field2d, lag=2):
    inc = np.concatenate([
        (field2d[:, lag:] - field2d[:, :-lag]).ravel(),
        (field2d[lag:, :] - field2d[:-lag, :]).ravel(),
    ])
    inc = inc - inc.mean()
    m2 = np.mean(inc ** 2)
    return float(np.mean(inc ** 4) / (m2 ** 2 + 1e-30))


def _wasserstein1d(a, b, n=512):
    qs = np.linspace(0, 1, n)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def amplitude_rms(ds) -> float:
    """RMS wind speed sqrt(mean(u**2 + v**2)) over the whole field, in m/s.

    A reference-free *fact* about a field's energy level. Scoring it requires a peer
    (see ``field_scores(..., ref_rms=...)``); on its own it is just the number to compare.
    """
    u, v = ds["u"].values, ds["v"].values
    return float(np.sqrt(np.mean(u ** 2 + v ** 2)))


def _amplitude_score(rms: float, ref_rms: float) -> float:
    """Symmetric ratio score in (0,1]: 1 at parity, penalising too-calm AND too-energetic.

    ``min(r, 1/r)`` with ``r = rms/ref_rms`` — a field at half the reference energy and one
    at double both score 0.5. Linear-in-ratio is interpretable (0.71 == "29% too calm").
    """
    if not (np.isfinite(rms) and np.isfinite(ref_rms)) or rms <= 0 or ref_rms <= 0:
        return float("nan")
    r = rms / ref_rms
    return float(min(r, 1.0 / r))


# ---------- objective scoring ----------

def field_scores(ds, *, ref_rms: float | None = None) -> dict:
    """Axis-1 scores. ``ref_rms`` (m/s): a peer RMS wind speed enabling the *scale* metric
    (``amplitude rms`` + ``score: amplitude``). The shape metrics + COMPOSITE are unaffected
    by it — COMPOSITE stays reference-free; amplitude is reported alongside as a diagnostic.
    """
    dx, dy = grid_spacing_m(ds)
    u, v = ds["u"].values, ds["v"].values  # (time, level, y, x)
    nt, nl = u.shape[0], u.shape[1]

    slope, rot, ratio, kurt = [], [], [], []
    for t in range(nt):
        for k in range(nl):
            slope.append(_spectrum_slope(u[t, k], dx, dy))
            slope.append(_spectrum_slope(v[t, k], dx, dy))
            rot.append(_helmholtz_rot_frac(u[t, k], v[t, k], dx, dy))
            ratio.append(_vort_div_ratio(u[t, k], v[t, k], dx, dy))
            kurt.append(_increment_kurtosis(u[t, k]))
            kurt.append(_increment_kurtosis(v[t, k]))

    raw = {
        "spectrum slope": float(np.nanmean(slope)),
        "Helmholtz rot. frac": float(np.nanmean(rot)),
        "vort/div ratio": float(np.nanmean(ratio)),
        "increment kurtosis": float(np.nanmean(kurt)),
    }
    # physically-anchored normalizations -> [0,1], higher = more wind-like
    s_spec = np.clip(-raw["spectrum slope"] / 3.0, 0, 1)        # flat=0, k^-3=1
    s_helm = np.clip((raw["Helmholtz rot. frac"] - 0.5) / 0.5, 0, 1)  # 0.5=0, 1.0=1
    s_int = np.clip((raw["increment kurtosis"] - 3.0) / 3.0, 0, 1)    # gaussian=0
    scores = {
        "score: spectrum": float(s_spec),
        "score: Helmholtz": float(s_helm),
        "score: intermittency": float(s_int),
        # COMPOSITE = spectrum (necessary gate) + intermittency (robust discriminator).
        # Helmholtz/vort-div are EXCLUDED: empirically weak here (rotational character is
        # largely spectrum-encoded, so phase-shuffle preserves it) AND the FFT-Helmholtz
        # assumes periodicity on a bounded box (spurious boundary divergence biases it low).
        # Kept as caveated diagnostics; promote once a non-periodic solver lands.
        "COMPOSITE": float(np.mean([s_spec, s_int])),
    }
    # amplitude (scale) — reported always; scored only against a supplied peer. Deliberately
    # NOT in COMPOSITE: it needs a reference, and the structure metrics stay reference-free.
    rms = float(np.sqrt(np.mean(u ** 2 + v ** 2)))
    raw["amplitude rms"] = rms
    if ref_rms is not None:
        scores["score: amplitude"] = _amplitude_score(rms, ref_rms)
    return {**raw, **scores, "speed": np.sqrt(u ** 2 + v ** 2).ravel()}
