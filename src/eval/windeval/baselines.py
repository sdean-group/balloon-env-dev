"""Structured stochastic baselines — "right statistics, zero weather".

Two rows that sit between the trivial anchors (white noise / phase shuffle) and the
trained models:

  simplex noise   Octaved (fBm) OpenSimplex noise over (x, y, level, time) — the noise
                  family the Balloon Learning Environment perturbs its wind forecasts
                  with (Bellemare et al. 2020, "Autonomous navigation of stratospheric
                  balloons using reinforcement learning" — 4D simplex wind noise),
                  promoted here to a standalone field generator.
  helmholtz gp    A stationary Gaussian process over the (u, v) vector field whose
                  kernel is built from a curl-free scalar potential plus a
                  divergence-free stream function (Helmholtz decomposition with SE
                  spectra — Berlinghieri et al. 2023, "Gaussian Processes at the
                  Helm(holtz)"), sampled spectrally on a padded grid (circulant
                  embedding) with separable SE correlation across level and time.

Fitting contract: every knob is matched to the SAME held-out half A that the other
anchor rows are derived from — per-level mean/std; adjacent-level and 4 h-lag
correlations; the divergence/vorticity energy split; the horizontal scale by a coarse
grid search minimizing SR_E — and the rows are then scored against the full reference
like every anchor. Read each row as "the best this noise family can do": a trained
model earns its place by beating structured noise at its best, not a strawman.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import artifact
from .artifact import grid_spacing_m
from .metrics.spectra import dataset_spectra, spectral_residual

DATA = Path(__file__).resolve().parent / "data"
FBM_OCTAVES = 4          # standard fBm stack: persistence 0.5, lacunarity 2
FBM_PERSISTENCE = 0.5
GP_PAD = 256             # circulant-embedding grid (>= domain + ~4·max length scale)
SR_FIT_FRAMES = 2        # frames per candidate in the horizontal-scale grid search
                         # (× 18 levels = 36 periodograms per candidate — plenty for a
                         # coarse argmin over octave-separated scales)


# ---------- shared fitting targets (all measured on held-out half A) ----------

def _target_stats(fit_ds) -> dict:
    """Everything the baselines are allowed to know about the reference."""
    u = fit_ds["u"].values.astype("float64")          # (t, L, y, x)
    v = fit_ds["v"].values.astype("float64")
    t, L = u.shape[:2]
    mean = {"u": u.mean(axis=(0, 2, 3)), "v": v.mean(axis=(0, 2, 3))}
    std = {"u": u.std(axis=(0, 2, 3)), "v": v.std(axis=(0, 2, 3))}

    ua = u - mean["u"][None, :, None, None]
    # adjacent-level correlation of u anomalies, averaged over level pairs
    flat = ua.transpose(1, 0, 2, 3).reshape(L, -1)
    rho_lev = float(np.mean([np.corrcoef(flat[l], flat[l + 1])[0, 1]
                             for l in range(L - 1)]))
    # 4 h-lag temporal autocorrelation (frames are 4-hourly within segments)
    hours = (fit_ds["time"].values - fit_ds["time"].values[0]) / np.timedelta64(1, "h")
    lag = np.diff(hours)
    pairs = [(i, i + 1) for i in range(t - 1) if lag[i] == 4.0]
    rho_4h = float(np.mean([np.corrcoef(ua[i].ravel(), ua[j].ravel())[0, 1]
                            for i, j in pairs]))

    # divergence vs vorticity energy split (physical derivatives)
    dx, dy = grid_spacing_m(fit_ds)
    du_dx = np.gradient(u, dx, axis=3)
    du_dy = np.gradient(u, dy, axis=2)
    dv_dx = np.gradient(v, dx, axis=3)
    dv_dy = np.gradient(v, dy, axis=2)
    div_frac = float(np.var(du_dx + dv_dy) / (np.var(du_dx + dv_dy)
                                              + np.var(dv_dx - du_dy)))

    return {
        "mean": mean, "std": std, "rho_lev": rho_lev, "rho_4h": rho_4h,
        "div_frac": div_frac, "hours": hours,
        "levels": fit_ds["level"].values, "lat": fit_ds["lat"].values,
        "lon": fit_ds["lon"].values, "times": fit_ds["time"].values,
        "T": t, "L": L, "H": fit_ds.sizes["y"], "W": fit_ds.sizes["x"],
    }


def _se_scale(dist: float, rho: float) -> float:
    """SE-kernel scale with correlation `rho` at distance `dist`: ρ=exp(-d²/2s²)."""
    rho = float(np.clip(rho, 1e-6, 1 - 1e-6))
    return dist / np.sqrt(-2.0 * np.log(rho))


def _moment_match(u_raw, v_raw, stats) -> tuple[np.ndarray, np.ndarray]:
    """Standardize per level, then apply half A's per-level std and mean."""
    out = []
    for raw, var in ((u_raw, "u"), (v_raw, "v")):
        a = raw - raw.mean(axis=(0, 2, 3), keepdims=True)
        sd = a.std(axis=(0, 2, 3), keepdims=True)
        a = a / np.where(sd > 0, sd, 1.0)
        out.append(a * stats["std"][var][None, :, None, None]
                   + stats["mean"][var][None, :, None, None])
    return out[0], out[1]


def _make_ds(u, v, stats):
    return artifact.make_field(u.astype("float32"), v.astype("float32"),
                               level=stats["levels"], lat=stats["lat"],
                               lon=stats["lon"], time=stats["times"])


def _fit_horizontal_scale(sample_fn, candidates, stats, ref_spec) -> tuple[float, dict]:
    """Coarse grid search: the candidate whose SR_E vs half A's spectrum is smallest.

    `sample_fn(scale)` returns a small moment-matched (u, v) sample (SR_FIT_FRAMES
    frames, all levels). SR_E is the exact benchmark objective, so the fitted row is
    "this family at its best" by construction.
    """
    tried = {}

    def sr(s):
        u, v = sample_fn(s)
        ds = _make_ds(u, v, {**stats, "times": stats["times"][:u.shape[0]]})
        tried[s] = spectral_residual(dataset_spectra(ds), ref_spec)["SR_E"]

    for s in candidates:
        sr(s)
    # one geometric refinement round around the coarse argmin
    best = min(tried, key=tried.get)
    for s in (round(best * 0.7, 1), round(best * 1.4, 1)):
        if s not in tried:
            sr(s)
    best = min(tried, key=tried.get)
    return best, tried


# ---------- simplex noise (the BLE noise family) ----------

def _octave_task(args) -> np.ndarray:
    """One (seed, octave) opensimplex call — top-level so process pools can run it.

    opensimplex's vectorized API is pure-Python-slow (~17 µs/point even with numba
    installed), but octaves × components are independent, so the callers fan them out
    over a process pool.
    """
    xs, ys, zs, ws, seed, octave = args
    import opensimplex
    opensimplex.seed(seed * 1000 + octave)
    f = 2.0 ** octave
    return opensimplex.noise4array(xs * f, ys * f, zs * f, ws * f).astype("float32")


def _fbm4(xs, ys, zs, ws, seed: int, pool=None) -> np.ndarray:
    """Octaved 4D OpenSimplex noise → array (len(ws), len(zs), len(ys), len(xs))."""
    tasks = [(xs, ys, zs, ws, seed, o) for o in range(FBM_OCTAVES)]
    layers = pool.map(_octave_task, tasks) if pool else map(_octave_task, tasks)
    total, amp, norm = 0.0, 1.0, 0.0
    for layer in layers:
        total = total + amp * layer
        norm += amp
        amp *= FBM_PERSISTENCE
    return total / norm


def _noise_autocorr_lag(rho_target: float, seed: int = 99) -> float:
    """Input-space lag at which unit-scale fBm autocorrelation equals rho_target.

    Measured empirically once (fBm autocorrelation is not SE), on a 1D transect of a
    modest 4D sample, then interpolated.
    """
    n = _fbm4(np.linspace(0, 40, 2048), np.zeros(4), np.zeros(2), np.zeros(2), seed)
    x = n[0, 0, 0] - n[0, 0, 0].mean()
    dx_in = 40 / 2047
    lags = np.arange(1, 512)
    rho = np.array([np.corrcoef(x[:-k], x[k:])[0, 1] for k in lags])
    below = rho <= rho_target
    if not below.any():
        return float(lags[-1] * dx_in)
    i = int(np.argmax(below))
    if i == 0:
        return float(lags[0] * dx_in)
    # linear interpolation across the first crossing
    r0, r1 = rho[i - 1], rho[i]
    frac = (r0 - rho_target) / (r0 - r1)
    return float((lags[i - 1] + frac) * dx_in)


def simplex_baseline(fit_ds, *, seed: int = 0) -> tuple:
    """(Dataset, fitted-knob dict). Knobs: horizontal feature scale (pixels, by SR_E
    grid search), level/time input scales (matched to half A's correlations)."""
    stats = _target_stats(fit_ds)
    ref_spec = dataset_spectra(fit_ds)

    # vertical & temporal input increments that reproduce half A's correlations
    d_lev = _noise_autocorr_lag(stats["rho_lev"])        # per level step
    d_4h = _noise_autocorr_lag(stats["rho_4h"])          # per 4 h step
    zs = np.arange(stats["L"]) * d_lev
    ws = stats["hours"] / 4.0 * d_4h

    from concurrent.futures import ProcessPoolExecutor

    def sample(scale_px, pool, frames=SR_FIT_FRAMES, sd=seed):
        xs = np.arange(stats["W"]) / scale_px
        ys = np.arange(stats["H"]) / scale_px
        # u and v octaves all fan out together through the shared pool
        tasks = ([(xs, ys, zs, ws[:frames], sd * 2 + 1, o) for o in range(FBM_OCTAVES)]
                 + [(xs, ys, zs, ws[:frames], sd * 2 + 2, o) for o in range(FBM_OCTAVES)])
        layers = list(pool.map(_octave_task, tasks))
        amps = [FBM_PERSISTENCE ** o for o in range(FBM_OCTAVES)]
        u = sum(a * l for a, l in zip(amps, layers[:FBM_OCTAVES])) / sum(amps)
        v = sum(a * l for a, l in zip(amps, layers[FBM_OCTAVES:])) / sum(amps)
        return _moment_match(u, v, stats)

    with ProcessPoolExecutor(max_workers=2 * FBM_OCTAVES) as pool:
        best, tried = _fit_horizontal_scale(lambda s: sample(s, pool),
                                            (4, 8, 16, 32, 64), stats, ref_spec)
        u, v = sample(best, pool, frames=stats["T"])
    knobs = {"scale_px": best, "sr_e_by_scale": tried, "d_lev": d_lev, "d_4h": d_4h,
             "octaves": FBM_OCTAVES, "persistence": FBM_PERSISTENCE}
    return _make_ds(u, v, stats), knobs


# ---------- Helmholtz-kernel GP ----------

def _helmholtz_unit_samples(n: int, hw: tuple[int, int], ell: float,
                            div_frac: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """n iid samples of a unit-variance Helmholtz GP (u, v), spectrally sampled.

    Curl-free part = ∇φ, div-free part = ∇×ψ; φ, ψ have SE spectra with shared scale
    `ell` (pixels). Components are empirically unit-normalized and combined so a
    `div_frac` share of small-scale energy is divergent. Sampled on a GP_PAD² periodic
    grid and cropped, so the domain sees no wraparound correlation.
    """
    H, W = hw
    k1 = 2 * np.pi * np.fft.fftfreq(GP_PAD)
    KX, KY = np.meshgrid(k1, k1)
    amp = np.exp(-(ell ** 2) * (KX ** 2 + KY ** 2) / 4.0)   # sqrt of SE spectrum

    us, vs = [], []
    for lo in range(0, n, 128):
        m = min(128, n - lo)
        w_phi = np.fft.fft2(rng.standard_normal((m, GP_PAD, GP_PAD)))
        w_psi = np.fft.fft2(rng.standard_normal((m, GP_PAD, GP_PAD)))
        phi, psi = w_phi * amp, w_psi * amp
        u_cf = np.fft.ifft2(1j * KX * phi).real[:, :H, :W]   # ∇φ (curl-free)
        v_cf = np.fft.ifft2(1j * KY * phi).real[:, :H, :W]
        u_df = -np.fft.ifft2(1j * KY * psi).real[:, :H, :W]  # ∇×ψ (div-free)
        v_df = np.fft.ifft2(1j * KX * psi).real[:, :H, :W]
        cf = np.sqrt(div_frac) / (np.std([u_cf, v_cf]) + 1e-12)
        df = np.sqrt(1 - div_frac) / (np.std([u_df, v_df]) + 1e-12)
        us.append(cf * u_cf + df * u_df)
        vs.append(cf * v_cf + df * v_df)
    return np.concatenate(us), np.concatenate(vs)


def helmholtz_baseline(fit_ds, *, seed: int = 0) -> tuple:
    """(Dataset, fitted-knob dict). Separable kernel k_helmholtz(x,y) ⊗ SE(level) ⊗
    SE(time): iid spatial samples are mixed across the (time, level) axis by the
    Cholesky factor of K_t ⊗ K_L (exact for a separable GP)."""
    stats = _target_stats(fit_ds)
    ref_spec = dataset_spectra(fit_ds)
    T, L, H, W = stats["T"], stats["L"], stats["H"], stats["W"]

    s_lev = _se_scale(1.0, stats["rho_lev"])
    s_time = _se_scale(4.0, stats["rho_4h"])
    li = np.arange(L)
    K_L = np.exp(-0.5 * ((li[:, None] - li[None, :]) / s_lev) ** 2)
    hh = stats["hours"]
    K_t = np.exp(-0.5 * ((hh[:, None] - hh[None, :]) / s_time) ** 2)
    C = np.linalg.cholesky(np.kron(K_t, K_L) + 1e-9 * np.eye(T * L))

    def sample(ell, frames=SR_FIT_FRAMES, sd=seed):
        rng = np.random.default_rng(sd)
        n = frames * L
        u0, v0 = _helmholtz_unit_samples(n, (H, W), ell, stats["div_frac"], rng)
        Cs = C[:n, :n] if frames < T else C
        u = (Cs @ u0.reshape(n, -1)).reshape(frames, L, H, W)
        v = (Cs @ v0.reshape(n, -1)).reshape(frames, L, H, W)
        return _moment_match(u, v, stats)

    best, tried = _fit_horizontal_scale(lambda s: sample(s), (2, 4, 8, 16, 32),
                                        stats, ref_spec)
    u, v = sample(best, frames=T)
    knobs = {"ell_px": best, "sr_e_by_scale": tried, "s_lev": s_lev,
             "s_time_h": s_time, "div_frac": stats["div_frac"]}
    return _make_ds(u, v, stats), knobs


# ---------- benchmark entry point ----------

def baseline_rows(fit_ds, *, regen: bool = False) -> dict:
    """{'simplex noise': ds, 'helmholtz gp': ds}, cached like the other artifacts."""
    out = {}
    for name, path, fn in (("simplex noise", DATA / "baseline_simplex.zarr",
                            simplex_baseline),
                           ("helmholtz gp", DATA / "baseline_helmholtz_gp.zarr",
                            helmholtz_baseline)):
        if regen or not path.exists():
            print(f"[bench] fitting + generating {name} baseline …", flush=True)
            ds, knobs = fn(fit_ds)
            print(f"[bench]   {name} knobs: { {k: v for k, v in knobs.items() if k != 'sr_e_by_scale'} }",
                  flush=True)
            artifact.write(ds, {
                "generator": {"name": name.replace(" ", "_"), "version": "v1"},
                "capabilities": {"extent": "bounded", "temporally_evolving": True},
                "conditioning": {"fit": {k: (v if np.isscalar(v) else str(v))
                                         for k, v in knobs.items()}},
            }, path)
        out[name] = artifact.read(path)
    return out
