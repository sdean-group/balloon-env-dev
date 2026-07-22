"""Reference-free field diagnostics for comparing generator configurations.

These are descriptive diagnostics, not substitutes for the benchmark-v2 distances to
held-out ERA5.  They revive the useful raw quantities from the first wind-eval harness
without the old clipped 0--1 composite score.
"""
from __future__ import annotations

import numpy as np

from ..artifact import grid_spacing_m


def _radial_slope(field: np.ndarray, dx: float, dy: float, n_bins: int = 16) -> float:
    centered = field - field.mean()
    power = np.abs(np.fft.fft2(centered)) ** 2
    fy = np.fft.fftfreq(field.shape[0], d=dy)
    fx = np.fft.fftfreq(field.shape[1], d=dx)
    kx, ky = np.meshgrid(fx, fy)
    radius = np.sqrt(kx**2 + ky**2).ravel()
    values = power.ravel()
    edges = np.linspace(0.0, radius.max(), n_bins + 1)
    index = np.digitize(radius, edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    shell_power = np.array(
        [values[index == i].mean() if np.any(index == i) else np.nan for i in range(n_bins)]
    )
    valid = np.isfinite(shell_power) & (centers > 0) & (shell_power > 0)
    if valid.sum() < 3:
        return float("nan")
    return float(np.polyfit(np.log10(centers[valid]), np.log10(shell_power[valid]), 1)[0])


def _tukey(n: int, alpha: float = 0.5) -> np.ndarray:
    window = np.ones(n)
    edge = int(np.floor(alpha * (n - 1) / 2.0))
    if edge < 1:
        return window
    t = np.arange(edge + 1)
    taper = 0.5 * (1 + np.cos(np.pi * (2 * t / (alpha * (n - 1)) - 1)))
    window[: edge + 1] = taper
    window[-(edge + 1) :] = taper[::-1]
    return window


def _rotational_fraction(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    taper = np.outer(_tukey(u.shape[0]), _tukey(u.shape[1]))
    U = np.fft.fft2((u - u.mean()) * taper)
    V = np.fft.fft2((v - v.mean()) * taper)
    ky = 2 * np.pi * np.fft.fftfreq(u.shape[0], d=dy)
    kx = 2 * np.pi * np.fft.fftfreq(u.shape[1], d=dx)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    k2[0, 0] = np.inf
    projection = (KX * U + KY * V) / k2
    Ud, Vd = projection * KX, projection * KY
    Ur, Vr = U - Ud, V - Vd
    divergent = np.sum(np.abs(Ud) ** 2 + np.abs(Vd) ** 2)
    rotational = np.sum(np.abs(Ur) ** 2 + np.abs(Vr) ** 2)
    return float(rotational / (rotational + divergent + 1e-30))


def _vorticity_divergence_ratio(
    u: np.ndarray, v: np.ndarray, dx: float, dy: float
) -> float:
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    return float(vorticity.std() / (divergence.std() + 1e-30))


def _increment_kurtosis(field: np.ndarray, lag: int = 2) -> float:
    increments = np.concatenate(
        [
            (field[:, lag:] - field[:, :-lag]).ravel(),
            (field[lag:, :] - field[:-lag, :]).ravel(),
        ]
    )
    increments -= increments.mean()
    variance = np.mean(increments**2)
    return float(np.mean(increments**4) / (variance**2 + 1e-30))


def field_diagnostics(ds) -> dict[str, float]:
    """Return unnormalized physical descriptors for one WindArtifact dataset."""
    dx, dy = grid_spacing_m(ds)
    u = np.asarray(ds["u"].values)
    v = np.asarray(ds["v"].values)
    slopes: list[float] = []
    rotational: list[float] = []
    ratios: list[float] = []
    kurtosis: list[float] = []
    for t in range(u.shape[0]):
        for level in range(u.shape[1]):
            slopes.extend(
                [_radial_slope(u[t, level], dx, dy), _radial_slope(v[t, level], dx, dy)]
            )
            rotational.append(_rotational_fraction(u[t, level], v[t, level], dx, dy))
            ratios.append(_vorticity_divergence_ratio(u[t, level], v[t, level], dx, dy))
            kurtosis.extend(
                [_increment_kurtosis(u[t, level]), _increment_kurtosis(v[t, level])]
            )
    speed = np.sqrt(u**2 + v**2)
    return {
        "spectrum slope": float(np.nanmean(slopes)),
        "Helmholtz rotational fraction": float(np.nanmean(rotational)),
        "vorticity/divergence ratio": float(np.nanmean(ratios)),
        "increment kurtosis": float(np.nanmean(kurtosis)),
        "wind RMS (m/s)": float(np.sqrt(np.mean(u**2 + v**2))),
        "speed mean (m/s)": float(speed.mean()),
        "speed p95 (m/s)": float(np.quantile(speed, 0.95)),
    }
