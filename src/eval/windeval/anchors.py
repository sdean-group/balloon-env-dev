"""Calibration anchors derived as pure functions of the real-ERA5 ingest.

- phase_shuffle: per-level 2D FFT phase randomization. Preserves the amplitude
  spectrum (PSD) EXACTLY while destroying spatial structure -> the 'realism trap'
  that catches a metric measuring only the spectrum.
- white_noise: Gaussian matched to per-level mean/variance -> lower bound.

These exist so we can validate the metric suite before trusting it on anything
new: real ERA5 should beat phase-shuffle should beat noise, with PSD specifically
*failing* to separate real from phase-shuffle. If the ordering is wrong, the
metric is broken, not the generator.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import artifact


def _phase_randomize(field2d: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomize phases while keeping the amplitude spectrum -> real output."""
    F = np.fft.fft2(field2d)
    amp = np.abs(F)
    # Phases from the FFT of real random noise are guaranteed Hermitian-symmetric,
    # so amp*exp(i*phase) inverts to a real field with an identical power spectrum.
    phases = np.angle(np.fft.fft2(rng.standard_normal(field2d.shape)))
    out = np.fft.ifft2(amp * np.exp(1j * phases)).real
    return out.astype("float32")


def phase_shuffle(real_path, out_path, *, seed: int = 0) -> Path:
    ds = artifact.read(real_path)
    rng = np.random.default_rng(seed)
    u, v = ds["u"].values.copy(), ds["v"].values.copy()  # (time, level, y, x)
    for t in range(u.shape[0]):
        for k in range(u.shape[1]):
            u[t, k] = _phase_randomize(u[t, k], rng)
            v[t, k] = _phase_randomize(v[t, k], rng)
    return _write_like(ds, u, v, out_path, name="anchor_phase_shuffle", seed=seed)


def white_noise(real_path, out_path, *, seed: int = 0) -> Path:
    ds = artifact.read(real_path)
    rng = np.random.default_rng(seed)
    u, v = ds["u"].values.copy(), ds["v"].values.copy()
    for arr in (u, v):
        for t in range(arr.shape[0]):
            for k in range(arr.shape[1]):
                mu, sd = arr[t, k].mean(), arr[t, k].std()
                arr[t, k] = rng.normal(mu, sd, arr[t, k].shape)
    return _write_like(ds, u, v, out_path, name="anchor_white_noise", seed=seed)


def _write_like(ds, u, v, out_path, *, name, seed) -> Path:
    new = ds.copy()
    new["u"] = (ds["u"].dims, u.astype("float32"))
    new["v"] = (ds["v"].dims, v.astype("float32"))
    attrs = dict(ds.attrs)
    attrs["generator"] = {"name": name, "config": {"derived_from": "era5_real", "seed": seed}}
    attrs["seed"] = seed
    return artifact.write(new, attrs, out_path)
