"""AdvectedField — a kinematic time axis on top of the static InfiniteDiffusion field.

Phase-4 temporal baseline (Route C). A balloon drifts for days, so the *truth* wind field
must evolve. The cheapest evolution that keeps the machinery's crown jewels — O(1)
random-access, seamlessness, seed-determinism — is **advection** (Taylor's frozen-flow
hypothesis): synoptic features are carried by the mean flow, so

    velocity_l(x, y, t) = field_static_l(x - U_l * t,  y - V_l * t)

i.e. evolving the field in time is just a *coordinate shift* on the query. That is O(1) in
``t`` (no rollout, no FFT), it is exactly deterministic, and querying a seamless field at
shifted coordinates is still seamless — so this wraps the static sampler with **zero changes
to ``sampler.py``**, the same way the toy denoiser stood in for the trained model.

Each vertical level advects at *its own* mean wind ``(U_l, V_l)`` (from the model's
``NormStats`` climatology), giving realistic vertical shear in the motion for free.

Honest limitation — this is a deliberately NAIVE floor, not a realism claim. Two reasons it
is unphysical for *wind*: (1) it is frozen-flow, so features translate but never grow/decay/
develop; (2) more fundamentally, a wind *pattern* is not a passive tracer — the jet is a
quasi-stationary feature that air flows *through*, so real stratospheric patterns propagate
far slower than the mean wind (the benchmark confirms advecting ERA5 by its mean wind predicts
the next frame *worse* than persistence). So this is the "temporal toy": an O(1), seamless,
deterministic baseline that exercises the temporal interface + metrics and gives the learned
models (autoregressive / spacetime) a floor to beat. Real temporal realism comes from those.
"""
from __future__ import annotations

import numpy as np


def velocity_from_stats(stats) -> np.ndarray:
    """Per-level mean wind (U, V) in m/s from a ``NormStats`` — the advecting background flow."""
    return np.stack([np.asarray(stats.mean_u, float), np.asarray(stats.mean_v, float)], axis=1)


def _bilinear(grid: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Sample ``grid`` (Hb, Wb) at the outer product of real coords ``ys`` (H,), ``xs`` (W,)."""
    Hb, Wb = grid.shape
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    y0 = np.clip(y0, 0, Hb - 2)
    x0 = np.clip(x0, 0, Wb - 2)
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x0 + 1)]
    g10 = grid[np.ix_(y0 + 1, x0)]
    g11 = grid[np.ix_(y0 + 1, x0 + 1)]
    return (g00 * (1 - fy) * (1 - fx) + g01 * (1 - fy) * fx
            + g10 * fy * (1 - fx) + g11 * fy * fx)


class AdvectedField:
    """Wrap an :class:`InfiniteDiffusion` sampler with a kinematic (advective) time axis.

    Args:
        sampler: the static InfiniteDiffusion sampler (the seamless O(1) field).
        vel_ms: per-level advecting wind, shape ``(n_levels, 2)`` in m/s (u, v). Use
            :func:`velocity_from_stats` for the trained model's climatology.
        pixel_km: lattice pixel size in km (≈28 for ERA5 0.25°).
        dt_seconds: seconds represented by one unit of ``t`` (the artifact time index).
    """

    def __init__(self, sampler, vel_ms, *, pixel_km: float = 28.0, dt_seconds: float = 3600.0):
        self.sampler = sampler
        self.vel_ms = np.asarray(vel_ms, dtype=float)            # (L, 2)
        self.L = self.vel_ms.shape[0]
        self.pixel_km = float(pixel_km)
        self.dt_seconds = float(dt_seconds)
        # advecting velocity in PIXELS per unit t (per level)
        px_per_ms = self.dt_seconds / (self.pixel_km * 1000.0)
        self.vel_px = self.vel_ms * px_per_ms                    # (L, 2) -> (u_px, v_px)

    # passthroughs so the generator can treat this like the static sampler
    @property
    def device(self):
        return self.sampler.device

    @property
    def window(self):
        return self.sampler.window

    @property
    def stride(self):
        return self.sampler.stride

    @property
    def T(self):
        return self.sampler.T

    def seam_lines(self, y0, y1, x0, x1):
        # advection is a smooth coordinate shift of an already-seamless field, so no new
        # seams appear; report the static stitch lines (t=0) as the diagnostic markers.
        return self.sampler.seam_lines(y0, y1, x0, x1)

    def clear_cache(self):
        self.sampler.clear_cache()

    def field_uv(self, y0: int, y1: int, x0: int, x1: int, t: float = 0.0):
        """Advected (u, v), each ``(n_levels, H, W)``, at integer-pixel window and time ``t``."""
        y0, y1, x0, x1 = int(y0), int(y1), int(x0), int(x1)
        H, W = y1 - y0, x1 - x0
        if t == 0:
            return self.sampler.field_uv(y0, y1, x0, x1)

        sx = self.vel_px[:, 0] * t      # per-level x shift (pixels)
        sy = self.vel_px[:, 1] * t      # per-level y shift
        # one bounding box covering every level's source window (+1px bilinear margin)
        by0 = int(np.floor(y0 - sy.max())) - 1
        by1 = int(np.ceil((y1 - 1) - sy.min())) + 2
        bx0 = int(np.floor(x0 - sx.max())) - 1
        bx1 = int(np.ceil((x1 - 1) - sx.min())) + 2
        su, sv = self.sampler.field_uv(by0, by1, bx0, bx1)       # (L, Hb, Wb)

        out_u = np.empty((self.L, H, W), dtype=su.dtype)
        out_v = np.empty((self.L, H, W), dtype=sv.dtype)
        ytar = np.arange(y0, y1, dtype=float)
        xtar = np.arange(x0, x1, dtype=float)
        for l in range(self.L):
            ys = ytar - sy[l] - by0
            xs = xtar - sx[l] - bx0
            out_u[l] = _bilinear(su[l], ys, xs)
            out_v[l] = _bilinear(sv[l], ys, xs)
        return out_u, out_v
