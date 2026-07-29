"""Shared geometry and coherence diagnostics for tile-scaling experiments."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

QUERY_SIZE = 64
GRID_SPACING_KM = 27.83


@dataclass(frozen=True)
class TilingProfile:
    """One 50%-overlap tiling of the fixed 64x64 evaluation field."""

    core_tiles: int
    tiles_per_axis: int
    window: int
    stride: int
    expected_final_windows: int

    @property
    def name(self) -> str:
        return f"tiles_{self.core_tiles}"


PROFILES = {
    4: TilingProfile(4, 2, 64, 32, 9),
    16: TilingProfile(16, 4, 32, 16, 25),
    64: TilingProfile(64, 8, 16, 8, 81),
}


def profile_for(core_tiles: int) -> TilingProfile:
    try:
        return PROFILES[int(core_tiles)]
    except KeyError as exc:
        raise ValueError(f"core_tiles must be one of {sorted(PROFILES)}") from exc


def boundary_indices(size: int, stride: int) -> list[int]:
    """Adjacent-difference indices immediately before internal core boundaries."""
    return [boundary - 1 for boundary in range(stride, size, stride)]


def _vector_jumps(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    return np.hypot(np.diff(u, axis=axis), np.diff(v, axis=axis))


def _select_boundaries(values: np.ndarray, indices: list[int], axis: int) -> np.ndarray:
    if not indices:
        return np.empty(0, dtype=np.float64)
    return np.take(values, indices, axis=axis)


def _adjacent_cosine(u: np.ndarray, v: np.ndarray, axis: int) -> np.ndarray:
    left = [slice(None)] * u.ndim
    right = [slice(None)] * u.ndim
    left[axis] = slice(None, -1)
    right[axis] = slice(1, None)
    ul, ur = u[tuple(left)], u[tuple(right)]
    vl, vr = v[tuple(left)], v[tuple(right)]
    denominator = np.hypot(ul, vl) * np.hypot(ur, vr)
    return (ul * ur + vl * vr) / np.maximum(denominator, 1e-6)


def boundary_coherence(u: np.ndarray, v: np.ndarray, stride: int) -> dict[str, float]:
    """Measure amplitude and directional continuity at tile-core boundaries.

    A jump ratio of one and a direction-cosine gap of zero mean boundaries look like
    ordinary adjacent grid-cell pairs. Values are descriptive rather than composites.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 4:
        raise ValueError("u and v must share shape (sample, level, y, x)")
    yi = boundary_indices(u.shape[-2], stride)
    xi = boundary_indices(u.shape[-1], stride)

    x_jump = _vector_jumps(u, v, axis=3)
    y_jump = _vector_jumps(u, v, axis=2)
    all_jump = np.concatenate((x_jump.ravel(), y_jump.ravel()))
    seam_jump = np.concatenate(
        (
            _select_boundaries(x_jump, xi, axis=3).ravel(),
            _select_boundaries(y_jump, yi, axis=2).ravel(),
        )
    )

    x_cos = _adjacent_cosine(u, v, axis=3)
    y_cos = _adjacent_cosine(u, v, axis=2)
    all_cos = np.concatenate((x_cos.ravel(), y_cos.ravel()))
    seam_cos = np.concatenate(
        (
            _select_boundaries(x_cos, xi, axis=3).ravel(),
            _select_boundaries(y_cos, yi, axis=2).ravel(),
        )
    )
    jump_mean = max(float(all_jump.mean()), 1e-12)
    jump_sq_mean = max(float(np.mean(all_jump**2)), 1e-12)
    return {
        "boundary jump ratio": float(seam_jump.mean() / jump_mean),
        "boundary squared-jump ratio": float(np.mean(seam_jump**2) / jump_sq_mean),
        "boundary direction cosine": float(seam_cos.mean()),
        "all-neighbor direction cosine": float(all_cos.mean()),
        "boundary direction gap": float(all_cos.mean() - seam_cos.mean()),
    }


def spatial_correlation_curve(
    u: np.ndarray,
    v: np.ndarray,
    *,
    max_lag: int | None = None,
) -> np.ndarray:
    """Mean zero-padded vector autocorrelation along x and y.

    FFT accumulation keeps this affordable for all samples and levels in the benchmark.
    The returned curve starts at lag zero and is normalized to one there.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 4:
        raise ValueError("u and v must share shape (sample, level, y, x)")
    batch = u.shape[0] * u.shape[1]
    height, width = u.shape[-2:]
    limit = min(height, width) // 2 if max_lag is None else int(max_lag)
    if not 1 <= limit < min(height, width):
        raise ValueError("max_lag must lie inside the spatial field")

    uf = u.reshape(batch, height, width)
    vf = v.reshape(batch, height, width)
    uf = uf - uf.mean(axis=(-2, -1), keepdims=True)
    vf = vf - vf.mean(axis=(-2, -1), keepdims=True)
    padded = (2 * height, 2 * width)
    power = np.zeros((padded[0], padded[1] // 2 + 1), dtype=np.float64)
    for values in (uf, vf):
        for start in range(0, batch, 32):
            spectrum = np.fft.rfft2(values[start:start + 32], s=padded)
            power += np.sum(np.abs(spectrum) ** 2, axis=0)
    covariance = np.fft.irfft2(power, s=padded)

    lags = np.arange(limit + 1)
    x_pairs = batch * height * (width - lags)
    y_pairs = batch * (height - lags) * width
    x_cov = covariance[0, lags] / x_pairs
    y_cov = covariance[lags, 0] / y_pairs
    curve = 0.5 * (x_cov + y_cov)
    return curve / max(float(curve[0]), 1e-12)


def coherence_length_km(
    u: np.ndarray,
    v: np.ndarray,
    *,
    spacing_km: float = GRID_SPACING_KM,
    threshold: float = 0.5,
) -> tuple[float, np.ndarray]:
    """First interpolated lag where vector correlation falls below ``threshold``."""
    curve = spatial_correlation_curve(u, v)
    below = np.flatnonzero(curve <= threshold)
    if not len(below):
        return float((len(curve) - 1) * spacing_km), curve
    hi = int(below[0])
    if hi == 0:
        return 0.0, curve
    lo = hi - 1
    denominator = curve[lo] - curve[hi]
    fraction = 0.0 if abs(denominator) < 1e-12 else (curve[lo] - threshold) / denominator
    return float((lo + fraction) * spacing_km), curve
