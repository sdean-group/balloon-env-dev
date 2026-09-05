from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

MODULE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion/tiling_scaling"
)
sys.path.insert(0, str(MODULE_ROOT))

from ensemble_diagnostics import (  # noqa: E402
    correlation_curve_rmse,
    gradient_w1,
    multiscale_patch_swd,
    structure_log_rmse,
    vector_structure_function,
)


def _fields(seed: int, samples: int = 6) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=(samples, 3, 32, 32)).astype(np.float32)
    v = rng.normal(size=(samples, 3, 32, 32)).astype(np.float32)
    return u, v


def test_structural_metrics_are_zero_for_identical_arrays() -> None:
    reference = _fields(3)
    patch, detail = multiscale_patch_swd(
        reference,
        reference,
        factors=(1, 2),
        patch=3,
        patches=128,
        projections=16,
    )
    structure, _, _ = structure_log_rmse(reference, reference)
    correlation, _, _ = correlation_curve_rmse(reference, reference)
    gradient = gradient_w1(reference, reference, samples=5_000)

    # Patch and gradient samplers intentionally draw independent subsets.
    assert patch < 0.4
    assert set(detail) == {1, 2}
    assert structure == 0.0
    assert correlation == 0.0
    assert gradient < 0.1


def test_metrics_detect_smoothing() -> None:
    reference = _fields(7)
    u, v = reference
    smooth = (
        0.25 * (
            u + np.roll(u, 1, axis=2) + np.roll(u, 1, axis=3)
            + np.roll(np.roll(u, 1, axis=2), 1, axis=3)
        ),
        0.25 * (
            v + np.roll(v, 1, axis=2) + np.roll(v, 1, axis=3)
            + np.roll(np.roll(v, 1, axis=2), 1, axis=3)
        ),
    )
    structure, _, _ = structure_log_rmse(smooth, reference)
    correlation, _, _ = correlation_curve_rmse(smooth, reference)

    assert structure > 0.5
    assert correlation > 0.05
    assert vector_structure_function(*smooth)[0] < vector_structure_function(*reference)[0]
