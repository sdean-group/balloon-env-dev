"""Tests for crop-matched training safeguards."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

TRAINING_DIR = (
    Path(__file__).resolve().parents[2]
    / "src/eval/windeval/generators/infinite_diffusion/tiling_scaling/training"
)
sys.path.insert(0, str(TRAINING_DIR))

from preflight import CONTROLLED_KEYS, validate_recipes  # noqa: E402


def _recipe(crop: int) -> dict:
    recipe = {
        "crop": crop,
        "levels": [49, 66],
        "spacetime": True,
        "conditional": True,
        "n_frames": 4,
        "frame_stride": 1,
        "temporal_kernel": 3,
        "model_channels": 128,
        "channel_mult": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "attn_resolutions": [8, 4],
        "sigma_data": 1.0,
        "batch_size": 16,
        "lr": 2e-4,
        "ema_decay": 0.9999,
        "n_steps": 100_000,
        "warmup_steps": 1_000,
        "ckpt_every": 2_000,
        "seed": 0,
    }
    assert set(CONTROLLED_KEYS) <= set(recipe)
    return recipe


def test_recipe_validation_allows_only_crop_to_change() -> None:
    reference = _recipe(64)
    validate_recipes(reference, {64: _recipe(64), 32: _recipe(32), 16: _recipe(16)})


def test_recipe_validation_rejects_optimizer_change() -> None:
    reference = _recipe(64)
    changed = _recipe(32)
    changed["lr"] = 1e-4
    with pytest.raises(ValueError, match="recipe differs"):
        validate_recipes(reference, {32: changed})


def test_recipe_validation_rejects_wrong_crop_label() -> None:
    with pytest.raises(ValueError, match="records crop"):
        validate_recipes(_recipe(64), {32: _recipe(16)})
