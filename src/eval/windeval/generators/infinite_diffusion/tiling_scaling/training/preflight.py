"""Validate the data and reference checkpoint for crop-matched tile training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml


CONTROLLED_KEYS = (
    "levels",
    "spacetime",
    "conditional",
    "n_frames",
    "frame_stride",
    "temporal_kernel",
    "model_channels",
    "channel_mult",
    "num_res_blocks",
    "attn_resolutions",
    "sigma_data",
    "batch_size",
    "lr",
    "ema_decay",
    "n_steps",
    "warmup_steps",
    "ckpt_every",
    "seed",
)


def _flat_yaml(path: Path) -> dict:
    raw = yaml.safe_load(path.read_text()) or {}
    flat = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    return flat


def _canonical(value):
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    return value


def validate_recipes(reference: dict, candidates: dict[int, dict]) -> None:
    for crop, candidate in candidates.items():
        if int(candidate.get("crop", -1)) != crop:
            raise ValueError(f"crop-{crop} config records crop={candidate.get('crop')}")
        differences = {
            key: (reference.get(key), candidate.get(key))
            for key in CONTROLLED_KEYS
            if _canonical(reference.get(key)) != _canonical(candidate.get(key))
        }
        if differences:
            raise ValueError(f"crop-{crop} recipe differs from reference: {differences}")


def validate_dataset(path: Path) -> dict:
    dataset = xr.open_zarr(path, consolidated=False)
    try:
        missing = {"u", "v"} - set(dataset.data_vars)
        if missing:
            raise ValueError(f"training dataset is missing variables {sorted(missing)}")
        for dimension in ("time", "level", "y", "x"):
            if dimension not in dataset.sizes:
                raise ValueError(f"training dataset is missing dimension {dimension}")
        if dataset.sizes["level"] != 18:
            raise ValueError(f"expected 18 model levels, found {dataset.sizes['level']}")
        if min(dataset.sizes["y"], dataset.sizes["x"]) < 64:
            raise ValueError("training grid is smaller than the 64x64 reference crop")

        times = np.asarray(dataset["time"].values).astype("datetime64[h]")
        years = np.unique(times.astype("datetime64[Y]").astype(int) + 1970)
        days = (times.astype("datetime64[D]") - times.astype("datetime64[M]")).astype(int) + 1
        contaminated = np.flatnonzero((days >= 8) & (days <= 14))
        if contaminated.size:
            raise ValueError(
                "training data contains held-out days 8-14; "
                f"first contaminated timestamp is {times[contaminated[0]]}"
            )
        if years.tolist() != [2023]:
            raise ValueError(f"expected the original 2023 training year, found {years.tolist()}")
        return {
            "path": str(path.resolve()),
            "sizes": {key: int(value) for key, value in dataset.sizes.items()},
            "first_time": str(times.min()),
            "last_time": str(times.max()),
            "years": years.tolist(),
            "held_out_days_absent": True,
        }
    finally:
        dataset.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint-64", type=Path, required=True)
    parser.add_argument("--config-64", type=Path, required=True)
    parser.add_argument("--config-32", type=Path, required=True)
    parser.add_argument("--config-16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint_64, map_location="cpu", weights_only=False)
    reference = dict(checkpoint["cfg"])
    if int(checkpoint.get("step", -1)) != 100_000:
        raise ValueError("64x64 reference checkpoint is not at step 100000")
    if int(reference.get("crop", -1)) != 64:
        raise ValueError("reference checkpoint was not trained on 64x64 crops")

    config64 = _flat_yaml(args.config_64)
    validate_recipes(
        reference,
        {
            64: config64,
            32: _flat_yaml(args.config_32),
            16: _flat_yaml(args.config_16),
        },
    )
    data_summary = validate_dataset(args.dataset)
    configured_path = Path(str(config64["data_path"]))
    report = {
        "status": "passed",
        "dataset": data_summary,
        "reference_checkpoint": str(args.checkpoint_64.resolve()),
        "reference_checkpoint_step": int(checkpoint["step"]),
        "reference_configured_data_path": str(configured_path),
        "controlled_keys": list(CONTROLLED_KEYS),
        "training_crops": [64, 32, 16],
        "primary_control": "fixed optimizer updates",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
