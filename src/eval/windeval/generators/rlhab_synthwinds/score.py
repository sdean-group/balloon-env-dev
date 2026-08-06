"""Score RL-HAB SynthWinds with the benchmark-v2 reference and metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ... import artifact
from ...benchmark import (
    DIST_METRICS,
    SPATIAL_METRICS,
    STRUCTURE_METRICS,
    SPATIAL_STRIDE_H,
    _cond_floor_groups,
    _level_hpa,
    _observation_condition_groups,
)
from ...metrics import METRIC_INFO, run_suite
from ...metrics.distributions import conditional_w1_grouped
from ...metrics.shear import climatological_dz
from ...reference import split
from .generate import radiosonde_layer_thickness


def _fmt(value) -> str:
    try:
        return "N/A" if not np.isfinite(value) else f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


DISPLAY_NAMES = {
    "W1 cond u (m/s)": "W1(u | c) (m/s)",
    "W1 cond v (m/s)": "W1(v | c) (m/s)",
    "W1 cond (m/s)": "mean W1(u,v | c) (m/s)",
}

EVAL_MONTHS = (1, 4, 7, 10)
EVAL_DAYS = tuple(range(8, 15))
EVAL_HOURS = tuple(range(0, 24, SPATIAL_STRIDE_H))


def _spatial_reference(reference):
    """Select and validate the exact held-out spatial-evaluation timestamps."""
    time = reference["time"]
    selected = reference.sel(
        time=time.dt.month.isin(EVAL_MONTHS)
        & time.dt.day.isin(EVAL_DAYS)
    ).sortby("time")
    hours = set(int(value) for value in selected.time.dt.hour.values.tolist())
    expected = len(EVAL_MONTHS) * len(EVAL_DAYS) * len(EVAL_HOURS)
    if hours == set(range(24)):
        selected = selected.sel(time=selected.time.dt.hour.isin(EVAL_HOURS))
    elif hours != set(EVAL_HOURS):
        raise ValueError(
            "ERA5 reference must contain either every hour or the standard 4-hour "
            f"timestamps; found hours={sorted(hours)}"
        )
    if selected.sizes["time"] != expected:
        found = {
            (int(t.month), int(t.day), int(t.hour))
            for t in selected.indexes["time"]
        }
        wanted = {
            (month, day, hour)
            for month in EVAL_MONTHS
            for day in EVAL_DAYS
            for hour in EVAL_HOURS
        }
        raise ValueError(
            "ERA5 reference is incomplete for evaluation v2: "
            f"found {len(found)}/{len(wanted)} timestamps; "
            f"first missing={sorted(wanted - found)[:4]}"
        )
    return selected.compute()


def _layer_thickness(
    reference,
    *,
    dz_file: Path | None,
    dz_source: Path | None,
    raw_dir: Path | None,
) -> tuple[np.ndarray | None, str]:
    if dz_file and dz_file.exists():
        return np.load(dz_file), f"benchmark cache: {dz_file}"
    if dz_source and dz_source.exists():
        return climatological_dz(dz_source), f"ERA5 stage-2 artifact: {dz_source}"
    if raw_dir and raw_dir.exists():
        return (
            radiosonde_layer_thickness(raw_dir, _level_hpa(reference)),
            f"RL-HAB radiosonde pressure-height profiles: {raw_dir}",
        )
    return None, "unavailable (vertical-shear metrics are N/A)"


def score(
    reference_path: Path,
    synth_path: Path,
    output_dir: Path,
    dz_source: Path | None,
    dz_file: Path | None = None,
    raw_dir: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    ref = _spatial_reference(artifact.read(reference_path))
    pred = artifact.read(synth_path).compute()
    ref_sp = ref
    half_a, half_b = split(ref)
    a_sp = half_a.compute()
    b_sp = half_b.compute()
    dz, dz_description = _layer_thickness(
        ref, dz_file=dz_file, dz_source=dz_source, raw_dir=raw_dir
    )

    floor, _ = run_suite(a_sp, b_sp, dz=dz, ref_temporal=b_sp)
    floor.update(conditional_w1_grouped(_cond_floor_groups(ref)))
    synth, _ = run_suite(pred, ref_sp, dz=dz)
    synth.update(conditional_w1_grouped(_observation_condition_groups(pred, ref)))
    rows = {"ERA5 self-split floor": floor, "RL-HAB SynthWinds": synth}

    json_path = output_dir / "rlhab_synthwinds_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, allow_nan=True))

    metrics = SPATIAL_METRICS + DIST_METRICS + STRUCTURE_METRICS
    lines = [
        "# RL-HAB SynthWinds evaluation",
        "",
        "The metric code, ERA5 reference, held-out dates, levels, and grid match evaluation v2. "
        "RL-HAB SynthWinds is an observation-driven radiosonde reconstruction at the scored "
        "timestamps, not the RL-HAB DQN policy and not a free-running generative model.",
        "",
        f"Reference frames after the standard 4-hour spatial subsampling: {ref_sp.sizes['time']}. "
        f"RL-HAB observation frames: {pred.sizes['time']}. Vertical spacing: {dz_description}.",
        "",
        "| Metric | ERA5 self-split floor | RL-HAB SynthWinds |",
        "|---|---:|---:|",
    ]
    for metric in metrics:
        direction, _ = METRIC_INFO[metric]
        display_name = DISPLAY_NAMES.get(metric, metric)
        lines.append(
            f"| {display_name} ({direction}) | {_fmt(floor.get(metric))} | "
            f"{_fmt(synth.get(metric))} |"
        )
    lines.extend(
        [
            "",
            "Conditioned W1 uses fixed location, month, and hour. The seven observation "
            "days are the RL-HAB samples for each condition; the ERA5 floor remains days "
            "8-10 versus days 11-14.",
            "",
        ]
    )
    markdown_path = output_dir / "rlhab_synthwinds_metrics.md"
    markdown_path.write_text("\n".join(lines))
    print(markdown_path.read_text())
    print(f"JSON: {json_path}")
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Score RL-HAB SynthWinds")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--synthwinds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dz-source", type=Path)
    parser.add_argument("--dz-file", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    args = parser.parse_args(argv)
    score(
        args.reference,
        args.synthwinds,
        args.output_dir,
        args.dz_source,
        args.dz_file,
        args.raw_dir,
    )


if __name__ == "__main__":
    main()
