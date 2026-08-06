"""Convert existing space-time runs to WindArtifacts and compare them without ERA5."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from . import artifact
from .metrics.diagnostics import field_diagnostics


def _load_run(path: Path, *, lat_origin: float, lon_origin: float, time_origin: str,
              stride: int, time_stride: int) -> tuple[object, dict]:
    npz_path = path / "wind.npz"
    metrics_path = path / "metrics.json"
    values = np.load(npz_path)
    run_metrics = json.loads(metrics_path.read_text())
    u, v, levels = values["u"], values["v"], values["levels"]

    # evaluate_spacetime_infinite queries from one stride inside the global origin.
    lat = lat_origin + 0.25 * np.arange(stride, stride + u.shape[2])
    lon = lon_origin + 0.25 * np.arange(stride, stride + u.shape[3])
    t0 = np.datetime64(time_origin) + np.timedelta64(time_stride, "h")
    times = (t0 + np.arange(u.shape[0]).astype("timedelta64[h]")).astype("datetime64[ns]")
    ds = artifact.make_field(u, v, level=levels, lat=lat, lon=lon, time=times)

    seams_x = list(range(stride, u.shape[3], stride))
    seams_y = list(range(stride, u.shape[2], stride))
    attrs = artifact.default_attrs(
        generator={
            "name": "conditional_spacetime_infinite_diffusion",
            "checkpoint": run_metrics.get("checkpoint"),
            "checkpoint_step": run_metrics.get("checkpoint_step"),
            "config": {
                "outer_depth": run_metrics.get("outer_depth", 1),
                "split_step": run_metrics.get("split_step"),
                "edm_steps": run_metrics.get("edm_steps"),
                "window": run_metrics.get("window"),
                "stride": run_metrics.get("stride"),
                "time_stride": run_metrics.get("time_stride"),
            },
        },
        capabilities={
            "extent": "unbounded",
            "tiled": True,
            "random_access": True,
            "temporally_evolving": True,
        },
        conditioning={
            "lat_origin": lat_origin,
            "lon_origin": lon_origin,
            "time_origin": time_origin,
        },
        model_levels=levels,
        seed=7,
    )
    attrs["seam_boundaries"] = {"x": seams_x, "y": seams_y}
    ds.attrs = attrs
    return ds, run_metrics


def _write_artifact_compatible(ds, path: Path) -> None:
    """Write format-v2 Zarr with both the cluster and older local xarray APIs."""
    try:
        artifact.write(ds, ds.attrs, path)
    except TypeError as exc:
        if "zarr_format" not in str(exc):
            raise
        if path.exists():
            shutil.rmtree(path)
        encoded = ds.copy()
        encoded.attrs = artifact._encode_attrs(ds.attrs)
        encoded.to_zarr(path, mode="w", consolidated=False, zarr_version=2)


def _format(value) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("outputs/spacetime_harness_report.md"))
    parser.add_argument("--lat-origin", type=float, default=25.0)
    parser.add_argument("--lon-origin", type=float, default=225.0)
    parser.add_argument("--time-origin", default="2023-01-15T00")
    args = parser.parse_args(argv)
    names = args.names or [p.name for p in args.runs]
    if len(names) != len(args.runs):
        parser.error("--names must have the same length as runs")

    rows: dict[str, dict] = {}
    for name, run in zip(names, args.runs):
        metrics = json.loads((run / "metrics.json").read_text())
        ds, run_metrics = _load_run(
            run,
            lat_origin=args.lat_origin,
            lon_origin=args.lon_origin,
            time_origin=args.time_origin,
            stride=int(metrics["stride"]),
            time_stride=int(metrics["time_stride"]),
        )
        artifact_path = run / "wind_artifact.zarr"
        _write_artifact_compatible(ds, artifact_path)
        row = field_diagnostics(ds)
        forward_evaluations = run_metrics.get("cold_model_forward_evaluations")
        if forward_evaluations is None and run_metrics.get("cold_model_window_calls") is not None:
            # A complete N-step Heun trajectory uses two forwards except at the last step.
            forward_evaluations = int(run_metrics["cold_model_window_calls"]) * (
                2 * int(run_metrics["edm_steps"]) - 1
            )
        row.update(
            {
                "x seam jump ratio": run_metrics.get("x_seam_jump_ratio"),
                "y seam jump ratio": run_metrics.get("y_seam_jump_ratio"),
                "time seam jump ratio": run_metrics.get("time_seam_jump_ratio"),
                "temporal vector change (m/s)": run_metrics.get(
                    "mean_temporal_vector_change_mps"
                ),
                "generation seconds": run_metrics.get("cold_seconds"),
                "model forward evaluations": forward_evaluations,
                "exact cached repeat": run_metrics.get("exact_cached_repeat"),
            }
        )
        rows[name] = row

    keys = list(next(iter(rows.values())))
    lines = [
        "# Space-time InfiniteDiffusion diagnostic comparison",
        "",
        "These are reference-free descriptors and procedural checks. They compare sampler "
        "configurations but do not measure distance to ERA5. The benchmark-v2 ERA5 rows "
        "remain the authoritative realism evaluation.",
        "",
        "| Diagnostic | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key in keys:
        lines.append(f"| {key} | " + " | ".join(_format(rows[n][key]) for n in names) + " |")
    lines += [
        "",
        "Interpretation notes:",
        "",
        "- Seam ratios: 1 means a tile boundary is statistically like an ordinary adjacent step; lower is better only insofar as it stays near 1.",
        "- Spectrum slope, rotational fraction, vorticity/divergence, and increment kurtosis are raw descriptors, not universal pass/fail targets.",
        "- Four generated frames are insufficient for the harness temporal PSD and trajectory-dispersion metrics, which require at least 16 contiguous frames.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
