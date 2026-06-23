"""Animate a passive balloon in a top-down 2D slice of a wind field.

Examples:
    pixi run python experiments/viz_passive_drift.py --field synthetic
    pixi run python experiments/viz_passive_drift.py --field era5 --data data/era5.npz
    pixi run python experiments/viz_passive_drift.py --field data-driven-gp --data data/era5.npz
    pixi run python experiments/viz_passive_drift.py --field all --data data/era5.npz
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import jax
import numpy as np
import plotly.graph_objects as go

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import (
    DataDrivenFlowField,
    GridConfig,
    GridPosition,
    ReanalysisFlowField,
    SyntheticFlowField,
)
from src.env.field.era5_data import load_era5


FIELD_CHOICES = ("synthetic", "era5", "data-driven-gp")


def _config_from_args(args) -> GridConfig:
    if args.data:
        winds = load_era5(args.data).winds
        spatial_shape = winds.shape[1:-1]
        return GridConfig.create(*spatial_shape)
    if args.field != "synthetic":
        raise ValueError("--data is required for ERA5 and data-driven GP fields")
    if len(args.grid) == 2:
        return GridConfig.create(args.grid[0], args.grid[1], args.default_levels)
    if len(args.grid) == 3:
        return GridConfig.create(*args.grid)
    raise ValueError("--grid needs either x y or x y z")


def _position(values: Sequence[float] | None, config: GridConfig) -> GridPosition:
    default_z = (config.n_z + 1.0) / 2.0 if config.ndim == 3 else None
    if values is None:
        coords = ((config.n_x + 1.0) / 2.0, (config.n_y + 1.0) / 2.0)
        if config.ndim == 3:
            coords = (*coords, default_z)
    else:
        if len(values) == 2 and config.ndim == 2:
            coords = (float(values[0]), float(values[1]))
        elif len(values) == 2 and config.ndim == 3:
            coords = (float(values[0]), float(values[1]), default_z)
        elif len(values) == 3 and config.ndim == 3:
            coords = tuple(float(value) for value in values)
        else:
            raise ValueError(
                f"--start needs x y or x y z coordinates, got {len(values)}"
            )
    for value, size in zip(coords, config.shape):
        if not 1.0 <= value <= size:
            raise ValueError(f"start coordinate {value} is outside [1, {size}]")
    if config.ndim == 2:
        return GridPosition(coords[0], coords[1], None)
    return GridPosition(coords[0], coords[1], coords[2])


def _build_field(name: str, config: GridConfig, args):
    if name == "synthetic":
        return SyntheticFlowField(
            config,
            sigma=args.synthetic_sigma,
            lengthscale=args.synthetic_lengthscale,
            num_features=args.num_features,
        )
    if name == "era5":
        return ReanalysisFlowField(
            config,
            args.data,
            scale=args.scale,
            slice_mode="fixed",
            fixed_index=args.time_index,
        )
    return DataDrivenFlowField.from_era5_cache(
        config,
        args.data,
        time_index=args.time_index,
        scale=args.scale,
        training_stride=args.training_stride,
        max_training_points=args.max_training_points,
        num_features=args.num_features,
        lengthscale=args.gp_lengthscale,
        noise_std=args.gp_noise_std,
        feature_seed=args.seed,
        sample_posterior=args.posterior_sample,
        )


def _clip_or_wrap(
    x: float,
    y: float,
    config: GridConfig,
    boundary: str,
) -> tuple[float, float, bool]:
    out_of_bounds = not (1.0 <= x <= config.n_x and 1.0 <= y <= config.n_y)
    if boundary == "clip":
        return (
            float(np.clip(x, 1.0, config.n_x)),
            float(np.clip(y, 1.0, config.n_y)),
            out_of_bounds,
        )
    if boundary == "periodic":
        width_x = float(config.n_x)
        width_y = float(config.n_y)
        return (
            float(((x - 1.0) % width_x) + 1.0),
            float(((y - 1.0) % width_y) + 1.0),
            out_of_bounds,
        )
    if boundary == "terminal":
        return float(x), float(y), out_of_bounds
    raise ValueError(f"unknown boundary mode {boundary!r}")


def _field_slice(field, config: GridConfig, z: float | None, grid_subsample: int):
    xs = np.arange(1, config.n_x + 1, grid_subsample, dtype=float)
    ys = np.arange(1, config.n_y + 1, grid_subsample, dtype=float)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")

    arr = field.velocity_field()
    if config.ndim == 2:
        u = arr[::grid_subsample, ::grid_subsample, 0].ravel()
        v = np.zeros_like(u)
    else:
        level_idx = int(np.clip(round(z), 1, config.n_z)) - 1
        u = arr[::grid_subsample, ::grid_subsample, level_idx, 0].ravel()
        v = arr[::grid_subsample, ::grid_subsample, level_idx, 1].ravel()
    return gx.ravel(), gy.ravel(), u, v


def _simulate(field, config: GridConfig, start: GridPosition, args):
    field.reset(jax.random.PRNGKey(args.seed))
    positions = [(float(start.i), float(start.j), None if start.k is None else float(start.k))]
    velocities = []
    terminal_step = None

    x, y, z = positions[0]
    for step in range(args.steps):
        u, v = field.velocity_at(GridPosition(x, y, z))
        u = float(np.clip(u, -args.max_displacement, args.max_displacement))
        v = 0.0 if v is None else v
        v = float(np.clip(v, -args.max_displacement, args.max_displacement))
        velocities.append((u, v))
        x_next, y_next = x + u, y + v
        x, y, out_of_bounds = _clip_or_wrap(x_next, y_next, config, args.boundary)
        positions.append((x, y, z))
        if args.boundary == "terminal" and out_of_bounds:
            terminal_step = step + 1
            break

    return positions, np.asarray(velocities), terminal_step


def _build_figure(name: str, field, config: GridConfig, positions, velocities, args):
    grid_subsample = args.grid_subsample
    z = positions[0][2]
    qx, qy, u, v = _field_slice(field, config, z, grid_subsample)
    speed = np.sqrt(u**2 + v**2)
    max_speed = float(speed.max()) if speed.size else 0.0
    arrow_scale = 0.7 * grid_subsample / max(max_speed, 1e-6)
    quiver_x = []
    quiver_y = []
    for x0, y0, du, dv in zip(qx, qy, u, v):
        quiver_x.extend([x0, x0 + arrow_scale * du, None])
        quiver_y.extend([y0, y0 + arrow_scale * dv, None])

    title_name = {
        "synthetic": "Synthetic GP",
        "era5": "ERA5 Linear Interpolation",
        "data-driven-gp": "Data-Driven GP",
    }[name]
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=quiver_x,
            y=quiver_y,
            mode="lines",
            line=dict(color="rgba(37,99,235,0.55)", width=1.5),
            name="wind",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[0][0]],
            y=[positions[0][1]],
            mode="markers",
            marker=dict(size=12, color="#f59e0b", line=dict(color="#92400e", width=2)),
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color="#0f766e", width=4),
            name="path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[0][0]],
            y=[positions[0][1]],
            mode="markers",
            marker=dict(size=15, color="#dc2626", symbol="diamond"),
            name="balloon",
        )
    )

    frames = []
    for idx in range(len(positions)):
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(x=quiver_x, y=quiver_y),
                    go.Scatter(x=[positions[0][0]], y=[positions[0][1]]),
                    go.Scatter(x=xs[: idx + 1], y=ys[: idx + 1]),
                    go.Scatter(x=[positions[idx][0]], y=[positions[idx][1]]),
                ],
            )
        )
    fig.frames = frames

    frame_ms = int(1000 / max(args.fps, 1))
    fig.update_layout(
        title=(
            f"{title_name} passive drift, top-down at z={z:.1f}"
            if z is not None
            else f"{title_name} passive drift, 2D field"
        ),
        width=960,
        height=760,
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#eef2f7",
        xaxis=dict(
            title="x",
            range=[0.5, config.n_x + 0.5],
            showgrid=True,
            gridcolor="rgba(15,23,42,0.12)",
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="y",
            range=[0.5, config.n_y + 0.5],
            showgrid=True,
            gridcolor="rgba(15,23,42,0.12)",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.0,
                y=1.08,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": frame_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": "step "},
                steps=[
                    dict(
                        label=str(idx),
                        method="animate",
                        args=[
                            [str(idx)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    )
                    for idx in range(len(positions))
                ],
            )
        ],
    )
    return fig


def _run_one(name: str, config: GridConfig, start: GridPosition, args) -> Path:
    field = _build_field(name, config, args)
    positions, velocities, terminal_step = _simulate(field, config, start, args)
    fig = _build_figure(name, field, config, positions, velocities, args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"passive_drift_{name.replace('-', '_')}.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_play=False)
    end = positions[-1]
    start_coords = (
        (round(start.i, 2), round(start.j, 2), round(start.k, 2))
        if start.k is not None
        else (round(start.i, 2), round(start.j, 2))
    )
    end_coords = (
        (round(float(end[0]), 2), round(float(end[1]), 2), round(float(end[2]), 2))
        if end[2] is not None
        else (round(float(end[0]), 2), round(float(end[1]), 2))
    )
    print(
        f"{name:>14}: start={start_coords} end={end_coords} -> {output_path}"
    )
    if terminal_step is not None:
        print(f"                terminated out of bounds at step {terminal_step}")
    if isinstance(field, DataDrivenFlowField):
        print(
            f"                trained on {field.training_points} points; "
            f"training RMSE={field.training_rmse:.4f} cells/step"
        )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        choices=(*FIELD_CHOICES, "all"),
        default="synthetic",
    )
    parser.add_argument("--data", help="ERA5-style .npz cache for era5/GP modes")
    parser.add_argument("--grid", type=int, nargs="+", default=[60, 60, 7])
    parser.add_argument("--default-levels", type=int, default=7)
    parser.add_argument("--start", type=float, nargs="+", help="drop coordinates x y [z]")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.06,
                        help="convert cached m/s values to grid cells per step")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--boundary", choices=["clip", "periodic", "terminal"], default="clip")
    parser.add_argument("--max-displacement", type=float, default=4.0)
    parser.add_argument("--grid-subsample", type=int, default=2)
    parser.add_argument("--num-features", type=int, default=256)
    parser.add_argument("--synthetic-sigma", type=float, default=1.5)
    parser.add_argument("--synthetic-lengthscale", type=float, default=10.0)
    parser.add_argument("--gp-lengthscale", type=float, default=8.0)
    parser.add_argument("--gp-noise-std", type=float, default=0.1)
    parser.add_argument("--training-stride", type=int, default=2)
    parser.add_argument("--max-training-points", type=int, default=5000)
    parser.add_argument("--posterior-sample", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "output"),
    )
    args = parser.parse_args()

    if args.field == "all" and not args.data:
        parser.error("--field all requires --data")
    try:
        config = _config_from_args(args)
        start = _position(args.start, config)
        names = FIELD_CHOICES if args.field == "all" else (args.field,)
        for name in names:
            _run_one(name, config, start, args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
