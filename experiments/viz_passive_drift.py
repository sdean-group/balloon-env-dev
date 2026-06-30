"""Animate a passive balloon in 2D views of a wind field.

Examples:
    pixi run python experiments/viz_passive_drift.py --field synthetic
    pixi run python experiments/viz_passive_drift.py --field era5 --data data/era5.npz
    pixi run python experiments/viz_passive_drift.py --field data-driven-gp --data data/era5.npz
    pixi run python experiments/viz_passive_drift.py --field all --data data/era5.npz
    pixi run python experiments/viz_passive_drift.py --field all --data data/era5.npz --view y-cross-section
"""

from __future__ import annotations

import argparse
import json
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
    HelmholtzDataDrivenFlowField,
    HelmholtzSyntheticFlowField,
    ReanalysisFlowField,
    SyntheticFlowField,
)
from src.env.field.era5_data import load_era5


FIELD_CHOICES = (
    "synthetic",
    "legacy-synthetic",
    "helmholtz-synthetic",
    "era5",
    "data-driven-gp",
    "helmholtz-data-driven-gp",
)
DEFAULT_DEMO_FIELDS = ("synthetic", "era5", "data-driven-gp")
VIEW_CHOICES = ("topdown", "y-cross-section")


def _config_from_args(args) -> GridConfig:
    if args.data:
        winds = load_era5(args.data).winds
        spatial_shape = winds.shape[1:-1]
        return GridConfig.create(*spatial_shape)
    if args.field not in ("synthetic", "legacy-synthetic", "helmholtz-synthetic"):
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
        return HelmholtzSyntheticFlowField(
            config,
            sigma=args.synthetic_sigma,
            lengthscale=args.synthetic_lengthscale,
            num_features=args.num_features,
        )
    if name == "legacy-synthetic":
        return SyntheticFlowField(
            config,
            sigma=args.synthetic_sigma,
            lengthscale=args.synthetic_lengthscale,
            num_features=args.num_features,
        )
    if name == "helmholtz-synthetic":
        return HelmholtzSyntheticFlowField(
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
    if name == "helmholtz-data-driven-gp":
        return HelmholtzDataDrivenFlowField.from_era5_cache(
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
        )
    if name == "data-driven-gp":
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
    raise ValueError(f"unknown field {name!r}")


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


def _y_cross_section_level(
    config: GridConfig, positions: list[tuple[float, float, float | None]], args
) -> float:
    if config.ndim != 3:
        raise ValueError("y-cross-section view requires a 3D field")
    y_level = positions[0][1] if args.cross_section_y is None else float(args.cross_section_y)
    if not 1.0 <= y_level <= config.n_y:
        raise ValueError(f"cross-section y={y_level} is outside [1, {config.n_y}]")
    return y_level


def _field_y_cross_section(
    field,
    config: GridConfig,
    y_level: float,
    grid_subsample: int,
):
    x_count = max(8, int(np.ceil(config.n_x / max(grid_subsample, 1))))
    z_count = max(4, config.n_z)
    xs = np.linspace(1.0, float(config.n_x), x_count)
    zs = np.linspace(1.0, float(config.n_z), z_count)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")

    u = np.zeros(gx.size, dtype=float)
    for idx, (x, z) in enumerate(zip(gx.ravel(), gz.ravel())):
        u_i, _ = field.velocity_at(GridPosition(float(x), y_level, float(z)))
        u[idx] = float(u_i)
    return gx.ravel(), gz.ravel(), u


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


def _title_name(name: str) -> str:
    return {
        "synthetic": "Helmholtz Synthetic GP",
        "legacy-synthetic": "Legacy Synthetic GP",
        "helmholtz-synthetic": "Helmholtz Synthetic GP",
        "era5": "ERA5 Linear Interpolation",
        "data-driven-gp": "Data-Driven GP",
        "helmholtz-data-driven-gp": "Helmholtz Data-Driven GP",
    }[name]


def _frame_controls(frame_count: int, fps: int):
    frame_ms = int(1000 / max(fps, 1))
    return dict(
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
                    for idx in range(frame_count)
                ],
            )
        ],
    )


def _build_topdown_figure(name: str, field, config: GridConfig, positions, velocities, args):
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

    fig.update_layout(
        title=(
            f"{_title_name(name)} passive drift, top-down at z={z:.1f}"
            if z is not None
            else f"{_title_name(name)} passive drift, 2D field"
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
        **_frame_controls(len(positions), args.fps),
    )
    return fig


def _build_interactive_topdown_html(
    name: str,
    field,
    config: GridConfig,
    start: GridPosition,
    args,
) -> str:
    wind = np.asarray(field.velocity_field(), dtype=float)
    components = 1 if config.ndim == 2 else 2
    initial_z = 1.0 if start.k is None else float(start.k)
    payload = {
        "title": f"{_title_name(name)} passive drift",
        "shape": list(config.shape),
        "ndim": config.ndim,
        "components": components,
        "wind": wind.tolist(),
        "start": [float(start.i), float(start.j), initial_z],
        "maxDisplacement": float(args.max_displacement),
        "boundary": args.boundary,
        "gridSubsample": int(args.grid_subsample),
        "fps": int(args.fps),
    }
    data_json = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_title_name(name)} passive drift</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #f8fafc;
      color: #172033;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: #f8fafc;
    }}
    .app {{
      max-width: 1120px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 260px;
      gap: 18px;
      align-items: start;
    }}
    h1 {{
      grid-column: 1 / -1;
      margin: 0 0 4px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    canvas {{
      width: 100%;
      aspect-ratio: 1.22;
      background: #eef3f8;
      border: 1px solid #cbd5e1;
      display: block;
    }}
    .panel {{
      border: 1px solid #cbd5e1;
      background: #ffffff;
      padding: 14px;
    }}
    .group {{
      display: grid;
      gap: 8px;
      margin-bottom: 14px;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 12px;
      font-weight: 650;
      color: #334155;
    }}
    input {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid #b9c3d0;
      padding: 7px 8px;
      font: inherit;
      border-radius: 3px;
      background: #fff;
    }}
    input[type="range"] {{
      padding: 0;
    }}
    .row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }}
    button {{
      border: 1px solid #0f766e;
      background: #0f766e;
      color: white;
      padding: 8px 10px;
      font: inherit;
      font-weight: 700;
      border-radius: 3px;
      cursor: pointer;
    }}
    button.secondary {{
      border-color: #94a3b8;
      background: #f8fafc;
      color: #172033;
    }}
    .status {{
      font-size: 12px;
      line-height: 1.45;
      color: #475569;
      border-top: 1px solid #e2e8f0;
      padding-top: 10px;
      white-space: pre-line;
    }}
    @media (max-width: 860px) {{
      body {{ padding: 12px; }}
      .app {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="app">
    <h1>{_title_name(name)} passive drift</h1>
    <canvas id="field" width="960" height="760"></canvas>
    <aside class="panel">
      <div class="group">
        <div class="row">
          <label>x<input id="xInput" type="number" step="0.1"></label>
          <label>y<input id="yInput" type="number" step="0.1"></label>
        </div>
        <label>z<input id="zInput" type="number" step="0.1"></label>
        <button id="deployBtn">Deploy</button>
      </div>
      <div class="group">
        <label>speed <input id="speedInput" type="range" min="1" max="30" step="1" value="8"></label>
        <div class="row">
          <button id="playBtn">Resume</button>
          <button id="pauseBtn" class="secondary">Pause</button>
        </div>
        <button id="resetBtn" class="secondary">Reset</button>
      </div>
      <div id="status" class="status"></div>
    </aside>
  </main>
  <script>
    const data = {data_json};
    const canvas = document.getElementById("field");
    const ctx = canvas.getContext("2d");
    const xInput = document.getElementById("xInput");
    const yInput = document.getElementById("yInput");
    const zInput = document.getElementById("zInput");
    const speedInput = document.getElementById("speedInput");
    const statusEl = document.getElementById("status");
    const pad = {{ left: 54, right: 24, top: 28, bottom: 50 }};
    const nx = data.shape[0];
    const ny = data.shape[1];
    const nz = data.ndim === 3 ? data.shape[2] : 1;
    let running = false;
    let timer = null;
    let step = 0;
    let pos = {{ x: data.start[0], y: data.start[1], z: data.start[2] }};
    let start = {{ ...pos }};
    let path = [{{ ...pos }}];

    xInput.min = 1; xInput.max = nx; xInput.value = pos.x.toFixed(1);
    yInput.min = 1; yInput.max = ny; yInput.value = pos.y.toFixed(1);
    zInput.min = 1; zInput.max = nz; zInput.value = pos.z.toFixed(1);
    zInput.disabled = data.ndim !== 3;

    function clamp(value, lo, hi) {{
      return Math.max(lo, Math.min(hi, value));
    }}

    function toCanvasX(x) {{
      return pad.left + (x - 1) / Math.max(nx - 1, 1) * (canvas.width - pad.left - pad.right);
    }}

    function toCanvasY(y) {{
      return canvas.height - pad.bottom - (y - 1) / Math.max(ny - 1, 1) * (canvas.height - pad.top - pad.bottom);
    }}

    function fromCanvas(px, py) {{
      const x = 1 + (px - pad.left) / (canvas.width - pad.left - pad.right) * Math.max(nx - 1, 1);
      const y = 1 + (canvas.height - pad.bottom - py) / (canvas.height - pad.top - pad.bottom) * Math.max(ny - 1, 1);
      return {{ x: clamp(x, 1, nx), y: clamp(y, 1, ny) }};
    }}

    function cell(value, maxN) {{
      const v = clamp(value, 1, maxN);
      const loCoord = Math.floor(v);
      const hiCoord = Math.min(loCoord + 1, maxN);
      return {{
        lo: loCoord - 1,
        hi: hiCoord - 1,
        t: hiCoord === loCoord ? 0 : v - loCoord
      }};
    }}

    function sampleAt(x, y, z) {{
      const cx = cell(x, nx);
      const cy = cell(y, ny);
      if (data.ndim === 2) {{
        const a = data.wind[cx.lo][cy.lo][0];
        const b = data.wind[cx.hi][cy.lo][0];
        const c = data.wind[cx.lo][cy.hi][0];
        const d = data.wind[cx.hi][cy.hi][0];
        const u = a * (1 - cx.t) * (1 - cy.t) + b * cx.t * (1 - cy.t) + c * (1 - cx.t) * cy.t + d * cx.t * cy.t;
        return [u, 0];
      }}
      const cz = cell(z, nz);
      const out = [0, 0];
      for (let ix of [0, 1]) for (let iy of [0, 1]) for (let iz of [0, 1]) {{
        const wx = ix ? cx.t : 1 - cx.t;
        const wy = iy ? cy.t : 1 - cy.t;
        const wz = iz ? cz.t : 1 - cz.t;
        const value = data.wind[ix ? cx.hi : cx.lo][iy ? cy.hi : cy.lo][iz ? cz.hi : cz.lo];
        out[0] += value[0] * wx * wy * wz;
        out[1] += value[1] * wx * wy * wz;
      }}
      return out;
    }}

    function drawArrow(x, y, u, v, scale) {{
      const x0 = toCanvasX(x);
      const y0 = toCanvasY(y);
      const x1 = toCanvasX(clamp(x + u * scale, 1, nx));
      const y1 = toCanvasY(clamp(y + v * scale, 1, ny));
      const angle = Math.atan2(y1 - y0, x1 - x0);
      const head = 6;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(angle - 0.55), y1 - head * Math.sin(angle - 0.55));
      ctx.lineTo(x1 - head * Math.cos(angle + 0.55), y1 - head * Math.sin(angle + 0.55));
      ctx.closePath();
      ctx.fill();
    }}

    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#eef3f8";
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      ctx.strokeStyle = "rgba(15,23,42,0.12)";
      ctx.lineWidth = 1;
      for (let i = 1; i <= nx; i += Math.max(5, data.gridSubsample)) {{
        ctx.beginPath();
        ctx.moveTo(toCanvasX(i), pad.top);
        ctx.lineTo(toCanvasX(i), canvas.height - pad.bottom);
        ctx.stroke();
      }}
      for (let j = 1; j <= ny; j += Math.max(5, data.gridSubsample)) {{
        ctx.beginPath();
        ctx.moveTo(pad.left, toCanvasY(j));
        ctx.lineTo(canvas.width - pad.right, toCanvasY(j));
        ctx.stroke();
      }}

      let maxMag = 0;
      const samples = [];
      const stride = Math.max(6, data.gridSubsample * 3);
      for (let x = 1; x <= nx; x += stride) {{
        for (let y = 1; y <= ny; y += stride) {{
          const [u, v] = sampleAt(x, y, pos.z);
          const mag = Math.hypot(u, v);
          maxMag = Math.max(maxMag, mag);
          samples.push([x, y, u, v]);
        }}
      }}
      const arrowScale = maxMag > 1e-9 ? 0.9 * stride / maxMag : 1;
      ctx.strokeStyle = "rgba(37,99,235,0.62)";
      ctx.fillStyle = "rgba(37,99,235,0.62)";
      ctx.lineWidth = 1.35;
      for (const [x, y, u, v] of samples) drawArrow(x, y, u, v, arrowScale);

      ctx.strokeStyle = "#0f766e";
      ctx.lineWidth = 4;
      ctx.beginPath();
      path.forEach((p, idx) => {{
        const x = toCanvasX(p.x);
        const y = toCanvasY(p.y);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      ctx.fillStyle = "#f59e0b";
      ctx.beginPath();
      ctx.arc(toCanvasX(start.x), toCanvasY(start.y), 7, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = "#dc2626";
      ctx.beginPath();
      const bx = toCanvasX(pos.x);
      const by = toCanvasY(pos.y);
      ctx.moveTo(bx, by - 9);
      ctx.lineTo(bx + 9, by);
      ctx.lineTo(bx, by + 9);
      ctx.lineTo(bx - 9, by);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "#334155";
      ctx.font = "13px Inter, sans-serif";
      ctx.fillText("x", canvas.width / 2, canvas.height - 16);
      ctx.save();
      ctx.translate(18, canvas.height / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("y", 0, 0);
      ctx.restore();

      statusEl.textContent = `step ${{step}}\\nx=${{pos.x.toFixed(2)}}  y=${{pos.y.toFixed(2)}}  z=${{pos.z.toFixed(2)}}\\nspeed=${{speedInput.value}} steps/sec`;
    }}

    function simStep() {{
      const [uRaw, vRaw] = sampleAt(pos.x, pos.y, pos.z);
      const u = clamp(uRaw, -data.maxDisplacement, data.maxDisplacement);
      const v = clamp(vRaw, -data.maxDisplacement, data.maxDisplacement);
      let nxp = pos.x + u;
      let nyp = pos.y + v;
      if (data.boundary === "periodic") {{
        nxp = ((nxp - 1) % nx + nx) % nx + 1;
        nyp = ((nyp - 1) % ny + ny) % ny + 1;
      }} else {{
        nxp = clamp(nxp, 1, nx);
        nyp = clamp(nyp, 1, ny);
      }}
      pos = {{ x: nxp, y: nyp, z: pos.z }};
      path.push({{ ...pos }});
      step += 1;
      draw();
    }}

    function setTimer() {{
      if (timer) clearInterval(timer);
      if (running) timer = setInterval(simStep, 1000 / Number(speedInput.value));
    }}

    function deploy() {{
      running = false;
      setTimer();
      pos = {{
        x: clamp(Number(xInput.value), 1, nx),
        y: clamp(Number(yInput.value), 1, ny),
        z: clamp(Number(zInput.value), 1, nz)
      }};
      start = {{ ...pos }};
      path = [{{ ...pos }}];
      step = 0;
      draw();
    }}

    canvas.addEventListener("click", (event) => {{
      const rect = canvas.getBoundingClientRect();
      const p = fromCanvas(
        (event.clientX - rect.left) * canvas.width / rect.width,
        (event.clientY - rect.top) * canvas.height / rect.height
      );
      xInput.value = p.x.toFixed(1);
      yInput.value = p.y.toFixed(1);
      deploy();
    }});

    document.getElementById("deployBtn").addEventListener("click", deploy);
    document.getElementById("playBtn").addEventListener("click", () => {{ running = true; setTimer(); }});
    document.getElementById("pauseBtn").addEventListener("click", () => {{ running = false; setTimer(); }});
    document.getElementById("resetBtn").addEventListener("click", deploy);
    speedInput.addEventListener("input", setTimer);
    zInput.addEventListener("change", deploy);
    draw();
  </script>
</body>
</html>
"""


def _build_y_cross_section_figure(
    name: str,
    field,
    config: GridConfig,
    positions,
    velocities,
    args,
):
    y_level = _y_cross_section_level(config, positions, args)
    qx, qz, u = _field_y_cross_section(
        field, config, y_level, args.grid_subsample
    )
    x_speed = np.abs(u)
    max_x_speed = float(x_speed.max()) if x_speed.size else 0.0
    arrow_scale = 0.55 * args.grid_subsample / max(max_x_speed, 1e-6)
    arrow_annotations = []
    for x0, z0, du in zip(qx, qz, u):
        x1 = float(np.clip(x0 + arrow_scale * du, 1.0, config.n_x))
        arrow_annotations.append(
            dict(
                x=x1,
                y=float(z0),
                ax=float(x0),
                ay=float(z0),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=0.85,
                arrowwidth=1.2,
                arrowcolor="rgba(37,99,235,0.65)",
                opacity=0.85,
            )
        )

    xs = [p[0] for p in positions]
    zs = [p[2] for p in positions]
    hover_y = [p[1] for p in positions]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=qx,
            y=qz,
            mode="markers",
            marker=dict(size=3, color="rgba(37,99,235,0.25)"),
            name="wind sample points",
            hovertemplate="x=%{x:.2f}<br>z=%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[0][0]],
            y=[positions[0][2]],
            mode="markers",
            marker=dict(size=12, color="#f59e0b", line=dict(color="#92400e", width=2)),
            name="start",
            hovertemplate=f"start<br>x=%{{x:.2f}}<br>z=%{{y:.2f}}<br>actual y={positions[0][1]:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=zs,
            mode="lines",
            line=dict(color="#0f766e", width=4),
            name="projected path",
            text=[f"actual y={value:.2f}" for value in hover_y],
            hovertemplate="x=%{x:.2f}<br>z=%{y:.2f}<br>%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[positions[0][0]],
            y=[positions[0][2]],
            mode="markers",
            marker=dict(size=15, color="#dc2626", symbol="diamond"),
            name="balloon",
            text=[f"actual y={positions[0][1]:.2f}"],
            hovertemplate="x=%{x:.2f}<br>z=%{y:.2f}<br>%{text}<extra></extra>",
        )
    )

    frames = []
    for idx in range(len(positions)):
        frames.append(
            go.Frame(
                name=str(idx),
                data=[
                    go.Scatter(x=qx, y=qz),
                    go.Scatter(x=[positions[0][0]], y=[positions[0][2]]),
                    go.Scatter(x=xs[: idx + 1], y=zs[: idx + 1]),
                    go.Scatter(
                        x=[positions[idx][0]],
                        y=[positions[idx][2]],
                        text=[f"actual y={positions[idx][1]:.2f}"],
                    ),
                ],
            )
        )
    fig.frames = frames

    fig.update_layout(
        title=f"{_title_name(name)} passive drift, x-z cross section at y={y_level:.1f}",
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
            title="z",
            range=[0.5, config.n_z + 0.5],
            showgrid=True,
            gridcolor="rgba(15,23,42,0.12)",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=30, t=80, b=60),
        annotations=arrow_annotations,
        **_frame_controls(len(positions), args.fps),
    )
    return fig


def _build_figure(name: str, field, config: GridConfig, positions, velocities, args):
    if args.view == "topdown":
        return _build_topdown_figure(name, field, config, positions, velocities, args)
    return _build_y_cross_section_figure(name, field, config, positions, velocities, args)


def _run_one(name: str, config: GridConfig, start: GridPosition, args) -> Path:
    field = _build_field(name, config, args)
    positions, velocities, terminal_step = _simulate(field, config, start, args)

    output_dir = Path(args.output_dir)
    if args.view == "y-cross-section":
        output_dir = output_dir / "y_cross_section"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"passive_drift_{name.replace('-', '_')}.html"
    if args.view == "topdown":
        output_path.write_text(
            _build_interactive_topdown_html(name, field, config, start, args),
            encoding="utf-8",
        )
    else:
        fig = _build_figure(name, field, config, positions, velocities, args)
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
    if isinstance(field, (DataDrivenFlowField, HelmholtzDataDrivenFlowField)):
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
    parser.add_argument("--view", choices=VIEW_CHOICES, default="topdown")
    parser.add_argument(
        "--cross-section-y",
        type=float,
        help="fixed y level for the x-z cross-section view; defaults to the drop y",
    )
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
        names = DEFAULT_DEMO_FIELDS if args.field == "all" else (args.field,)
        for name in names:
            _run_one(name, config, start, args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
