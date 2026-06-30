"""Navigation tasks with a simple first-order MPC baseline.

The controller is intentionally plain: at every step, it tries a small set of
fixed-altitude rollouts and chooses the altitude whose simulated endpoint is
best for the task. This is the baseline we should expect learned policies to beat
later.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import GridConfig, GridPosition, ReanalysisFlowField
from src.env.field.era5_data import load_era5


def _config_from_cache(path: str) -> GridConfig:
    winds = load_era5(path).winds
    return GridConfig.create(*winds.shape[1:-1])


def _clip(x: float, y: float, config: GridConfig) -> tuple[float, float]:
    return float(np.clip(x, 1.0, config.n_x)), float(np.clip(y, 1.0, config.n_y))


def _distance(x: float, y: float, target: tuple[float, float]) -> float:
    return float(np.hypot(x - target[0], y - target[1]))


def _objective(x, y, args) -> float:
    if args.task == "cross-country":
        return -_distance(x, y, (float(args.target[0]), float(args.target[1])))
    return _distance(x, y, (float(args.start[0]), float(args.start[1])))


def _rollout_altitude(field, config, x, y, z, horizon, time_index, args):
    for h in range(horizon):
        position = GridPosition(x, y, z)
        if hasattr(field, "velocity_at_time"):
            u, v = field.velocity_at_time(position, min(time_index + h, field._T - 1))
        else:
            u, v = field.velocity_at(position)
        x, y = _clip(x + float(u), y + float(v), config)
    return _objective(x, y, args)


def _choose_altitude(field, config, x, y, candidates, horizon, time_index, args):
    scores = [
        (_rollout_altitude(field, config, x, y, z, horizon, time_index, args), z)
        for z in candidates
    ]
    return max(scores, key=lambda item: item[0])[1]


def _simulate(args):
    config = _config_from_cache(args.data)
    if config.ndim != 3:
        raise ValueError("cross-country MPC demo expects a 3D wind cache")

    field = ReanalysisFlowField(
        config,
        args.data,
        scale=args.scale,
        slice_mode="fixed",
        fixed_index=float(args.time_index),
    )
    field.reset(jax.random.PRNGKey(args.seed))

    x, y, z = args.start
    target = (float(args.target[0]), float(args.target[1]))
    candidates = np.linspace(1.0, float(config.n_z), args.altitude_candidates)
    path = []
    for step in range(args.steps + 1):
        dist = _distance(x, y, target)
        start_dist = _distance(x, y, (float(args.start[0]), float(args.start[1])))
        path.append(dict(step=step, x=x, y=y, z=z, distance=dist, start_distance=start_dist))
        if args.task == "cross-country" and dist <= args.target_radius:
            break
        if step == args.steps:
            break
        time_index = min(float(args.time_index) + step * args.time_delta, field._T - 1)
        z = _choose_altitude(
            field,
            config,
            x,
            y,
            candidates,
            args.horizon,
            time_index,
            args,
        )
        u, v = field.velocity_at_time(GridPosition(x, y, z), time_index)
        x, y = _clip(x + float(u), y + float(v), config)
    return config, path


def _write_html(config: GridConfig, path, args, output_path: Path) -> None:
    payload = dict(
        shape=list(config.shape),
        start=list(args.start),
        target=list(args.target),
        task=args.task,
        path=path,
        title=(
            "Cross-country navigation: first-order MPC"
            if args.task == "cross-country"
            else "Max-distance navigation: first-order MPC"
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{payload["title"]}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="plot" style="width:100%;height:92vh;"></div>
  <script>
    const data = {json.dumps(payload)};
    const xs = data.path.map(p => p.x);
    const ys = data.path.map(p => p.y);
    const zs = data.path.map(p => p.z);
    Plotly.newPlot("plot", [
      {{
        x: xs,
        y: ys,
        mode: "lines+markers",
        line: {{color: "#0f766e", width: 4}},
        marker: {{size: 6, color: zs, colorscale: "Viridis", colorbar: {{title: "z"}}}},
        name: "MPC path"
      }},
      {{
        x: [data.start[0]],
        y: [data.start[1]],
        mode: "markers",
        marker: {{size: 14, color: "#f59e0b"}},
        name: "start"
      }},
      ...(data.task === "cross-country" ? [{{
        x: [data.target[0]],
        y: [data.target[1]],
        mode: "markers",
        marker: {{size: 16, color: "#dc2626", symbol: "x"}},
        name: "target"
      }}] : [])
    ], {{
      title: data.title,
      paper_bgcolor: "#f8fafc",
      plot_bgcolor: "#eef3f8",
      xaxis: {{title: "x", range: [0.5, data.shape[0] + 0.5]}},
      yaxis: {{title: "y", range: [0.5, data.shape[1] + 0.5], scaleanchor: "x"}},
      margin: {{l: 60, r: 20, t: 60, b: 50}}
    }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cross-country", "max-distance"], default="cross-country")
    parser.add_argument("--data", required=True)
    parser.add_argument("--start", type=float, nargs=3, default=[8.0, 20.0, 4.0])
    parser.add_argument("--target", type=float, nargs=2, default=[36.0, 32.0])
    parser.add_argument("--target-radius", type=float, default=2.0)
    parser.add_argument("--time-index", type=float, default=0.0)
    parser.add_argument("--time-delta", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--altitude-candidates", type=int, default=7)
    parser.add_argument("--scale", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/output/cross_country_mpc.html")
    args = parser.parse_args()

    config, path = _simulate(args)
    output_path = Path(args.output)
    _write_html(config, path, args, output_path)
    final = path[-1]
    metric = final["distance"] if args.task == "cross-country" else final["start_distance"]
    print(f"wrote {output_path}")
    print(
        f"final=({final['x']:.2f}, {final['y']:.2f}, z={final['z']:.2f}) "
        f"metric={metric:.2f} steps={final['step']}"
    )


if __name__ == "__main__":
    main()
