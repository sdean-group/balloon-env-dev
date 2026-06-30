"""Compare MPC and PPO on the max-distance-from-start task.

The output is one HTML page with both trajectories overlaid. This is meant as a
demo/diagnostic view, not a statistically rigorous benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.cross_country_navigation_mpc import _simulate as simulate_mpc
from experiments.navigation_ppo import _train as train_ppo


def _metric(path, start):
    final = path[-1]
    return ((final["x"] - start[0]) ** 2 + (final["y"] - start[1]) ** 2) ** 0.5


def _ppo_args(args) -> Namespace:
    return Namespace(
        task="max-distance",
        data=args.data,
        start=args.start,
        target=args.target,
        target_radius=2.0,
        time_index=args.time_index,
        time_delta=args.time_delta,
        steps=args.steps,
        scale=args.scale,
        altitude_candidates=args.altitude_candidates,
        updates=args.ppo_updates,
        episodes_per_update=args.ppo_episodes_per_update,
        epochs=args.ppo_epochs,
        hidden=args.ppo_hidden,
        learning_rate=args.ppo_learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        seed=args.seed,
        output=str(args.output),
    )


def _mpc_args(args) -> Namespace:
    return Namespace(
        task="max-distance",
        data=args.data,
        start=args.start,
        target=args.target,
        target_radius=2.0,
        time_index=args.time_index,
        time_delta=args.time_delta,
        steps=args.steps,
        horizon=args.mpc_horizon,
        altitude_candidates=args.altitude_candidates,
        scale=args.scale,
        seed=args.seed,
        output=str(args.output),
    )


def _write_html(config, mpc_path, ppo_path, args, output: Path) -> None:
    payload = {
        "shape": list(config.shape),
        "start": list(args.start),
        "mpcPath": mpc_path,
        "ppoPath": ppo_path,
        "mpcDistance": _metric(mpc_path, args.start),
        "ppoDistance": _metric(ppo_path, args.start),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Max-distance: MPC vs PPO</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <div id="plot" style="width:100%;height:92vh;"></div>
  <script>
    const data = {json.dumps(payload)};
    function trace(path, name, color) {{
      return {{
        x: path.map(p => p.x),
        y: path.map(p => p.y),
        mode: "lines+markers",
        line: {{color, width: 4}},
        marker: {{size: 6}},
        name
      }};
    }}
    Plotly.newPlot("plot", [
      trace(data.mpcPath, `MPC (${{data.mpcDistance.toFixed(2)}} cells)`, "#0f766e"),
      trace(data.ppoPath, `PPO (${{data.ppoDistance.toFixed(2)}} cells)`, "#7c3aed"),
      {{
        x: [data.start[0]],
        y: [data.start[1]],
        mode: "markers",
        marker: {{size: 16, color: "#f59e0b", line: {{color: "#92400e", width: 2}}}},
        name: "start"
      }}
    ], {{
      title: "Max-distance from start: first-order MPC vs PPO",
      paper_bgcolor: "#f8fafc",
      plot_bgcolor: "#eef3f8",
      xaxis: {{title: "x", range: [0.5, data.shape[0] + 0.5]}},
      yaxis: {{title: "y", range: [0.5, data.shape[1] + 0.5], scaleanchor: "x"}},
      legend: {{orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1}},
      margin: {{l: 60, r: 20, t: 70, b: 50}}
    }});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/era5_ithaca_3d.npz")
    parser.add_argument("--start", type=float, nargs=3, default=[8.0, 20.0, 4.0])
    parser.add_argument("--target", type=float, nargs=2, default=[35.0, 35.0])
    parser.add_argument("--time-index", type=float, default=0.0)
    parser.add_argument("--time-delta", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--scale", type=float, default=0.03)
    parser.add_argument("--altitude-candidates", type=int, default=7)
    parser.add_argument("--mpc-horizon", type=int, default=8)
    parser.add_argument("--ppo-updates", type=int, default=15)
    parser.add_argument("--ppo-episodes-per-update", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--ppo-hidden", type=int, default=64)
    parser.add_argument("--ppo-learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="experiments/output/max_distance_mpc_vs_ppo.html")
    args = parser.parse_args()

    config, mpc_path = simulate_mpc(_mpc_args(args))
    ppo_config, ppo_path = train_ppo(_ppo_args(args))
    if tuple(config.shape) != tuple(ppo_config.shape):
        raise RuntimeError("MPC and PPO configs do not match")

    output = Path(args.output)
    _write_html(config, mpc_path, ppo_path, args, output)
    print(f"wrote {output}")
    print(f"mpc_distance={_metric(mpc_path, args.start):.2f}")
    print(f"ppo_distance={_metric(ppo_path, args.start):.2f}")


if __name__ == "__main__":
    main()

