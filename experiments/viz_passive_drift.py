"""Animate a passive balloon in synthetic, ERA5, or data-driven GP wind.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import (
    DataDrivenFlowField,
    GridActor,
    GridConfig,
    GridEnvironment,
    GridPosition,
    NavigationArena,
    NavigationRenderer,
    NavigationReward,
    ReanalysisFlowField,
    SyntheticFlowField,
)
from src.env.field.era5_data import load_era5


STAY = 1
FIELD_CHOICES = ("synthetic", "era5", "data-driven-gp")


def _config_from_args(args) -> GridConfig:
    if args.data:
        winds = load_era5(args.data).winds
        spatial_shape = winds.shape[1:-1]
        return GridConfig.create(*spatial_shape)
    if args.field != "synthetic":
        raise ValueError("--data is required for ERA5 and data-driven GP fields")
    return GridConfig.create(*args.grid)


def _position(values: Sequence[float] | None, config: GridConfig) -> GridPosition:
    if values is None:
        coords = tuple((n + 1.0) / 2.0 for n in config.shape)
    else:
        if len(values) != config.ndim:
            raise ValueError(
                f"--start needs {config.ndim} coordinates for this cache, got {len(values)}"
            )
        coords = tuple(float(value) for value in values)
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


def _run_one(name: str, config: GridConfig, start: GridPosition, args) -> Path:
    field = _build_field(name, config, args)

    # z_max=0 and noise_std=0 make the actor inert. The repeated STAY action is
    # only required by the environment API; it cannot change the balloon state.
    actor = GridActor(scale=1.0, noise_std=0.0, z_max=0.0)
    target_coords = [max(1.0, n - 2.0) for n in config.shape]
    target = (
        GridPosition(target_coords[0], target_coords[1], None)
        if config.ndim == 2
        else GridPosition(target_coords[0], target_coords[1], start.k)
    )
    reward_fn = NavigationReward(
        target_position=target,
        vicinity_radius=2.0,
        peak_reward=10.0,
        step_cost=0.1,
        proximity_scale=0.1,
    )
    max_displacement = min(
        float(args.max_displacement),
        float(min(config.n_ambient)) - 1e-6,
    )
    arena = NavigationArena(
        realized_field=field,
        observed_field=field,
        actor=actor,
        config=config,
        initial_position=start,
        target_position=target,
        vicinity_radius=2.0,
        max_displacement=max_displacement,
        boundary_mode=args.boundary,
        reward_fn=reward_fn,
        terminate_on_reach=False,
        process_noise_std=0.0,
        obs_noise_std=0.0,
    )
    renderer = NavigationRenderer(
        config=config,
        width=960,
        height=760,
        show_grid_points=True,
        field=field,
        show_field=True,
    )
    env = GridEnvironment(
        arena=arena,
        max_steps=args.steps,
        seed=args.seed,
        renderer=renderer,
    )
    _, info = env.reset(seed=args.seed)
    for _ in range(args.steps):
        _, _, terminated, truncated, info = env.step(STAY)
        if terminated or truncated:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"passive_drift_{name.replace('-', '_')}.html"
    renderer.save_animated_html(str(output_path), fps=args.fps)
    end = info["position"]
    print(
        f"{name:>14}: start={tuple(round(v, 2) for v in start.ambient)} "
        f"end={tuple(round(v, 2) for v in end.ambient)} -> {output_path}"
    )
    if isinstance(field, DataDrivenFlowField):
        print(
            f"                trained on {field.training_points} points; "
            f"training RMSE={field.training_rmse:.4f} cells/step"
        )
    env.close()
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--field",
        choices=(*FIELD_CHOICES, "all"),
        default="synthetic",
    )
    parser.add_argument("--data", help="ERA5-style .npz cache for era5/GP modes")
    parser.add_argument("--grid", type=int, nargs="+", default=[60, 60])
    parser.add_argument("--start", type=float, nargs="+", help="drop coordinates x y [z]")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--scale", type=float, default=0.06,
                        help="convert cached m/s values to grid cells per step")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--boundary", choices=["clip", "periodic", "terminal"], default="clip")
    parser.add_argument("--max-displacement", type=float, default=4.0)
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
