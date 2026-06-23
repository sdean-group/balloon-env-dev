"""Watch an inert agent drift through a REAL-data (linear-interpolation) wind field.

Identical setup to ``viz_wind_drift.py`` (the GP demo), but the wind source is a
``ReanalysisFlowField`` backed by a regridded ERA5 ``.npz`` and queried via linear
interpolation. The agent always takes action 1 ("STAY"), so every bit of motion comes from
the interpolated wind dragging it along the ambient axis -- this is the quickest way to
confirm the real-data field works end to end inside the environment.

If the cache file is missing, this script generates a realistic DEMO cache (smooth, coherent
synthetic winds in m/s, no credentials needed) so you can run it immediately. Swap in a real
ERA5 cache by running ``fetch_era5.py --source cds`` and passing ``--data`` to this script.

Run:
    pixi run python experiments/viz_real_wind_drift.py                  # demo (auto-generated)
    pixi run python experiments/viz_real_wind_drift.py --data data/era5_sf_2d.npz --scale 0.06

Output: an interactive animated HTML (Play / Pause / scrubber) you open in a browser.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments", "field_estimation", "scripts"))

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
from src.env.field import ReanalysisFlowField
from fetch_era5 import make_demo_cache, save_cache  # noqa: E402

STAY = 1  # action 1 = do nothing on the controllable axis

N_X, N_Y = 80, 80
DEFAULT_DATA = os.path.join(ROOT, "data", "era5_demo_2d.npz")
# m/s -> grid cells/step. Demo winds average ~12 m/s; 0.06 gives ~0.7 cells/step of drift.
DEFAULT_SCALE = 0.06


def ensure_cache(data_path: str) -> None:
    """Generate the demo cache on first run -- but only for the default demo path.

    If you point --data at a real ERA5 cache that doesn't exist, that's an error (we won't
    silently fabricate demo data in its place).
    """
    if os.path.exists(data_path):
        return
    if os.path.abspath(data_path) != os.path.abspath(DEFAULT_DATA):
        raise FileNotFoundError(
            f"no cache at {data_path}. Build one with:\n"
            f"  pixi run python experiments/field_estimation/scripts/fetch_era5.py "
            f"--source cds --grid {N_X} {N_Y} --out {data_path}"
        )
    print(f"no cache at {data_path} -- generating a demo ERA5-shaped cache ...")
    winds, meta = make_demo_cache((N_X, N_Y), T=24, seed=7, mean_wind=12.0, std_wind=6.0)
    save_cache(data_path, winds, meta)
    print(f"  wrote demo cache, winds shape {winds.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA,
                        help="path to a wind .npz (demo or real ERA5 cache)")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE,
                        help="m/s -> grid cells/step multiplier")
    parser.add_argument("--random", action="store_true",
                        help="draw a random time slice each reset (default: fixed slice 0)")
    args = parser.parse_args()

    ensure_cache(args.data)
    config = GridConfig.create(n_x=N_X, n_y=N_Y)

    # Real-data wind field: ERA5 (or demo stand-in) via linear interpolation.
    slice_mode = "random" if args.random else "fixed"
    field = ReanalysisFlowField(config, args.data, scale=args.scale, slice_mode=slice_mode)
    print(f"field source : {args.data}  (scale={args.scale}, slice_mode={slice_mode})")

    # Actor with NO noise: with action=STAY the agent truly does nothing, so 100% of the
    # motion comes from the interpolated field.
    actor = GridActor(scale=1.0, noise_std=0.0, z_max=3.0)

    start = GridPosition(20, 40, None)
    target = GridPosition(70, 40, None)  # only matters for the reward overlay
    reward_fn = NavigationReward(
        target_position=target, vicinity_radius=10.0,
        peak_reward=10.0, step_cost=0.1, proximity_scale=0.1,
    )

    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=start, target_position=target,
        vicinity_radius=10.0, max_displacement=4.0,
        boundary_mode="clip", reward_fn=reward_fn,
        terminate_on_reach=False,
        process_noise_std=0.2, obs_noise_std=0.2,
    )

    renderer = NavigationRenderer(
        config=config, width=900, height=800,
        show_grid_points=True, field=field, show_field=True,
    )

    env = GridEnvironment(arena=arena, max_steps=60, seed=7, renderer=renderer)
    obs, info = env.reset(seed=7)

    print(f"start position : ({info['position'].i:.2f}, {info['position'].j:.2f})")
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(STAY)
        if terminated or truncated:
            break
    pos = info["position"]
    print(f"end position   : ({pos.i:.2f}, {pos.j:.2f})   <- fractional => continuous drift")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "real_wind_drift_animated.html")
    renderer.save_animated_html(out_path, fps=8)
    print(f"\nOpen this in a browser:\n  {out_path}")
    env.close()


if __name__ == "__main__":
    main()
