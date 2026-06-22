"""Watch an inert agent drift through a REAL-data (linear-interpolation) wind field.

Identical setup to ``viz_wind_drift.py`` (the GP demo), but the wind source is a
``ReanalysisFlowField`` backed by a regridded ERA5 ``.npz`` and queried via linear
interpolation. The agent always takes action 1 ("STAY"), so every bit of motion comes from
the interpolated wind dragging it along the ambient axis -- this is the quickest way to
confirm the real-data field works end to end inside the environment.

If the cache file is missing, this script generates a realistic DEMO cache (smooth, coherent
synthetic winds in m/s, no credentials needed) so you can run it immediately. Swap in a real
ERA5 cache by running ``fetch_era5.py --source cds`` and pointing ``DATA_PATH`` at it.

Run:
    pixi run python experiments/viz_real_wind_drift.py

Output: an interactive animated HTML (Play / Pause / scrubber) you open in a browser.
"""

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
DATA_PATH = os.path.join(ROOT, "data", "era5_demo_2d.npz")
# m/s -> grid cells/step. Demo winds average ~12 m/s; 0.06 gives ~0.7 cells/step of drift.
SCALE = 0.06


def ensure_cache() -> None:
    """Generate the demo cache on first run so the script is zero-setup."""
    if os.path.exists(DATA_PATH):
        return
    print(f"no cache at {DATA_PATH} -- generating a demo ERA5-shaped cache ...")
    winds, meta = make_demo_cache((N_X, N_Y), T=24, seed=7, mean_wind=12.0, std_wind=6.0)
    save_cache(DATA_PATH, winds, meta)
    print(f"  wrote demo cache, winds shape {winds.shape}")


def main() -> None:
    ensure_cache()
    config = GridConfig.create(n_x=N_X, n_y=N_Y)

    # Real-data wind field: ERA5 (here, demo stand-in) via linear interpolation.
    # slice_mode="fixed" -> always use time slice 0, so the demo is reproducible.
    field = ReanalysisFlowField(config, DATA_PATH, scale=SCALE, slice_mode="fixed")

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
