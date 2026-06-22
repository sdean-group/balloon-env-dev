"""Watch the GP wind field push an inert agent around.

The agent always takes action 1 ("STAY"), so it does *nothing* on its own
(controllable) axis. Every bit of motion you see is the environmental field
dragging it along the ambient axis. This is the simplest way to confirm the
continuous (float) dynamics work: the trajectory drifts smoothly through
fractional positions instead of hopping between integer cells.

Run:
    pixi run python experiments/viz_wind_drift.py

Output: an interactive animated HTML (Play / Pause / scrubber) you open in a
browser. Blue arrows = the field's mean displacement (the "wind"); the red
diamond = the agent; the blue line = its drift trajectory.

2D setting note: there is ONE ambient axis (x) and one controllable axis (y).
The wind therefore points along x only, so a do-nothing agent drifts
horizontally while holding its y position. (For a full 2-component (u, v) wind
map, build a 3D config instead -- ambient (x, y), controllable z = altitude.)
"""

import os
import sys

# Allow running directly (`python experiments/viz_wind_drift.py`) by putting the
# project root on the import path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    NavigationRenderer,
    GridConfig,
    GridPosition,
)
from src.env.field import SyntheticFlowField

STAY = 1  # action 1 = do nothing on the controllable axis


def main() -> None:
    config = GridConfig.create(n_x=80, n_y=80)

    # GP wind field: smooth, clearly non-zero mean so the drift is visible.
    # Low noise_std so the motion is dominated by the (fixed-per-episode) field
    # rather than observation jitter -- makes the drift easy to read.
    field = SyntheticFlowField(
        config,
        sigma=3.0,          # wind amplitude
        lengthscale=20.0,   # large => smooth, slowly-varying wind
        nu=2.5,
        num_features=400,
    )

    # Actor with NO noise: with action=STAY the agent truly does nothing,
    # so 100% of the motion comes from the field.
    actor = GridActor(scale=1.0, noise_std=0.0, z_max=3.0)

    start = GridPosition(40, 40, None)
    target = GridPosition(70, 40, None)  # only matters for the reward overlay
    reward_fn = NavigationReward(
        target_position=target, vicinity_radius=10.0,
        peak_reward=10.0, step_cost=0.1, proximity_scale=0.1,
    )

    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=start, target_position=target,
        vicinity_radius=10.0, max_displacement=4.0,  # continuous clip bound per step
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
    out_path = os.path.join(out_dir, "wind_drift_animated.html")
    renderer.save_animated_html(out_path, fps=8)
    print(f"\nOpen this in a browser:\n  {out_path}")
    env.close()


if __name__ == "__main__":
    main()
