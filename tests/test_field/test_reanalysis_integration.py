"""Integration tests: ReanalysisFlowField inside the real environment (criterion 10).

These confirm the field drops into NavigationArena exactly where SyntheticFlowField does
(see experiments/viz_wind_drift.py) and that a do-nothing agent is genuinely carried by the
interpolated wind. Process/observation noise is zeroed so motion is 100% field-driven and the
assertions are deterministic.
"""

import numpy as np

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    GridConfig,
    GridPosition,
)
from src.env.field import ReanalysisFlowField

STAY = 1  # action 1 = do nothing on the controllable axis


def _build_env(field, config, start, target, *, max_steps=8):
    actor = GridActor(scale=1.0, noise_std=0.0, z_max=3.0)
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
        process_noise_std=0.0, obs_noise_std=0.0,
    )
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=7)


def test_inert_agent_drifts(affine_npz_2d):
    """The affine field is strictly positive on the ambient axis -> STAY must drift +x."""
    config = GridConfig.create(affine_npz_2d["n_x"], affine_npz_2d["n_y"])
    field = ReanalysisFlowField(config, affine_npz_2d["path"], scale=0.05, slice_mode="fixed")
    start, target = GridPosition(3.0, 4.0), GridPosition(10.0, 4.0)
    env = _build_env(field, config, start, target)

    _, info = env.reset(seed=7)
    start_i = info["position"].i
    for _ in range(8):
        _, _, terminated, truncated, info = env.step(STAY)
        if terminated or truncated:
            break
    end = info["position"]
    assert end.i - start_i > 1e-3            # carried downwind on the ambient axis
    assert abs(end.j - start.j) < 1e-6       # STAY -> controllable axis unchanged
    env.close()


def test_trajectory_stays_in_bounds_under_clip(affine_npz_2d):
    config = GridConfig.create(affine_npz_2d["n_x"], affine_npz_2d["n_y"])
    field = ReanalysisFlowField(config, affine_npz_2d["path"], scale=0.2, slice_mode="fixed")
    start, target = GridPosition(3.0, 4.0), GridPosition(10.0, 4.0)
    env = _build_env(field, config, start, target, max_steps=20)

    _, info = env.reset(seed=7)
    for _ in range(20):
        _, _, terminated, truncated, info = env.step(STAY)
        p = info["position"]
        assert 1.0 <= p.i <= affine_npz_2d["n_x"]
        assert 1.0 <= p.j <= affine_npz_2d["n_y"]
        assert np.isfinite(p.i) and np.isfinite(p.j)
        if terminated or truncated:
            break
    env.close()


def test_swaps_in_for_synthetic(affine_npz_2d):
    """Same arena wiring as the GP demo, just a different field source -> runs, stays finite."""
    config = GridConfig.create(affine_npz_2d["n_x"], affine_npz_2d["n_y"])
    field = ReanalysisFlowField(config, affine_npz_2d["path"], scale=0.05, slice_mode="fixed")
    env = _build_env(field, config, GridPosition(3.0, 4.0), GridPosition(10.0, 4.0))

    obs, info = env.reset(seed=7)
    assert np.all(np.isfinite(np.asarray(obs)))
    for _ in range(8):
        obs, reward, terminated, truncated, info = env.step(STAY)
        assert np.all(np.isfinite(np.asarray(obs)))
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    env.close()
