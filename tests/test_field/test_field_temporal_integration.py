"""Integration tests: time-varying fields inside the real environment.

These prove the end-to-end wiring added for in-episode weather evolution: the arena
passes its step count as ``t`` into ``velocity_at``, so a field that depends on ``t``
produces a trajectory that *evolves during the episode*. Process/observation noise is
zeroed so motion is 100% field-driven and assertions are deterministic.
"""

import jax
import numpy as np

from src.env import (
    GridEnvironment,
    NavigationArena,
    NavigationReward,
    GridActor,
    GridConfig,
    GridPosition,
)
from src.env.field import ReanalysisFlowField, SyntheticFlowField

STAY = 1  # action 1 = do nothing on the controllable axis


def _build_env(field, config, *, max_steps=6, seed=7):
    actor = GridActor(scale=1.0, noise_std=0.0, z_max=3.0)
    reward_fn = NavigationReward(
        target_position=GridPosition(10.0, 4.0), vicinity_radius=10.0,
        peak_reward=10.0, step_cost=0.1, proximity_scale=0.1,
    )
    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=GridPosition(3.0, 4.0), target_position=GridPosition(10.0, 4.0),
        vicinity_radius=10.0, max_displacement=4.0, boundary_mode="clip",
        reward_fn=reward_fn, terminate_on_reach=False,
        process_noise_std=0.0, obs_noise_std=0.0,
    )
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


def _run(env, n):
    info = env.reset(seed=7)[1]
    positions = [info["position"].i]
    for _ in range(n):
        info = env.step(STAY)[4]
        positions.append(info["position"].i)
    env.close()
    return np.array(positions)


def test_arena_drives_field_time(affine_npz_2d):
    """A temporal field drifts differently than a frozen one -> proves the arena feeds t in.

    The affine fixture grows +10 per slice, so as episode time advances the ambient wind
    strengthens. If the arena left t pinned at 0 the two trajectories would be identical.
    """
    case = affine_npz_2d
    config = GridConfig.create(case["n_x"], case["n_y"])

    static = ReanalysisFlowField(config, case["path"], scale=0.05, slice_mode="fixed")
    temporal = ReanalysisFlowField(
        config, case["path"], scale=0.05, slice_mode="fixed", steps_per_slice=2.0
    )

    static_traj = _run(_build_env(static, config), 6)
    temporal_traj = _run(_build_env(temporal, config), 6)

    # Same start (t=0 -> slice 0 for both); diverging thereafter as the temporal wind grows.
    assert static_traj[0] == temporal_traj[0]
    assert temporal_traj[-1] > static_traj[-1] + 1e-3


def test_temporal_trajectory_is_reproducible(affine_npz_2d):
    """Identical seeds -> identical evolving trajectory (determinism survives the time axis)."""
    case = affine_npz_2d
    config = GridConfig.create(case["n_x"], case["n_y"])

    def fresh():
        f = ReanalysisFlowField(
            config, case["path"], scale=0.05, slice_mode="fixed", steps_per_slice=2.0
        )
        return _run(_build_env(f, config), 6)

    np.testing.assert_array_equal(fresh(), fresh())


def test_temporal_gp_runs_in_arena_and_evolves(affine_npz_2d):
    """A time-varying SyntheticFlowField drives the arena: finite, reproducible, evolving."""
    config = GridConfig.create(12, 9)
    field = SyntheticFlowField(
        config, sigma=1.5, lengthscale=4.0, num_features=128, lengthscale_t=5.0
    )
    traj_a = _run(_build_env(field, config, max_steps=10), 10)

    field_b = SyntheticFlowField(
        config, sigma=1.5, lengthscale=4.0, num_features=128, lengthscale_t=5.0
    )
    traj_b = _run(_build_env(field_b, config, max_steps=10), 10)

    assert np.all(np.isfinite(traj_a))
    np.testing.assert_array_equal(traj_a, traj_b)          # same seed -> same trajectory
    # The field evolves, so steps are not a single constant drift repeated.
    steps = np.diff(traj_a)
    assert np.ptp(steps) > 1e-4
