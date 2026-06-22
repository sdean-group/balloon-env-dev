"""Tests for continuous (float) positions and displacements.

These verify the int->float structural change: positions and displacements are
continuous floats end to end (no rounding in the live step path). The
SyntheticFlowField is the primary wind source under test since it is the one used
in real experiments.

Per the clean wind-field design, the field is a pure spatial source; the arena
owns clipping (``max_displacement``) and optional per-step noise.
"""

import jax
import numpy as np
import pytest

from src.env.environment import GridEnvironment
from src.env.arena.grid_arena import GridArena
from src.env.arena.navigation_arena import NavigationArena
from src.env.arena.reward import NavigationReward
from src.env.field.composite import ZeroField
from src.env.field.synthetic import SyntheticFlowField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_integer_valued(x: float) -> bool:
    return float(x) == round(float(x))


def _make_rff_nav_env(*, noise_std=0.5, max_displacement=3.0, n=20, max_steps=30,
                      seed=0, boundary_mode="clip"):
    config = GridConfig.create(n, n)
    # One synthetic field shared as realized + observed (perfect-forecast baseline).
    field = SyntheticFlowField(config, sigma=2.0, lengthscale=5.0, nu=2.5,
                               num_features=200)
    actor = GridActor(scale=1.5, noise_std=noise_std, z_max=3.0)
    target = GridPosition(n - 3, n - 3, None)
    reward_fn = NavigationReward(target_position=target, vicinity_radius=2.0,
                                 peak_reward=10.0, step_cost=0.1, proximity_scale=0.1)
    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=GridPosition(4, 4, None), target_position=target,
        vicinity_radius=2.0, max_displacement=max_displacement,
        boundary_mode=boundary_mode, reward_fn=reward_fn,
        process_noise_std=noise_std, obs_noise_std=noise_std,
    )
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


# ---------------------------------------------------------------------------
# 1. GridPosition holds floats
# ---------------------------------------------------------------------------

def test_grid_position_accepts_floats():
    p2 = GridPosition(3.7, 5.2, None)
    assert p2.ndim == 2
    assert p2.controllable == pytest.approx(5.2)
    assert p2.ambient == (pytest.approx(3.7),)

    p3 = GridPosition(3.7, 5.2, 1.5)
    assert p3.ndim == 3
    assert p3.controllable == pytest.approx(1.5)
    assert p3.ambient == (pytest.approx(3.7), pytest.approx(5.2))


# ---------------------------------------------------------------------------
# 2. Actor produces continuous displacement (no rounding)
# ---------------------------------------------------------------------------

def test_actor_deterministic_displacement_is_continuous():
    """noise_std=0, scale=1.5 => displacement exactly +/-1.5 (no rounding)."""
    actor = GridActor(scale=1.5, noise_std=0.0, z_max=3.0)
    key = jax.random.PRNGKey(0)

    up = actor.step_controllable(GridPosition(5.0, 5.0, None), action=2, rng_key=key)
    down = actor.step_controllable(GridPosition(5.0, 5.0, None), action=0, rng_key=key)
    stay = actor.step_controllable(GridPosition(5.0, 5.0, None), action=1, rng_key=key)

    assert up.controllable == pytest.approx(6.5)    # 5 + 1.5
    assert down.controllable == pytest.approx(3.5)  # 5 - 1.5
    assert stay.controllable == pytest.approx(5.0)


def test_actor_noisy_displacement_is_fractional():
    """With noise, displacement is generally non-integer."""
    actor = GridActor(scale=1.0, noise_std=0.5, z_max=3.0)
    fractional_seen = False
    for s in range(20):
        new = actor.step_controllable(
            GridPosition(5.0, 5.0, None), action=2, rng_key=jax.random.PRNGKey(s)
        )
        # respects clip bound
        disp = new.controllable - 5.0
        assert -3.0 <= disp <= 3.0
        if not _is_integer_valued(disp):
            fractional_seen = True
    assert fractional_seen


def test_actor_clip_to_z_max():
    """Huge intended displacement is clipped to z_max (continuous)."""
    actor = GridActor(scale=10.0, noise_std=0.0, z_max=2.5)
    new = actor.step_controllable(GridPosition(5.0, 5.0, None), action=2,
                                  rng_key=jax.random.PRNGKey(0))
    assert new.controllable == pytest.approx(7.5)  # 5 + clip(10, ., 2.5)


# ---------------------------------------------------------------------------
# 3. SyntheticFlowField works at fractional positions
# ---------------------------------------------------------------------------

def test_field_velocity_at_fractional_position():
    config = GridConfig.create(20, 20)
    field = SyntheticFlowField(config, sigma=2.0, lengthscale=5.0, nu=2.5,
                               num_features=200)
    field.reset(jax.random.PRNGKey(1))

    frac = GridPosition(7.3, 11.8, None)
    u, v = field.velocity_at(frac)

    assert v is None
    assert np.isfinite(u)

    # velocity_at matches the differentiable core at the same continuous point.
    u_point, _ = field.velocity_at_point(7.3, 11.8)
    assert u == pytest.approx(float(u_point), abs=1e-5)


def test_precompute_matches_continuous_at_integer_points():
    """At integer cells, continuous eval matches the precomputed grid velocity."""
    config = GridConfig.create(15, 15)
    field = SyntheticFlowField(config, sigma=1.5, lengthscale=4.0, nu=1.5,
                               num_features=256)
    field.reset(jax.random.PRNGKey(3))
    vel_field = field.velocity_field()  # precomputed grid

    for (i, j) in [(1, 1), (8, 8), (15, 15)]:
        u_cont, _ = field.velocity_at_point(float(i), float(j))
        assert float(u_cont) == pytest.approx(float(vel_field[i - 1, j - 1, 0]),
                                              rel=1e-4, abs=1e-6)


# ---------------------------------------------------------------------------
# 4. Arena step yields continuous (non-snapped) positions
# ---------------------------------------------------------------------------

def test_arena_step_position_is_continuous():
    env = _make_rff_nav_env(noise_std=0.5, seed=7)
    env.reset(seed=7)
    obs, _, _, _, info = env.step(2)
    pos = info["position"]
    # at least one coordinate should be fractional after a step
    assert not (_is_integer_valued(pos.i) and _is_integer_valued(pos.j))
    assert np.all(np.isfinite(obs))


def test_episode_visits_fractional_positions():
    env = _make_rff_nav_env(noise_std=0.5, max_steps=30, seed=11)
    env.reset(seed=11)
    saw_fractional = False
    for t in range(30):
        _, _, term, trunc, info = env.step(t % 3)
        pos = info["position"]
        if not _is_integer_valued(pos.i) or not _is_integer_valued(pos.j):
            saw_fractional = True
        if term or trunc:
            break
    assert saw_fractional


# ---------------------------------------------------------------------------
# 5. Boundary enforcement on fractional positions
# ---------------------------------------------------------------------------

def test_clip_boundary_on_fractional_positions():
    config = GridConfig.create(10, 8)
    field = ZeroField(config)
    arena = GridArena(realized_field=field, observed_field=field,
                      actor=GridActor(noise_std=0.0),
                      config=config, initial_position=GridPosition(5.0, 4.0, None),
                      max_displacement=3.0, boundary_mode="clip")

    # interior fractional stays put
    pos_in, oob_in = arena._enforce_boundaries_2d(GridPosition(7.3, 5.6, None))
    assert pos_in.i == pytest.approx(7.3)
    assert pos_in.j == pytest.approx(5.6)
    assert oob_in is False

    # fractional overshoot clamps to the float boundary
    pos_out, oob_out = arena._enforce_boundaries_2d(GridPosition(10.5, 8.5, None))
    assert pos_out.i == pytest.approx(10.0)
    assert pos_out.j == pytest.approx(8.0)
    assert oob_out is True


def test_terminal_boundary_flags_fractional_oob():
    config = GridConfig.create(10, 8)
    field = ZeroField(config)
    arena = GridArena(realized_field=field, observed_field=field,
                      actor=GridActor(noise_std=0.0),
                      config=config, initial_position=GridPosition(5.0, 4.0, None),
                      max_displacement=3.0, boundary_mode="terminal")
    _, oob = arena._enforce_boundaries_2d(GridPosition(0.5, 4.0, None))
    assert oob is True
    _, oob_in = arena._enforce_boundaries_2d(GridPosition(1.5, 4.0, None))
    assert oob_in is False


# ---------------------------------------------------------------------------
# 6. Clip bound is enforced by the arena step
# ---------------------------------------------------------------------------

def test_arena_clips_displacement_to_max_displacement():
    """The realized displacement applied per step never exceeds max_displacement."""
    n = 30
    config = GridConfig.create(n, n)
    field = SyntheticFlowField(config, sigma=10.0, lengthscale=2.0, nu=2.5,
                               num_features=200)
    # action=1 (stay) keeps the controllable axis fixed regardless of scale;
    # we only check the ambient (i) displacement, which comes from the field.
    actor = GridActor(scale=1.5, noise_std=0.0, z_max=3.0)
    target = GridPosition(n - 3, n - 3, None)
    reward_fn = NavigationReward(target_position=target, vicinity_radius=2.0,
                                 peak_reward=10.0, step_cost=0.1, proximity_scale=0.1)
    max_disp = 1.0
    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=GridPosition(15, 15, None), target_position=target,
        vicinity_radius=2.0, max_displacement=max_disp, boundary_mode="periodic",
        reward_fn=reward_fn,
    )
    arena.reset(jax.random.PRNGKey(0))
    for _ in range(10):
        prev = arena.position
        arena.step(1)
        di = abs(arena.position.i - prev.i)
        # account for periodic wrap on i
        di = min(di, n - di)
        assert di <= max_disp + 1e-6
        assert abs(arena.last_displacement.u) <= max_disp + 1e-6


# ---------------------------------------------------------------------------
# 7. Determinism still holds with continuous values
# ---------------------------------------------------------------------------

def test_continuous_trajectory_is_deterministic():
    actions = [t % 3 for t in range(20)]
    trajs = []
    for _ in range(2):
        env = _make_rff_nav_env(noise_std=0.5, seed=999)
        obs, _ = env.reset(seed=999)
        traj = [obs.copy()]
        for a in actions:
            obs, _, term, trunc, _ = env.step(a)
            traj.append(obs.copy())
            if term or trunc:
                break
        trajs.append(traj)
    for t1, t2 in zip(trajs[0], trajs[1]):
        np.testing.assert_array_equal(t1, t2)
