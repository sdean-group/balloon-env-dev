"""Parameterized end-to-end episode integration tests.

Reward under test is provided by reward_helpers (make_reward_fn). Change
reward_helpers.py to switch reward; tests stay reward-agnostic.
Covers: grid/observation contracts, termination, determinism, state, actor PMF, constructor validation.
"""

import pytest
import numpy as np
import jax

from src.env.environment import GridEnvironment
from src.env.arena.grid_arena import GridArena
from src.env.arena.navigation_arena import NavigationArena
from src.env.field.simple_field import UniformDriftField
from src.env.field.composite import ZeroField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition

from tests.test_arena.reward_helpers import make_reward_fn


# =============================================================================
# Helpers (reward-agnostic: use reward_helpers for nav env reward)
# =============================================================================

def _make_grid_env(config, initial, *, boundary_mode="clip", d_max=1,
                   max_steps=100, seed=42, **actor_kw) -> GridEnvironment:
    field = UniformDriftField(config, max_drift=d_max)
    actor = GridActor(**{"noise_std": 0.0, **actor_kw})
    arena = GridArena(realized_field=field, observed_field=field, actor=actor,
                      config=config, initial_position=initial,
                      max_displacement=d_max, boundary_mode=boundary_mode)
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


_REWARD_KEYS = frozenset(("peak_reward", "step_cost", "proximity_scale"))


def _make_nav_env(config, initial, target, *, vicinity_radius=2.0,
                  reward_fn=None, d_max=1, max_steps=100, seed=42, boundary_mode="clip",
                  terminate_on_reach=False, **kwargs) -> GridEnvironment:
    """Build nav env; reward comes from make_reward_fn unless reward_fn is given.
    kwargs: reward params (peak_reward, step_cost, proximity_scale) go to make_reward_fn;
    the rest go to GridActor (e.g. noise_std).
    """
    field = UniformDriftField(config, max_drift=d_max)
    reward_kw = {k: v for k, v in kwargs.items() if k in _REWARD_KEYS}
    actor_kw = {k: v for k, v in kwargs.items() if k not in _REWARD_KEYS}
    actor = GridActor(**{"noise_std": 0.0, **actor_kw})
    if reward_fn is None:
        reward_fn = make_reward_fn(target, vicinity_radius, **reward_kw)
    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=initial, target_position=target,
        vicinity_radius=vicinity_radius, max_displacement=d_max,
        boundary_mode=boundary_mode,
        reward_fn=reward_fn,
        terminate_on_reach=terminate_on_reach,
    )
    return GridEnvironment(arena=arena, max_steps=max_steps, seed=seed)


# =============================================================================
# Grid shape / observation contract (parameterized)
# =============================================================================

GRID_CASES = [
    # (n_x, n_y, n_z, d_max, desc)
    (1, 1, None, 0, "1x1-2d"),
    (1, 1, 1, 0, "1x1x1-3d"),
    (1, 1, 10, 0, "1x1x10-corridor"),
    (20, 1, None, 1, "20x1-horiz-2d"),
    (5, 5, None, 2, "5x5-2d"),
    (5, 5, 5, 2, "5x5x5-3d"),
    (100, 100, None, 10, "100x100-2d"),
    (100, 5, 3, 2, "100x5x3-wide-3d"),
    (5, 5, 100, 2, "5x5x100-tall-3d"),
    (200, 100, None, 10, "200x100-large-2d"),
]


@pytest.mark.parametrize(
    ("n_x", "n_y", "n_z", "d_max", "desc"), GRID_CASES, ids=lambda *a: str(a[-1])
)
def test_observation_and_bounds_contract(n_x, n_y, n_z, d_max, desc):
    """Observation shape, dtype, bounds, and action space for all grid configs."""
    config = GridConfig.create(n_x, n_y, n_z)
    ndim = config.ndim
    if ndim == 3:
        initial = GridPosition((n_x+1)//2, (n_y+1)//2, (n_z+1)//2)
    else:
        initial = GridPosition((n_x+1)//2, (n_y+1)//2, None)

    env = _make_grid_env(config, initial, d_max=d_max, max_steps=20)
    obs, info = env.reset(seed=0)

    # Shape
    expected_dim = 5 if ndim == 3 else 3
    assert obs.shape == (expected_dim,)
    assert obs.dtype == np.float32
    assert env.observation_space.shape == (expected_dim,)
    assert env.action_space.n == 3

    # Observation space bounds
    low = env.observation_space.low
    high = env.observation_space.high
    assert low[0] == 1 and high[0] == n_x
    assert low[1] == 1 and high[1] == n_y
    if ndim == 3:
        assert low[2] == 1 and high[2] == n_z
        assert low[3] == -d_max and high[3] == d_max
        assert low[4] == -d_max and high[4] == d_max
    else:
        assert low[2] == -d_max and high[2] == d_max

    # Run 10 steps: position always in bounds (clip mode)
    for _ in range(10):
        obs, _, terminated, truncated, info = env.step(np.random.randint(0, 3))
        pos = info["position"]
        assert 1 <= pos.i <= n_x
        assert 1 <= pos.j <= n_y
        if ndim == 3:
            assert 1 <= pos.k <= n_z
        if terminated or truncated:
            break


# =============================================================================
# Termination contracts (parameterized)
# =============================================================================

@pytest.mark.parametrize("max_steps", [1, 5, 50])
def test_truncation_at_max_steps(max_steps):
    """Episode truncates exactly at max_steps."""
    config = GridConfig.create(10, 10, 10)
    env = _make_grid_env(config, GridPosition(5, 5, 5), d_max=1, max_steps=max_steps)
    env.reset(seed=0)

    steps = 0
    truncated = False
    while not truncated:
        _, _, terminated, truncated, _ = env.step(1)
        steps += 1
        assert not terminated
    assert steps == max_steps


def test_terminal_boundary_triggers_termination():
    """Terminal mode ends episode when position leaves grid."""
    config = GridConfig.create(3, 3)
    initial = GridPosition(1, 1, None)  # corner — easy to push OOB
    env = _make_grid_env(config, initial, boundary_mode="terminal", d_max=2,
                         max_steps=200, seed=42)
    env.reset(seed=42)

    terminated = False
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(0)  # push down
        if terminated:
            assert info["out_of_bounds"] is True
            break
    assert terminated or truncated


@pytest.mark.parametrize("ndim", [2, 3])
def test_terminate_on_reach(ndim):
    """Episode terminates when target vicinity is reached."""
    n = 10
    if ndim == 2:
        config = GridConfig.create(n, n)
        initial = GridPosition(5, 1, None)
        target = GridPosition(5, 5, None)
    else:
        config = GridConfig.create(n, n, n)
        initial = GridPosition(5, 5, 1, )
        target = GridPosition(5, 5, 5)

    env = _make_nav_env(config, initial, target, vicinity_radius=1.5,
                        terminate_on_reach=True, max_steps=100, d_max=0)
    env.reset(seed=0)

    terminated = False
    for _ in range(20):
        _, _, terminated, _, info = env.step(2)  # move up on controllable axis
        if terminated:
            assert info["target_reached"] is True
            break
    assert terminated


# =============================================================================
# Determinism
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_deterministic_replay(ndim):
    """Same seed + same actions = identical trajectory."""
    n = 10
    if ndim == 2:
        config = GridConfig.create(n, n)
        initial = GridPosition(5, 5, None)
        target = GridPosition(8, 8, None)
    else:
        config = GridConfig.create(n, n, n)
        initial = GridPosition(5, 5, 5)
        target = GridPosition(8, 8, 8)

    actions = [step % 3 for step in range(20)]
    trajs = []
    for _ in range(2):
        env = _make_nav_env(config, initial, target, seed=12345, d_max=2)
        obs, _ = env.reset(seed=12345)
        traj = [obs.copy()]
        for a in actions:
            obs, _, _, _, _ = env.step(a)
            traj.append(obs.copy())
        trajs.append(traj)

    for t1, t2 in zip(trajs[0], trajs[1]):
        np.testing.assert_array_equal(t1, t2)


@pytest.mark.parametrize("ndim", [2, 3])
def test_different_seeds_diverge(ndim):
    """Different seeds produce different trajectories."""
    n = 10
    if ndim == 2:
        config = GridConfig.create(n, n)
        init = GridPosition(5, 5, None)
    else:
        config = GridConfig.create(n, n, n)
        init = GridPosition(5, 5, 5)

    obs_list = []
    for seed in [111, 222]:
        env = _make_grid_env(config, init, d_max=2, seed=seed)
        env.reset(seed=seed)
        for _ in range(10):
            obs, _, _, _, _ = env.step(1)
        obs_list.append(obs)

    assert not np.allclose(obs_list[0], obs_list[1])


# =============================================================================
# State consistency
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_step_count_and_reset(ndim):
    """step_count increments each step and resets to 0."""
    n = 10
    if ndim == 2:
        config = GridConfig.create(n, n)
        init = GridPosition(5, 5, None)
        targ = GridPosition(8, 8, None)
    else:
        config = GridConfig.create(n, n, n)
        init = GridPosition(5, 5, 5)
        targ = GridPosition(8, 8, 8)

    env = _make_nav_env(config, init, targ, d_max=1)
    _, info = env.reset(seed=0)
    assert info["step_count"] == 0

    for expected in range(1, 11):
        _, _, _, _, info = env.step(1)
        assert info["step_count"] == expected

    _, info = env.reset(seed=0)
    assert info["step_count"] == 0
    assert info["cumulative_reward"] == 0.0
    assert info["target_reached"] is False


# =============================================================================
# Actor PMF contract (new clipped-Gaussian model)
# =============================================================================

ACTOR_CASES = [
    # (scale, noise_std, z_max, desc)
    (1.0, 0.0, 1, "deterministic-default"),
    (1.0, 0.1, 1, "noisy-default"),
    (1.0, 0.0, 0, "z_max_zero"),
    (1.0, 0.5, 3, "wide-support"),
    (2.0, 0.1, 2, "scale2"),
    (0.5, 0.3, 2, "scale_half"),
    (1.0, 1e-8, 1, "tiny-noise"),
    (1.0, 5.0, 1, "huge-noise"),
]


@pytest.mark.parametrize(
    ("scale", "noise_std", "z_max", "desc"), ACTOR_CASES, ids=lambda *a: str(a[-1])
)
def test_actor_pmf_contracts(scale, noise_std, z_max, desc):
    """Actor PMF: shape, non-negativity, normalisation, symmetry for stay action."""
    actor = GridActor(scale=scale, noise_std=noise_std, z_max=z_max)
    pmf = actor.get_controllable_displacement_pmf()

    n_actions = 3
    n_displacements = 2 * z_max + 1

    # Shape
    assert pmf.shape == (n_actions, n_displacements)

    # Non-negative and finite
    assert np.all(np.isfinite(pmf))
    assert np.all(pmf >= 0.0)

    # Each row sums to 1
    for a in range(n_actions):
        assert float(np.sum(pmf[a])) == pytest.approx(1.0, abs=1e-5)

    # Deterministic case: all mass on one bin
    if noise_std == 0.0 and z_max > 0:
        for a in range(n_actions):
            assert np.count_nonzero(pmf[a]) == 1

    # Stay action (a=1) is symmetric around 0 when scale is any value
    # (mean = scale * 0 = 0, noise is symmetric)
    if z_max > 0 and noise_std > 0:
        stay_pmf = pmf[1]
        assert np.allclose(stay_pmf, stay_pmf[::-1], atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"scale": 0.0}, "scale must be positive"),
        ({"scale": -1.0}, "scale must be positive"),
        ({"noise_std": -0.1}, "noise_std must be non-negative"),
        ({"z_max": -1}, "z_max must be non-negative"),
    ],
)
def test_actor_rejects_invalid_params(kwargs, match):
    with pytest.raises(ValueError, match=match):
        GridActor(**kwargs)


# =============================================================================
# Station-keeping scenario
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_station_keeping_accumulates(ndim):
    """Starting at target with d_max=0: constant reward each step."""
    n = 10
    bonus = 7.0
    if ndim == 2:
        config = GridConfig.create(n, n)
        pos = GridPosition(5, 5, None)
    else:
        config = GridConfig.create(n, n, n)
        pos = GridPosition(5, 5, 5)

    env = _make_nav_env(config, pos, pos, peak_reward=bonus + 0.5, step_cost=0.5,
                        proximity_scale=0.1, d_max=0, max_steps=20)
    env.reset(seed=0)

    total = 0.0
    for _ in range(10):
        _, reward, _, _, _ = env.step(1)
        total += reward
        assert reward == pytest.approx(bonus)
    assert total == pytest.approx(bonus * 10)


# =============================================================================
# Constructor validation
# =============================================================================

@pytest.mark.parametrize("max_steps", [0, -1, -100])
def test_environment_rejects_invalid_max_steps(max_steps):
    config = GridConfig.create(5, 5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    arena = GridArena(realized_field=field, observed_field=field, actor=actor,
                      config=config, initial_position=GridPosition(1, 1, None),
                      max_displacement=1, boundary_mode="clip")
    with pytest.raises(ValueError, match="max_steps must be positive"):
        GridEnvironment(arena=arena, max_steps=max_steps)


@pytest.mark.parametrize(
    ("max_displacement", "match"),
    [
        (-1, "max_displacement must be non-negative"),
        (5, "max_displacement must be smaller"),   # max_displacement >= n_x=5
        (10, "max_displacement must be smaller"),
    ],
)
def test_arena_rejects_invalid_max_displacement(max_displacement, match):
    """max_displacement validation now lives on the arena (clip + obs bound)."""
    config = GridConfig.create(n_x=5, n_y=5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    with pytest.raises(ValueError, match=match):
        GridArena(realized_field=field, observed_field=field, actor=actor,
                  config=config, initial_position=GridPosition(1, 1, None),
                  max_displacement=max_displacement, boundary_mode="clip")


@pytest.mark.parametrize("max_displacement", [2.5, 1.7, 2.0, 0.4])
def test_arena_accepts_float_max_displacement(max_displacement):
    """Float max_displacement is a valid continuous clip bound."""
    config = GridConfig.create(n_x=5, n_y=5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    arena = GridArena(realized_field=field, observed_field=field, actor=actor,
                      config=config, initial_position=GridPosition(1, 1, None),
                      max_displacement=max_displacement, boundary_mode="clip")
    assert arena.max_displacement == pytest.approx(max_displacement)
