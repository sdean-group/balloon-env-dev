"""Parameterized contract tests for NavigationArena reward computation.

Reward under test is provided by reward_helpers (make_reward_fn, expected_reward_at_distance).
Change reward_helpers.py to switch reward; tests stay reward-agnostic.
Covers: reward contracts, vicinity detection, cumulative tracking, constructor validation — 2D and 3D.
"""

import pytest
import numpy as np
import jax

from src.env.arena.navigation_arena import NavigationArena
from src.env.arena.reward import NavigationReward
from src.env.field.composite import ZeroField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition

from tests.test_arena.reward_helpers import make_reward_fn, expected_reward_at_distance, DEFAULT_REWARD_KWARGS

RNG = np.random.default_rng(9999)


# =============================================================================
# Helpers (reward-agnostic: use reward_helpers for reward construction)
# =============================================================================

def _make_arena(
    ndim: int, n: int, initial: GridPosition, target: GridPosition, *,
    vicinity_radius: float = 2.0,
    reward_fn=None,
    d_max: int = 1,
    **reward_kwargs,
) -> NavigationArena:
    """Build NavigationArena; reward comes from make_reward_fn unless reward_fn is given."""
    if ndim == 3:
        config = GridConfig.create(n, n, n)
    else:
        config = GridConfig.create(n, n)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    if reward_fn is None:
        reward_fn = make_reward_fn(target, vicinity_radius, **reward_kwargs)
    return NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=initial, target_position=target,
        vicinity_radius=vicinity_radius, max_displacement=d_max,
        boundary_mode="clip",
        reward_fn=reward_fn,
        terminate_on_reach=False,
    )


# =============================================================================
# Parameterized distance + reward contract
# =============================================================================

REWARD_CASES = [
    (2, (5, 5), (5, 5), 2.0, "2d-at-target"),
    (2, (1, 1), (5, 5), 2.0, "2d-far-from-target"),
    (2, (7, 5), (5, 5), 2.0, "2d-on-boundary"),
    (2, (8, 5), (5, 5), 2.0, "2d-just-outside"),
    (2, (1, 1), (10, 10), 1.0, "2d-max-distance"),
    (3, (5, 5, 5), (5, 5, 5), 2.0, "3d-at-target"),
    (3, (1, 1, 1), (5, 5, 5), 2.0, "3d-far-from-target"),
    (3, (7, 5, 5), (5, 5, 5), 2.0, "3d-on-boundary"),
    (3, (8, 5, 5), (5, 5, 5), 2.0, "3d-just-outside"),
    (3, (1, 1, 1), (10, 10, 10), 0.5, "3d-max-distance"),
]


@pytest.mark.parametrize(
    ("ndim", "init_tup", "targ_tup", "radius", "desc"), REWARD_CASES,
    ids=lambda *a: a[-1] if isinstance(a[-1], str) else str(a)
)
def test_reward_contracts(ndim, init_tup, targ_tup, radius, desc):
    """Reward from make_reward_fn: non-negative, matches expected_reward_at_distance, vicinity set."""
    n = 10
    reward_kw = {"peak_reward": 10.0, "step_cost": 0.5, "proximity_scale": 0.1}
    if ndim == 2:
        initial = GridPosition(init_tup[0], init_tup[1], None)
        target = GridPosition(targ_tup[0], targ_tup[1], None)
        dist = np.sqrt((init_tup[0] - targ_tup[0])**2 + (init_tup[1] - targ_tup[1])**2)
    else:
        initial = GridPosition(*init_tup)
        target = GridPosition(*targ_tup)
        dist = np.sqrt(sum((a - b)**2 for a, b in zip(init_tup, targ_tup)))

    arena = _make_arena(ndim, n, initial, target, vicinity_radius=radius, d_max=0, **reward_kw)
    arena.reset(jax.random.PRNGKey(0))

    assert arena._compute_distance(initial, target) == pytest.approx(dist, abs=1e-10)

    reward = arena.compute_reward()
    expected = expected_reward_at_distance(dist, **reward_kw)
    assert reward == pytest.approx(expected, rel=1e-6)
    assert reward >= 0.0
    if dist <= radius:
        assert arena._target_reached is True
    if dist == 0:
        assert reward == pytest.approx(reward_kw["peak_reward"] - reward_kw["step_cost"], rel=1e-6)


# =============================================================================
# Proximity scale contract (reward decreases with distance)
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_reward_decreases_with_distance(ndim):
    """Larger distance => smaller reward (reward from make_reward_fn)."""
    n = 10
    if ndim == 2:
        target = GridPosition(5, 5, None)
        positions = [
            GridPosition(5, 5, None),   # dist 0
            GridPosition(6, 5, None),   # dist 1
            GridPosition(8, 5, None),   # dist 3
            GridPosition(1, 1, None),  # dist ~5.66
        ]
    else:
        target = GridPosition(5, 5, 5)
        positions = [
            GridPosition(5, 5, 5),
            GridPosition(6, 5, 5),
            GridPosition(8, 5, 5),
            GridPosition(1, 1, 1),
        ]

    reward_fn = make_reward_fn(target, vicinity_radius=10.0, peak_reward=10.0, step_cost=0.2, proximity_scale=0.1)
    config = GridConfig.create(n, n, n) if ndim == 3 else GridConfig.create(n, n)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    arena = NavigationArena(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=positions[0], target_position=target,
        vicinity_radius=10.0, max_displacement=0, boundary_mode="clip",
        reward_fn=reward_fn,
    )

    rewards = [arena.reward_fn.compute_scalar(p) for p in positions]
    for i in range(len(rewards) - 1):
        assert rewards[i] >= rewards[i + 1], (rewards[i], rewards[i + 1])


# =============================================================================
# Cumulative reward + reset contract
# =============================================================================

@pytest.mark.parametrize("ndim", [2, 3])
def test_cumulative_reward_and_reset(ndim):
    """Cumulative reward accumulates then resets to zero (reward from make_reward_fn)."""
    n = 10
    reward_kw = {"peak_reward": 10.0, "step_cost": 0.5}
    if ndim == 2:
        pos = GridPosition(5, 5, None)
    else:
        pos = GridPosition(5, 5, 5)

    arena = _make_arena(ndim, n, pos, pos, d_max=0, **reward_kw)
    arena.reset(jax.random.PRNGKey(0))

    expected_per_step = expected_reward_at_distance(0.0, **reward_kw)
    for k in range(1, 6):
        arena.compute_reward()
        assert arena.get_cumulative_reward() == pytest.approx(expected_per_step * k)

    arena.reset(jax.random.PRNGKey(1))
    assert arena.get_cumulative_reward() == 0.0
    assert arena._target_reached is False


# =============================================================================
# Constructor validation
# =============================================================================

@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"vicinity_radius": -1.0}, "vicinity_radius must be non-negative"),
    ],
)
def test_navigation_arena_rejects_invalid_params(kwargs, match):
    config = GridConfig.create(n_x=10, n_y=10)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    target = GridPosition(5, 5, None)
    reward_fn = make_reward_fn(target, vicinity_radius=2.0)
    defaults = dict(
        realized_field=field, observed_field=field, actor=actor, config=config,
        initial_position=target,
        target_position=target,
        vicinity_radius=2.0,
        max_displacement=1,
        reward_fn=reward_fn,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        NavigationArena(**defaults)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"peak_reward": 0.5, "step_cost": 1.0}, "peak_reward must be > step_cost"),
        ({"step_cost": -0.1}, "step_cost >= 0"),
        ({"proximity_scale": 0.0}, "proximity_scale > 0"),
        ({"proximity_scale": -0.1}, "proximity_scale > 0"),
    ],
)
def test_navigation_reward_rejects_invalid_params(kwargs, match):
    """NavigationReward constructor rejects invalid reward parameters (reward-class-specific)."""
    target = GridPosition(5, 5, None)
    defaults = dict(
        target_position=target,
        vicinity_radius=2.0,
        **DEFAULT_REWARD_KWARGS,
    )
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=match):
        NavigationReward(**defaults)


@pytest.mark.parametrize(
    ("target", "ndim", "match"),
    [
        (GridPosition(0, 5, None), 2, "target_position.*outside grid"),
        (GridPosition(11, 5, None), 2, "target_position.*outside grid"),
        (GridPosition(5, 0, None), 2, "target_position.*outside grid"),
        (GridPosition(0, 5, 5), 3, "target_position.*outside grid"),
        (GridPosition(5, 5, 0), 3, "target_position.k.*outside grid"),
        (GridPosition(5, 5, 11), 3, "target_position.k.*outside grid"),
    ],
)
def test_navigation_arena_rejects_oob_target(target, ndim, match):
    if ndim == 2:
        config = GridConfig.create(n_x=10, n_y=10)
        init = GridPosition(5, 5, None)
    else:
        config = GridConfig.create(n_x=10, n_y=10, n_z=10)
        init = GridPosition(5, 5, 5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    reward_fn = make_reward_fn(init, vicinity_radius=2.0)
    with pytest.raises(ValueError, match=match):
        NavigationArena(
            realized_field=field, observed_field=field, actor=actor, config=config,
            initial_position=init, target_position=target,
            vicinity_radius=2.0, max_displacement=1, reward_fn=reward_fn,
        )
