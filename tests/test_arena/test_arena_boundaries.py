"""Parameterized contract tests for arena boundary enforcement.

Tests _enforce_boundaries for clip, periodic, terminal modes across
randomized grid configs, extreme positions, and edge cases.
"""

import pytest
import numpy as np

from src.env.arena.grid_arena import GridArena
from src.env.field.composite import ZeroField
from src.env.actor.grid_actor import GridActor
from src.env.utils.types import GridConfig, GridPosition

RNG = np.random.default_rng(7777)


# =============================================================================
# Case generators
# =============================================================================

def _random_grid_config(ndim: int) -> dict:
    n_x = int(RNG.integers(2, 50))
    n_y = int(RNG.integers(2, 50))
    n_z = int(RNG.integers(2, 50)) if ndim == 3 else None
    return {"n_x": n_x, "n_y": n_y, "n_z": n_z, "ndim": ndim}


def _random_interior_pos(cfg: dict) -> GridPosition:
    """Position strictly inside the grid (not on boundary)."""
    i = int(RNG.integers(2, cfg["n_x"]))  # [2, n_x-1]
    j = int(RNG.integers(2, cfg["n_y"]))
    k = int(RNG.integers(2, cfg["n_z"])) if cfg["ndim"] == 3 else None
    return GridPosition(i, j, k)


def _random_oob_pos(cfg: dict) -> GridPosition:
    """Position guaranteed to be out-of-bounds on at least one axis."""
    axis = int(RNG.integers(0, cfg["ndim"]))
    vals = [int(RNG.integers(1, cfg["n_x"] + 1)),
            int(RNG.integers(1, cfg["n_y"] + 1))]
    if cfg["ndim"] == 3:
        vals.append(int(RNG.integers(1, cfg["n_z"] + 1)))
    # Force one axis OOB
    sign = int(RNG.choice([-1, 1]))
    if sign == -1:
        vals[axis] = int(RNG.integers(-10, 1))  # <=0
    else:
        maxes = [cfg["n_x"], cfg["n_y"]] + ([cfg["n_z"]] if cfg["ndim"] == 3 else [])
        vals[axis] = maxes[axis] + int(RNG.integers(1, 11))
    if cfg["ndim"] == 2:
        return GridPosition(vals[0], vals[1], None)
    return GridPosition(vals[0], vals[1], vals[2])


GRID_CASES = [_random_grid_config(2) for _ in range(5)] + [_random_grid_config(3) for _ in range(5)]

# Build (config, mode, position, should_be_in_bounds) tuples
BOUNDARY_CASES = []
for cfg in GRID_CASES:
    interior = _random_interior_pos(cfg)
    oob = _random_oob_pos(cfg)
    for mode in ["clip", "periodic", "terminal"]:
        BOUNDARY_CASES.append({"cfg": cfg, "mode": mode, "pos": interior, "in_bounds": True})
        BOUNDARY_CASES.append({"cfg": cfg, "mode": mode, "pos": oob, "in_bounds": False})


def _make_arena(cfg: dict, mode: str) -> GridArena:
    config = GridConfig.create(cfg["n_x"], cfg["n_y"], cfg["n_z"])
    d_max = max(0, min(cfg["n_x"], cfg["n_y"] if cfg["ndim"] == 3 else cfg["n_x"]) - 1)
    d_max = min(d_max, 3)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    if cfg["ndim"] == 3:
        init = GridPosition(1, 1, 1)
    else:
        init = GridPosition(1, 1, None)
    return GridArena(realized_field=field, observed_field=field, actor=actor,
                     config=config, initial_position=init,
                     max_displacement=d_max, boundary_mode=mode)


def _case_id(case):
    c = case["cfg"]
    return f'{c["ndim"]}d-{c["n_x"]}x{c["n_y"]}{"x"+str(c["n_z"]) if c["n_z"] else ""}-{case["mode"]}-{"in" if case["in_bounds"] else "oob"}'


# =============================================================================
# Parameterized contract tests
# =============================================================================

@pytest.mark.parametrize("case", BOUNDARY_CASES, ids=_case_id)
def test_boundary_contracts(case):
    """Comprehensive boundary enforcement contract for all modes and positions."""
    cfg = case["cfg"]
    arena = _make_arena(cfg, case["mode"])
    pos = case["pos"]
    config = arena.config

    if cfg["ndim"] == 3:
        new_pos, oob = arena._enforce_boundaries_3d(pos)
    else:
        new_pos, oob = arena._enforce_boundaries_2d(pos)

    if case["mode"] == "clip":
        # Contract: position is always clamped inside grid
        assert 1 <= new_pos.i <= config.n_x
        assert 1 <= new_pos.j <= config.n_y
        if cfg["ndim"] == 3:
            assert 1 <= new_pos.k <= config.n_z
        # oob flag reflects whether clipping was needed
        if case["in_bounds"]:
            assert new_pos == pos
            assert oob is False
        else:
            assert oob is True

    elif case["mode"] == "periodic":
        # Contract: ambient axes wrap, controllable clips
        assert 1 <= new_pos.i <= config.n_x
        if cfg["ndim"] == 3:
            assert 1 <= new_pos.j <= config.n_y
            assert 1 <= new_pos.k <= config.n_z
            # Verify ambient wrapping formula
            assert new_pos.i == ((pos.i - 1) % config.n_x) + 1
            assert new_pos.j == ((pos.j - 1) % config.n_y) + 1
            assert new_pos.k == max(1, min(pos.k, config.n_z))
        else:
            assert 1 <= new_pos.j <= config.n_y
            assert new_pos.i == ((pos.i - 1) % config.n_x) + 1
            assert new_pos.j == max(1, min(pos.j, config.n_y))

    elif case["mode"] == "terminal":
        # Contract: position is never modified; oob flag is set iff truly OOB
        assert new_pos == pos
        is_inside = (1 <= pos.i <= config.n_x and 1 <= pos.j <= config.n_y)
        if cfg["ndim"] == 3:
            is_inside = is_inside and (1 <= pos.k <= config.n_z)
        assert oob == (not is_inside)


# =============================================================================
# Extreme position edge cases
# =============================================================================

EXTREME_POSITIONS_2D = [
    GridPosition(0, 0, None),           # just below lower bound
    GridPosition(-100, -100, None),     # far below lower bound
    GridPosition(1, 1, None),           # lower corner (in bounds)
    GridPosition(10, 8, None),          # upper corner (in bounds for 10x8)
    GridPosition(11, 9, None),          # just above upper bound
    GridPosition(1000, 1000, None),     # far above upper bound
    GridPosition(0, 5, None),           # i OOB, j OK
    GridPosition(5, 0, None),           # i OK, j OOB
]

EXTREME_POSITIONS_3D = [
    GridPosition(0, 0, 0),
    GridPosition(-50, -50, -50),
    GridPosition(1, 1, 1),
    GridPosition(10, 10, 5),
    GridPosition(11, 11, 6),
    GridPosition(500, 500, 500),
    GridPosition(0, 5, 3),              # single axis OOB
    GridPosition(5, 0, 3),
    GridPosition(5, 5, 0),
]


@pytest.mark.parametrize("pos", EXTREME_POSITIONS_2D, ids=lambda p: f"2d-{p.i},{p.j}")
@pytest.mark.parametrize("mode", ["clip", "periodic", "terminal"])
def test_extreme_positions_2d(pos, mode):
    """Extreme 2D positions never crash; clip always returns in-bounds."""
    config = GridConfig.create(n_x=10, n_y=8)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    arena = GridArena(realized_field=field, observed_field=field, actor=actor,
                      config=config, initial_position=GridPosition(5, 4, None),
                      max_displacement=3, boundary_mode=mode)

    new_pos, oob = arena._enforce_boundaries_2d(pos)

    if mode == "clip":
        assert 1 <= new_pos.i <= 10
        assert 1 <= new_pos.j <= 8
    elif mode == "terminal":
        assert new_pos == pos


@pytest.mark.parametrize("pos", EXTREME_POSITIONS_3D, ids=lambda p: f"3d-{p.i},{p.j},{p.k}")
@pytest.mark.parametrize("mode", ["clip", "periodic", "terminal"])
def test_extreme_positions_3d(pos, mode):
    """Extreme 3D positions never crash; clip always returns in-bounds."""
    config = GridConfig.create(n_x=10, n_y=10, n_z=5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    arena = GridArena(realized_field=field, observed_field=field, actor=actor,
                      config=config, initial_position=GridPosition(5, 5, 3),
                      max_displacement=3, boundary_mode=mode)

    new_pos, oob = arena._enforce_boundaries_3d(pos)

    if mode == "clip":
        assert 1 <= new_pos.i <= 10
        assert 1 <= new_pos.j <= 10
        assert 1 <= new_pos.k <= 5
    elif mode == "terminal":
        assert new_pos == pos


# =============================================================================
# Constructor validation
# =============================================================================

@pytest.mark.parametrize("mode", ["invalid", "wrap", "absorb", "", "CLIP"])
def test_invalid_boundary_mode_rejected(mode):
    config = GridConfig.create(n_x=5, n_y=5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    with pytest.raises(ValueError, match="boundary_mode must be one of"):
        GridArena(realized_field=field, observed_field=field, actor=actor,
                  config=config, initial_position=GridPosition(1, 1, None),
                  max_displacement=1, boundary_mode=mode)


@pytest.mark.parametrize(
    ("pos", "ndim", "match"),
    [
        (GridPosition(0, 1, None), 2, "initial_position.*outside grid"),
        (GridPosition(6, 1, None), 2, "initial_position.*outside grid"),
        (GridPosition(1, 0, None), 2, "initial_position.*outside grid"),
        (GridPosition(1, 6, None), 2, "initial_position.*outside grid"),
        (GridPosition(0, 1, 1), 3, "initial_position.*outside grid"),
        (GridPosition(1, 1, 0), 3, "initial_position.k.*invalid"),
        (GridPosition(1, 1, 6), 3, "initial_position.k.*invalid"),
        (GridPosition(1, 1, None), 3, "initial_position.k.*invalid"),
        (GridPosition(1, 1, 1), 2, "initial_position.k must be None"),
    ],
)
def test_invalid_initial_position_rejected(pos, ndim, match):
    if ndim == 2:
        config = GridConfig.create(n_x=5, n_y=5)
    else:
        config = GridConfig.create(n_x=5, n_y=5, n_z=5)
    field = ZeroField(config)
    actor = GridActor(noise_std=0.0)
    with pytest.raises(ValueError, match=match):
        GridArena(realized_field=field, observed_field=field, actor=actor,
                  config=config, initial_position=pos,
                  max_displacement=1, boundary_mode="clip")
