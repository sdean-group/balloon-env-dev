"""Dynamic Programming All-Pairs Stochastic Shortest Path Agent.

Computes the optimal policy and expected step cost for a stochastic shortest
path formulation to any goal on the grid, via Batch Value Iteration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from .agent import Agent, AgentConfig
from ..env.arena.navigation_arena import NavigationArena
from ..env.utils.types import GridConfig, GridPosition


@dataclass
class APSSPAgentConfig(AgentConfig):
    """Configuration for the AP-SSP agent."""

    max_iters: int = 2000
    rel_tol: float = 1e-3
    warmstart: bool = True
    debug: bool = False
    #: Finite penalty for transitions that leave the grid (terminal mode) and
    #: initial cost for unreachable states. If ``None``, uses ``n_x * n_y`` from
    #: the arena grid at ``plan()`` time (must exceed optimal H for correctness).
    oob_penalty_max: Optional[float] = None


def _ap_ssp_value_iteration_2d(
    field_pmf: jnp.ndarray,
    actor_pmf: jnp.ndarray,
    vicinity_radius: float,
    vicinity_metric: str,
    d_max: int,
    z_max: int,
    boundary_mode: str,
    max_cost: float,
    max_iters: int = 2000,
    rel_tol: float = 1e-3,
    H_init_warm: Optional[jnp.ndarray] = None,
    debug: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray, int]:
    """Compute All-Pairs Stochastic Shortest Path (AP-SSP) tables for 2D.

    Parameters
    ----------
    max_cost : float
    max_iters : int
    rel_tol : float
    H_init_warm : Optional[jnp.ndarray]
    debug : bool

    Returns
    -------
    H : jnp.ndarray shape (n_x, n_y, n_x, n_y)
        Expected Cost (number of steps).
    Pi : jnp.ndarray shape (n_x, n_y, n_x, n_y)
        Optimal policy (action index; stay=1 in goal vicinity).
    iters : int
        Number of value-iteration sweeps performed.
    """
    n_x, n_y, _ = field_pmf.shape

    all_i = jnp.arange(n_x)
    all_j = jnp.arange(n_y)

    # Grid coordinates for distance calculation
    I_grid = all_i[:, None, None, None]
    J_grid = all_j[None, :, None, None]
    GI = all_i[None, None, :, None]
    GJ = all_j[None, None, None, :]

    # Distance metrics
    if vicinity_metric == "euclidean":
        dist = jnp.sqrt((I_grid - GI) ** 2 + (J_grid - GJ) ** 2)
    elif vicinity_metric == "l1":
        dist = jnp.abs(I_grid - GI) + jnp.abs(J_grid - GJ)
    elif vicinity_metric == "linf":
        dist = jnp.maximum(jnp.abs(I_grid - GI), jnp.abs(J_grid - GJ))
    else:
        # Fallback to euclidean if somehow wrong string passed
        warnings.warn(f"Unknown vicinity_metric: {vicinity_metric}, using euclidean")
        dist = jnp.sqrt((I_grid - GI) ** 2 + (J_grid - GJ) ** 2)

    reached_mask = dist <= vicinity_radius

    # Initialize H. Reached states have 0 cost. Others have high cost,
    # unless a warmstart H is provided — then reuse it (vicinity overrides).
    if H_init_warm is None:
        H_init = jnp.where(reached_mask, 0.0, max_cost)
    else:
        H_init = jnp.where(reached_mask, 0.0, H_init_warm)

    u_offsets = jnp.arange(-d_max, d_max + 1)
    v_offsets = jnp.arange(-z_max, z_max + 1)

    def _apply_boundary_i(idx):
        if boundary_mode == "clip":
            return jnp.clip(idx, 0, n_x - 1)
        elif boundary_mode == "periodic":
            return idx % n_x
        else:
            return jnp.clip(idx, 0, n_x - 1)

    def _apply_boundary_j(idx):
        return jnp.clip(idx, 0, n_y - 1)

    def _oob_mask_i(idx):
        return (idx < 0) | (idx >= n_x)

    def _oob_mask_j(idx):
        return (idx < 0) | (idx >= n_y)

    raw_next_i = all_i[:, None] + u_offsets[None, :]
    next_i = _apply_boundary_i(raw_next_i)
    oob_i = _oob_mask_i(raw_next_i) if boundary_mode == "terminal" else None

    raw_next_j = all_j[:, None] + v_offsets[None, :]
    next_j = _apply_boundary_j(raw_next_j)
    oob_j = _oob_mask_j(raw_next_j) if boundary_mode == "terminal" else None

    # Pre-compute indexing arrays outside the loop
    idx_i = next_i[:, None, :, None]
    idx_j = next_j[None, :, None, :]

    if boundary_mode == "terminal":
        oob_mask = oob_i[:, None, :, None] | oob_j[None, :, None, :]
    else:
        oob_mask = None

    def body_fn(val):
        iter_count, H, _Pi, _abs_diff, _rel_diff = val

        H_lookup = H[idx_i, idx_j]

        if oob_mask is not None:
            H_lookup = jnp.where(oob_mask[..., None, None], max_cost, H_lookup)

        W = jnp.einsum("iju,ijuvgh->ijvgh", field_pmf, H_lookup)
        Q = 1.0 + jnp.einsum("av,ijvgh->ijagh", actor_pmf, W)

        H_new = jnp.min(Q, axis=2)
        Pi_new = jnp.argmin(Q, axis=2).astype(jnp.int32)
        # In vicinity of goal, prefer stay (action 1) to avoid drift when not terminating on reach.
        Pi_new = jnp.where(reached_mask, jnp.int32(1), Pi_new)

        H_new = jnp.where(reached_mask, 0.0, H_new)
        abs_diff = jnp.max(jnp.abs(H - H_new)).astype(jnp.float32)
        # Denominator: max |H_new| over non-sentinel cells, so OOB
        # max_cost cells don't deflate the relative error.
        finite_mask = H_new < 0.99 * max_cost
        denom = jnp.max(jnp.where(finite_mask, jnp.abs(H_new), 0.0)) + 1e-8
        rel_diff = (abs_diff / denom).astype(jnp.float32)
        return (iter_count + 1, H_new, Pi_new, abs_diff, rel_diff)

    Pi_init = jnp.zeros_like(H_init, dtype=jnp.int32)
    init_val = (
        jnp.array(0),
        H_init,
        Pi_init,
        jnp.array(1e6, dtype=jnp.float32),
        jnp.array(1e6, dtype=jnp.float32),
    )

    if debug:
        from tqdm import tqdm

        step_fn = jax.jit(body_fn)
        val = init_val
        with tqdm(total=max_iters, desc="AP-SSP Value Iteration") as pbar:
            while val[0] < max_iters and val[4] > rel_tol:
                val = step_fn(val)
                pbar.update(1)
                pbar.set_postfix(
                    {"rel": f"{float(val[4]):.4g}", "abs": f"{float(val[3]):.4g}"}
                )
        iters, H, Pi, abs_diff, rel_diff = val
    else:

        @jax.jit
        def _solve(init_val):
            def cond_fn(val):
                iter_count, _H, _Pi, _abs_diff, rel_diff = val
                return (iter_count < max_iters) & (rel_diff > rel_tol)

            return jax.lax.while_loop(cond_fn, body_fn, init_val)

        iters, H, Pi, abs_diff, rel_diff = _solve(init_val)

    iters = int(iters)
    abs_diff = float(abs_diff)
    rel_diff = float(rel_diff)
    print(f"AP-SSP converged in {iters} iters (rel={rel_diff:.4g}, abs={abs_diff:.4g})")

    return H, Pi, iters


class APSSPAgent(Agent):
    """Agent that plans stochastic shortest paths to any goal."""

    def __init__(
        self,
        config: APSSPAgentConfig | None = None,
        num_actions: int = 3,
        obs_shape: tuple[int, ...] = (3,),
    ) -> None:
        if config is None:
            config = APSSPAgentConfig()
        super().__init__(config, num_actions, obs_shape)

        self._cost_table: Optional[jnp.ndarray] = None
        self._policy_table: Optional[jnp.ndarray] = None
        self._ndim: Optional[int] = None
        self.target_position: Optional[GridPosition] = None
        self.last_value_iteration_iters: Optional[int] = None
        self.oob_penalty_max: Optional[float] = None

    def prepare_episode(self, env) -> None:
        """Run batch value iteration for the current field."""
        self.plan(env.arena)

    def plan(self, arena: NavigationArena) -> None:
        """Run AP-SSP Value Iteration."""
        cfg: GridConfig = arena.config
        if cfg.ndim != 2:
            raise NotImplementedError(
                "APSSPAgent currently only supports 2D environments."
            )

        # Integer displacement resolutions for the discretized transition model.
        # Positions/displacements are continuous at runtime; AP-SSP discretizes them.
        d_max = arena.field.disp_levels
        z_max = arena.actor.ctrl_levels
        self._ndim = cfg.ndim

        field_pmf = arena.field.get_displacement_pmf_grid()
        actor_pmf = jnp.asarray(arena.actor.get_controllable_displacement_pmf())

        vicinity_radius = arena.vicinity_radius
        vicinity_metric = getattr(arena, "vicinity_metric", "euclidean")
        boundary_mode = arena.boundary_mode

        H_warm: Optional[jnp.ndarray] = None
        if self.config.warmstart and self._cost_table is not None:
            expected_shape = (cfg.n_x, cfg.n_y, cfg.n_x, cfg.n_y)
            if self._cost_table.shape == expected_shape:
                H_warm = jnp.asarray(self._cost_table)
            else:
                warnings.warn(
                    f"Warmstart H shape {self._cost_table.shape} does not match "
                    f"expected {expected_shape}; falling back to cold start."
                )

        resolved_max = self.config.oob_penalty_max
        if resolved_max is None:
            resolved_max = float(cfg.n_x * cfg.n_y)
        else:
            resolved_max = float(resolved_max)
            if resolved_max <= 0.0:
                raise ValueError(
                    f"oob_penalty_max must be positive, got {resolved_max}"
                )

        self.oob_penalty_max = resolved_max

        H, Pi, vi_iters = _ap_ssp_value_iteration_2d(
            field_pmf,
            actor_pmf,
            vicinity_radius,
            vicinity_metric,
            d_max,
            z_max,
            boundary_mode,
            resolved_max,
            max_iters=self.config.max_iters,
            rel_tol=self.config.rel_tol,
            H_init_warm=H_warm,
            debug=self.config.debug,
        )
        self._cost_table = np.array(H)
        self._policy_table = np.array(Pi)
        self.last_value_iteration_iters = int(vi_iters)

    def get_expected_cost(self, pos: GridPosition, target: GridPosition) -> float:
        """Look up the expected cost to reach target from pos."""
        if self._cost_table is None:
            raise RuntimeError("Agent must be planned first.")
        i = int(round(float(pos.i))) - 1
        j = int(round(float(pos.j))) - 1
        gi = int(round(float(target.i))) - 1
        gj = int(round(float(target.j))) - 1
        return float(self._cost_table[i, j, gi, gj])

    def set_target(self, target: GridPosition) -> None:
        """Set the target position for the inner routine."""
        self.target_position = target

    def _action_from_obs(self, observation: np.ndarray) -> int:
        if self._policy_table is None:
            raise RuntimeError("Agent must be planned first.")
        if self.target_position is None:
            raise RuntimeError("Target position must be set before stepping.")

        # Continuous positions -> nearest integer cell (clamped to the table).
        n_x, n_y = self._policy_table.shape[0], self._policy_table.shape[1]
        i_idx = max(0, min(int(round(float(observation[0]))) - 1, n_x - 1))
        j_idx = max(0, min(int(round(float(observation[1]))) - 1, n_y - 1))
        g_i = max(0, min(int(round(float(self.target_position.i))) - 1, n_x - 1))
        g_j = max(0, min(int(round(float(self.target_position.j))) - 1, n_y - 1))

        return int(self._policy_table[i_idx, j_idx, g_i, g_j])

    def begin_episode(self, observation: np.ndarray) -> int:
        return self._action_from_obs(observation)

    def step(self, reward: float, observation: np.ndarray) -> int:
        return self._action_from_obs(observation)

    def end_episode(self, reward: float, terminal: bool) -> None:
        pass
