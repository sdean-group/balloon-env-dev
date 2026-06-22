"""Dynamic Programming / Value Iteration agent.

Computes the optimal policy for a finite-horizon MDP via backward induction,
given full access to the analytical transition model (field + actor PMFs) and
the reward function.  Serves as an oracle upper-bound baseline.

Supports both 2D and 3D NavigationArena environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .agent import Agent, AgentConfig

from ..env.arena.navigation_arena import NavigationArena
from ..env.utils.types import GridConfig


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DPAgentConfig(AgentConfig):
    """Configuration for the DP agent.  No learnable hyper-parameters --
    everything is derived from the arena."""
    pass


# ---------------------------------------------------------------------------
# Backward induction  -- 2D
# ---------------------------------------------------------------------------

def _backward_induction_2d(
    reward: jnp.ndarray,
    field_pmf: jnp.ndarray,
    actor_pmf: jnp.ndarray,
    horizon: int,
    d_max: int,
    z_max: int,
    boundary_mode: str,
) -> jnp.ndarray:
    """Compute the time-dependent optimal policy for a 2D grid.

    Parameters
    ----------
    reward : (n_x, n_y)
    field_pmf : (n_x, n_y, 2*d_max+1)
    actor_pmf : (n_actions, 2*z_max+1)
    horizon : H
    d_max, z_max : displacement bounds
    boundary_mode : 'clip' | 'periodic' | 'terminal'

    Returns
    -------
    policy : jnp.ndarray, shape (H, n_x, n_y), dtype int32
        policy[t, i, j] = optimal action at time t for 0-indexed (i,j).
    """
    n_x, n_y = reward.shape

    # displacement index arrays
    u_offsets = jnp.arange(-d_max, d_max + 1)  # ambient displacements
    v_offsets = jnp.arange(-z_max, z_max + 1)  # controllable displacements

    # Coordinate arrays (0-indexed)
    all_i = jnp.arange(n_x)
    all_j = jnp.arange(n_y)

    # ---- Boundary helper: map raw next-indices to valid 0-indexed coords ----
    def _apply_boundary_i(idx):
        """Boundary on ambient axis."""
        if boundary_mode == "clip":
            return jnp.clip(idx, 0, n_x - 1)
        elif boundary_mode == "periodic":
            return idx % n_x
        else:  # terminal -- out-of-bounds mapped to 0 (value = 0)
            return jnp.clip(idx, 0, n_x - 1)

    def _apply_boundary_j(idx):
        """Boundary on controllable axis (always clip, even for periodic)."""
        return jnp.clip(idx, 0, n_y - 1)

    def _oob_mask_i(idx):
        """True where ambient index is out of bounds (terminal mode)."""
        return (idx < 0) | (idx >= n_x)

    def _oob_mask_j(idx):
        """True where controllable index is out of bounds."""
        return (idx < 0) | (idx >= n_y)

    # ---- Pre-compute next-state index tables ----
    # next_i[i, u] = boundary(i + u_offset[u])   shape (n_x, 2d+1)
    raw_next_i = all_i[:, None] + u_offsets[None, :]  # (n_x, 2d+1)
    next_i = _apply_boundary_i(raw_next_i)
    oob_i = _oob_mask_i(raw_next_i) if boundary_mode == "terminal" else None

    # next_j[j, v] = boundary(j + v_offset[v])   shape (n_y, 2z+1)
    raw_next_j = all_j[:, None] + v_offsets[None, :]  # (n_y, 2z+1)
    next_j = _apply_boundary_j(raw_next_j)
    oob_j = _oob_mask_j(raw_next_j) if boundary_mode == "terminal" else None

    # ---- Backward induction via jax.lax.scan ----

    def _bellman_step(V_next, _t):
        """One backward Bellman step.  Returns (V_t, policy_t)."""
        # We need V_lookup[i, j, u, v] = V_next[next_i[i,u], next_j[j,v]]
        # where next_i: (n_x, n_du) and next_j: (n_y, n_dv).
        #
        # Use explicit meshgrid-style indexing to avoid broadcast ambiguity:
        #   idx_i: (n_x, 1, n_du, 1)    -- varies over i and u
        #   idx_j: (1, n_y, 1, n_dv)    -- varies over j and v
        idx_i = next_i[:, None, :, None]   # (n_x, 1, 2d+1, 1)
        idx_j = next_j[None, :, None, :]   # (1, n_y, 1, 2z+1)
        V_lookup = V_next[idx_i, idx_j]    # (n_x, n_y, 2d+1, 2z+1)

        # Zero out values at out-of-bounds next states (terminal mode)
        if boundary_mode == "terminal":
            # oob_i: (n_x, 2d+1), oob_j: (n_y, 2z+1)
            oob_mask = (oob_i[:, None, :, None]
                        | oob_j[None, :, None, :])
            V_lookup = jnp.where(oob_mask, 0.0, V_lookup)

        # W[i, j, v] = sum_u field_pmf[i, j, u] * V_lookup[i, j, u, v]
        W = jnp.einsum("iju,ijuv->ijv", field_pmf, V_lookup)
        # W shape: (n_x, n_y, 2z+1)

        # Q[i,j,a] = R[i,j] + sum_v actor_pmf[a,v] * W[i,j,v]
        Q = reward[:, :, None] + jnp.einsum("av,ijv->ija", actor_pmf, W)

        V_t = jnp.max(Q, axis=-1)           # (n_x, n_y)
        pi_t = jnp.argmax(Q, axis=-1)       # (n_x, n_y) int32

        return V_t, pi_t

    # Terminal value
    V_H = jnp.zeros((n_x, n_y), dtype=jnp.float32)

    # Scan backward from t = H-1 down to 0
    # jax.lax.scan goes *forward* over the sequence, so we just
    # use it with length=H steps; each step ignores its input index.
    _V_0, policy = jax.lax.scan(_bellman_step, V_H, None, length=horizon)
    # policy shape: (H, n_x, n_y), where policy[0] = pi_{H-1}, ..., policy[H-1] = pi_0.
    # Reverse so that policy[t] = pi_t.
    policy = policy[::-1]

    return policy


# ---------------------------------------------------------------------------
# Backward induction  -- 3D
# ---------------------------------------------------------------------------

def _backward_induction_3d(
    reward: jnp.ndarray,
    field_pmf: jnp.ndarray,
    actor_pmf: jnp.ndarray,
    horizon: int,
    d_max: int,
    z_max: int,
    boundary_mode: str,
) -> jnp.ndarray:
    """Compute the time-dependent optimal policy for a 3D grid.

    Parameters
    ----------
    reward : (n_x, n_y, n_z)
    field_pmf : (n_x, n_y, n_z, 2*d_max+1, 2*d_max+1)
    actor_pmf : (n_actions, 2*z_max+1)
    horizon : H
    d_max, z_max : displacement bounds
    boundary_mode : 'clip' | 'periodic' | 'terminal'

    Returns
    -------
    policy : jnp.ndarray, shape (H, n_x, n_y, n_z), dtype int32
    """
    n_x, n_y, n_z = reward.shape

    u_offsets = jnp.arange(-d_max, d_max + 1)  # ambient-i displacements
    v_offsets = jnp.arange(-d_max, d_max + 1)  # ambient-j displacements
    w_offsets = jnp.arange(-z_max, z_max + 1)  # controllable displacements

    all_i = jnp.arange(n_x)
    all_j = jnp.arange(n_y)
    all_k = jnp.arange(n_z)

    # ---- Boundary helpers ----
    def _bnd_ambient_i(idx):
        if boundary_mode == "clip":
            return jnp.clip(idx, 0, n_x - 1)
        elif boundary_mode == "periodic":
            return idx % n_x
        else:
            return jnp.clip(idx, 0, n_x - 1)

    def _bnd_ambient_j(idx):
        if boundary_mode == "clip":
            return jnp.clip(idx, 0, n_y - 1)
        elif boundary_mode == "periodic":
            return idx % n_y
        else:
            return jnp.clip(idx, 0, n_y - 1)

    def _bnd_ctrl(idx):
        return jnp.clip(idx, 0, n_z - 1)

    def _oob(idx, n):
        return (idx < 0) | (idx >= n)

    # ---- Pre-compute next-state index tables ----
    raw_ni = all_i[:, None] + u_offsets[None, :]     # (n_x, 2d+1)
    raw_nj = all_j[:, None] + v_offsets[None, :]     # (n_y, 2d+1)
    raw_nk = all_k[:, None] + w_offsets[None, :]     # (n_z, 2z+1)

    ni = _bnd_ambient_i(raw_ni)
    nj = _bnd_ambient_j(raw_nj)
    nk = _bnd_ctrl(raw_nk)

    if boundary_mode == "terminal":
        oob_i = _oob(raw_ni, n_x)  # (n_x, 2d+1)
        oob_j = _oob(raw_nj, n_y)  # (n_y, 2d+1)
        oob_k = _oob(raw_nk, n_z)  # (n_z, 2z+1)

    # ---- Backward induction ----

    def _bellman_step(V_next, _t):
        # V_lookup[i, j, k, u, v, w] = V_next[ni[i,u], nj[j,v], nk[k,w]]
        # Build via advanced indexing:
        #   ni[:, :] -> (n_x, 2d+1)
        #   nj[:, :] -> (n_y, 2d+1)
        #   nk[:, :] -> (n_z, 2z+1)
        # Result shape: (n_x, 2d+1, n_y, 2d+1, n_z, 2z+1)
        V_lookup = V_next[ni[:, :, None, None, None, None],
                          nj[None, None, :, :, None, None],
                          nk[None, None, None, None, :, :]]
        # Transpose to (n_x, n_y, n_z, 2d+1, 2d+1, 2z+1)
        V_lookup = jnp.transpose(V_lookup, (0, 2, 4, 1, 3, 5))

        if boundary_mode == "terminal":
            oob_mask = (oob_i[:, None, None, :, None, None]
                        | oob_j[None, :, None, None, :, None]
                        | oob_k[None, None, :, None, None, :])
            V_lookup = jnp.where(oob_mask, 0.0, V_lookup)

        # W[i,j,k,w] = sum_{u,v} field_pmf[i,j,k,u,v] * V_lookup[i,j,k,u,v,w]
        W = jnp.einsum("ijkuv,ijkuvw->ijkw", field_pmf, V_lookup)
        # W shape: (n_x, n_y, n_z, 2z+1)

        # Q[i,j,k,a] = R[i,j,k] + sum_w actor_pmf[a,w] * W[i,j,k,w]
        Q = reward[..., None] + jnp.einsum("aw,ijkw->ijka", actor_pmf, W)

        V_t = jnp.max(Q, axis=-1)
        pi_t = jnp.argmax(Q, axis=-1)

        return V_t, pi_t

    V_H = jnp.zeros((n_x, n_y, n_z), dtype=jnp.float32)
    _V_0, policy = jax.lax.scan(_bellman_step, V_H, None, length=horizon)
    policy = policy[::-1]
    return policy


# ---------------------------------------------------------------------------
# DP Agent
# ---------------------------------------------------------------------------

class DPAgent(Agent):
    """Oracle agent that solves for the optimal policy via dynamic programming.

    Usage
    -----
    After ``env.reset()``, call ``dp_agent.plan(arena, horizon)`` to run
    backward induction for the current episode's field realisation.
    Then use the standard ``begin_episode`` / ``step`` / ``end_episode`` loop.
    """

    def __init__(
        self,
        config: DPAgentConfig | None = None,
        num_actions: int = 3,
        obs_shape: tuple[int, ...] = (3,),
    ) -> None:
        if config is None:
            config = DPAgentConfig()
        super().__init__(config, num_actions, obs_shape)

        # Populated by plan()
        self._policy: Optional[jnp.ndarray] = None  # (H, *grid_shape)
        self._ndim: Optional[int] = None
        self._grid_shape: Optional[tuple] = None
        self._step_t: int = 0

    # ------------------------------------------------------------------
    # Episode preparation (called by runner after env.reset)
    # ------------------------------------------------------------------

    def prepare_episode(self, env) -> None:
        """Run backward induction for the current episode's field realisation."""
        self.plan(env.arena, env.max_steps)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, arena: NavigationArena, horizon: int) -> None:
        """Run backward induction for the current episode.

        Parameters
        ----------
        arena : NavigationArena
            The arena whose field has already been reset (so PMFs reflect the
            current episode's field realisation).
        horizon : int
            Episode horizon H (number of steps).
        """
        cfg: GridConfig = arena.config
        # Integer displacement resolutions for the discretized transition model.
        # Positions/displacements are continuous at runtime; DP discretizes them.
        d_levels = arena.field.disp_levels
        z_levels = arena.actor.ctrl_levels
        self._ndim = cfg.ndim
        self._grid_shape = cfg.shape  # for clamping rounded observations to cells

        reward = arena.reward_fn.compute_grid(cfg)
        field_pmf = arena.field.get_displacement_pmf_grid()
        actor_pmf = jnp.asarray(arena.actor.get_controllable_displacement_pmf())

        # Validate that actor PMF is consistent with the agent's action count
        n_actions_from_pmf = actor_pmf.shape[0]
        if n_actions_from_pmf != self._num_actions:
            raise ValueError(
                f"Actor PMF has {n_actions_from_pmf} actions but DPAgent was "
                f"constructed with num_actions={self._num_actions}. "
                f"These must match."
            )

        if cfg.ndim == 2:
            self._policy = _backward_induction_2d(
                reward, field_pmf, actor_pmf,
                horizon, d_levels, z_levels, arena.boundary_mode,
            )
        else:
            self._policy = _backward_induction_3d(
                reward, field_pmf, actor_pmf,
                horizon, d_levels, z_levels, arena.boundary_mode,
            )

        self._step_t = 0

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def _action_from_obs(self, observation: np.ndarray) -> int:
        """Look up the optimal action in the pre-computed policy table."""
        if self._policy is None:
            raise RuntimeError(
                "DPAgent.plan() must be called before begin_episode / step."
            )

        # Observation layout: [i, j, (k,) u, (v)]
        # Positions are continuous and 1-indexed; the policy table is discrete and
        # 0-indexed, so round each coordinate to the nearest cell and clamp.
        def _cell(value: float, axis: int) -> int:
            idx = int(round(float(value))) - 1
            return max(0, min(idx, self._grid_shape[axis] - 1))

        if self._ndim == 2:
            i_idx = _cell(observation[0], 0)
            j_idx = _cell(observation[1], 1)
            action = int(self._policy[self._step_t, i_idx, j_idx])
        else:
            i_idx = _cell(observation[0], 0)
            j_idx = _cell(observation[1], 1)
            k_idx = _cell(observation[2], 2)
            action = int(self._policy[self._step_t, i_idx, j_idx, k_idx])

        return action

    def begin_episode(self, observation: np.ndarray) -> int:
        self._step_t = 0
        action = self._action_from_obs(observation)
        self._step_t += 1
        return action

    def step(self, reward: float, observation: np.ndarray) -> int:
        action = self._action_from_obs(observation)
        self._step_t += 1
        return action

    def end_episode(self, reward: float, terminal: bool) -> None:
        pass
