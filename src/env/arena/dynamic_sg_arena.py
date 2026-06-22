"""Dynamic Start-Goal Arena for multi-segment simulation experiments.

Supports changing the target and/or start position between segments
without re-sampling the ambient field, enabling multi-waypoint navigation
with a single planning phase.
"""

from __future__ import annotations

import numpy as np

from .grid_arena import GridArena
from .reward import RewardFunction
from ..field.flow_field import FlowField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import (
    GridPosition,
    GridConfig,
    ArenaState,
    GridArenaState,
    DynamicSGArenaState,
)


class DynamicSGArena(GridArena):
    """Arena with mutable start position and goal for simulation.

    Designed for multi-segment experiments where the agent navigates a
    sequence of targets under a static ambient field.  The arena never
    terminates on target reach — that decision is left to the outer
    simulation loop which reads ``target_reached`` from the state.

    Two levels of bookkeeping are maintained:

    *Segment-level* (reset on ``soft_reset``):
        ``segment_step_count``, ``segment_cumulative_reward``,
        ``target_reached``

    *Global-level* (reset only on full ``reset``):
        ``global_step_count``, ``global_cumulative_reward``
    """

    def __init__(
        self,
        realized_field: FlowField,
        observed_field: FlowField,
        actor: AbstractActor,
        config: GridConfig,
        initial_position: GridPosition,
        target_position: GridPosition,
        vicinity_radius: float,
        max_displacement: float,
        boundary_mode: str = "clip",
        vicinity_metric: str = "euclidean",
        *,
        reward_fn: RewardFunction,
        process_noise_std: float = 0.0,
        obs_noise_std: float = 0.0,
    ):
        super().__init__(
            realized_field=realized_field,
            observed_field=observed_field,
            actor=actor,
            config=config,
            initial_position=initial_position,
            max_displacement=max_displacement,
            boundary_mode=boundary_mode,
            process_noise_std=process_noise_std,
            obs_noise_std=obs_noise_std,
        )

        if vicinity_radius < 0.0:
            raise ValueError(
                f"vicinity_radius must be non-negative, got {vicinity_radius}"
            )
        self._validate_position(target_position, "target_position")

        self.target_position = target_position
        self.vicinity_radius = vicinity_radius
        self.vicinity_metric = vicinity_metric
        self.reward_fn = reward_fn

        # Segment-level counters
        self._segment_step_count = 0
        self._segment_cumulative_reward = 0.0
        self._target_reached = False
        self._segment_index = 0

        # Global-level counters
        self._global_step_count = 0
        self._global_cumulative_reward = 0.0

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_position(self, pos: GridPosition, label: str) -> None:
        if not (1 <= pos.i <= self.config.n_x and 1 <= pos.j <= self.config.n_y):
            raise ValueError(
                f"{label} {pos} is outside grid ({self.config.n_x}, {self.config.n_y})"
            )
        if self.config.ndim == 3 and not (1 <= pos.k <= self.config.n_z):
            raise ValueError(
                f"{label}.k={pos.k} is outside grid [1, {self.config.n_z}]"
            )

    # ------------------------------------------------------------------
    # Mutation methods (called between segments)
    # ------------------------------------------------------------------

    def set_target(
        self, target: GridPosition, vicinity_radius: float | None = None
    ) -> None:
        """Update the active target (and optionally the vicinity radius).

        Also forwards the change to the reward function via
        ``reward_fn.set_target`` so that subsequent ``compute_scalar`` /
        ``compute_grid`` calls reflect the new goal.
        """
        self._validate_position(target, "target")
        self.target_position = target
        if vicinity_radius is not None:
            self.vicinity_radius = vicinity_radius
        self.reward_fn.set_target(target, vicinity_radius)

    def set_position(self, position: GridPosition) -> None:
        """Teleport the actor to *position* (hard start-position reset)."""
        self._validate_position(position, "position")
        self.position = position
        self.initial_position = position

    # ------------------------------------------------------------------
    # Reset methods
    # ------------------------------------------------------------------

    def reset(self, rng_key) -> np.ndarray:
        """Full reset: re-sample the field and zero all counters."""
        obs = super().reset(rng_key)
        self._segment_step_count = 0
        self._segment_cumulative_reward = 0.0
        self._target_reached = False
        self._segment_index = 0
        self._global_step_count = 0
        self._global_cumulative_reward = 0.0
        return obs

    def reset_counters_and_position(
        self, position: GridPosition | None = None
    ) -> np.ndarray:
        """Reset all step/reward counters and move to a position without re-sampling the field.

        Use at the start of a new strategy or re-run so that global and
        segment counters start at zero. If position is None, uses
        initial_position.
        """
        pos = position if position is not None else self.initial_position
        self.set_position(pos)
        self.last_position = self.position
        self.step_count = 0
        self._segment_step_count = 0
        self._segment_cumulative_reward = 0.0
        self._target_reached = False
        self._segment_index = 0
        self._global_step_count = 0
        self._global_cumulative_reward = 0.0
        self._last_action = None
        self._last_reward = 0.0
        self.last_displacement = self._zero_displacement()
        return self._get_observation()

    def soft_reset(self) -> np.ndarray:
        """Start a new segment without re-sampling the field.

        Resets segment-level counters and ``target_reached``.  Global
        counters and the RNG state are untouched.  The actor stays at its
        current position (use ``set_position`` beforehand to relocate).
        """
        self._segment_step_count = 0
        self._segment_cumulative_reward = 0.0
        self._target_reached = False
        self._segment_index += 1
        self._last_reward = 0.0
        self.last_displacement = self._zero_displacement()
        self._last_action = None
        return self._get_observation()

    # ------------------------------------------------------------------
    # Per-step
    # ------------------------------------------------------------------

    def step(self, action: int) -> np.ndarray:
        obs = super().step(action)
        self._segment_step_count += 1
        self._global_step_count += 1
        return obs

    def compute_reward(self) -> float:
        reward = self.reward_fn.compute_scalar(self.position)

        dist = self._compute_distance(self.position, self.target_position)
        if dist <= self.vicinity_radius and not self._target_reached:
            self._target_reached = True

        self._segment_cumulative_reward += reward
        self._global_cumulative_reward += reward
        self._last_reward = reward
        return reward

    def is_terminal(self) -> bool:
        """Boundary-based termination only; never terminates on reach."""
        return super().is_terminal()

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def _compute_distance(self, pos1: GridPosition, pos2: GridPosition) -> float:
        diffs = [pos1.i - pos2.i, pos1.j - pos2.j]
        if self.ndim == 3:
            diffs.append(pos1.k - pos2.k)
        diffs = np.array(diffs)

        if self.vicinity_metric == "euclidean":
            return float(np.sqrt(np.sum(diffs**2)))
        elif self.vicinity_metric == "l1":
            return float(np.sum(np.abs(diffs)))
        elif self.vicinity_metric == "linf":
            return float(np.max(np.abs(diffs)))
        raise ValueError(f"Unknown vicinity_metric: {self.vicinity_metric}")

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def get_state(self) -> DynamicSGArenaState:
        base = super().get_state()
        return DynamicSGArenaState(
            # ArenaState
            step_count=base.step_count,
            last_action=base.last_action,
            last_reward=base.last_reward,
            rng_key=base.rng_key,
            # GridArenaState
            position=base.position,
            last_position=base.last_position,
            last_displacement=base.last_displacement,
            out_of_bounds=base.out_of_bounds,
            initial_position=base.initial_position,
            # Segment-level
            segment_step_count=self._segment_step_count,
            segment_cumulative_reward=self._segment_cumulative_reward,
            target_reached=self._target_reached,
            # Global-level
            global_step_count=self._global_step_count,
            global_cumulative_reward=self._global_cumulative_reward,
            # Task config
            target_position=self.target_position,
            vicinity_radius=self.vicinity_radius,
            segment_index=self._segment_index,
        )

    def set_state(self, state: ArenaState) -> None:
        super().set_state(state)
        if isinstance(state, DynamicSGArenaState):
            self._segment_step_count = state.segment_step_count
            self._segment_cumulative_reward = state.segment_cumulative_reward
            self._target_reached = state.target_reached
            self._global_step_count = state.global_step_count
            self._global_cumulative_reward = state.global_cumulative_reward
            self.target_position = state.target_position
            self.vicinity_radius = state.vicinity_radius
            self._segment_index = state.segment_index
        else:
            self._segment_step_count = 0
            self._segment_cumulative_reward = 0.0
            self._target_reached = False
            self._global_step_count = 0
            self._global_cumulative_reward = 0.0
            self._segment_index = 0
