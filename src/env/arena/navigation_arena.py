"""Navigation arena for target-reaching tasks."""

import numpy as np

from .grid_arena import GridArena
from .reward import RewardFunction
from ..field.flow_field import FlowField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import GridPosition, GridConfig, ArenaState, NavigationArenaState


class NavigationArena(GridArena):
    """Arena for navigation and station-keeping tasks.

    Supports both 2D and 3D settings (inherits from GridArena).

    Reward is provided by the injected reward_fn (RewardFunction).
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
        boundary_mode: str = "terminal",
        vicinity_metric: str = "euclidean",
        *,
        reward_fn: RewardFunction,
        terminate_on_reach: bool = False,
        process_noise_std: float = 0.0,
        obs_noise_std: float = 0.0,
    ):
        """Initialize navigation arena.

        Args:
            realized_field: Wind source W that moves the balloon.
            observed_field: Wind source W_hat the agent observes.
            actor: Actor with controllable axis dynamics.
            config: Grid configuration.
            initial_position: Starting position.
            target_position: Goal position to reach.
            vicinity_radius: Radius around target that counts as "reached".
            max_displacement: Per-step displacement magnitude bound.
            boundary_mode: Boundary handling ('clip', 'periodic', 'terminal').
            vicinity_metric: Distance metric for vicinity ('euclidean', 'l1', 'linf').
            reward_fn: Reward function (e.g. NavigationReward); caller constructs with desired params.
            terminate_on_reach: If True, episode ends when target is first reached.
            process_noise_std: Std of optional per-step jitter on the realized field.
            obs_noise_std: Std of optional per-step jitter on the observed field.
        """
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

        if not (
            1 <= target_position.i <= config.n_x
            and 1 <= target_position.j <= config.n_y
        ):
            raise ValueError(
                f"target_position {target_position} is outside grid "
                f"({config.n_x}, {config.n_y}, {config.n_z})"
            )
        if config.ndim == 3 and not (1 <= target_position.k <= config.n_z):
            raise ValueError(
                f"target_position.k={target_position.k} is outside grid "
                f"[1, {config.n_z}]"
            )

        self.target_position = target_position
        self.vicinity_radius = vicinity_radius
        self.vicinity_metric = vicinity_metric
        self.terminate_on_reach = terminate_on_reach
        self.reward_fn = reward_fn

        self._target_reached = False
        self._cumulative_reward = 0.0

    def reset(self, rng_key):
        """Reset arena and navigation state."""
        obs = super().reset(rng_key)
        self._target_reached = False
        self._cumulative_reward = 0.0
        return obs

    def _compute_distance(self, pos1: GridPosition, pos2: GridPosition) -> float:
        """Compute distance between two positions using configured metric."""
        diffs = [pos1.i - pos2.i, pos1.j - pos2.j]
        if self.ndim == 3:
            diffs.append(pos1.k - pos2.k)

        diffs = np.array(diffs)

        if self.vicinity_metric == "euclidean":
            return np.sqrt(np.sum(diffs**2))
        elif self.vicinity_metric == "l1":
            return np.sum(np.abs(diffs))
        elif self.vicinity_metric == "linf":
            return np.max(np.abs(diffs))
        else:
            raise ValueError(f"Unknown vicinity_metric: {self.vicinity_metric}")

    def compute_reward(self) -> float:
        """Compute reward (proximity minus step cost)."""
        reward = self.reward_fn.compute_scalar(self.position)

        distance_to_target = self._compute_distance(self.position, self.target_position)
        if distance_to_target <= self.vicinity_radius and not self._target_reached:
            self._target_reached = True

        self._cumulative_reward += reward
        self._last_reward = reward

        return reward

    def is_terminal(self) -> bool:
        """Check if episode should terminate."""
        if self.terminate_on_reach and self._target_reached:
            return True
        return super().is_terminal()

    def get_cumulative_reward(self) -> float:
        """Get cumulative reward for current episode."""
        return self._cumulative_reward

    def get_state(self) -> NavigationArenaState:
        """Get complete navigation arena state."""
        base_state = super().get_state()
        # Create extended navigation state with full config
        return NavigationArenaState(
            # Universal state
            step_count=base_state.step_count,
            last_action=base_state.last_action,
            last_reward=base_state.last_reward,
            rng_key=base_state.rng_key,
            # Grid state
            position=base_state.position,
            last_position=base_state.last_position,
            last_displacement=base_state.last_displacement,
            out_of_bounds=base_state.out_of_bounds,
            initial_position=base_state.initial_position,
            # Navigation dynamic state
            cumulative_reward=self._cumulative_reward,
            target_reached=self._target_reached,
            # Navigation static config
            target_position=self.target_position,
            vicinity_radius=self.vicinity_radius,
        )

    def set_state(self, state: ArenaState) -> None:
        """Restore navigation arena state."""
        super().set_state(state)
        if isinstance(state, NavigationArenaState):
            self._cumulative_reward = state.cumulative_reward
            self._target_reached = state.target_reached
        else:
            self._cumulative_reward = 0.0
            self._target_reached = False
