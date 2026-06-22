"""Basic grid arena implementation with boundary handling."""

from typing import Tuple
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp

from .abstract_arena import AbstractArena
from ..field.abstract_field import AbstractField
from ..actor.abstract_actor import AbstractActor
from ..utils.types import (
    GridPosition, DisplacementObservation, GridConfig, ArenaState, GridArenaState
)


class GridArena(AbstractArena):
    """Basic grid arena with configurable boundary handling.
    
    Supports both 2D and 3D settings:
    - 3D: Actor controls z-axis, field controls (x, y)
    - 2D: Actor controls y-axis, field controls (x,)
    
    Supports different boundary conditions and serves as base for specific tasks.
    """
    
    def __init__(
        self,
        field: AbstractField,
        actor: AbstractActor,
        config: GridConfig,
        initial_position: GridPosition,
        boundary_mode: str = 'terminal'
    ):
        """Initialize grid arena.
        
        Args:
            field: Environmental field providing ambient displacements.
            actor: Actor with controllable axis dynamics.
            config: Grid configuration.
            initial_position: Starting position for reset.
            boundary_mode: How to handle boundaries:
                - 'clip': Clamp position to valid range
                - 'periodic': Wrap around on ambient axes, clip on controllable
                - 'terminal': Mark as terminal when crossing boundary (default)
        """
        self.field = field
        self.actor = actor
        self.config = config
        self.initial_position = initial_position
        self.boundary_mode = boundary_mode
        
        # Validate boundary mode
        valid_modes = ['clip', 'periodic', 'terminal']
        if boundary_mode not in valid_modes:
            raise ValueError(
                f"boundary_mode must be one of {valid_modes}, got {boundary_mode}"
            )
        
        # Validate initial_position is within grid
        if not (1 <= initial_position.i <= config.n_x and
                1 <= initial_position.j <= config.n_y):
            raise ValueError(
                f"initial_position {initial_position} is outside grid "
                f"({config.n_x}, {config.n_y}, {config.n_z})"
            )
        if config.ndim == 3 and (
            initial_position.k is None or
            not (1 <= initial_position.k <= config.n_z)
        ):
            raise ValueError(
                f"initial_position.k={initial_position.k} is invalid for 3D grid "
                f"[1, {config.n_z}]"
            )
        if config.ndim == 2 and initial_position.k is not None:
            raise ValueError(
                f"initial_position.k must be None for 2D grid, got {initial_position.k}"
            )
        
        # Arena state (updated in reset and step)
        self.position = initial_position
        self.last_position = initial_position
        self.last_displacement = self._zero_displacement()
        self.step_count = 0
        self._out_of_bounds = False
        self._rng = None
        self._last_action = None
        self._last_reward = 0.0
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.config.ndim
    
    def _zero_displacement(self) -> DisplacementObservation:
        """Create zero displacement appropriate for dimensionality."""
        if self.ndim == 3:
            return DisplacementObservation(0.0, 0.0)
        else:
            return DisplacementObservation(0.0, None)
    
    def reset(self, rng_key: jnp.ndarray) -> np.ndarray:
        """Reset arena to initial state."""
        # Split RNG for field and future use
        self._rng = rng_key
        field_key, self._rng = jax.random.split(self._rng)
        
        # Reset field
        self.field.reset(field_key)
        
        # Reset state
        self.position = self.initial_position
        self.last_position = self.initial_position
        self.last_displacement = self._zero_displacement()
        self.step_count = 0
        self._out_of_bounds = False
        self._last_action = None
        self._last_reward = 0.0
        
        return self._get_observation()
    
    def step(self, action: int) -> np.ndarray:
        """Execute one simulation step."""
        # Track action
        self._last_action = action
        
        # Split RNG keys for field and actor
        field_key, actor_key, self._rng = jax.random.split(self._rng, 3)
        
        # 1. Sample ambient displacement from field (continuous)
        displacement_obs = self.field.sample_displacement(
            self.position, field_key
        )
        
        # 2. Apply ambient displacement (discrete state transition)
        # Store last position before update
        self.last_position = self.position
        
        if self.ndim == 3:
            # 3D: update (i, j) from field (continuous), keep k
            new_i = self.position.i + displacement_obs.u
            new_j = self.position.j + displacement_obs.v
            self.position = GridPosition(new_i, new_j, self.position.k)
        else:
            # 2D: update (i,) from field (continuous), keep j
            new_i = self.position.i + displacement_obs.u
            self.position = GridPosition(new_i, self.position.j, None)
        
        # 3. Apply controllable action
        self.position = self.actor.step_controllable(self.position, action, actor_key)
        
        # 4. Enforce boundaries
        self.position, self._out_of_bounds = self._enforce_boundaries(
            self.position
        )
        
        # Store displacement observation for next observation
        self.last_displacement = displacement_obs
        
        self.step_count += 1
        
        return self._get_observation()
    
    def get_state(self) -> GridArenaState:
        """Get complete grid arena state."""
        return GridArenaState(
            # Universal state
            step_count=self.step_count,
            last_action=self._last_action,
            last_reward=self._last_reward,
            rng_key=self._rng,
            # Grid-specific dynamic state
            position=self.position,
            last_position=self.last_position,
            last_displacement=self.last_displacement,
            out_of_bounds=self._out_of_bounds,
            # Static config
            initial_position=self.initial_position
        )
    
    def set_state(self, state: ArenaState) -> None:
        """Restore grid arena state."""
        # Restore universal state
        self.step_count = state.step_count
        self._last_action = state.last_action
        self._last_reward = state.last_reward
        self._rng = state.rng_key
        
        # Restore grid-specific fields if available
        if isinstance(state, GridArenaState):
            self.position = state.position
            self.last_position = state.last_position
            self.last_displacement = state.last_displacement
            self._out_of_bounds = state.out_of_bounds
            # Note: initial_position is static config, not restored
        else:
            # Fallback for base ArenaState (shouldn't happen in practice)
            import warnings
            warnings.warn(
                f"GridArena.set_state() received {type(state).__name__} instead of "
                f"GridArenaState. Resetting grid-specific fields to defaults.",
                UserWarning,
                stacklevel=2
            )
            self.position = self.initial_position
            self.last_position = self.initial_position
            self.last_displacement = self._zero_displacement()
            self._out_of_bounds = False
    
    def compute_reward(self) -> float:
        """Default reward (override in subclasses for specific tasks)."""
        return 0.0
    
    def is_terminal(self) -> bool:
        """Default termination logic (override in subclasses)."""
        if self.boundary_mode == 'terminal':
            return self._out_of_bounds
        return False
    
    @property
    def observation_space(self) -> gym.Space:
        """Observation space based on dimensionality.
        
        - 3D: [i, j, k, u_obs, v_obs] (5 dimensions)
        - 2D: [i, j, u_obs] (3 dimensions)
        """
        d_max = self.field.d_max
        
        if self.ndim == 3:
            return gym.spaces.Box(
                low=np.array([1, 1, 1, -d_max, -d_max], dtype=np.float32),
                high=np.array([
                    self.config.n_x, self.config.n_y, self.config.n_z,
                    d_max, d_max
                ], dtype=np.float32),
                dtype=np.float32
            )
        else:
            return gym.spaces.Box(
                low=np.array([1, 1, -d_max], dtype=np.float32),
                high=np.array([self.config.n_x, self.config.n_y, d_max], dtype=np.float32),
                dtype=np.float32
            )
    
    def _get_observation(self) -> np.ndarray:
        """Construct flat observation array."""
        if self.ndim == 3:
            return np.array([
                float(self.position.i),
                float(self.position.j),
                float(self.position.k),
                self.last_displacement.u,
                self.last_displacement.v
            ], dtype=np.float32)
        else:
            return np.array([
                float(self.position.i),
                float(self.position.j),
                self.last_displacement.u
            ], dtype=np.float32)
    
    def _enforce_boundaries(
        self, position: GridPosition
    ) -> Tuple[GridPosition, bool]:
        """Enforce boundary conditions based on mode.
        
        Returns:
            (new_position, out_of_bounds_flag)
        """
        out_of_bounds = False
        
        if self.ndim == 3:
            return self._enforce_boundaries_3d(position)
        else:
            return self._enforce_boundaries_2d(position)
    
    def _enforce_boundaries_3d(
        self, position: GridPosition
    ) -> Tuple[GridPosition, bool]:
        """Enforce boundaries for 3D setting."""
        out_of_bounds = False
        
        if self.boundary_mode == 'clip':
            new_i = float(max(1.0, min(position.i, self.config.n_x)))
            new_j = float(max(1.0, min(position.j, self.config.n_y)))
            new_k = float(max(1.0, min(position.k, self.config.n_z)))

            out_of_bounds = (
                new_i != position.i or
                new_j != position.j or
                new_k != position.k
            )
            position = GridPosition(new_i, new_j, new_k)

        elif self.boundary_mode == 'periodic':
            # Wrap around on ambient axes (i, j) over the domain [1, n], clip on
            # controllable (k). Float modulo keeps the result in [1, n).
            new_i = ((position.i - 1) % self.config.n_x) + 1
            new_j = ((position.j - 1) % self.config.n_y) + 1
            new_k = float(max(1.0, min(position.k, self.config.n_z)))
            position = GridPosition(new_i, new_j, new_k)
            
        elif self.boundary_mode == 'terminal':
            out_of_bounds = (
                position.i < 1 or position.i > self.config.n_x or
                position.j < 1 or position.j > self.config.n_y or
                position.k < 1 or position.k > self.config.n_z
            )
        
        return position, out_of_bounds
    
    def _enforce_boundaries_2d(
        self, position: GridPosition
    ) -> Tuple[GridPosition, bool]:
        """Enforce boundaries for 2D setting."""
        out_of_bounds = False
        
        if self.boundary_mode == 'clip':
            new_i = float(max(1.0, min(position.i, self.config.n_x)))
            new_j = float(max(1.0, min(position.j, self.config.n_y)))

            out_of_bounds = (new_i != position.i or new_j != position.j)
            position = GridPosition(new_i, new_j, None)

        elif self.boundary_mode == 'periodic':
            # Wrap around on ambient axis (i) over the domain [1, n], clip on
            # controllable (j). Float modulo keeps the result in [1, n).
            new_i = ((position.i - 1) % self.config.n_x) + 1
            new_j = float(max(1.0, min(position.j, self.config.n_y)))
            position = GridPosition(new_i, new_j, None)
            
        elif self.boundary_mode == 'terminal':
            out_of_bounds = (
                position.i < 1 or position.i > self.config.n_x or
                position.j < 1 or position.j > self.config.n_y
            )
        
        return position, out_of_bounds

