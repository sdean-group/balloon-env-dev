"""Type definitions for the grid environment.

Supports both 2D and 3D settings:
- 3D: Agent controls z-axis (vertical), field controls x-y plane (ambient)
- 2D: Agent controls y-axis (controllable), field controls x-axis (ambient)

Terminology:
- Ambient axes: Controlled by the environmental field (1 axis for 2D, 2 for 3D)
- Controllable axis: Controlled by the agent's actions (always 1 axis)
"""

from typing import NamedTuple, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
from dataclasses import asdict


class GridPosition(NamedTuple):
    """Grid position coordinates (continuous, 1-indexed domain).

    Positions are continuous floats over the domain [1, n] on each axis;
    integer values are valid but no rounding is applied to the live dynamics.

    Supports both 2D and 3D settings:
    - 3D: (i, j, k) where k is controllable axis, (i, j) are ambient
    - 2D: (i, j, None) where j is controllable axis, (i,) is ambient

    Args:
        i: First ambient axis coordinate [1, n_x]
        j: Second ambient (3D) or controllable (2D) axis [1, n_y]
        k: Controllable axis for 3D [1, n_z], None for 2D
    """
    i: float  # Ambient axis 1 (always present)
    j: float  # Ambient axis 2 (3D) OR controllable axis (2D)
    k: Optional[float] = None  # Controllable axis (3D only)
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 3 if self.k is not None else 2
    
    @property
    def controllable(self) -> int:
        """Position on controllable axis (agent-controlled)."""
        return self.k if self.k is not None else self.j
    
    @property
    def ambient(self) -> Tuple[int, ...]:
        """Position on ambient axes (field-controlled)."""
        return (self.i, self.j) if self.k is not None else (self.i,)


class DisplacementObservation(NamedTuple):
    """Observed displacement on ambient axes (field-controlled).
    
    Supports both 2D and 3D settings:
    - 3D: (u, v) displacement in x-y plane
    - 2D: (u, None) displacement in x direction only
    
    Following BLE's units.py pattern: stores continuous values but provides
    integer properties for discrete state transitions.
    
    Args:
        u: Displacement on ambient axis 1 (always present)
        v: Displacement on ambient axis 2 (3D only, None for 2D)
    """
    u: float  # Ambient axis 1 displacement
    v: Optional[float] = None  # Ambient axis 2 displacement (3D only)
    
    @property
    def ndim(self) -> int:
        """Number of ambient dimensions."""
        return 2 if self.v is not None else 1
    
    @property
    def as_tuple(self) -> Tuple[float, ...]:
        """Displacement as tuple (handles both 2D and 3D)."""
        return (self.u, self.v) if self.v is not None else (self.u,)
    
    @property
    def as_int_tuple(self) -> Tuple[int, ...]:
        """Discrete displacement as tuple."""
        return (self.u_int, self.v_int) if self.v is not None else (self.u_int,)
    
    @property
    def u_int(self) -> int:
        """Get discrete displacement on ambient axis 1."""
        return int(round(self.u))
    
    @property
    def v_int(self) -> Optional[int]:
        """Get discrete displacement on ambient axis 2 (None for 2D)."""
        return int(round(self.v)) if self.v is not None else None


class GridConfig(NamedTuple):
    """Grid environment configuration.
    
    Supports both 2D and 3D settings:
    - 3D: n_z is set, agent controls z, field controls (x, y)
    - 2D: n_z is None, agent controls y, field controls (x,)
    
    Args:
        n_x: Grid size on ambient axis 1 (always present)
        n_y: Grid size on ambient axis 2 (3D) or controllable axis (2D)
        n_z: Grid size on controllable axis (3D only, None for 2D)
    """
    n_x: int  # Ambient axis 1 size
    n_y: int  # Ambient axis 2 (3D) or controllable axis (2D)
    n_z: Optional[int] = None  # Controllable axis size (3D only, None for 2D)
    
    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 3 if self.n_z is not None else 2
    
    @property
    def n_controllable(self) -> int:
        """Size of controllable axis."""
        return self.n_z if self.n_z is not None else self.n_y
    
    @property
    def n_ambient(self) -> Tuple[int, ...]:
        """Sizes of ambient axes."""
        return (self.n_x, self.n_y) if self.n_z is not None else (self.n_x,)
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Full grid shape."""
        if self.n_z is not None:
            return (self.n_x, self.n_y, self.n_z)
        else:
            return (self.n_x, self.n_y)

    @classmethod
    def create(cls, n_x: int, n_y: int, n_z: Optional[int] = None) -> 'GridConfig':
        """Create GridConfig with validation.
        
        Args:
            n_x: Ambient axis 1 size
            n_y: Ambient axis 2 (3D) or controllable axis (2D)
            n_z: Controllable axis size for 3D (None for 2D)
        
        Returns:
            Validated GridConfig instance
        """
        # Validate dimensions
        if n_x <= 0 or n_y <= 0:
            raise ValueError("Grid dimensions must be positive integers")
        if n_z is not None and n_z <= 0:
            raise ValueError("n_z must be positive if provided")
        
        return cls(n_x, n_y, n_z)


@dataclass(frozen=True)
class ArenaState:
    """Base arena state (common across all arena types).
    
    Contains only truly universal fields that apply to ANY arena implementation.
    Using frozen dataclass for immutability and inheritance support.
    
    Fields:
        step_count: Number of steps taken in current episode
        last_action: Most recent action (None at episode start)
        last_reward: Reward from last step
        rng_key: JAX PRNG key for reproducibility
    """
    step_count: int
    last_action: Optional[int]
    last_reward: float
    rng_key: jnp.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (includes all fields from subclasses)."""
        return asdict(self)


@dataclass(frozen=True)
class GridArenaState(ArenaState):
    """Grid arena state (adds spatial state and grid-specific fields).
    
    Extends base state with grid world spatial information.
    
    Dynamic state:
        position: Current grid position
        last_position: Previous position (for trajectory tracking)
        last_displacement: Last observed field displacement
        out_of_bounds: Whether position violates boundaries
    
    Static config (for visualization/reproducibility):
        initial_position: Starting position for episode
    """
    position: GridPosition
    last_position: Optional[GridPosition]
    last_displacement: Optional[DisplacementObservation]
    out_of_bounds: bool
    initial_position: GridPosition  # Static: needed for visualization


@dataclass(frozen=True)
class NavigationArenaState(GridArenaState):
    """Navigation arena state (adds navigation task state and configuration).
    
    Extends grid state with navigation-specific information.
    
    Dynamic state:
        cumulative_reward: Total reward accumulated in episode
        target_reached: Whether target vicinity has been reached
    
    Static config (for visualization/analysis):
        target_position: Goal position
        vicinity_radius: Radius defining "reached" region
    """
    # Dynamic navigation state
    cumulative_reward: float
    target_reached: bool
    
    # Static task configuration (for visualization/analysis)
    target_position: GridPosition
    vicinity_radius: float


@dataclass(frozen=True)
class DynamicSGArenaState(GridArenaState):
    """State for the Dynamic Start-Goal Arena.

    Extends grid state with dual-level bookkeeping (segment and global)
    for multi-segment simulation runs.

    Segment-level (reset on each new segment):
        segment_step_count: Steps taken in the current segment
        segment_cumulative_reward: Reward accumulated in the current segment
        target_reached: Whether the current target vicinity has been reached

    Global-level (never reset except on full reset):
        global_step_count: Total steps across all segments
        global_cumulative_reward: Total reward across all segments

    Task config (for visualization/analysis):
        target_position: Current goal position
        vicinity_radius: Radius defining "reached" region
        segment_index: Zero-based index of the current segment
    """
    # Segment-level
    segment_step_count: int
    segment_cumulative_reward: float
    target_reached: bool
    # Global-level
    global_step_count: int
    global_cumulative_reward: float
    # Task config
    target_position: GridPosition
    vicinity_radius: float
    segment_index: int