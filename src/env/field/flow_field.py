"""FlowField: a pure spatial wind source.

A ``FlowField`` answers exactly one question: "what is the wind at point ``p``?"
It is the *source* of velocities (synthetic GP, real ERA5, learned model). It is
deterministic after ``reset()`` and knows nothing about forecasts, noise, clipping,
or agents -- all of that lives in the arena.

Everything the old ``AbstractField`` did beyond "give me the wind" (per-step noise,
clipping, displacement PMFs, ``disp_levels``) has moved to the arena or been deleted.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp

from ..utils.types import GridPosition, GridConfig


class FlowField(ABC):
    """A deterministic continuous velocity field, fixed after ``reset()``. Nothing else.

    Subclasses differ ONLY in where velocities come from.
    """

    def __init__(self, config: GridConfig):
        self.config = config

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return self.config.ndim

    @abstractmethod
    def reset(self, rng_key: jnp.ndarray) -> None:
        """Draw one realization of the field (called once per episode)."""

    @abstractmethod
    def velocity_at(self, position: GridPosition) -> Tuple[float, Optional[float]]:
        """Deterministic (u, v) at a continuous position. v is None in 2D."""

    def velocity_field(self) -> Optional[np.ndarray]:
        """(n_x, n_y[, n_z], ndim) grid of velocities -- for plotting. Optional."""
        return None

    def sub_fields(self) -> Tuple["FlowField", ...]:
        """Child fields this field is composed from (empty for leaf sources).

        Used by the arena to walk the field tree and reset each unique field
        exactly once, so shared sub-fields stay correlated.
        """
        return ()

    # Composition sugar (see composite.py)
    def __add__(self, other: "FlowField") -> "FlowField":
        from .composite import SumField

        return SumField(self, other)

    def __mul__(self, scalar: float) -> "FlowField":
        from .composite import ScaledField

        return ScaledField(self, scalar)

    __rmul__ = __mul__


def unique_fields(*roots: FlowField) -> Tuple[FlowField, ...]:
    """Dedupe every field reachable from ``roots`` by ``id()``.

    Walks the field tree via :meth:`FlowField.sub_fields` so that a field object
    shared between the realized and observed fields is returned exactly once.
    """
    seen: dict[int, FlowField] = {}
    stack = list(roots)
    while stack:
        f = stack.pop()
        if f is None or id(f) in seen:
            continue
        seen[id(f)] = f
        stack.extend(f.sub_fields())
    return tuple(seen.values())
