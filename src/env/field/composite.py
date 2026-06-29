"""Composite flow fields: build forecast/reality relationships by composition.

Three tiny combinators express the whole forecast/reality relationship without any
dedicated "coupling" class:

    SumField(a, b)      a + b
    ScaledField(a, s)   s * a
    ZeroField(config)   always (0, 0) -- an explicit "nothing"

Correlation between the realized wind ``W`` and the observed wind ``W_hat`` comes from
*sharing a field object* between them::

    shared = SyntheticFlowField(config, sigma=5, lengthscale=10)
    error  = SyntheticFlowField(config, sigma=1, lengthscale=2)

    observed = shared             # the forecast
    realized = shared + error     # truth = forecast + structured error (reuses shared!)

Composite ``reset`` is a no-op: the arena resets each *unique* leaf reachable via
``sub_fields`` exactly once, so shared sub-fields are drawn once and stay correlated.
"""

from typing import Optional, Tuple

import numpy as np
import jax.numpy as jnp

from .flow_field import FlowField
from ..utils.types import GridPosition, GridConfig


class SumField(FlowField):
    """Sum of two fields: ``a + b``."""

    def __init__(self, a: FlowField, b: FlowField):
        if a.config.ndim != b.config.ndim:
            raise ValueError("Cannot add fields with different ndim")
        super().__init__(a.config)
        self.a = a
        self.b = b

    def reset(self, rng_key: jnp.ndarray) -> None:
        # No own randomness: children are reset by the arena's unique-field walk.
        pass

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        ua, va = self.a.velocity_at(position, t)
        ub, vb = self.b.velocity_at(position, t)
        u = ua + ub
        if va is None or vb is None:
            return (u, None)
        return (u, va + vb)

    def velocity_field(self, t: float = 0.0) -> Optional[np.ndarray]:
        fa = self.a.velocity_field(t)
        fb = self.b.velocity_field(t)
        if fa is None or fb is None:
            return None
        return fa + fb

    def sub_fields(self) -> Tuple[FlowField, ...]:
        return (self.a, self.b)

    @property
    def time_varying(self) -> bool:
        return self.a.time_varying or self.b.time_varying


class ScaledField(FlowField):
    """Field scaled by a constant: ``scalar * field``."""

    def __init__(self, field: FlowField, scalar: float):
        super().__init__(field.config)
        self.field = field
        self.scalar = float(scalar)

    def reset(self, rng_key: jnp.ndarray) -> None:
        pass

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        u, v = self.field.velocity_at(position, t)
        if v is None:
            return (self.scalar * u, None)
        return (self.scalar * u, self.scalar * v)

    def velocity_field(self, t: float = 0.0) -> Optional[np.ndarray]:
        f = self.field.velocity_field(t)
        if f is None:
            return None
        return self.scalar * f

    def sub_fields(self) -> Tuple[FlowField, ...]:
        return (self.field,)

    @property
    def time_varying(self) -> bool:
        return self.field.time_varying


class ZeroField(FlowField):
    """A field that is always zero -- an explicit "nothing"."""

    def __init__(self, config: GridConfig):
        super().__init__(config)

    def reset(self, rng_key: jnp.ndarray) -> None:
        pass

    def velocity_at(
        self, position: GridPosition, t: float = 0.0
    ) -> Tuple[float, Optional[float]]:
        if self.ndim == 3:
            return (0.0, 0.0)
        return (0.0, None)

    def velocity_field(self, t: float = 0.0) -> np.ndarray:
        if self.ndim == 3:
            return np.zeros((self.config.n_x, self.config.n_y, self.config.n_z, 2))
        return np.zeros((self.config.n_x, self.config.n_y, 1))
