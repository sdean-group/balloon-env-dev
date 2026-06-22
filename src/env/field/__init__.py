"""Field implementations for environmental wind sources.

A ``FlowField`` is a pure spatial wind source ("what is the wind at point p?").
Forecast/reality relationships are built by composing and sharing fields; the arena
owns all dynamics (noise, clipping, displacement).
"""

from .flow_field import FlowField, unique_fields
from .composite import SumField, ScaledField, ZeroField
from .synthetic import SyntheticFlowField
from .simple_field import ConstantDriftField, UniformDriftField

__all__ = [
    'FlowField',
    'unique_fields',
    'SumField',
    'ScaledField',
    'ZeroField',
    'SyntheticFlowField',
    'ConstantDriftField',
    'UniformDriftField',
]
