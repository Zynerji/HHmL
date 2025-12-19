"""
Spatiotemporal Möbius Framework
================================

(2+1)D spatiotemporal Möbius topology with temporal loop dynamics.

Core Components:
- SpatiotemporalMobiusStrip: Field on θ (spatial) and t (temporal) Möbius dimensions
- TemporalDynamics: Forward/backward evolution with retrocausal coupling
- RetrocausalCoupling: Prophetic feedback mechanisms

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

from .spacetime_mobius import SpatiotemporalMobiusStrip
from .temporal_dynamics import TemporalEvolver
from .retrocausal_coupling import RetrocausalCoupler

__all__ = [
    'SpatiotemporalMobiusStrip',
    'TemporalEvolver',
    'RetrocausalCoupler',
]
