"""
HHmL: Holo-Harmonic Möbius Lattice Framework

A computational research platform for exploring holographic resonance
on Möbius strip topology.

This is a specialized fork of iVHL (Vibrational Helical Lattice) focused
on closed-loop topological structures for enhanced vortex stability and
harmonic richness.

Modules:
- mobius: Möbius-specific training and topology
- resonance: Holographic boundary dynamics
- gft: Group Field Theory condensate
- tensor_networks: MERA holography and RT formula
- utils: Utilities and validation

Author: Zynerji / iVHL Framework
Date: 2025-12-16
"""

__version__ = "0.1.0"
__author__ = "Zynerji"
__parent_project__ = "iVHL (Vibrational Helical Lattice)"

# Core modules
from . import mobius
from . import resonance
from . import gft
from . import tensor_networks
from . import utils

__all__ = [
    "mobius",
    "resonance",
    "gft",
    "tensor_networks",
    "utils",
]
