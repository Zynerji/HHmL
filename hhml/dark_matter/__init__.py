"""
Dark Matter as Multiversal Pruning Residue - HHmL Module
==========================================================

This module implements a test of the theory that dark matter emerges
as informational residue from the holographic universe pruning
discordant multiverse branches.

Core components:
- multiverse_generator: Create perturbed Möbius strip branches
- pruning_simulator: Coherence-based branch pruning
- residue_analyzer: Measure dark matter signatures
- cosmological_validator: Test against observations

Author: HHmL Project
Date: 2025-12-17
"""

from .multiverse_generator import generate_multiverse_branches, MultiverseConfig
from .pruning_simulator import prune_discordant, compute_coherence, PruningResult
from .residue_analyzer import measure_dark_residue, DarkMatterMetrics
from .cosmological_validator import validate_theory, CosmologicalTests

__all__ = [
    'generate_multiverse_branches',
    'MultiverseConfig',
    'prune_discordant',
    'compute_coherence',
    'PruningResult',
    'measure_dark_residue',
    'DarkMatterMetrics',
    'validate_theory',
    'CosmologicalTests',
]

__version__ = '0.1.0'
