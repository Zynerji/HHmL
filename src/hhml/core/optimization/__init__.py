"""
HHmL Optimization Module
========================

Production-ready optimization algorithms leveraging recursive topology and
spectral graph methods.

Key Components:
- MultiStripMobiusSolver: 18-strip Möbius SAT solver (0.9262 satisfaction) **NEW**
- HybridSATSolver: Optimized hybrid SAT solver (0.8943 satisfaction)
- RecursiveOptimizer: General recursive topology optimization
- FiedlerPartitioner: Constraint-aware graph partitioning

Performance Comparison:
- Multi-strip Möbius: 0.9262 (WINNER - Investigation 11)
- Optimized Hybrid: 0.8943 (Investigation 6)
- Uniform baseline: 0.8786

References:
- Hash Quine Investigation Suite (HASH-QUINE/investigations/)
- Investigation 11: Ultimate Hybrid Möbius SAT (18 strips optimal)
"""

# Import Möbius SAT (winner of Investigation 11) as default
from .mobius_sat_solver import (
    MultiStripMobiusSolver,
    solve_mobius_sat,
    SATInstance  # Möbius version is the default (0.9262 satisfaction)
)

# Import hybrid solver (Investigation 6) with explicit naming
from .hybrid_sat_solver import (
    HybridSATSolver,
    SATInstance as HybridSATInstance  # Legacy hybrid version (0.8943 satisfaction)
)
from .recursive_optimizer import RecursiveOptimizer, OptimizationProblem
from .fiedler_partitioner import FiedlerPartitioner, PartitioningConfig

__all__ = [
    # Möbius SAT (default - winner of Investigation 11)
    'MultiStripMobiusSolver',
    'solve_mobius_sat',
    'SATInstance',  # Möbius version

    # Hybrid SAT (legacy - Investigation 6)
    'HybridSATSolver',
    'HybridSATInstance',  # Hybrid version for backward compatibility

    # General optimization
    'RecursiveOptimizer',
    'OptimizationProblem',
    'FiedlerPartitioner',
    'PartitioningConfig',
]

__version__ = '0.1.0'
