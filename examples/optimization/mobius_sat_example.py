#!/usr/bin/env python3
"""
Multi-Strip Möbius SAT Solver Example
======================================

Demonstrates the production-ready multi-strip Möbius SAT solver.

Performance: 0.9262 satisfaction ratio (92.62% of clauses satisfied)
Winner of Investigation 11 (beats all other methods)

This example shows:
1. Basic usage with default parameters (18 strips)
2. Custom strip counts
3. Parameter tuning (omega, coupling, iterations)
4. Comparison to standard hybrid solver

Author: tHHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import time

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.optimization.mobius_sat_solver import (
    SATInstance as MobiusSATInstance,
    MultiStripMobiusSolver,
    solve_mobius_sat
)
from src.hhml.core.optimization.hybrid_sat_solver import (
    SATInstance as HybridSATInstance,
    HybridSATSolver
)


def example_basic():
    """Example 1: Basic usage with defaults."""
    print("="*80)
    print("EXAMPLE 1: Basic Möbius SAT Solving")
    print("="*80)
    print()

    # Create random 3-SAT instance at phase transition
    n_vars = 100
    m_clauses = int(4.2 * n_vars)

    print(f"Generating instance: {n_vars} variables, {m_clauses} clauses")
    instance = MobiusSATInstance.random(n_vars, m_clauses, seed=42)
    print()

    # Solve with defaults (18 strips)
    print("Solving with multi-strip Möbius solver (18 strips)...")
    start = time.time()

    result = solve_mobius_sat(instance, seed=42)

    duration = time.time() - start

    print()
    print("Results:")
    print(f"  Satisfaction ratio: {result['satisfaction_ratio']:.4f}")
    print(f"  Satisfied clauses: {result['num_satisfied']}/{result['total_clauses']}")
    print(f"  Number of strips: {result['num_strips']}")
    print(f"  Solve time: {duration:.3f}s")
    print()


def example_custom_strips():
    """Example 2: Custom strip counts."""
    print("="*80)
    print("EXAMPLE 2: Testing Different Strip Counts")
    print("="*80)
    print()

    instance = MobiusSATInstance.random(n_vars=100, m_clauses=420, seed=42)

    strip_counts = [7, 9, 18, 23, 30]

    print(f"Testing strip counts: {strip_counts}")
    print()

    for num_strips in strip_counts:
        result = solve_mobius_sat(instance, num_strips=num_strips, seed=42)
        print(f"  {num_strips:2d} strips: rho={result['satisfaction_ratio']:.4f}")

    print()


def example_parameter_tuning():
    """Example 3: Parameter tuning."""
    print("="*80)
    print("EXAMPLE 3: Parameter Tuning")
    print("="*80)
    print()

    instance = MobiusSATInstance.random(n_vars=100, m_clauses=420, seed=42)

    # Test different omega values
    print("Testing helical strength (omega):")
    for omega in [0.05, 0.1, 0.15, 0.2]:
        result = solve_mobius_sat(instance, omega=omega, seed=42)
        print(f"  omega={omega:.2f}: rho={result['satisfaction_ratio']:.4f}")

    print()

    # Test different iteration counts
    print("Testing iteration counts:")
    for iters in [1, 3, 5, 10]:
        result = solve_mobius_sat(instance, num_iterations=iters, seed=42)
        print(f"  {iters:2d} iterations: rho={result['satisfaction_ratio']:.4f}")

    print()


def example_comparison():
    """Example 4: Comparison to hybrid solver."""
    print("="*80)
    print("EXAMPLE 4: Möbius vs Hybrid Comparison")
    print("="*80)
    print()

    # Create instances (same problem, different classes for different solvers)
    instance_mobius = MobiusSATInstance.random(n_vars=100, m_clauses=420, seed=42)
    instance_hybrid = HybridSATInstance.random(n_vars=100, m_clauses=420, seed=42)

    # Multi-strip Möbius
    print("Running multi-strip Möbius solver...")
    start = time.time()
    result_mobius = solve_mobius_sat(instance_mobius, seed=42)
    time_mobius = time.time() - start

    # Hybrid (from Investigation 6)
    print("Running optimized hybrid solver...")
    solver_hybrid = HybridSATSolver(instance_hybrid)
    start = time.time()
    result_hybrid = solver_hybrid.solve(max_depth=3, num_iterations=3, omega=0.1, seed=42)
    time_hybrid = time.time() - start

    print()
    print("Results:")
    print()
    print(f"  Multi-strip Möbius: {result_mobius['satisfaction_ratio']:.4f} ({time_mobius:.3f}s)")
    print(f"  Optimized Hybrid:   {result_hybrid['satisfaction_ratio']:.4f} ({time_hybrid:.3f}s)")
    print()

    improvement = ((result_mobius['satisfaction_ratio'] - result_hybrid['satisfaction_ratio']) /
                   result_hybrid['satisfaction_ratio']) * 100

    if result_mobius['satisfaction_ratio'] > result_hybrid['satisfaction_ratio']:
        print(f"  -> Möbius WINS by {improvement:+.2f}%")
    elif result_mobius['satisfaction_ratio'] < result_hybrid['satisfaction_ratio']:
        print(f"  -> Hybrid WINS by {-improvement:+.2f}%")
    else:
        print(f"  -> TIE")

    print()


def example_solver_class():
    """Example 5: Using solver class directly."""
    print("="*80)
    print("EXAMPLE 5: Using MultiStripMobiusSolver Class")
    print("="*80)
    print()

    instance = MobiusSATInstance.random(n_vars=50, m_clauses=210, seed=42)

    # Create solver
    solver = MultiStripMobiusSolver(instance, num_strips=18)

    print(f"Created solver with {solver.num_strips} Möbius strips")
    print(f"Variables per strip: ~{solver.n_vars // solver.num_strips}")
    print()

    # Solve
    print("Solving...")
    result = solver.solve(omega=0.1, coupling_strength=0.5, num_iterations=3, seed=42)

    print()
    print("Results:")
    print(f"  Satisfaction: {result['satisfaction_ratio']:.4f}")
    print()

    # Iteration breakdown
    print("Iteration history:")
    for r in result['iteration_results']:
        print(f"  Iteration {r['iteration']}: rho={r['satisfaction']:.4f}")

    print()

    # Get solution
    assignment, satisfaction = solver.get_solution()
    print(f"Best solution: {satisfaction:.4f} satisfaction")
    print(f"Assignment vector shape: {assignment.shape}")

    print()


def main():
    """Run all examples."""
    import argparse

    parser = argparse.ArgumentParser(description='Möbius SAT Solver Examples')
    parser.add_argument('--example', type=int, default=None,
                       help='Run specific example (1-5), or all if not specified')

    args = parser.parse_args()

    examples = {
        1: ("Basic Usage", example_basic),
        2: ("Custom Strips", example_custom_strips),
        3: ("Parameter Tuning", example_parameter_tuning),
        4: ("Comparison", example_comparison),
        5: ("Solver Class", example_solver_class)
    }

    if args.example is not None:
        if args.example in examples:
            name, func = examples[args.example]
            print(f"\nRunning Example {args.example}: {name}\n")
            func()
        else:
            print(f"Error: Example {args.example} not found. Available: 1-5")
            return 1
    else:
        # Run all examples
        for num, (name, func) in examples.items():
            func()
            print("\n")

    print("="*80)
    print("Examples complete!")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
