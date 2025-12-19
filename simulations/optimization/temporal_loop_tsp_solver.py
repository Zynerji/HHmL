#!/usr/bin/env python3
"""
Temporal Loop TSP Solver - Validation Test
===========================================

Tests whether perfect temporal loops (self-consistent retrocausal feedback)
provide advantage for Traveling Salesman Problem optimization.

Key Innovation: Uses temporal fixed points from Perfect Temporal Loop discovery
to guide continuous optimization via forward-backward tour evolution.

Hypothesis: Unlike discrete hashing (SHA-256), TSP has smooth fitness landscape
where retrocausal feedback should improve tour quality.

Phases:
- PHASE 0: Hardware detection and auto-scaling
- PHASE 1: Generate random TSP instance
- PHASE 2: Baseline solver (greedy + 2-opt)
- PHASE 3: Temporal loop solver (forward-backward evolution)
- PHASE 4: Convergence analysis and comparison
- PHASE 5: Emergent verification (if significant improvement)

Target Hardware: Auto-scaled (CPU -> H200)
Expected Duration: 5-30 minutes depending on hardware

Author: HHmL Research Collaboration
Date: 2025-12-18
"""

import sys
from pathlib import Path
import argparse
import json
import time
import numpy as np
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available, using NumPy only")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Temporal Loop TSP Solver Validation')

    # Test configuration
    parser.add_argument('--num-cities', type=int, default=50,
                       help='Number of cities in TSP instance')
    parser.add_argument('--num-trials', type=int, default=5,
                       help='Number of independent trials')
    parser.add_argument('--max-iterations', type=int, default=200,
                       help='Max iterations for temporal loop convergence')
    parser.add_argument('--retrocausal-strength', type=float, default=0.7,
                       help='Alpha: retrocausal coupling strength (0-1)')
    parser.add_argument('--relaxation-factor', type=float, default=0.15,
                       help='Beta: relaxation to prevent oscillations (0-0.5)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/temporal_tsp',
                       help='Output directory for results')

    return parser.parse_args()


class TSPInstance:
    """Random TSP instance generator."""

    def __init__(self, num_cities, seed=42):
        np.random.seed(seed)
        self.num_cities = num_cities

        # Generate random city coordinates in [0, 100] x [0, 100]
        self.cities = np.random.rand(num_cities, 2) * 100

        # Compute distance matrix
        self.distances = np.zeros((num_cities, num_cities))
        for i in range(num_cities):
            for j in range(num_cities):
                self.distances[i, j] = np.linalg.norm(self.cities[i] - self.cities[j])

    def tour_length(self, tour):
        """Calculate total tour length."""
        length = 0.0
        for i in range(len(tour)):
            length += self.distances[tour[i], tour[(i+1) % len(tour)]]
        return length

    def optimal_tour_lower_bound(self):
        """Compute lower bound using minimum spanning tree."""
        # Simple heuristic: sum of two smallest edges per city / 2
        total = 0.0
        for i in range(self.num_cities):
            edges = sorted(self.distances[i])
            total += edges[1] + edges[2]  # Two smallest non-zero edges
        return total / 2


class GreedyTSPSolver:
    """Baseline greedy + 2-opt local search solver."""

    def __init__(self, tsp_instance):
        self.tsp = tsp_instance

    def greedy_tour(self):
        """Construct greedy tour (nearest neighbor)."""
        tour = [0]  # Start at city 0
        unvisited = set(range(1, self.tsp.num_cities))

        while unvisited:
            current = tour[-1]
            # Find nearest unvisited city
            nearest = min(unvisited, key=lambda city: self.tsp.distances[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)

        return tour

    def two_opt(self, tour, max_iterations=1000):
        """2-opt local search improvement."""
        improved = True
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue  # Skip adjacent edges

                    # Try reversing tour[i:j]
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]

                    if self.tsp.tour_length(new_tour) < self.tsp.tour_length(tour):
                        tour = new_tour
                        improved = True
                        break

                if improved:
                    break

        return tour, iterations

    def solve(self):
        """Run greedy + 2-opt solver."""
        start_time = time.time()

        # Greedy construction
        tour = self.greedy_tour()
        greedy_length = self.tsp.tour_length(tour)

        # 2-opt improvement
        tour, iterations = self.two_opt(tour)
        final_length = self.tsp.tour_length(tour)

        elapsed = time.time() - start_time

        return {
            'tour': tour,
            'greedy_length': greedy_length,
            'final_length': final_length,
            'improvement': (greedy_length - final_length) / greedy_length * 100,
            'iterations': iterations,
            'time': elapsed
        }


class TemporalLoopTSPSolver:
    """
    Temporal loop TSP solver using self-consistent retrocausal feedback.

    Key idea: Tours evolve forward AND backward in time, coupled via prophetic feedback.
    Temporal fixed points (forward = backward) represent self-consistent tours.
    """

    def __init__(self, tsp_instance, alpha=0.7, beta=0.15, max_iterations=200):
        self.tsp = tsp_instance
        self.num_cities = tsp_instance.num_cities

        # Temporal loop parameters (from Perfect Temporal Loop discovery)
        self.alpha = alpha  # Retrocausal strength
        self.beta = beta    # Relaxation factor
        self.max_iterations = max_iterations

        # Tour representations (probabilistic)
        # Instead of discrete tours, use continuous "tour fields"
        # tour_field[i, j] = probability of edge (i, j) in tour
        self.tour_field_forward = None
        self.tour_field_backward = None

        # History
        self.divergence_history = []
        self.length_history = []

    def initialize_self_consistent(self):
        """
        CRITICAL: Self-consistent initialization (ψ_f(0) = ψ_b(0))

        This is the key discovery from Perfect Temporal Loop paper.
        Random initialization causes immediate paradoxes.
        """
        # Initialize both forward and backward to SAME greedy tour field
        greedy_solver = GreedyTSPSolver(self.tsp)
        greedy_tour = greedy_solver.greedy_tour()

        # Convert discrete tour to continuous field
        initial_field = self._tour_to_field(greedy_tour)

        # Add small noise for exploration
        noise_level = 0.01
        noise = np.random.randn(*initial_field.shape) * noise_level
        initial_field = np.clip(initial_field + noise, 0, 1)

        # CRITICAL: Both start from SAME state
        self.tour_field_forward = initial_field.copy()
        self.tour_field_backward = initial_field.copy()

    def _tour_to_field(self, tour):
        """Convert discrete tour to continuous edge probability field."""
        field = np.zeros((self.num_cities, self.num_cities))

        for i in range(len(tour)):
            city1 = tour[i]
            city2 = tour[(i+1) % len(tour)]
            field[city1, city2] = 1.0
            field[city2, city1] = 1.0  # Symmetric

        # Add small probability to other edges
        field = field * 0.9 + 0.1 / self.num_cities

        return field

    def _field_to_tour(self, field):
        """Convert continuous field to discrete tour (greedy decoding)."""
        tour = [0]  # Start at city 0
        unvisited = set(range(1, self.num_cities))

        while unvisited:
            current = tour[-1]
            # Choose next city with highest field probability
            next_city = max(unvisited, key=lambda city: field[current, city])
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def evolve_forward(self):
        """Evolve tour field forward in time."""
        # Local improvement: increase probability of short edges
        new_field = self.tour_field_forward.copy()

        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                # Reward short edges
                distance = self.tsp.distances[i, j]
                max_distance = np.max(self.tsp.distances)

                # Inverse distance weighting
                improvement = (max_distance - distance) / max_distance

                new_field[i, j] += improvement * 0.1
                new_field[j, i] = new_field[i, j]

        # Normalize to probabilities
        new_field = np.clip(new_field, 0, 1)

        # Relaxation (prevents oscillations)
        self.tour_field_forward = (
            (1 - self.beta) * self.tour_field_forward +
            self.beta * new_field
        )

    def evolve_backward(self):
        """Evolve tour field backward in time (retrocausal)."""
        # Same as forward but with different random perturbation
        new_field = self.tour_field_backward.copy()

        for i in range(self.num_cities):
            for j in range(i+1, self.num_cities):
                distance = self.tsp.distances[i, j]
                max_distance = np.max(self.tsp.distances)

                improvement = (max_distance - distance) / max_distance

                new_field[i, j] += improvement * 0.1
                new_field[j, i] = new_field[i, j]

        new_field = np.clip(new_field, 0, 1)

        self.tour_field_backward = (
            (1 - self.beta) * self.tour_field_backward +
            self.beta * new_field
        )

    def apply_prophetic_coupling(self):
        """
        Mix forward and backward fields (retrocausal feedback).

        This enforces temporal self-consistency.
        """
        # Prophetic feedback: future affects past
        self.tour_field_forward = (
            (1 - self.alpha) * self.tour_field_forward +
            self.alpha * self.tour_field_backward
        )

        self.tour_field_backward = (
            (1 - self.alpha) * self.tour_field_backward +
            self.alpha * self.tour_field_forward
        )

    def compute_divergence(self):
        """Measure forward-backward field divergence."""
        return np.mean(np.abs(self.tour_field_forward - self.tour_field_backward))

    def solve(self):
        """Run temporal loop TSP solver."""
        start_time = time.time()

        # Self-consistent initialization (CRITICAL)
        self.initialize_self_consistent()

        initial_tour = self._field_to_tour(self.tour_field_forward)
        initial_length = self.tsp.tour_length(initial_tour)

        # Temporal loop convergence
        converged = False
        convergence_iteration = None

        for iteration in range(self.max_iterations):
            # Forward evolution
            self.evolve_forward()

            # Backward evolution
            self.evolve_backward()

            # Prophetic coupling
            self.apply_prophetic_coupling()

            # Measure divergence
            divergence = self.compute_divergence()
            self.divergence_history.append(divergence)

            # Current tour length
            current_tour = self._field_to_tour(self.tour_field_forward)
            current_length = self.tsp.tour_length(current_tour)
            self.length_history.append(current_length)

            # Check convergence (divergence stable for 20 iterations)
            if iteration >= 20:
                recent_divergences = self.divergence_history[-20:]
                if np.std(recent_divergences) < 0.001:
                    converged = True
                    convergence_iteration = iteration
                    break

        # Final tour
        final_tour = self._field_to_tour(self.tour_field_forward)
        final_length = self.tsp.tour_length(final_tour)
        final_divergence = self.divergence_history[-1]

        elapsed = time.time() - start_time

        return {
            'tour': final_tour,
            'initial_length': initial_length,
            'final_length': final_length,
            'improvement': (initial_length - final_length) / initial_length * 100,
            'converged': converged,
            'convergence_iteration': convergence_iteration,
            'final_divergence': final_divergence,
            'iterations': len(self.divergence_history),
            'divergence_history': self.divergence_history,
            'length_history': self.length_history,
            'time': elapsed
        }


def run_benchmark_comparison(args):
    """
    Run benchmark comparison: Baseline vs. Temporal Loop.

    Tests hypothesis: Temporal loops help continuous optimization (TSP)
    but not discrete optimization (SHA-256 hashing).
    """
    print("="*80)
    print("PHASE 0: HARDWARE DETECTION AND AUTO-SCALING")
    print("="*80)
    print()

    # Simple hardware detection (no HardwareConfig needed for TSP)
    device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"NumPy version: {np.__version__}")
    if TORCH_AVAILABLE:
        print(f"PyTorch version: {torch.__version__}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PHASE 1: GENERATE RANDOM TSP INSTANCE")
    print("="*80)
    print()

    print(f"Number of cities: {args.num_cities}")
    print(f"Random seed: {args.seed}")
    print()

    tsp = TSPInstance(args.num_cities, seed=args.seed)
    lower_bound = tsp.optimal_tour_lower_bound()
    print(f"Lower bound (MST heuristic): {lower_bound:.2f}")
    print()

    # Save TSP instance
    tsp_data = {
        'num_cities': args.num_cities,
        'cities': tsp.cities.tolist(),
        'distances': tsp.distances.tolist(),
        'lower_bound': lower_bound
    }

    with open(run_dir / 'tsp_instance.json', 'w') as f:
        json.dump(tsp_data, f, indent=2)

    print("="*80)
    print("PHASE 2: BASELINE SOLVER (GREEDY + 2-OPT)")
    print("="*80)
    print()

    baseline_results = []

    for trial in range(args.num_trials):
        print(f"Baseline trial {trial+1}/{args.num_trials}...")

        solver = GreedyTSPSolver(tsp)
        result = solver.solve()
        baseline_results.append(result)

        print(f"  Greedy: {result['greedy_length']:.2f}")
        print(f"  Final: {result['final_length']:.2f}")
        print(f"  Improvement: {result['improvement']:.2f}%")
        print(f"  Time: {result['time']:.3f}s")
        print()

    # Baseline statistics
    baseline_lengths = [r['final_length'] for r in baseline_results]
    baseline_mean = np.mean(baseline_lengths)
    baseline_std = np.std(baseline_lengths)
    baseline_best = np.min(baseline_lengths)

    print(f"Baseline summary ({args.num_trials} trials):")
    print(f"  Mean length: {baseline_mean:.2f} +/- {baseline_std:.2f}")
    print(f"  Best length: {baseline_best:.2f}")
    print(f"  Gap from lower bound: {(baseline_best - lower_bound) / lower_bound * 100:.2f}%")
    print()

    print("="*80)
    print("PHASE 3: TEMPORAL LOOP SOLVER (FORWARD-BACKWARD EVOLUTION)")
    print("="*80)
    print()

    print(f"Retrocausal strength (alpha): {args.retrocausal_strength}")
    print(f"Relaxation factor (beta): {args.relaxation_factor}")
    print(f"Max iterations: {args.max_iterations}")
    print()

    temporal_results = []

    for trial in range(args.num_trials):
        print(f"Temporal loop trial {trial+1}/{args.num_trials}...")

        # Use different seed for each trial
        np.random.seed(args.seed + trial + 1000)

        solver = TemporalLoopTSPSolver(
            tsp,
            alpha=args.retrocausal_strength,
            beta=args.relaxation_factor,
            max_iterations=args.max_iterations
        )
        result = solver.solve()
        temporal_results.append(result)

        print(f"  Initial: {result['initial_length']:.2f}")
        print(f"  Final: {result['final_length']:.2f}")
        print(f"  Improvement: {result['improvement']:.2f}%")
        print(f"  Converged: {result['converged']} (iteration {result['convergence_iteration']})")
        print(f"  Final divergence: {result['final_divergence']:.6f}")
        print(f"  Time: {result['time']:.3f}s")
        print()

    # Temporal loop statistics
    temporal_lengths = [r['final_length'] for r in temporal_results]
    temporal_mean = np.mean(temporal_lengths)
    temporal_std = np.std(temporal_lengths)
    temporal_best = np.min(temporal_lengths)

    print(f"Temporal loop summary ({args.num_trials} trials):")
    print(f"  Mean length: {temporal_mean:.2f} +/- {temporal_std:.2f}")
    print(f"  Best length: {temporal_best:.2f}")
    print(f"  Gap from lower bound: {(temporal_best - lower_bound) / lower_bound * 100:.2f}%")
    print()

    print("="*80)
    print("PHASE 4: CONVERGENCE ANALYSIS AND COMPARISON")
    print("="*80)
    print()

    # Statistical comparison
    from scipy import stats

    # Mann-Whitney U test (non-parametric)
    u_stat, p_value = stats.mannwhitneyu(
        baseline_lengths,
        temporal_lengths,
        alternative='greater'  # Test if baseline > temporal (i.e., temporal better)
    )

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((baseline_std**2 + temporal_std**2) / 2)
    cohens_d = (baseline_mean - temporal_mean) / pooled_std if pooled_std > 0 else 0

    improvement_vs_baseline = (baseline_mean - temporal_mean) / baseline_mean * 100

    print("Statistical Comparison:")
    print(f"  Baseline mean: {baseline_mean:.2f}")
    print(f"  Temporal mean: {temporal_mean:.2f}")
    print(f"  Improvement: {improvement_vs_baseline:.2f}%")
    print(f"  Mann-Whitney U: {u_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d: {cohens_d:.3f}")
    print()

    # Interpretation
    if p_value < 0.05:
        if improvement_vs_baseline > 0:
            print("RESULT: Temporal loops provide SIGNIFICANT IMPROVEMENT over baseline")
            print(f"  (p < 0.05, {improvement_vs_baseline:.2f}% better)")
            significance = "significant_improvement"
        else:
            print("RESULT: Temporal loops are SIGNIFICANTLY WORSE than baseline")
            print(f"  (p < 0.05, {abs(improvement_vs_baseline):.2f}% worse)")
            significance = "significant_degradation"
    else:
        print("RESULT: NO SIGNIFICANT DIFFERENCE between temporal loops and baseline")
        print(f"  (p = {p_value:.4f} > 0.05)")
        significance = "no_difference"
    print()

    # Convergence analysis
    convergence_rates = [r['converged'] for r in temporal_results]
    convergence_iterations = [r['convergence_iteration'] for r in temporal_results if r['converged']]

    print("Temporal Loop Convergence:")
    print(f"  Convergence rate: {np.mean(convergence_rates)*100:.1f}% ({sum(convergence_rates)}/{len(convergence_rates)})")
    if convergence_iterations:
        print(f"  Mean iterations to convergence: {np.mean(convergence_iterations):.1f}")
    print()

    # Summary
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'tsp_instance': {
            'num_cities': args.num_cities,
            'lower_bound': lower_bound
        },
        'baseline': {
            'mean_length': float(baseline_mean),
            'std_length': float(baseline_std),
            'best_length': float(baseline_best),
            'trials': baseline_results
        },
        'temporal_loop': {
            'mean_length': float(temporal_mean),
            'std_length': float(temporal_std),
            'best_length': float(temporal_best),
            'convergence_rate': float(np.mean(convergence_rates)),
            'mean_convergence_iterations': float(np.mean(convergence_iterations)) if convergence_iterations else None,
            'trials': temporal_results
        },
        'comparison': {
            'improvement_percent': float(improvement_vs_baseline),
            'mann_whitney_u': float(u_stat),
            'p_value': float(p_value),
            'cohens_d': float(cohens_d),
            'significance': significance
        }
    }

    # Save summary
    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {run_dir}")
    print()

    return summary, significance


def main():
    """Run temporal loop TSP solver validation."""
    args = parse_args()

    print("="*80)
    print("TEMPORAL LOOP TSP SOLVER - VALIDATION TEST")
    print("="*80)
    print()
    print("Hypothesis: Temporal loops help continuous optimization (TSP)")
    print("            but not discrete optimization (SHA-256)")
    print()
    print("Method: Self-consistent initialization + forward-backward evolution")
    print("        from Perfect Temporal Loop discovery (2025-12-18)")
    print()

    # Run benchmark
    summary, significance = run_benchmark_comparison(args)

    # Final verdict
    print("="*80)
    print("FINAL VERDICT")
    print("="*80)
    print()

    if significance == "significant_improvement":
        print("SUCCESS: Temporal loops provide measurable advantage for TSP!")
        print()
        print("Next steps:")
        print("  1. Integrate into HHmL as spatiotemporal framework")
        print("  2. Test on larger TSP instances (100+ cities)")
        print("  3. Try other continuous optimization (protein folding, path planning)")
        print("  4. Generate publication-quality whitepaper")

    elif significance == "significant_degradation":
        print("FAILURE: Temporal loops are WORSE than baseline for TSP")
        print()
        print("Implications:")
        print("  1. Retrocausal feedback may introduce harmful noise")
        print("  2. Continuous optimization might still need gradients, not just smoothness")
        print("  3. Document as rigorous negative result (like SHA-256)")

    else:
        print("INCONCLUSIVE: No significant difference detected")
        print()
        print("Possible explanations:")
        print("  1. TSP instance too small (try 100+ cities)")
        print("  2. Temporal loop parameters not optimal (tune alpha, beta)")
        print("  3. Baseline (greedy + 2-opt) already near-optimal for this size")
        print("  4. More trials needed for statistical power")
        print()
        print("Next steps:")
        print("  1. Scale up to larger instances")
        print("  2. Hyperparameter search (alpha, beta)")
        print("  3. Increase number of trials (10-20)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
