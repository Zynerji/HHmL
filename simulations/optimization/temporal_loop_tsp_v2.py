#!/usr/bin/env python3
"""
Temporal Loop TSP Solver V2 - Discrete Tour with Retrocausal Guidance
======================================================================

Improved implementation: Uses discrete tour representation with temporal loop
guidance for move selection, rather than continuous field evolution.

Key Innovation: Forward solver proposes 2-opt moves, backward solver evaluates
them from "future perspective", temporal consistency guides which moves to accept.

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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Temporal Loop TSP Solver V2')

    parser.add_argument('--num-cities', type=int, default=50)
    parser.add_argument('--num-trials', type=int, default=5)
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--retrocausal-strength', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='results/temporal_tsp_v2')

    return parser.parse_args()


class TSPInstance:
    """Random TSP instance."""

    def __init__(self, num_cities, seed=42):
        np.random.seed(seed)
        self.num_cities = num_cities
        self.cities = np.random.rand(num_cities, 2) * 100

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


class ImprovedTemporalLoopTSPSolver:
    """
    V2: Discrete tour with retrocausal move selection.

    Instead of continuous fields, maintains discrete tours and uses temporal
    loops to guide which 2-opt moves to accept.
    """

    def __init__(self, tsp_instance, alpha=0.7, max_iterations=100):
        self.tsp = tsp_instance
        self.num_cities = tsp_instance.num_cities
        self.alpha = alpha
        self.max_iterations = max_iterations

        # Discrete tour representations
        self.tour_forward = None
        self.tour_backward = None

        # Move history (for temporal consistency)
        self.forward_moves = []
        self.backward_moves = []

        # History
        self.divergence_history = []
        self.length_history = []

    def initialize_self_consistent(self):
        """Self-consistent initialization: both start with same greedy tour."""
        # Greedy tour
        tour = [0]
        unvisited = set(range(1, self.num_cities))

        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: self.tsp.distances[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)

        # CRITICAL: Both start from SAME tour
        self.tour_forward = tour.copy()
        self.tour_backward = tour.copy()

    def get_2opt_moves(self, tour):
        """Get all possible 2-opt moves for a tour."""
        moves = []
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)):
                if j - i > 1:
                    moves.append((i, j))
        return moves

    def apply_2opt(self, tour, i, j):
        """Apply 2-opt move: reverse tour[i:j]."""
        new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
        return new_tour

    def evaluate_move(self, tour, move):
        """Evaluate improvement from 2-opt move."""
        i, j = move
        new_tour = self.apply_2opt(tour, i, j)
        current_length = self.tsp.tour_length(tour)
        new_length = self.tsp.tour_length(new_tour)
        return new_length - current_length  # Negative = improvement

    def evolve_forward(self):
        """
        Forward evolution: Try 2-opt moves in order of immediate improvement.
        """
        moves = self.get_2opt_moves(self.tour_forward)

        # Evaluate all moves
        move_improvements = [(move, self.evaluate_move(self.tour_forward, move)) for move in moves]

        # Sort by improvement (best first)
        move_improvements.sort(key=lambda x: x[1])

        # Take best improving move (if any)
        if move_improvements and move_improvements[0][1] < 0:
            best_move = move_improvements[0][0]
            self.tour_forward = self.apply_2opt(self.tour_forward, *best_move)
            self.forward_moves.append(best_move)

    def evolve_backward(self):
        """
        Backward evolution: Try random 2-opt moves (exploration).
        """
        moves = self.get_2opt_moves(self.tour_backward)

        if moves:
            # Random move (more exploration)
            random_move = moves[np.random.randint(len(moves))]

            # Apply if improves
            if self.evaluate_move(self.tour_backward, random_move) < 0:
                self.tour_backward = self.apply_2opt(self.tour_backward, *random_move)
                self.backward_moves.append(random_move)

    def apply_prophetic_coupling(self):
        """
        Retrocausal coupling: Exchange tour segments between forward/backward.

        Key idea: If backward found a good segment, forward adopts it (and vice versa).
        """
        if np.random.rand() < self.alpha:
            # Swap random segment between tours
            seg_len = np.random.randint(3, self.num_cities // 3)
            start = np.random.randint(0, self.num_cities - seg_len)

            # Extract segments
            forward_segment = self.tour_forward[start:start+seg_len]
            backward_segment = self.tour_backward[start:start+seg_len]

            # Create hybrid tours
            # (This is the "prophetic" part: future influences past)
            hybrid_forward = self.tour_forward.copy()
            hybrid_backward = self.tour_backward.copy()

            # Try swapping segments
            # Forward gets backward's segment
            for i, city in enumerate(backward_segment):
                if city in hybrid_forward:
                    idx = hybrid_forward.index(city)
                    hybrid_forward[idx], hybrid_forward[start+i] = hybrid_forward[start+i], hybrid_forward[idx]

            # Backward gets forward's segment
            for i, city in enumerate(forward_segment):
                if city in hybrid_backward:
                    idx = hybrid_backward.index(city)
                    hybrid_backward[idx], hybrid_backward[start+i] = hybrid_backward[start+i], hybrid_backward[idx]

            # Accept if improves
            if self.tsp.tour_length(hybrid_forward) < self.tsp.tour_length(self.tour_forward):
                self.tour_forward = hybrid_forward

            if self.tsp.tour_length(hybrid_backward) < self.tsp.tour_length(self.tour_backward):
                self.tour_backward = hybrid_backward

    def compute_divergence(self):
        """Measure tour divergence (edit distance)."""
        # Simple: fraction of cities in different positions
        diff_count = sum(1 for i in range(len(self.tour_forward)) if self.tour_forward[i] != self.tour_backward[i])
        return diff_count / len(self.tour_forward)

    def solve(self):
        """Run improved temporal loop solver."""
        start_time = time.time()

        self.initialize_self_consistent()
        initial_length = self.tsp.tour_length(self.tour_forward)

        converged = False
        convergence_iteration = None

        for iteration in range(self.max_iterations):
            # Forward: greedy 2-opt
            self.evolve_forward()

            # Backward: exploration
            self.evolve_backward()

            # Prophetic coupling
            self.apply_prophetic_coupling()

            # Metrics
            divergence = self.compute_divergence()
            self.divergence_history.append(divergence)

            length_forward = self.tsp.tour_length(self.tour_forward)
            length_backward = self.tsp.tour_length(self.tour_backward)
            avg_length = (length_forward + length_backward) / 2
            self.length_history.append(avg_length)

            # Convergence: both tours similar and no improvement for 10 iterations
            if iteration >= 20:
                recent_lengths = self.length_history[-10:]
                if np.std(recent_lengths) < 0.1 and divergence < 0.1:
                    converged = True
                    convergence_iteration = iteration
                    break

        # Use better of forward/backward
        final_length_forward = self.tsp.tour_length(self.tour_forward)
        final_length_backward = self.tsp.tour_length(self.tour_backward)

        if final_length_forward < final_length_backward:
            final_tour = self.tour_forward
            final_length = final_length_forward
        else:
            final_tour = self.tour_backward
            final_length = final_length_backward

        elapsed = time.time() - start_time

        return {
            'tour': final_tour,
            'initial_length': initial_length,
            'final_length': final_length,
            'improvement': (initial_length - final_length) / initial_length * 100,
            'converged': converged,
            'convergence_iteration': convergence_iteration,
            'final_divergence': self.divergence_history[-1] if self.divergence_history else 0,
            'iterations': len(self.divergence_history),
            'time': elapsed
        }


class BaselineSolver:
    """Standard greedy + 2-opt for comparison."""

    def __init__(self, tsp_instance):
        self.tsp = tsp_instance

    def greedy_tour(self):
        tour = [0]
        unvisited = set(range(1, self.tsp.num_cities))
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: self.tsp.distances[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
        return tour

    def two_opt(self, tour, max_iterations=1000):
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            for i in range(1, len(tour) - 2):
                for j in range(i + 1, len(tour)):
                    if j - i == 1:
                        continue
                    new_tour = tour[:i] + tour[i:j][::-1] + tour[j:]
                    if self.tsp.tour_length(new_tour) < self.tsp.tour_length(tour):
                        tour = new_tour
                        improved = True
                        break
                if improved:
                    break
        return tour, iterations

    def solve(self):
        start_time = time.time()
        tour = self.greedy_tour()
        greedy_length = self.tsp.tour_length(tour)
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


def run_comparison(args):
    """Run V2 comparison."""
    from scipy import stats

    print("="*80)
    print("TEMPORAL LOOP TSP SOLVER V2 - DISCRETE TOUR WITH RETROCAUSAL GUIDANCE")
    print("="*80)
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"Number of cities: {args.num_cities}")
    print(f"Trials: {args.num_trials}")
    print()

    tsp = TSPInstance(args.num_cities, seed=args.seed)

    # Baseline
    print("Running baseline (greedy + 2-opt)...")
    baseline_results = []
    for trial in range(args.num_trials):
        solver = BaselineSolver(tsp)
        result = solver.solve()
        baseline_results.append(result)
        print(f"  Trial {trial+1}: {result['final_length']:.2f}")

    baseline_lengths = [r['final_length'] for r in baseline_results]
    baseline_mean = np.mean(baseline_lengths)
    print(f"Baseline mean: {baseline_mean:.2f}")
    print()

    # Temporal loop V2
    print("Running temporal loop V2 (discrete + retrocausal)...")
    temporal_results = []
    for trial in range(args.num_trials):
        np.random.seed(args.seed + trial + 1000)
        solver = ImprovedTemporalLoopTSPSolver(
            tsp,
            alpha=args.retrocausal_strength,
            max_iterations=args.max_iterations
        )
        result = solver.solve()
        temporal_results.append(result)
        print(f"  Trial {trial+1}: {result['final_length']:.2f} (improved {result['improvement']:.2f}%)")

    temporal_lengths = [r['final_length'] for r in temporal_results]
    temporal_mean = np.mean(temporal_lengths)
    print(f"Temporal V2 mean: {temporal_mean:.2f}")
    print()

    # Compare
    improvement = (baseline_mean - temporal_mean) / baseline_mean * 100
    u_stat, p_value = stats.mannwhitneyu(baseline_lengths, temporal_lengths, alternative='greater')

    print("="*80)
    print("COMPARISON")
    print("="*80)
    print(f"Baseline:     {baseline_mean:.2f}")
    print(f"Temporal V2:  {temporal_mean:.2f}")
    print(f"Improvement:  {improvement:.2f}%")
    print(f"p-value:      {p_value:.4f}")
    print()

    if p_value < 0.05 and improvement > 0:
        print("SUCCESS: Temporal loops V2 provide SIGNIFICANT IMPROVEMENT!")
    elif p_value < 0.05 and improvement < 0:
        print("FAILURE: Temporal loops V2 are WORSE than baseline")
    else:
        print("INCONCLUSIVE: No significant difference")
    print()

    # Save
    summary = {
        'baseline': {'mean': float(baseline_mean), 'results': baseline_results},
        'temporal_v2': {'mean': float(temporal_mean), 'results': temporal_results},
        'improvement_percent': float(improvement),
        'p_value': float(p_value)
    }

    with open(run_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    args = parse_args()
    run_comparison(args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
