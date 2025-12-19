#!/usr/bin/env python3
"""
Optimized Hybrid SAT Solver
============================

Based on Investigation 5 results showing recursive topology WINS (rho=0.8924),
we create an OPTIMIZED hybrid that uses the best of both approaches more intelligently.

Key Insights from Results:
1. Recursive topology (0.8924) > Hybrid (0.8829) > Uniform (0.8790) > Helical (0.8686)
2. Recursive decomposition finds better local structure
3. Helical weighting may be too aggressive (loses global constraints)

New Hybrid Strategy:
1. **Adaptive recursion depth** - deeper for larger problems
2. **Constraint-aware partitioning** - preserve clause connectivity
3. **Minimal helical weighting** - only on partition boundaries
4. **Iterative refinement** - multiple passes with different partitions
5. **Clause-guided splitting** - partition to minimize clause-splitting

Expected Result: rho > 0.90 (beating all previous methods)

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import numpy as np
import json
from datetime import datetime
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import networkx as nx
from itertools import combinations
from typing import List, Tuple, Dict

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def random_3sat(n_vars: int, m_clauses: int, seed: int = 42) -> List[List[int]]:
    """Generate random 3-SAT instance."""
    np.random.seed(seed)
    clauses = []
    for _ in range(m_clauses):
        vars_selected = np.random.choice(n_vars, size=3, replace=False)
        signs = np.random.choice([1, -1], size=3)
        clause = [(vars_selected[i] + 1) * signs[i] for i in range(3)]
        clauses.append(clause)
    return clauses


def evaluate_sat(clauses: List[List[int]], assign: np.ndarray) -> float:
    """Evaluate satisfaction ratio."""
    sat_count = 0
    for clause in clauses:
        clause_satisfied = any(
            (lit > 0 and assign[abs(lit) - 1] > 0) or
            (lit < 0 and assign[abs(lit) - 1] < 0)
            for lit in clause
        )
        if clause_satisfied:
            sat_count += 1
    return sat_count / len(clauses)


class OptimizedHybridSATSolver:
    """
    Optimized hybrid combining best strategies from recursive and helical approaches.
    """

    def __init__(self, clauses: List[List[int]], n_vars: int):
        self.clauses = clauses
        self.n_vars = n_vars
        self.assignment = np.zeros(n_vars)

        # Build clause-variable bipartite graph
        self.G = nx.Graph()
        for i in range(n_vars):
            self.G.add_node(f"v{i}", type='var')

        for c_idx, clause in enumerate(clauses):
            self.G.add_node(f"c{c_idx}", type='clause')
            for lit in clause:
                var_idx = abs(lit) - 1
                self.G.add_edge(f"v{var_idx}", f"c{c_idx}")

    def compute_clause_connectivity(self, var_indices: List[int]) -> float:
        """
        Compute how many clauses are fully contained within this partition.

        Higher is better (fewer clauses split across partitions).
        """
        contained_clauses = 0

        for clause in self.clauses:
            vars_in_clause = [abs(lit) - 1 for lit in clause]
            if all(v in var_indices for v in vars_in_clause):
                contained_clauses += 1

        return contained_clauses / len(self.clauses)

    def constraint_aware_partition(
        self,
        var_indices: List[int],
        depth: int,
        max_depth: int
    ) -> List[List[int]]:
        """
        Partition variables while preserving clause connectivity.

        Uses Fiedler vector but evaluates quality by clause containment.
        """
        if len(var_indices) <= 5 or depth >= max_depth:
            return [var_indices]

        # Build subgraph for these variables
        var_subgraph = [f"v{i}" for i in var_indices]
        clause_neighbors = set()

        for v_node in var_subgraph:
            for neighbor in self.G.neighbors(v_node):
                if self.G.nodes[neighbor]['type'] == 'clause':
                    clause_neighbors.add(neighbor)

        # Build adjacency matrix (variable-variable via shared clauses)
        n = len(var_indices)
        adjacency = np.zeros((n, n))

        for i, var_i in enumerate(var_indices):
            for j, var_j in enumerate(var_indices):
                if i >= j:
                    continue

                # Count shared clauses
                clauses_i = set(self.G.neighbors(f"v{var_i}"))
                clauses_j = set(self.G.neighbors(f"v{var_j}"))
                shared = len(clauses_i & clauses_j)

                adjacency[i, j] = shared
                adjacency[j, i] = shared

        # Laplacian
        degree = adjacency.sum(axis=1)
        laplacian = np.diag(degree) - adjacency
        laplacian_sparse = csr_matrix(laplacian)

        # Compute Fiedler vector
        try:
            _, eigenvectors = eigsh(laplacian_sparse, k=2, which='SM')
            fiedler = eigenvectors[:, 1]
        except:
            fiedler = np.random.randn(n)

        # Try multiple partition thresholds to maximize clause containment
        best_partitions = None
        best_score = -1

        for threshold in np.linspace(-1, 1, 11):
            cluster_a = [var_indices[i] for i in range(n) if fiedler[i] >= threshold]
            cluster_b = [var_indices[i] for i in range(n) if fiedler[i] < threshold]

            if len(cluster_a) == 0 or len(cluster_b) == 0:
                continue

            # Score by clause containment
            score_a = self.compute_clause_connectivity(cluster_a) if cluster_a else 0
            score_b = self.compute_clause_connectivity(cluster_b) if cluster_b else 0
            total_score = score_a + score_b

            if total_score > best_score:
                best_score = total_score
                best_partitions = (cluster_a, cluster_b)

        if best_partitions is None:
            # Fallback to median split
            median = np.median(fiedler)
            cluster_a = [var_indices[i] for i in range(n) if fiedler[i] >= median]
            cluster_b = [var_indices[i] for i in range(n) if fiedler[i] < median]
            best_partitions = (cluster_a, cluster_b)

        cluster_a, cluster_b = best_partitions

        # Recurse
        partitions_a = self.constraint_aware_partition(cluster_a, depth + 1, max_depth)
        partitions_b = self.constraint_aware_partition(cluster_b, depth + 1, max_depth)

        return partitions_a + partitions_b

    def solve_partition(self, var_indices: List[int], omega: float = 0.1):
        """
        Solve a partition using LIGHT helical weighting.

        omega=0.1 (much lower than standard 0.3) to avoid over-biasing.
        """
        n = len(var_indices)
        if n == 0:
            return

        # Build helical-weighted graph for partition
        G_sub = nx.Graph()
        for i in var_indices:
            G_sub.add_node(i)

        # Add edges with minimal helical weighting
        N = 20000
        for clause in self.clauses:
            vars_in_clause = [abs(lit) - 1 for lit in clause if abs(lit) - 1 in var_indices]

            for u, v in combinations(set(vars_in_clause), 2):
                theta_u = 2 * np.pi * np.log(u + 1) / N
                theta_v = 2 * np.pi * np.log(v + 1) / N
                w = np.cos(omega * (theta_u - theta_v))

                if G_sub.has_edge(u, v):
                    G_sub[u][v]['weight'] += w
                else:
                    G_sub.add_edge(u, v, weight=w)

        # Compute Laplacian
        if G_sub.number_of_edges() > 0:
            L_sub = nx.laplacian_matrix(G_sub, weight='weight').tocsc().astype(float)

            try:
                _, vec = eigsh(L_sub, k=1, which='SM', maxiter=200)
                local_assign = np.sign(vec[:, 0])
            except:
                local_assign = np.random.choice([1, -1], size=n)
        else:
            local_assign = np.random.choice([1, -1], size=n)

        # Assign
        for i, var_idx in enumerate(var_indices):
            self.assignment[var_idx] = local_assign[i] if local_assign[i] != 0 else np.random.choice([1, -1])

    def solve(self, max_depth: int = 3, num_iterations: int = 3) -> Tuple[float, Dict]:
        """
        Solve using optimized hybrid approach.

        Args:
            max_depth: Adaptive recursion depth (deeper for larger problems)
            num_iterations: Multiple passes with different random seeds

        Returns:
            Best satisfaction ratio and metadata
        """
        best_rho = 0.0
        best_assignment = None

        for iteration in range(num_iterations):
            # Reset
            self.assignment = np.zeros(self.n_vars)

            # Set random seed for this iteration
            np.random.seed(42 + iteration)

            # Adaptive depth (deeper for larger problems)
            adaptive_depth = min(max_depth, int(np.log2(self.n_vars / 10)) + 1)

            # Partition
            var_indices = list(range(self.n_vars))
            partitions = self.constraint_aware_partition(var_indices, 0, adaptive_depth)

            # Solve each partition
            for partition in partitions:
                self.solve_partition(partition, omega=0.1)

            # Handle unassigned
            self.assignment[self.assignment == 0] = np.random.choice([1, -1], size=(self.assignment == 0).sum())

            # Evaluate
            rho = evaluate_sat(self.clauses, self.assignment)

            if rho > best_rho:
                best_rho = rho
                best_assignment = self.assignment.copy()

        # Restore best
        self.assignment = best_assignment

        metadata = {
            'num_iterations': num_iterations,
            'adaptive_depth': adaptive_depth,
            'num_partitions': len(partitions),
            'partition_sizes': [len(p) for p in partitions]
        }

        return best_rho, metadata


def run_benchmark(n_vars: int, m_clauses: int, num_seeds: int = 5, max_depth: int = 3):
    """Run optimized hybrid benchmark."""

    results = []

    for seed in range(42, 42 + num_seeds):
        clauses = random_3sat(n_vars, m_clauses, seed)

        # Solve with optimized hybrid
        solver = OptimizedHybridSATSolver(clauses, n_vars)

        start = time.time()
        rho, metadata = solver.solve(max_depth=max_depth, num_iterations=3)
        solve_time = time.time() - start

        results.append({
            'rho': rho,
            'time': solve_time,
            'metadata': metadata
        })

    # Statistics
    rhos = [r['rho'] for r in results]
    times = [r['time'] for r in results]

    summary = {
        'mean_rho': np.mean(rhos),
        'std_rho': np.std(rhos, ddof=1),
        'ci_rho': 1.96 * np.std(rhos, ddof=1) / np.sqrt(num_seeds),
        'mean_time': np.mean(times),
        'max_rho': np.max(rhos),
        'min_rho': np.min(rhos)
    }

    return summary, results


def main():
    parser = argparse.ArgumentParser(description='Optimized Hybrid SAT Solver')
    parser.add_argument('--n-vars', type=int, default=50)
    parser.add_argument('--m-clauses', type=int, default=210)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--num-seeds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='HASH-QUINE/investigations/results')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("OPTIMIZED HYBRID SAT SOLVER")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Variables: {args.n_vars}")
    print(f"  Clauses: {args.m_clauses} (ratio: {args.m_clauses/args.n_vars:.2f})")
    print(f"  Max recursion depth: {args.max_depth}")
    print(f"  Random seeds: {args.num_seeds}")
    print()

    print("Running optimized hybrid...")
    summary, results = run_benchmark(args.n_vars, args.m_clauses, args.num_seeds, args.max_depth)

    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print()

    print(f"Satisfaction Ratio: {summary['mean_rho']:.4f} +/- {summary['ci_rho']:.4f}")
    print(f"  Range: [{summary['min_rho']:.4f}, {summary['max_rho']:.4f}]")
    print(f"Avg Time: {summary['mean_time']:.4f}s")
    print()

    # Comparison to previous results
    print("="*80)
    print("COMPARISON TO INVESTIGATION 5")
    print("="*80)
    print()

    # Previous results from Investigation 5
    prev_results = {
        'Helical': 0.8686,
        'Recursive': 0.8924,
        'Hybrid': 0.8829,
        'Uniform': 0.8790
    }

    print("Previous Methods (Investigation 5):")
    for method, rho in prev_results.items():
        improvement = ((summary['mean_rho'] - rho) / rho) * 100
        print(f"  {method:<12}: {rho:.4f}  (Optimized Hybrid: {improvement:+.1f}%)")

    print()

    # Winner?
    best_prev = max(prev_results.values())
    if summary['mean_rho'] > best_prev:
        improvement_pct = ((summary['mean_rho'] - best_prev) / best_prev) * 100
        print(f"NEW WINNER: Optimized Hybrid beats all previous methods by {improvement_pct:+.1f}%")
    else:
        gap_pct = ((best_prev - summary['mean_rho']) / best_prev) * 100
        print(f"Optimized Hybrid did not beat recursive ({best_prev:.4f}), gap: {gap_pct:.1f}%")

    print()

    # Save results
    output = {
        'timestamp': timestamp,
        'config': vars(args),
        'summary': summary,
        'results': results,
        'comparison_to_investigation_5': prev_results
    }

    results_path = output_dir / f'optimized_hybrid_sat_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Results saved: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
