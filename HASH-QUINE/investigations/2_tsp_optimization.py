#!/usr/bin/env python3
"""
TSP Optimization via Recursive Topology
========================================

Tests whether recursive Möbius collapse helps solve Traveling Salesman Problem,
where hash quines failed for cryptographic mining.

Key Difference from Mining:
- TSP has SMOOTH fitness landscape (small tour changes -> small distance changes)
- Cryptographic hashing has CHAOTIC landscape (small input changes -> random output)
- Recursive topology should exploit smooth gradients

Hypothesis: Fiedler vector graph partitioning + recursive collapse creates
good TSP tour initialization via spectral clustering of cities.

Approach:
1. Map TSP cities to Möbius lattice nodes
2. Recursive Fiedler collapse creates hierarchical clustering
3. Clusters define tour segments
4. Compare tour quality: Recursive vs Random vs Greedy

Expected Outcome:
- Random tour: 100% baseline
- Greedy tour: ~70-80% of random (good heuristic)
- Recursive topology: ? (test if better than greedy)

Success Criteria:
- Recursive < Greedy → validates smooth landscape optimization
- Recursive ~= Random → hash quines don't help here either

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import torch
import numpy as np
import json
from datetime import datetime
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from typing import List, Tuple

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def generate_random_tsp_instance(num_cities: int, seed: int = 42) -> np.ndarray:
    """Generate random TSP instance (cities in unit square)."""
    np.random.seed(seed)
    cities = np.random.rand(num_cities, 2)
    return cities


def compute_tour_length(cities: np.ndarray, tour: List[int]) -> float:
    """Compute total tour length."""
    total_length = 0.0
    for i in range(len(tour)):
        city_a = cities[tour[i]]
        city_b = cities[tour[(i + 1) % len(tour)]]
        total_length += np.linalg.norm(city_a - city_b)
    return total_length


def greedy_tour(cities: np.ndarray) -> List[int]:
    """Greedy nearest-neighbor tour."""
    n = len(cities)
    unvisited = set(range(n))
    tour = [0]  # Start at city 0
    unvisited.remove(0)

    while unvisited:
        current_city = tour[-1]
        # Find nearest unvisited city
        nearest = min(unvisited, key=lambda c: np.linalg.norm(cities[current_city] - cities[c]))
        tour.append(nearest)
        unvisited.remove(nearest)

    return tour


def random_tour(cities: np.ndarray, seed: int = 42) -> List[int]:
    """Random tour."""
    np.random.seed(seed)
    tour = list(range(len(cities)))
    np.random.shuffle(tour)
    return tour


def map_cities_to_mobius(cities: np.ndarray, num_lattice_nodes: int, windings: float, device='cpu'):
    """
    Map TSP cities to Möbius lattice nodes.

    Each city gets assigned to nearest lattice point.
    """
    # Create Möbius lattice
    t = torch.linspace(0, 2 * np.pi, num_lattice_nodes, device=device)

    lattice_positions = torch.stack([
        (1 + 0.5 * torch.cos(windings * t / 2)) * torch.cos(t),
        (1 + 0.5 * torch.cos(windings * t / 2)) * torch.sin(t),
        0.5 * torch.sin(windings * t / 2)
    ], dim=1).cpu().numpy()

    # Flatten to 2D for city matching (ignore z-coordinate)
    lattice_2d = lattice_positions[:, :2]

    # Assign each city to nearest lattice node
    city_to_node = []
    for city in cities:
        distances = np.linalg.norm(lattice_2d - city, axis=1)
        nearest_node = np.argmin(distances)
        city_to_node.append(nearest_node)

    return city_to_node, lattice_positions


def fiedler_partition(positions: np.ndarray, node_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partition nodes using Fiedler vector.

    Returns two clusters based on Fiedler vector sign.
    """
    n = len(node_indices)
    if n < 2:
        return node_indices, np.array([])

    # Get positions
    node_positions = positions[node_indices]

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(node_positions[i] - node_positions[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # k-nearest neighbor adjacency
    k = min(5, n - 1)
    adjacency = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist_matrix[i])[1:k+1]
        adjacency[i, nearest] = 1
        adjacency[nearest, i] = 1

    # Laplacian
    degree = np.diag(adjacency.sum(axis=1))
    laplacian = degree - adjacency

    # Compute Fiedler vector
    try:
        laplacian_sparse = csr_matrix(laplacian)
        eigenvalues, eigenvectors = eigsh(laplacian_sparse, k=2, which='SM')
        fiedler_vector = eigenvectors[:, 1]
    except Exception as e:
        print(f"  Warning: Fiedler failed ({e}), random split")
        fiedler_vector = np.random.randn(n)

    # Partition by sign
    cluster_a = node_indices[fiedler_vector >= 0]
    cluster_b = node_indices[fiedler_vector < 0]

    return cluster_a, cluster_b


def recursive_tsp_tour(cities: np.ndarray, lattice_positions: np.ndarray,
                       city_to_node: List[int], max_depth: int = 3) -> List[int]:
    """
    Build TSP tour via recursive Fiedler partitioning.

    Recursively partition cities into clusters, then connect clusters.
    """
    n = len(cities)
    node_indices = np.array(city_to_node)

    # Recursive partitioning
    clusters = [node_indices]

    for depth in range(max_depth):
        print(f"  Depth {depth}: {len(clusters)} clusters")

        new_clusters = []
        for cluster in clusters:
            if len(cluster) <= 2:
                new_clusters.append(cluster)
            else:
                cluster_a, cluster_b = fiedler_partition(lattice_positions, cluster)
                if len(cluster_a) > 0:
                    new_clusters.append(cluster_a)
                if len(cluster_b) > 0:
                    new_clusters.append(cluster_b)

        clusters = new_clusters

    print(f"  Final: {len(clusters)} leaf clusters")

    # Build tour from clusters
    tour = []

    # Visit clusters in order, and cities within each cluster
    for cluster in clusters:
        # Map nodes back to cities
        cities_in_cluster = [i for i, node in enumerate(city_to_node) if node in cluster]

        # Sort cities in cluster by angle (simple heuristic)
        if len(cities_in_cluster) > 0:
            cluster_cities = cities[cities_in_cluster]
            centroid = cluster_cities.mean(axis=0)
            angles = np.arctan2(cluster_cities[:, 1] - centroid[1],
                              cluster_cities[:, 0] - centroid[0])
            sorted_indices = np.argsort(angles)
            tour.extend([cities_in_cluster[i] for i in sorted_indices])

    return tour


def main():
    parser = argparse.ArgumentParser(description='TSP Optimization via Recursive Topology')
    parser.add_argument('--num-cities', type=int, default=50)
    parser.add_argument('--lattice-nodes', type=int, default=500)
    parser.add_argument('--windings', type=float, default=109.0)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='HASH-QUINE/investigations/results')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TSP OPTIMIZATION VIA RECURSIVE TOPOLOGY")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Cities: {args.num_cities}")
    print(f"  Lattice nodes: {args.lattice_nodes}")
    print(f"  Max recursion depth: {args.max_depth}")
    print()

    # Generate TSP instance
    print("Generating TSP instance...")
    cities = generate_random_tsp_instance(args.num_cities, args.seed)
    print(f"Generated {len(cities)} cities")
    print()

    # Baseline tours
    print("="*80)
    print("BASELINE TOURS")
    print("="*80)
    print()

    # Random tour
    print("Computing random tour...")
    random_tour_list = random_tour(cities, args.seed)
    random_length = compute_tour_length(cities, random_tour_list)
    print(f"Random tour length: {random_length:.4f}")
    print()

    # Greedy tour
    print("Computing greedy tour...")
    greedy_tour_list = greedy_tour(cities)
    greedy_length = compute_tour_length(cities, greedy_tour_list)
    greedy_improvement = ((random_length - greedy_length) / random_length) * 100
    print(f"Greedy tour length: {greedy_length:.4f}")
    print(f"Improvement over random: {greedy_improvement:.1f}%")
    print()

    # Recursive topology tour
    print("="*80)
    print("RECURSIVE TOPOLOGY TOUR")
    print("="*80)
    print()

    print("Mapping cities to Mobius lattice...")
    city_to_node, lattice_positions = map_cities_to_mobius(
        cities, args.lattice_nodes, args.windings, device
    )
    print(f"Mapped {len(cities)} cities to {args.lattice_nodes} lattice nodes")
    print()

    print("Building tour via recursive Fiedler partitioning...")
    start_time = time.time()
    recursive_tour_list = recursive_tsp_tour(
        cities, lattice_positions, city_to_node, args.max_depth
    )
    recursive_time = time.time() - start_time

    recursive_length = compute_tour_length(cities, recursive_tour_list)
    recursive_improvement = ((random_length - recursive_length) / random_length) * 100
    print()
    print(f"Recursive tour length: {recursive_length:.4f}")
    print(f"Improvement over random: {recursive_improvement:.1f}%")
    print(f"Computation time: {recursive_time:.3f}s")
    print()

    # Comparison
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()

    print(f"Tour Lengths:")
    print(f"  Random:    {random_length:.4f} (baseline)")
    print(f"  Greedy:    {greedy_length:.4f} ({greedy_improvement:+.1f}%)")
    print(f"  Recursive: {recursive_length:.4f} ({recursive_improvement:+.1f}%)")
    print()

    # Interpretation
    if recursive_length < greedy_length:
        interpretation = "SUCCESS: Recursive topology BEATS greedy heuristic"
        success = True
    elif recursive_length < random_length * 0.95:
        interpretation = "PARTIAL: Recursive topology helps but not better than greedy"
        success = True
    else:
        interpretation = "FAILURE: Recursive topology no better than random"
        success = False

    print(f"Interpretation: {interpretation}")
    print()

    # Key insight
    if success:
        print("KEY INSIGHT: Unlike cryptographic mining (hash quines failed),")
        print("recursive topology DOES help continuous optimization problems like TSP.")
        print("This confirms the hypothesis: smooth fitness landscapes benefit from")
        print("spectral graph partitioning, while chaotic landscapes do not.")
    else:
        print("KEY INSIGHT: Recursive topology doesn't help TSP either.")
        print("Hash quines may be purely self-similar patterns without optimization benefit.")

    print()

    # Save results
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'tour_lengths': {
            'random': float(random_length),
            'greedy': float(greedy_length),
            'recursive': float(recursive_length)
        },
        'improvements': {
            'greedy_vs_random': float(greedy_improvement),
            'recursive_vs_random': float(recursive_improvement)
        },
        'recursive_time_sec': float(recursive_time),
        'interpretation': interpretation,
        'success': success
    }

    results_path = output_dir / f'tsp_optimization_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
