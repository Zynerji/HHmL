#!/usr/bin/env python3
"""
Topology Comparison: Möbius vs. Torus vs. Sphere
=================================================

Tests whether hash quine emergence is unique to Möbius topology or a general
property of recursive topological structures.

Hypothesis: Möbius single-sided surface creates stronger self-similarity than
orientable topologies (torus, sphere) due to twist-induced feedback.

Phases:
1. Möbius recursive collapse (baseline - known to produce hash quines)
2. Torus recursive collapse (control - orientable, no twist)
3. Sphere recursive collapse (control - orientable, different geometry)
4. Compare pattern repetition ratios

Expected Outcome:
- Möbius: 300-400× pattern repetition (established)
- Torus: ? (test if comparable or lower)
- Sphere: ? (test if topology matters or just recursion)

If Möbius ≈ Torus ≈ Sphere → hash quines are general recursive property
If Möbius >> Torus ≈ Sphere → Möbius twist is critical

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import hashlib
import torch
import numpy as np
import json
from datetime import datetime
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from typing import Dict, List

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MobiusTopology:
    """Möbius strip with twist parameter."""

    def __init__(self, num_nodes: int, windings: float, device='cpu'):
        self.num_nodes = num_nodes
        self.windings = windings
        self.device = device

        t = torch.linspace(0, 2 * np.pi, num_nodes, device=device)

        # Standard Möbius parameterization with twist
        self.positions = torch.stack([
            (1 + 0.5 * torch.cos(windings * t / 2)) * torch.cos(t),
            (1 + 0.5 * torch.cos(windings * t / 2)) * torch.sin(t),
            0.5 * torch.sin(windings * t / 2)  # Twist creates z-variation
        ], dim=1)

        self.field = torch.randn(num_nodes, dtype=torch.complex64, device=device) * 0.1


class TorusTopology:
    """Standard torus (orientable, no twist)."""

    def __init__(self, num_nodes: int, windings: float, device='cpu'):
        self.num_nodes = num_nodes
        self.windings = windings  # Not used for torus, but kept for API consistency
        self.device = device

        # Torus parameterization (R=1.0, r=0.5)
        t = torch.linspace(0, 2 * np.pi, num_nodes, device=device)
        R = 1.0  # Major radius
        r = 0.5  # Minor radius

        # Standard torus (no twist)
        self.positions = torch.stack([
            (R + r * torch.cos(t)) * torch.cos(t),
            (R + r * torch.cos(t)) * torch.sin(t),
            r * torch.sin(t)
        ], dim=1)

        self.field = torch.randn(num_nodes, dtype=torch.complex64, device=device) * 0.1


class SphereTopology:
    """Sphere (orientable, different geometry)."""

    def __init__(self, num_nodes: int, windings: float, device='cpu'):
        self.num_nodes = num_nodes
        self.windings = windings  # Not used for sphere
        self.device = device

        # Fibonacci sphere sampling for uniform distribution
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle

        positions = []
        for i in range(num_nodes):
            y = 1 - (i / float(num_nodes - 1)) * 2  # y from 1 to -1
            radius = np.sqrt(1 - y * y)
            theta = phi * i

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            positions.append([x, y, z])

        self.positions = torch.tensor(positions, dtype=torch.float32, device=device)
        self.field = torch.randn(num_nodes, dtype=torch.complex64, device=device) * 0.1


def propagate_field(topology, cycles=20, depth=0):
    """Evolve field on any topology."""
    field = topology.field

    # Depth-dependent parameters (deeper = stronger nonlinearity)
    nonlinearity_strength = 0.1 * (1.0 + depth * 0.2)
    damping_strength = 0.05 * (1.0 - depth * 0.1)

    for _ in range(cycles):
        # Neighbor averaging (circular for simplicity)
        neighbor_sum = (
            torch.roll(field, -1, dims=0) +
            torch.roll(field, 1, dims=0)
        )

        # Nonlinear term
        nonlinearity = -nonlinearity_strength * torch.abs(field)**2 * field

        # Damping
        damping = -damping_strength * field

        # Update
        field = field + 0.01 * (neighbor_sum + nonlinearity + damping)

    topology.field = field
    return field


def detect_vortices(topology, threshold=0.3):
    """Detect vortex cores."""
    magnitudes = torch.abs(topology.field)
    vortex_mask = magnitudes < threshold
    vortex_indices = torch.where(vortex_mask)[0]
    return vortex_indices.cpu().numpy()


def fiedler_collapse(positions, vortices, target_size=None):
    """
    Helical SAT collapse via Fiedler vector.

    Returns indices of collapsed singularity points.
    """
    n = len(vortices)
    if n < 10:
        return vortices[:min(5, n)]

    if target_size is None:
        target_size = max(n // 5, 10)

    # Get vortex positions
    vortex_positions = positions[vortices].cpu().numpy()

    # Build distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(vortex_positions[i] - vortex_positions[j])
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
        print(f"    Warning: Fiedler computation failed ({e}), using random")
        fiedler_vector = np.random.randn(n)

    # Select top nodes by Fiedler magnitude
    magnitudes = np.abs(fiedler_vector)
    selected_indices = np.argsort(magnitudes)[-target_size:]

    return vortices[selected_indices]


def recursive_collapse(topology_class, num_nodes, windings, max_depth, device='cpu'):
    """
    Perform recursive collapse on given topology.

    Returns list of singularity points from deepest layer.
    """
    print(f"\nRecursive collapse ({topology_class.__name__}):")

    layers = []
    current_nodes = num_nodes

    for depth in range(max_depth + 1):
        print(f"  Depth {depth}: {current_nodes} nodes")

        # Create layer
        layer = topology_class(current_nodes, windings, device)
        layers.append(layer)

        # Propagate field
        propagate_field(layer, cycles=20, depth=depth)

        # Detect vortices
        vortices = detect_vortices(layer, threshold=0.3)
        print(f"    Detected {len(vortices)} vortices")

        if len(vortices) < 10 or depth >= max_depth:
            print(f"    Stopping at depth {depth}")
            return vortices

        # Collapse to next layer
        collapsed = fiedler_collapse(layer.positions, vortices, target_size=current_nodes // 5)
        print(f"    Collapsed to {len(collapsed)} singularities")

        current_nodes = len(collapsed)

    return layers[-1] if layers else np.array([])


def compute_hash_quine_metric(nonces: np.ndarray, num_samples=1000):
    """
    Compute hash quine pattern repetition ratio.

    Returns (max_pattern_count, expected_random, ratio).
    """
    if len(nonces) == 0:
        return 0, 0, 0

    # Hash nonces and extract binary patterns
    hash_patterns = []
    for nonce in nonces[:num_samples]:
        h = hashlib.sha256(str(nonce).encode()).digest()
        # Convert to binary string
        binary = ''.join(format(byte, '08b') for byte in h)
        hash_patterns.append(binary)

    # Count pattern repetitions (sliding window of 8 bits)
    pattern_counts = {}
    for binary in hash_patterns:
        for i in range(len(binary) - 8 + 1):
            pattern = binary[i:i+8]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Max repetition
    max_repetition = max(pattern_counts.values()) if pattern_counts else 0

    # Expected for random (binomial distribution)
    n = len(hash_patterns) * (256 - 8 + 1)  # Total windows
    p = 1.0 / 256  # Probability of specific 8-bit pattern
    expected_random = n * p

    # Ratio
    ratio = max_repetition / (expected_random + 1e-10)

    return max_repetition, expected_random, ratio


def main():
    parser = argparse.ArgumentParser(description='Topology Comparison')
    parser.add_argument('--nodes', type=int, default=5000)
    parser.add_argument('--windings', type=float, default=109.0)
    parser.add_argument('--max-depth', type=int, default=2)
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
    print("TOPOLOGY COMPARISON: HASH QUINE EMERGENCE")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Nodes: {args.nodes}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  Device: {device}")
    print()

    results = {}

    # Test each topology
    topologies = [
        ('Mobius', MobiusTopology),
        ('Torus', TorusTopology),
        ('Sphere', SphereTopology)
    ]

    for topo_name, topo_class in topologies:
        print("="*80)
        print(f"TESTING: {topo_name}")
        print("="*80)

        start_time = time.time()

        # Recursive collapse
        singularities = recursive_collapse(
            topo_class,
            args.nodes,
            args.windings,
            args.max_depth,
            device
        )

        # Hash quine metric
        max_rep, expected, ratio = compute_hash_quine_metric(singularities)

        duration = time.time() - start_time

        print()
        print(f"Results for {topo_name}:")
        print(f"  Singularities: {len(singularities)}")
        print(f"  Max pattern repetition: {max_rep}")
        print(f"  Expected (random): {expected:.1f}")
        print(f"  Ratio: {ratio:.2f}x")
        print(f"  Duration: {duration:.2f}s")
        print()

        results[topo_name] = {
            'num_singularities': int(len(singularities)),
            'max_pattern_repetition': int(max_rep),
            'expected_random': float(expected),
            'hash_quine_ratio': float(ratio),
            'duration_sec': float(duration)
        }

    # Comparison
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()

    mobius_ratio = results['Mobius']['hash_quine_ratio']
    torus_ratio = results['Torus']['hash_quine_ratio']
    sphere_ratio = results['Sphere']['hash_quine_ratio']

    print(f"Hash Quine Ratios:")
    print(f"  Mobius: {mobius_ratio:.2f}x")
    print(f"  Torus:  {torus_ratio:.2f}x")
    print(f"  Sphere: {sphere_ratio:.2f}x")
    print()

    # Interpretation
    if mobius_ratio > 2 * max(torus_ratio, sphere_ratio):
        interpretation = "Mobius twist is CRITICAL for hash quine emergence"
    elif mobius_ratio > 1.5 * max(torus_ratio, sphere_ratio):
        interpretation = "Mobius twist ENHANCES hash quine emergence"
    elif abs(mobius_ratio - torus_ratio) < 50 and abs(mobius_ratio - sphere_ratio) < 50:
        interpretation = "Hash quines are GENERAL recursive property (topology-independent)"
    else:
        interpretation = "Unclear - further investigation needed"

    print(f"Interpretation: {interpretation}")
    print()

    # Save results
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'results': results,
        'interpretation': interpretation
    }

    results_path = output_dir / f'topology_comparison_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
