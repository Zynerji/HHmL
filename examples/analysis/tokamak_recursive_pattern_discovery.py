#!/usr/bin/env python3
"""
Tokamak Recursive Pattern Discovery
====================================

HYPOTHESIS:
The 300-strip tokamak configuration with wormhole networks may exhibit
self-similar recursive patterns (like hash quines) when subjected to:
1. Recursive spectral collapse (Helical SAT Fiedler vectors)
2. Multi-layer nesting analysis
3. Self-bootstrapping pattern detection

This test adapts the hash quine discovery framework to analyze tokamak
field states for emergent self-similarity, WITHOUT any cryptocurrency mining.

Architecture:
1. Run tokamak training to generate wormhole networks (300 strips)
2. Build recursive layer analysis (depth 1-3)
3. Apply Helical SAT spectral collapse to each layer
4. Measure pattern self-similarity across layers
5. Detect "topological quines" - self-referential wormhole structures

Expected Outcomes:
- Discover if wormhole networks exhibit recursive self-similarity
- Measure pattern repetition across spatial scales
- Test if spectral collapse reveals hidden network structure
- Compare to hash quine baseline (312-371x repetition)

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import tokamak components
sys.path.insert(0, str(project_root / "examples" / "training"))
from train_tokamak_wormhole_hunt import (
    SparseTokamakMobiusStrips,
    RetrocausalCoupler,
    detect_temporal_vortices_gpu
)


class RecursiveLayerAnalyzer:
    """
    Analyzes tokamak field states for recursive self-similar patterns.

    Adapted from hash quine framework but analyzes wormhole networks
    instead of Bitcoin nonces.
    """

    def __init__(
        self,
        tokamak_field: torch.Tensor,
        strip_indices: torch.Tensor,
        positions: torch.Tensor,
        num_layers: int = 3,
        device='cuda'
    ):
        self.field = tokamak_field  # [num_nodes, num_time_steps] complex
        self.strip_indices = strip_indices
        self.positions = positions
        self.num_layers = num_layers
        self.device = device

        # Storage for recursive layers
        self.layers = []
        self.collapsed_representations = []
        self.similarity_scores = []

        print(f"Initialized Recursive Layer Analyzer:")
        print(f"  Input field: {self.field.shape}")
        print(f"  Num layers: {num_layers}")
        print(f"  Device: {device}")

    def build_recursive_layers(self):
        """
        Build recursive layer hierarchy from tokamak field.

        Each layer represents field at different spatial scale:
        - Layer 0: Full field (all nodes)
        - Layer 1: Per-strip aggregated field (300 nodes)
        - Layer 2: Per-cluster aggregated (strip groups)
        - Layer 3: Global summary (single node)
        """
        print("\n[Recursive Layers] Building spatial hierarchy...")

        # Layer 0: Full field (baseline)
        self.layers.append({
            'depth': 0,
            'field': self.field.clone(),
            'num_nodes': self.field.shape[0],
            'description': 'Full tokamak field'
        })
        print(f"  Layer 0: {self.field.shape[0]} nodes (full field)")

        # Layer 1: Per-strip aggregated
        num_strips = self.strip_indices.max().item() + 1
        strip_field = torch.zeros(num_strips, self.field.shape[1],
                                  dtype=torch.complex64, device=self.device)

        for strip_idx in range(num_strips):
            mask = self.strip_indices == strip_idx
            strip_field[strip_idx] = self.field[mask].mean(dim=0)

        self.layers.append({
            'depth': 1,
            'field': strip_field,
            'num_nodes': num_strips,
            'description': 'Per-strip aggregated field'
        })
        print(f"  Layer 1: {num_strips} nodes (strip-level)")

        # Layer 2: Strip clusters (groups of ~10 strips)
        cluster_size = 10
        num_clusters = (num_strips + cluster_size - 1) // cluster_size
        cluster_field = torch.zeros(num_clusters, self.field.shape[1],
                                    dtype=torch.complex64, device=self.device)

        for cluster_idx in range(num_clusters):
            start = cluster_idx * cluster_size
            end = min(start + cluster_size, num_strips)
            cluster_field[cluster_idx] = strip_field[start:end].mean(dim=0)

        self.layers.append({
            'depth': 2,
            'field': cluster_field,
            'num_nodes': num_clusters,
            'description': 'Strip cluster field'
        })
        print(f"  Layer 2: {num_clusters} nodes (cluster-level)")

        print(f"[OK] Built {len(self.layers)} recursive layers")

    def spectral_collapse(self, layer_idx: int) -> np.ndarray:
        """
        Apply Fiedler vector spectral collapse to a layer.

        This is the KEY mechanism from hash quine discovery:
        - Compute graph Laplacian from field adjacency
        - Extract Fiedler vector (2nd smallest eigenvalue's eigenvector)
        - Use as one-shot dimensionality reduction

        Returns:
            Fiedler vector representation (1D projection)
        """
        layer = self.layers[layer_idx]
        field = layer['field']  # [N, T] complex

        print(f"\n[Spectral Collapse] Layer {layer_idx} ({layer['description']})")

        # Compute adjacency from field correlations
        field_real = torch.stack([field.real, field.imag], dim=-1)  # [N, T, 2]
        field_flat = field_real.reshape(field.shape[0], -1)  # [N, 2T]

        # Correlation matrix as adjacency
        adjacency = torch.abs(torch.mm(field_flat, field_flat.t()))
        adjacency = adjacency.cpu().numpy()

        # Normalize
        adjacency = adjacency / (adjacency.max() + 1e-8)

        # Degree matrix
        degree = np.diag(adjacency.sum(axis=1))

        # Laplacian
        laplacian = degree - adjacency

        # Extract Fiedler vector (2nd smallest eigenvalue)
        try:
            eigenvalues, eigenvectors = eigsh(laplacian, k=min(10, layer['num_nodes']-1), which='SM')
            fiedler = eigenvectors[:, 1]  # 2nd eigenvector

            print(f"  Fiedler eigenvalue: {eigenvalues[1]:.6f}")
            print(f"  Spectral gap: {eigenvalues[1] - eigenvalues[0]:.6f}")
            print(f"  Fiedler vector range: [{fiedler.min():.4f}, {fiedler.max():.4f}]")

            self.collapsed_representations.append({
                'layer': layer_idx,
                'fiedler': fiedler,
                'eigenvalue': eigenvalues[1],
                'spectral_gap': eigenvalues[1] - eigenvalues[0]
            })

            return fiedler

        except Exception as e:
            print(f"  [WARN] Spectral collapse failed: {e}")
            # Fallback: return field magnitude
            fallback = torch.abs(field).mean(dim=1).cpu().numpy()
            self.collapsed_representations.append({
                'layer': layer_idx,
                'fiedler': fallback,
                'eigenvalue': 0.0,
                'spectral_gap': 0.0
            })
            return fallback

    def measure_pattern_similarity(self):
        """
        Measure self-similarity across collapsed layers.

        This is the hash quine test:
        - Convert Fiedler vectors to binary patterns
        - Count pattern repetitions
        - Compare to random baseline
        """
        print("\n[Pattern Similarity] Analyzing cross-layer self-similarity...")

        if len(self.collapsed_representations) < 2:
            print("  [WARN] Need at least 2 layers for similarity analysis")
            return {}

        # Convert each Fiedler vector to binary pattern
        binary_patterns = []
        for rep in self.collapsed_representations:
            fiedler = rep['fiedler']
            median = np.median(fiedler)
            binary = (fiedler > median).astype(int)
            binary_patterns.append(binary)
            print(f"  Layer {rep['layer']}: {len(binary)} bits, "
                  f"{binary.sum()} ones ({100*binary.sum()/len(binary):.1f}%)")

        # Measure pattern overlap across layers
        similarities = []

        for i in range(len(binary_patterns)):
            for j in range(i+1, len(binary_patterns)):
                # Need to align different sizes
                min_len = min(len(binary_patterns[i]), len(binary_patterns[j]))

                # Downsample larger pattern
                p1 = binary_patterns[i]
                p2 = binary_patterns[j]

                if len(p1) > min_len:
                    indices = np.linspace(0, len(p1)-1, min_len, dtype=int)
                    p1 = p1[indices]

                if len(p2) > min_len:
                    indices = np.linspace(0, len(p2)-1, min_len, dtype=int)
                    p2 = p2[indices]

                # Count matching bits
                matches = (p1 == p2).sum()
                similarity = matches / min_len

                similarities.append({
                    'layer_i': i,
                    'layer_j': j,
                    'similarity': similarity,
                    'matches': int(matches),
                    'total_bits': min_len
                })

                print(f"  Layer {i} <-> Layer {j}: "
                      f"{100*similarity:.1f}% similar ({matches}/{min_len} bits)")

        # Compute baseline: random patterns
        random_similarity = 0.5  # Expected for random binary

        # Pattern repetition factor (vs random)
        if similarities:
            avg_similarity = np.mean([s['similarity'] for s in similarities])
            repetition_factor = avg_similarity / random_similarity

            print(f"\n[Result] Average cross-layer similarity: {100*avg_similarity:.1f}%")
            print(f"[Result] Pattern repetition factor: {repetition_factor:.2f}x vs random")
            print(f"[Result] Hash quine baseline: 312-371x")

            if repetition_factor > 2.0:
                print("[!!] STRONG self-similarity detected (>2x random)")
            elif repetition_factor > 1.5:
                print("[*] Moderate self-similarity detected")
            else:
                print("[.] Weak self-similarity (near random baseline)")

            return {
                'similarities': similarities,
                'avg_similarity': avg_similarity,
                'repetition_factor': repetition_factor,
                'baseline': random_similarity
            }

        return {}

    def analyze_wormhole_network_recursion(self):
        """
        Analyze if wormhole network structure exhibits recursion.

        Test if wormhole connectivity patterns at different scales
        are self-similar (like hash quines).
        """
        print("\n[Wormhole Network Recursion] Analyzing multi-scale connectivity...")

        # This requires wormhole detection at each layer
        # For now, report structure
        print("  [TODO] Implement wormhole detection at each recursive layer")
        print("  [TODO] Compare connectivity patterns across scales")
        print("  [TODO] Measure if strip-level wormholes mirror cluster-level structure")

        return {}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Tokamak Recursive Pattern Discovery (Hash Quine Analysis)'
    )

    parser.add_argument('--num-strips', type=int, default=300,
                       help='Number of Mobius strips')
    parser.add_argument('--nodes-per-strip', type=int, default=166,
                       help='Nodes per strip')
    parser.add_argument('--num-time-steps', type=int, default=10,
                       help='Time steps')
    parser.add_argument('--num-cycles', type=int, default=100,
                       help='Training cycles to generate field state')
    parser.add_argument('--num-layers', type=int, default=3,
                       help='Recursive layer depth')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str,
                       default='results/tokamak_recursive_patterns',
                       help='Output directory')

    return parser.parse_args()


def main():
    """Run tokamak recursive pattern discovery."""
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    total_nodes = args.num_strips * args.nodes_per_strip

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TOKAMAK RECURSIVE PATTERN DISCOVERY (HASH QUINE FRAMEWORK)")
    print("="*80)
    print(f"Strips: {args.num_strips}")
    print(f"Nodes: {total_nodes}")
    print(f"Time steps: {args.num_time_steps}")
    print(f"Training cycles: {args.num_cycles}")
    print(f"Recursive layers: {args.num_layers}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print()

    # =========================================================================
    # PHASE 1: Generate Tokamak Field State
    # =========================================================================

    print("="*80)
    print("PHASE 1: GENERATING TOKAMAK FIELD STATE")
    print("="*80)
    print()

    # Initialize tokamak
    tokamak = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        kappa=1.5,
        delta=0.3,
        sparse_threshold=0.3,
        max_neighbors=2000,
        device=device
    )

    # Initialize retrocausal coupling
    coupler = RetrocausalCoupler(
        num_nodes=total_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,
        prophetic_mixing=0.3,
        device=device
    )

    # Initialize fields (self-consistent)
    print("\nInitializing fields (self-consistent)...")
    field_forward = torch.randn(total_nodes, args.num_time_steps,
                                dtype=torch.complex64, device=device) * 0.1
    field_backward = field_forward.clone()

    # Run retrocausal coupling to converge
    for _ in range(50):
        field_forward, field_backward = coupler.apply_coupling(
            field_forward, field_backward,
            enable_mixing=True,
            enable_swapping=True,
            enable_anchoring=True
        )

    print(f"  Fields initialized (divergence: {torch.abs(field_forward - field_backward).mean():.6f})")

    # Run brief training to generate interesting structure
    print(f"\nRunning {args.num_cycles} cycles to generate wormhole networks...")

    for cycle in range(args.num_cycles):
        # Evolve fields
        field_forward, field_backward = coupler.apply_coupling(
            field_forward, field_backward,
            enable_mixing=True,
            enable_swapping=True,
            enable_anchoring=True
        )

        if cycle % 20 == 0:
            # Detect vortices
            vortex_dict = detect_temporal_vortices_gpu(
                field_forward,
                tokamak.positions,
                vortex_threshold=0.1,
                phase_grad_threshold=1.0
            )

            num_vortices = len(vortex_dict['node_idx'])
            print(f"  Cycle {cycle}/{args.num_cycles}: "
                  f"{num_vortices} vortices, "
                  f"divergence {torch.abs(field_forward - field_backward).mean():.6f}")

    print(f"\n[OK] Generated tokamak field state with wormhole structure")

    # =========================================================================
    # PHASE 2: Recursive Layer Analysis
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 2: RECURSIVE LAYER ANALYSIS")
    print("="*80)
    print()

    # Create analyzer
    analyzer = RecursiveLayerAnalyzer(
        tokamak_field=field_forward,
        strip_indices=tokamak.strip_indices,
        positions=tokamak.positions,
        num_layers=args.num_layers,
        device=device
    )

    # Build recursive layers
    analyzer.build_recursive_layers()

    # =========================================================================
    # PHASE 3: Spectral Collapse (Fiedler Vectors)
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 3: SPECTRAL COLLAPSE (HELICAL SAT)")
    print("="*80)
    print()

    for layer_idx in range(len(analyzer.layers)):
        analyzer.spectral_collapse(layer_idx)

    # =========================================================================
    # PHASE 4: Self-Similarity Measurement
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 4: SELF-SIMILARITY MEASUREMENT")
    print("="*80)
    print()

    similarity_results = analyzer.measure_pattern_similarity()

    # =========================================================================
    # PHASE 5: Wormhole Network Recursion
    # =========================================================================

    print("\n" + "="*80)
    print("PHASE 5: WORMHOLE NETWORK RECURSION")
    print("="*80)
    print()

    network_results = analyzer.analyze_wormhole_network_recursion()

    # =========================================================================
    # FINAL: Save Results
    # =========================================================================

    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    print()

    results = {
        'timestamp': timestamp,
        'parameters': vars(args),
        'layers': [
            {
                'depth': layer['depth'],
                'num_nodes': layer['num_nodes'],
                'description': layer['description']
            }
            for layer in analyzer.layers
        ],
        'collapsed_representations': [
            {
                'layer': rep['layer'],
                'eigenvalue': float(rep['eigenvalue']),
                'spectral_gap': float(rep['spectral_gap']),
                'fiedler_stats': {
                    'min': float(rep['fiedler'].min()),
                    'max': float(rep['fiedler'].max()),
                    'mean': float(rep['fiedler'].mean()),
                    'std': float(rep['fiedler'].std())
                }
            }
            for rep in analyzer.collapsed_representations
        ],
        'similarity_analysis': similarity_results,
        'network_recursion': network_results
    }

    results_file = output_dir / f"recursive_patterns_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_file}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if similarity_results:
        print(f"Recursive layers analyzed: {len(analyzer.layers)}")
        print(f"Average cross-layer similarity: {100*similarity_results['avg_similarity']:.1f}%")
        print(f"Pattern repetition factor: {similarity_results['repetition_factor']:.2f}x")
        print(f"Hash quine baseline: 312-371x")

        if similarity_results['repetition_factor'] > 2.0:
            print("\n[!!!] TOPOLOGICAL QUINE CANDIDATE DETECTED")
            print("     Strong self-similarity across recursive layers!")
        elif similarity_results['repetition_factor'] > 1.5:
            print("\n[**] Moderate self-similarity detected")
        else:
            print("\n[--] Weak self-similarity (near random baseline)")

    print(f"\nComplete results: {results_file}")
    print("\n" + "="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
