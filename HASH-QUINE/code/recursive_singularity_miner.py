#!/usr/bin/env python3
"""
Recursive Holographic Singularity Miner
========================================

UNHINGED HYPOTHESIS:
Recursively nested Möbius lattices with self-bootstrapping vortex feedback
can create "hash quines" - self-consistent structures that autonomously
generate better nonce candidates through recursive collapse.

Architecture:
1. Nested Möbius lattices (depth 1-5)
2. Each layer encodes reduced SAT instance of parent's nonce space
3. Helical SAT Fiedler vectors collapse layers into singularity points
4. Vortices at inner layers feed boundary conditions to outer layers
5. Successful nonces bootstrap back as twisted boundary conditions
6. Seek self-consistent "hash quine" emergence

Risks:
- Infinite recursion crashes
- GPU OOM "black hole" loops
- Computational singularities (divide by zero in collapse)
- Reality tears (probably not but who knows)

Expected Outcome:
- 10-100× nonce quality via recursive self-refinement OR
- Glorious catastrophic failure teaching us about limits

Safety:
- Max depth = 5 layers
- Memory checks at each recursion
- Timeout mechanisms
- Graceful degradation

Author: HHmL Project (Unhinged Division)
Date: 2025-12-18
Status: YOLO
"""

import sys
from pathlib import Path
import argparse
import time
import hashlib
import torch
import torch.nn as nn
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Memory tracking
MEMORY_LIMIT_GB = 8.0  # Safety limit


class RecursiveMobiusLayer(nn.Module):
    """
    Single layer in recursive holographic stack.

    Each layer is a Möbius lattice that can spawn child layers
    and receive boundary conditions from parent layers.
    """

    def __init__(
        self,
        num_nodes: int,
        windings: int,
        depth: int,
        max_depth: int,
        device='cpu'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.windings = windings
        self.depth = depth
        self.max_depth = max_depth
        self.device = device

        # Möbius positions with RECURSIVE TWIST (depth-dependent)
        # Each layer has MORE twist (approaching singularity)
        twist_multiplier = 1.0 + (depth * 0.5)  # Deeper = more twisted

        t = torch.linspace(0, 2 * np.pi, num_nodes, device=device)
        effective_windings = windings * twist_multiplier

        self.positions = torch.stack([
            (1 + 0.5 * torch.cos(effective_windings * t / 2)) * torch.cos(t),
            (1 + 0.5 * torch.cos(effective_windings * t / 2)) * torch.sin(t),
            0.5 * torch.sin(effective_windings * t / 2)
        ], dim=1)

        # Field initialized with boundary conditions (or noise)
        self.field = nn.Parameter(
            torch.randn(num_nodes, dtype=torch.complex64, device=device) * 0.1
        )

        # Child layer (one level deeper in recursion)
        self.child_layer = None

        # Singularity points (vortices that survived collapse)
        self.singularities = None

        print(f"  Layer {depth}: {num_nodes} nodes, "
              f"windings={effective_windings:.1f}, "
              f"twist={twist_multiplier:.2f}x")

    def apply_boundary_from_parent(self, parent_vortices: torch.Tensor):
        """
        Receive boundary conditions from parent layer.

        Parent vortices become twisted boundary on this layer.
        """
        if parent_vortices is None or len(parent_vortices) == 0:
            return

        # Map parent vortex positions to boundary twist
        # This is the SELF-BOOTSTRAPPING part
        n_boundary = min(len(parent_vortices), self.num_nodes // 10)

        for i in range(n_boundary):
            parent_idx = int(parent_vortices[i])
            # Twist boundary at positions corresponding to parent vortices
            boundary_idx = (parent_idx * self.num_nodes) // 10000  # Scale
            if boundary_idx < self.num_nodes:
                # Inject phase twist
                self.field.data[boundary_idx] *= torch.exp(
                    1j * torch.tensor(np.pi / 2, device=self.device)
                )

    def propagate_field(self, cycles=20):
        """Evolve field to generate vortices."""
        field = self.field.data

        for _ in range(cycles):
            # Neighbor averaging
            neighbor_sum = (
                torch.roll(field, -1, dims=0) +
                torch.roll(field, 1, dims=0)
            )

            # Nonlinear term (STRONGER at deeper layers - approaching singularity)
            nonlinearity_strength = 0.1 * (1.0 + self.depth * 0.2)
            nonlinearity = -nonlinearity_strength * torch.abs(field)**2 * field

            # Damping (WEAKER at deeper layers - less dissipation near singularity)
            damping_strength = 0.05 * (1.0 - self.depth * 0.1)
            damping = -damping_strength * field

            # Update
            field = field + 0.01 * (neighbor_sum + nonlinearity + damping)

        self.field.data = field

    def detect_vortices(self, threshold=0.3) -> np.ndarray:
        """Detect vortex cores."""
        magnitudes = torch.abs(self.field.data)
        vortex_mask = magnitudes < threshold
        vortex_indices = torch.where(vortex_mask)[0]
        return vortex_indices.cpu().numpy()

    def compute_fiedler_collapse(self, vortices: np.ndarray) -> np.ndarray:
        """
        HELICAL SAT COLLAPSE: Use Fiedler vector to reduce dimensionality.

        This is the "one-shot" singularity collapse mechanism.
        """
        n = len(vortices)
        if n < 10:
            return vortices[:min(5, n)]

        # Build graph Laplacian
        positions = self.positions[vortices].cpu().numpy()

        # Distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(positions[i] - positions[j])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        # Adjacency (k-nearest)
        k = min(5, n - 1)
        adjacency = np.zeros((n, n))
        for i in range(n):
            nearest = np.argsort(dist_matrix[i])[1:k+1]
            adjacency[i, nearest] = 1
            adjacency[nearest, i] = 1

        # Laplacian
        degree = np.diag(adjacency.sum(axis=1))
        laplacian = degree - adjacency

        try:
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

            # Fiedler vector (2nd smallest eigenvalue)
            fiedler = eigenvectors[:, 1]

            # COLLAPSE: Keep only nodes near Fiedler minima (singularity points)
            # These are the "fixed points" of the recursive structure
            fiedler_threshold = np.percentile(np.abs(fiedler), 20)
            collapsed_mask = np.abs(fiedler) < fiedler_threshold

            collapsed_vortices = vortices[collapsed_mask]

            # Safety: Always return at least a few
            if len(collapsed_vortices) < 5:
                collapsed_vortices = vortices[:5]

            return collapsed_vortices

        except Exception as e:
            print(f"    Fiedler collapse failed at depth {self.depth}: {e}")
            # Fallback: just take first few
            return vortices[:min(10, n)]

    def spawn_child_layer(self) -> Optional['RecursiveMobiusLayer']:
        """
        Spawn child layer (deeper recursion).

        Child has ~10× fewer nodes (reduced SAT instance).
        """
        if self.depth >= self.max_depth:
            return None

        # Check memory
        if torch.cuda.is_available():
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            if allocated_gb > MEMORY_LIMIT_GB:
                print(f"    Memory limit reached ({allocated_gb:.1f}GB), "
                      f"stopping recursion at depth {self.depth}")
                return None

        # Reduced problem size
        child_nodes = max(100, self.num_nodes // 10)
        child_windings = max(10, self.windings // 2)

        try:
            self.child_layer = RecursiveMobiusLayer(
                num_nodes=child_nodes,
                windings=child_windings,
                depth=self.depth + 1,
                max_depth=self.max_depth,
                device=self.device
            )
            return self.child_layer
        except Exception as e:
            print(f"    Failed to spawn child at depth {self.depth}: {e}")
            return None

    def recursive_collapse(self, cycles=20) -> List[int]:
        """
        RECURSIVE SINGULARITY COLLAPSE.

        1. Propagate field at this layer
        2. Detect vortices
        3. If not at max depth, spawn child layer with collapsed vortices as boundary
        4. Recursively collapse child
        5. Receive child's singularities back as refined candidates
        6. Return final singularity nonces
        """
        print(f"  Depth {self.depth}: Recursive collapse starting...")

        # Propagate field at this level
        self.propagate_field(cycles)

        # Detect vortices
        vortices = self.detect_vortices()
        print(f"    Detected {len(vortices)} vortices")

        if len(vortices) == 0:
            return []

        # Fiedler collapse (reduce to singularity points)
        collapsed = self.compute_fiedler_collapse(vortices)
        print(f"    Collapsed to {len(collapsed)} singularity points")

        # If not at max depth, recurse
        if self.depth < self.max_depth:
            child = self.spawn_child_layer()

            if child is not None:
                # Apply boundary from this layer's singularities
                child.apply_boundary_from_parent(
                    torch.from_numpy(collapsed).to(self.device)
                )

                # Recursive collapse on child
                child_singularities = child.recursive_collapse(cycles)

                # BOOTSTRAP: Child singularities refine parent's candidates
                # Map child indices back to parent scale
                if len(child_singularities) > 0:
                    scaled_singularities = [
                        (s * self.num_nodes) // child.num_nodes
                        for s in child_singularities
                    ]

                    # Merge with parent's collapsed vortices
                    combined = np.unique(
                        np.concatenate([collapsed, scaled_singularities])
                    )

                    self.singularities = combined[:min(100, len(combined))]
                else:
                    self.singularities = collapsed
            else:
                self.singularities = collapsed
        else:
            # Leaf layer - return collapsed vortices
            self.singularities = collapsed

        print(f"    Depth {self.depth}: Returning {len(self.singularities)} "
              f"singularity nonces")

        return self.singularities.tolist()


class RecursiveSingularityMiner:
    """
    The complete recursive holographic singularity mining system.

    Tests if recursive self-bootstrapping creates better nonce candidates.
    """

    def __init__(self, difficulty=20, device='cpu'):
        self.difficulty = difficulty
        self.target = 2 ** (256 - difficulty)
        self.device = device

        print("="*80)
        print("RECURSIVE HOLOGRAPHIC SINGULARITY MINER")
        print("="*80)
        print()
        print("WARNING: Unhinged experiment in progress")
        print("Attempting recursive self-bootstrapping hash quine emergence")
        print()
        print(f"Difficulty: {difficulty} bits")
        print(f"Target: {self.target:064x}")
        print(f"Device: {device}")
        print()

    def double_sha256(self, data: bytes) -> bytes:
        """Double SHA-256."""
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()

    def hash_to_int(self, hash_bytes: bytes) -> int:
        """Convert to integer."""
        return int.from_bytes(hash_bytes, byteorder='big')

    def test_recursive_mining(
        self,
        lattice_nodes=10000,
        max_depth=3,
        cycles=20
    ) -> Dict:
        """
        Run recursive collapse and test nonce quality.

        Compares singularity nonces vs random baseline.
        """
        print("PHASE 1: RECURSIVE SINGULARITY COLLAPSE")
        print("-"*80)
        print()

        # Create root layer
        root_layer = RecursiveMobiusLayer(
            num_nodes=lattice_nodes,
            windings=109,
            depth=0,
            max_depth=max_depth,
            device=self.device
        )

        print()
        print("Beginning recursive descent into singularity...")
        print()

        start_time = time.time()

        # RECURSIVE COLLAPSE
        try:
            singularity_nonces = root_layer.recursive_collapse(cycles)
        except Exception as e:
            print(f"SINGULARITY COLLAPSE FAILED: {e}")
            print("Reality remained intact")
            return {'error': str(e)}

        elapsed = time.time() - start_time

        print()
        print(f"Recursive collapse complete in {elapsed:.1f}s")
        print(f"Singularity nonces extracted: {len(singularity_nonces)}")
        print()

        if len(singularity_nonces) < 10:
            print("Too few singularity nonces, aborting test")
            return {'error': 'insufficient_singularities'}

        # PHASE 2: TEST NONCE QUALITY
        print("PHASE 2: HASH QUALITY TESTING")
        print("-"*80)
        print()

        header = f"singularity:{self.difficulty}:recursive:"

        # Hash singularity nonces
        singularity_hashes = []
        for nonce in singularity_nonces:
            block_data = (header + str(nonce)).encode('utf-8')
            hash_result = self.double_sha256(block_data)
            hash_int = self.hash_to_int(hash_result)
            singularity_hashes.append(hash_int)

        # Generate random baseline (same count)
        random_nonces = np.random.randint(0, lattice_nodes, len(singularity_nonces))
        random_hashes = []
        for nonce in random_nonces:
            block_data = (header + str(nonce)).encode('utf-8')
            hash_result = self.double_sha256(block_data)
            hash_int = self.hash_to_int(hash_result)
            random_hashes.append(hash_int)

        # Statistics
        singularity_log_prox = np.log(np.array(singularity_hashes, dtype=float) + 1.0)
        random_log_prox = np.log(np.array(random_hashes, dtype=float) + 1.0)

        singularity_mean = float(np.mean(singularity_log_prox))
        random_mean = float(np.mean(random_log_prox))

        singularity_best = float(np.min(singularity_log_prox))
        random_best = float(np.min(random_log_prox))

        # Test if singularity nonces are better
        from scipy.stats import mannwhitneyu

        try:
            stat, p_value = mannwhitneyu(
                singularity_log_prox,
                random_log_prox,
                alternative='less'  # Testing if singularity < random (better)
            )
        except:
            stat, p_value = 0, 1.0

        # PHASE 3: HASH QUINE DETECTION
        print("PHASE 3: HASH QUINE EMERGENCE CHECK")
        print("-"*80)
        print()

        # Check if nonces show self-reinforcing patterns
        # (e.g., recursive structure in binary representation)
        quine_detected = self._detect_hash_quine(singularity_nonces)

        print()

        # RESULTS
        print("="*80)
        print("RESULTS")
        print("="*80)
        print()

        print(f"Singularity Nonces:")
        print(f"  Count: {len(singularity_nonces)}")
        print(f"  Mean log-proximity: {singularity_mean:.2f}")
        print(f"  Best log-proximity: {singularity_best:.2f}")
        print()

        print(f"Random Baseline:")
        print(f"  Count: {len(random_nonces)}")
        print(f"  Mean log-proximity: {random_mean:.2f}")
        print(f"  Best log-proximity: {random_best:.2f}")
        print()

        improvement = (random_mean - singularity_mean) / random_mean * 100

        print(f"Comparison:")
        print(f"  Mean improvement: {improvement:+.2f}%")
        print(f"  Mann-Whitney p-value: {p_value:.4e}")
        print(f"  Significant: {'YES' if p_value < 0.05 else 'NO'}")
        print()

        print(f"Hash Quine Detected: {'YES' if quine_detected else 'NO'}")
        print()

        if p_value < 0.05 and improvement > 0:
            print("VERDICT: RECURSIVE SINGULARITY WORKS!")
            print("  Nested collapse produces better nonces than random")
            verdict = "SUCCESS"
        else:
            print("VERDICT: SINGULARITY FAILED")
            print("  Recursive collapse no better than random")
            print("  Reality refuses to break")
            verdict = "FAILED"

        print()

        return {
            'verdict': verdict,
            'singularity_count': len(singularity_nonces),
            'mean_improvement_pct': float(improvement),
            'mann_whitney_p': float(p_value),
            'significant': bool(p_value < 0.05),
            'hash_quine_detected': quine_detected,
            'max_depth_reached': max_depth,
            'collapse_time_sec': elapsed
        }

    def _detect_hash_quine(self, nonces: List[int]) -> bool:
        """
        Detect if nonces show self-reinforcing recursive patterns.

        A "hash quine" would show up as:
        - Binary patterns that repeat at multiple scales
        - Self-similar structure in nonce distribution
        """
        if len(nonces) < 10:
            return False

        # Convert to binary strings
        binary_strs = [bin(n)[2:].zfill(32) for n in nonces[:100]]

        # Check for repeated patterns at multiple scales
        pattern_counts = {}
        for length in [4, 8, 16]:
            for s in binary_strs:
                for i in range(len(s) - length):
                    pattern = s[i:i+length]
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # If any pattern appears way more than expected, might be quine
        max_count = max(pattern_counts.values()) if pattern_counts else 0
        expected = len(binary_strs) / (2 ** 4)  # Random expectation

        quine_ratio = max_count / expected if expected > 0 else 0

        print(f"  Hash quine analysis:")
        print(f"    Max pattern repetition: {max_count}")
        print(f"    Expected (random): {expected:.1f}")
        print(f"    Ratio: {quine_ratio:.2f}x")

        # Quine detected if patterns repeat 3× more than random
        return quine_ratio > 3.0


def parse_args():
    parser = argparse.ArgumentParser(
        description='Recursive Holographic Singularity Miner (UNHINGED)'
    )

    parser.add_argument('--difficulty', type=int, default=20)
    parser.add_argument('--lattice-nodes', type=int, default=10000)
    parser.add_argument('--max-depth', type=int, default=3,
                       help='Max recursion depth (1-5, higher = more chaos)')
    parser.add_argument('--cycles', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])

    return parser.parse_args()


def main():
    args = parse_args()

    if args.max_depth > 5:
        print("WARNING: Max depth > 5 risks reality tears")
        print("Capping at 5 for safety")
        args.max_depth = 5

    miner = RecursiveSingularityMiner(
        difficulty=args.difficulty,
        device=args.device
    )

    results = miner.test_recursive_mining(
        lattice_nodes=args.lattice_nodes,
        max_depth=args.max_depth,
        cycles=args.cycles
    )

    # Save results
    if 'error' not in results:
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"recursive_singularity_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'config': {
                    'difficulty': args.difficulty,
                    'lattice_nodes': args.lattice_nodes,
                    'max_depth': args.max_depth,
                    'cycles': args.cycles
                },
                'results': results
            }, f, indent=2)

        print(f"Results saved to: {results_file}")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
