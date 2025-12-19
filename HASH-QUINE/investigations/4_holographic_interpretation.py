#!/usr/bin/env python3
"""
Holographic Interpretation of Recursive Hash Quines
====================================================

Tests whether recursive Möbius layers exhibit bulk-boundary correspondence
similar to AdS/CFT holographic duality.

Holographic Hypothesis:
- Outer layers (boundary) = low-dimensional encoding of high-dimensional bulk
- Inner layers (bulk) = emergent higher-energy modes
- Recursive collapse = holographic projection from bulk to boundary
- Self-bootstrapping = consistency condition (like boundary determines bulk)

Testable Predictions:
1. Information content: Inner layers encode MORE information than outer layers
2. Energy scaling: Field energy increases toward center (UV/IR correspondence)
3. Correlation length: Decreases toward center (bulk has finer structure)
4. Entanglement entropy: Scales with boundary area (holographic entropy bound)

Comparison to AdS/CFT:
- AdS/CFT: 3D boundary ↔ 4D bulk (gravity emerges from gauge theory)
- Hash Quines: Outer Möbius ↔ Inner Möbius layers (patterns emerge from recursion)

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
from typing import List, Dict
from scipy.stats import pearsonr

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class RecursiveMobiusHolography:
    """
    Recursive Möbius layers with holographic analysis.
    """

    def __init__(self, num_nodes: int, max_depth: int, device='cpu'):
        self.num_nodes = num_nodes
        self.max_depth = max_depth
        self.device = device

        # Store all layers
        self.layers = []

        # Metrics per layer
        self.layer_metrics = []

    def build_layers(self, windings: float = 109.0):
        """Build recursive Möbius layers from boundary to bulk."""

        current_nodes = self.num_nodes

        for depth in range(self.max_depth + 1):
            print(f"  Layer {depth}: {current_nodes} nodes")

            # Twist increases toward center (approaching singularity)
            twist_multiplier = 1.0 + (depth * 0.5)
            effective_windings = windings * twist_multiplier

            # Möbius positions
            t = torch.linspace(0, 2 * np.pi, current_nodes, device=self.device)

            positions = torch.stack([
                (1 + 0.5 * torch.cos(effective_windings * t / 2)) * torch.cos(t),
                (1 + 0.5 * torch.cos(effective_windings * t / 2)) * torch.sin(t),
                0.5 * torch.sin(effective_windings * t / 2)
            ], dim=1)

            # Initialize field
            field = torch.randn(current_nodes, dtype=torch.complex64, device=self.device) * 0.1

            # Evolve field
            field = self._propagate_field(field, depth, cycles=20)

            # Store layer
            self.layers.append({
                'depth': depth,
                'num_nodes': current_nodes,
                'windings': effective_windings,
                'positions': positions,
                'field': field
            })

            # Compute layer metrics
            metrics = self._compute_layer_metrics(field, positions, depth)
            self.layer_metrics.append(metrics)

            # Next layer (collapse to bulk)
            current_nodes = max(current_nodes // 5, 50)

    def _propagate_field(self, field: torch.Tensor, depth: int, cycles: int = 20):
        """Evolve field with depth-dependent parameters."""

        # Nonlinearity and damping scale with depth
        nonlinearity_strength = 0.1 * (1.0 + depth * 0.2)
        damping_strength = 0.05 * (1.0 - depth * 0.1)

        for _ in range(cycles):
            # Neighbor averaging
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

        return field

    def _compute_layer_metrics(self, field: torch.Tensor, positions: torch.Tensor, depth: int) -> Dict:
        """Compute holographic metrics for a layer."""

        # 1. Field energy (should increase toward bulk - UV/IR correspondence)
        energy = torch.mean(torch.abs(field) ** 2).item()

        # 2. Vortex density (topological objects)
        magnitudes = torch.abs(field)
        vortex_mask = magnitudes < 0.3
        vortex_density = vortex_mask.sum().item() / len(field)

        # 3. Correlation length (should decrease toward bulk)
        # Measure autocorrelation of field magnitude
        mag_np = magnitudes.cpu().numpy()
        if len(mag_np) > 1:
            autocorr = np.corrcoef(mag_np[:-1], mag_np[1:])[0, 1]
        else:
            autocorr = 0.0

        # 4. Entropy (Shannon entropy of magnitude distribution)
        mag_hist, _ = np.histogram(mag_np, bins=20, range=(0, mag_np.max() + 1e-6))
        mag_probs = mag_hist / (mag_hist.sum() + 1e-10)
        mag_probs = mag_probs[mag_probs > 0]
        shannon_entropy = -np.sum(mag_probs * np.log2(mag_probs + 1e-10))

        # 5. Effective temperature (variance of field)
        temperature = torch.var(torch.abs(field)).item()

        return {
            'depth': depth,
            'num_nodes': len(field),
            'energy': energy,
            'vortex_density': vortex_density,
            'autocorrelation': autocorr,
            'entropy': shannon_entropy,
            'temperature': temperature
        }

    def test_holographic_predictions(self):
        """
        Test holographic duality predictions.

        Returns dict with test results and p-values.
        """
        print("\nTesting holographic predictions...")

        depths = [m['depth'] for m in self.layer_metrics]
        energies = [m['energy'] for m in self.layer_metrics]
        entropies = [m['entropy'] for m in self.layer_metrics]
        temps = [m['temperature'] for m in self.layer_metrics]
        autocorrs = [m['autocorrelation'] for m in self.layer_metrics]

        tests = {}

        # Test 1: Energy increases toward bulk (UV/IR correspondence)
        if len(depths) > 2:
            r_energy, p_energy = pearsonr(depths, energies)
            tests['energy_scaling'] = {
                'correlation': r_energy,
                'p_value': p_energy,
                'prediction': 'positive correlation (higher energy toward bulk)',
                'result': 'PASS' if r_energy > 0 and p_energy < 0.05 else 'FAIL'
            }
            print(f"  Energy scaling: r={r_energy:.3f}, p={p_energy:.3e} -> {tests['energy_scaling']['result']}")

        # Test 2: Entropy increases toward bulk (more information in bulk)
        if len(depths) > 2:
            r_entropy, p_entropy = pearsonr(depths, entropies)
            tests['entropy_scaling'] = {
                'correlation': r_entropy,
                'p_value': p_entropy,
                'prediction': 'positive correlation (higher entropy toward bulk)',
                'result': 'PASS' if r_entropy > 0 and p_entropy < 0.05 else 'FAIL'
            }
            print(f"  Entropy scaling: r={r_entropy:.3f}, p={p_entropy:.3e} -> {tests['entropy_scaling']['result']}")

        # Test 3: Temperature increases toward bulk
        if len(depths) > 2:
            r_temp, p_temp = pearsonr(depths, temps)
            tests['temperature_scaling'] = {
                'correlation': r_temp,
                'p_value': p_temp,
                'prediction': 'positive correlation (higher T toward bulk)',
                'result': 'PASS' if r_temp > 0 and p_temp < 0.05 else 'FAIL'
            }
            print(f"  Temperature scaling: r={r_temp:.3f}, p={p_temp:.3e} -> {tests['temperature_scaling']['result']}")

        # Test 4: Correlation length decreases toward bulk
        if len(depths) > 2:
            r_autocorr, p_autocorr = pearsonr(depths, autocorrs)
            tests['correlation_length_scaling'] = {
                'correlation': r_autocorr,
                'p_value': p_autocorr,
                'prediction': 'negative correlation (shorter range toward bulk)',
                'result': 'PASS' if r_autocorr < 0 and p_autocorr < 0.05 else 'FAIL'
            }
            print(f"  Correlation length: r={r_autocorr:.3f}, p={p_autocorr:.3e} -> {tests['correlation_length_scaling']['result']}")

        return tests


def main():
    parser = argparse.ArgumentParser(description='Holographic Interpretation of Hash Quines')
    parser.add_argument('--nodes', type=int, default=5000)
    parser.add_argument('--max-depth', type=int, default=4)
    parser.add_argument('--windings', type=float, default=109.0)
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
    print("HOLOGRAPHIC INTERPRETATION OF RECURSIVE HASH QUINES")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Boundary nodes: {args.nodes}")
    print(f"  Max depth (layers): {args.max_depth}")
    print(f"  Windings: {args.windings}")
    print()

    # Build recursive holographic structure
    print("="*80)
    print("BUILDING RECURSIVE HOLOGRAPHIC STRUCTURE")
    print("="*80)
    print()

    holography = RecursiveMobiusHolography(args.nodes, args.max_depth, device)

    start_time = time.time()
    holography.build_layers(args.windings)
    build_time = time.time() - start_time

    print(f"\nBuilt {len(holography.layers)} layers in {build_time:.2f}s")

    # Display layer metrics
    print()
    print("="*80)
    print("LAYER METRICS (Boundary -> Bulk)")
    print("="*80)
    print()

    print(f"{'Depth':<8} {'Nodes':<10} {'Energy':<12} {'Entropy':<12} {'Temp':<12} {'Autocorr':<12}")
    print("-" * 80)

    for metrics in holography.layer_metrics:
        print(f"{metrics['depth']:<8} {metrics['num_nodes']:<10} "
              f"{metrics['energy']:<12.6f} {metrics['entropy']:<12.4f} "
              f"{metrics['temperature']:<12.6f} {metrics['autocorrelation']:<12.4f}")

    # Test holographic predictions
    print()
    print("="*80)
    print("HOLOGRAPHIC DUALITY TESTS")
    print("="*80)

    tests = holography.test_holographic_predictions()

    # Summary
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()

    num_passed = sum(1 for t in tests.values() if t['result'] == 'PASS')
    total_tests = len(tests)

    print(f"Passed: {num_passed}/{total_tests} holographic predictions")
    print()

    if num_passed >= 3:
        print("STRONG EVIDENCE for holographic interpretation:")
        print("  - Recursive Mobius layers exhibit bulk-boundary structure")
        print("  - Inner layers = higher energy/entropy (bulk)")
        print("  - Outer layers = lower energy/entropy (boundary)")
        print("  - Self-bootstrapping may implement holographic consistency")
        print()
        print("Hash quines could be HOLOGRAPHIC PROJECTIONS from bulk to boundary")

    elif num_passed >= 2:
        print("PARTIAL EVIDENCE for holographic interpretation:")
        print("  - Some holographic signatures present")
        print("  - Requires further investigation")

    else:
        print("WEAK EVIDENCE for holographic interpretation:")
        print("  - Recursive structure may not be holographic")
        print("  - Could be purely mathematical artifact")

    print()

    # Key insight
    print("KEY INSIGHT:")
    print()
    print("If hash quines exhibit holographic structure (boundary ↔ bulk),")
    print("this provides a NEW INTERPRETATION:")
    print()
    print("  Hash quines = holographic encoding of recursive topology")
    print("  Pattern repetition = projection from high-dimensional bulk")
    print("  Self-similarity = holographic consistency condition")
    print()
    print("This connects to AdS/CFT duality (boundary theory ↔ bulk gravity)")
    print("and suggests recursive Mobius could be a toy model of holography")

    print()

    # Save results
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'layer_metrics': holography.layer_metrics,
        'holographic_tests': tests,
        'summary': {
            'num_layers': len(holography.layers),
            'tests_passed': num_passed,
            'total_tests': total_tests,
            'holographic_evidence': 'strong' if num_passed >= 3 else ('partial' if num_passed >= 2 else 'weak')
        }
    }

    results_path = output_dir / f'holographic_interpretation_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
