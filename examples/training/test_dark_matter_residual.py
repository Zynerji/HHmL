#!/usr/bin/env python3
"""
Dark Matter Residual Test - RNN Tokamak Implementation
========================================================

Tests for dark matter signatures in RNN-controlled tokamak vortex dynamics.

Dark matter hypothesis: Pruned/annihilated vortices leave residual field structure
that behaves like "dark" (non-interacting but gravitationally present) matter.

Metrics:
- Annihilated vortex count (dark matter candidates)
- Residual field amplitude at annihilation sites
- Energy density of annihilated regions
- Correlation with total field mass

Target: 100 cycles to establish baseline dark matter production rate

Author: HHmL Project
Date: 2025-12-19
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
from src.hhml.core.spatiotemporal.retrocausal_coupling import RetrocausalCoupler

# Import from training scripts in same directory
sys.path.insert(0, str(Path(__file__).parent))
from train_tokamak_wormhole_hunt import detect_temporal_vortices_gpu
from train_tokamak_rnn_control import TokamakRNN


def analyze_dark_matter_residue(field, annihilated_nodes, tokamak):
    """
    Analyze field residue at annihilated vortex sites (dark matter candidates).

    Args:
        field: Complex field tensor [num_nodes, num_time_steps]
        annihilated_nodes: Tensor of node indices that were annihilated
        tokamak: Tokamak geometry

    Returns:
        dict with dark matter metrics
    """
    if len(annihilated_nodes) == 0:
        # Compute total energy even when no annihilations
        total_energy = (torch.abs(field) ** 2).mean().item()
        return {
            'num_dark_candidates': 0,
            'residual_amplitude_mean': 0.0,
            'residual_amplitude_std': 0.0,
            'energy_density_dark': 0.0,
            'dark_matter_fraction': 0.0,
            'total_energy': total_energy
        }

    # Extract field at annihilated sites
    dark_field = field[annihilated_nodes, :]  # [num_annihilated, num_time_steps]

    # Compute residual amplitude (should be low but non-zero)
    residual_amplitude = torch.abs(dark_field)
    residual_mean = residual_amplitude.mean().item()
    residual_std = residual_amplitude.std().item()

    # Compute energy density at dark sites
    energy_density_dark = (residual_amplitude ** 2).mean().item()

    # Compute total field energy
    total_energy = (torch.abs(field) ** 2).mean().item()

    # Dark matter fraction (energy at annihilated sites / total energy)
    dark_fraction = energy_density_dark / (total_energy + 1e-10)

    return {
        'num_dark_candidates': len(annihilated_nodes),
        'residual_amplitude_mean': residual_mean,
        'residual_amplitude_std': residual_std,
        'energy_density_dark': energy_density_dark,
        'dark_matter_fraction': dark_fraction,
        'total_energy': total_energy
    }


def train_cycle_with_dark_tracking(tokamak, coupler, rnn, optimizer, scaler, args, device, prev_metrics):
    """Train one cycle and track dark matter signatures."""

    # Get RNN parameters - construct state with proper time dimension
    # Create state features (8 features from previous metrics)
    state_vec = torch.zeros(8, dtype=torch.float32, device=device)
    if prev_metrics is not None:
        state_vec[0] = prev_metrics.get('vortex_density', 0.0)
        state_vec[1] = prev_metrics.get('fixed_point_pct', 100.0) / 100.0
        state_vec[2] = prev_metrics.get('divergence', 0.0)
        state_vec[3] = prev_metrics.get('dark_matter_fraction', 0.0)
        state_vec[4] = prev_metrics.get('dark_candidates', 0) / args.total_nodes
        state_vec[5] = prev_metrics.get('num_annihilated', 0) / args.total_nodes
        state_vec[6] = prev_metrics.get('dark_residual_amplitude', 0.0)
        state_vec[7] = prev_metrics.get('dark_energy_density', 0.0)

    # Repeat across time dimension
    state_features = state_vec.unsqueeze(0).repeat(args.num_time_steps, 1)

    # Pad to state_dim=128
    padding = torch.zeros(args.num_time_steps, 120, device=device)
    state_input = torch.cat([state_features, padding], dim=-1)  # (num_time_steps, 128)
    state_input = state_input.unsqueeze(0)  # (1, num_time_steps, 128)

    # RNN forward pass
    with torch.no_grad():
        params = rnn(state_input)

    # Extract scalar values (remove batch dimension)
    params_scalar = {k: v.item() if v.numel() == 1 else v.squeeze(0).item()
                     for k, v in params.items() if k != 'value'}

    # Update retrocausal coupling with RNN parameters
    with torch.no_grad():
        coupler.retrocausal_strength = params_scalar['retrocausal_alpha']
        coupler.prophetic_mixing = params_scalar['prophetic_gamma']

    # Initialize temporal fields (2D: nodes x time_steps)
    # Self-consistent initialization: ψ_f(t=0) = ψ_b(t=0)
    field_forward = torch.randn(args.total_nodes, args.num_time_steps,
                               dtype=torch.complex64, device=device) * 1.0
    field_backward = field_forward.clone()

    # Run temporal evolution with RNN-controlled coupling
    field_forward, field_backward = coupler.apply_coupling(
        field_forward,
        field_backward,
        enable_mixing=True,
        enable_swapping=True,
        enable_anchoring=True
    )

    # Compute convergence metrics
    divergence = torch.mean(torch.abs(field_forward - field_backward)).item()
    fixed_point_mask = torch.abs(field_forward - field_backward) < 1e-5
    fixed_point_pct = (fixed_point_mask.sum().item() / fixed_point_mask.numel()) * 100.0

    # Use forward field as final state
    field_final = field_forward

    # Detect vortices (use spatial for simplicity)
    vortex_dict = detect_temporal_vortices_gpu(
        field_final[:, 0],
        tokamak.positions,
        vortex_threshold=params_scalar['vortex_quality_threshold']
    )

    num_vortices = len(vortex_dict['node_idx'])
    vortex_density = num_vortices / args.total_nodes

    # Vortex annihilation (dark matter generation)
    num_annihilated = 0
    annihilated_nodes = torch.tensor([], dtype=torch.long, device=device)
    field_annihilated = field_final.clone()

    if num_vortices > 0:
        vortex_amplitudes = vortex_dict['amplitude']
        low_quality_mask = vortex_amplitudes < params_scalar['vortex_quality_threshold']
        num_low_quality = low_quality_mask.sum().item()
        num_to_annihilate = int(num_low_quality * (1.0 - params_scalar['preserve_ratio']))

        if num_to_annihilate > 0:
            low_quality_indices = torch.where(low_quality_mask)[0]
            qualities = vortex_amplitudes[low_quality_indices]
            sorted_order = torch.argsort(qualities)
            annihilate_indices = low_quality_indices[sorted_order[:num_to_annihilate]]
            annihilated_nodes = vortex_dict['node_idx'][annihilate_indices]

            # Inject antivortices
            for node_idx in annihilated_nodes:
                field_annihilated[node_idx, :] *= (1.0 - params_scalar['antivortex_strength'])

            num_annihilated = len(annihilated_nodes)

    # DARK MATTER ANALYSIS
    dark_metrics = analyze_dark_matter_residue(field_annihilated, annihilated_nodes, tokamak)

    # Simple reward
    reward = (
        100.0 * min(vortex_density / 0.8, 1.0) +  # Vortex density target
        100.0 * (fixed_point_pct / 100.0)         # Fixed points
    )

    # Metrics
    metrics = {
        'reward': reward,
        'vortex_count': num_vortices,
        'vortex_density': vortex_density,
        'num_annihilated': num_annihilated,
        'fixed_point_pct': fixed_point_pct,
        'divergence': divergence,
        # Dark matter metrics
        'dark_candidates': dark_metrics['num_dark_candidates'],
        'dark_residual_amplitude': dark_metrics['residual_amplitude_mean'],
        'dark_energy_density': dark_metrics['energy_density_dark'],
        'dark_matter_fraction': dark_metrics['dark_matter_fraction'],
        'total_energy': dark_metrics['total_energy']
    }

    return metrics, params_scalar


def main():
    parser = argparse.ArgumentParser(description='Dark Matter Residual Test')

    parser.add_argument('--num-strips', type=int, default=300)
    parser.add_argument('--nodes-per-strip', type=int, default=166)
    parser.add_argument('--num-time-steps', type=int, default=10)
    parser.add_argument('--num-cycles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='~/results/dark_matter_residual_test')

    args = parser.parse_args()
    args.total_nodes = args.num_strips * args.nodes_per_strip

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser() / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Initialize tokamak
    print("Initializing tokamak...")
    tokamak = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        kappa=1.5,
        delta=0.3,
        sparse_threshold=0.3,
        max_neighbors=2000,
        device=device
    )

    coupler = RetrocausalCoupler(
        num_nodes=args.total_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,
        prophetic_mixing=0.3,
        device=device
    )

    print(f"Initialized tokamak: {args.num_strips} strips, {args.total_nodes} nodes\n")

    # Initialize RNN
    print("Initializing RNN controller...")
    rnn = TokamakRNN(
        state_dim=128,
        hidden_dim=2048,
        num_params=11,
        device=device
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print("RNN initialized\n")

    # Training loop with dark matter tracking
    print(f"Starting dark matter residual test: {args.num_cycles} cycles")
    print("=" * 80)
    print()

    start_time = time.time()

    history = {
        'rewards': [],
        'vortex_counts': [],
        'vortex_densities': [],
        'num_annihilated': [],
        'dark_candidates': [],
        'dark_residual_amplitudes': [],
        'dark_energy_densities': [],
        'dark_matter_fractions': [],
        'total_energies': [],
        'fixed_point_pcts': [],
        'divergences': []
    }

    prev_metrics = None

    for cycle in range(args.num_cycles):
        cycle_start = time.time()

        metrics, params_used = train_cycle_with_dark_tracking(
            tokamak, coupler, rnn, optimizer, scaler, args, device, prev_metrics
        )

        prev_metrics = metrics

        # Record history
        history['rewards'].append(metrics['reward'])
        history['vortex_counts'].append(metrics['vortex_count'])
        history['vortex_densities'].append(metrics['vortex_density'])
        history['num_annihilated'].append(metrics['num_annihilated'])
        history['dark_candidates'].append(metrics['dark_candidates'])
        history['dark_residual_amplitudes'].append(metrics['dark_residual_amplitude'])
        history['dark_energy_densities'].append(metrics['dark_energy_density'])
        history['dark_matter_fractions'].append(metrics['dark_matter_fraction'])
        history['total_energies'].append(metrics['total_energy'])
        history['fixed_point_pcts'].append(metrics['fixed_point_pct'])
        history['divergences'].append(metrics['divergence'])

        cycle_time = time.time() - cycle_start

        # Print progress
        if cycle % 10 == 0 or cycle == args.num_cycles - 1:
            print(f"Cycle {cycle}/{args.num_cycles}")
            print(f"  Reward: {metrics['reward']:.2f}")
            print(f"  Vortex density: {metrics['vortex_density']:.2%} ({metrics['vortex_count']} vortices)")
            print(f"  Annihilated: {metrics['num_annihilated']}")
            print(f"  DARK MATTER METRICS:")
            print(f"    Dark candidates: {metrics['dark_candidates']}")
            print(f"    Residual amplitude: {metrics['dark_residual_amplitude']:.6f}")
            print(f"    Dark energy density: {metrics['dark_energy_density']:.6f}")
            print(f"    Dark matter fraction: {metrics['dark_matter_fraction']:.6f}")
            print(f"  Fixed points: {metrics['fixed_point_pct']:.1f}%")
            print(f"  Divergence: {metrics['divergence']:.6f}")
            print(f"  Time: {cycle_time:.3f}s")
            print()

    total_time = time.time() - start_time

    print("=" * 80)
    print(f"Test complete: {args.num_cycles} cycles in {total_time:.2f}s")
    print(f"Average time/cycle: {total_time/args.num_cycles:.3f}s")
    print()

    # Compute summary statistics
    avg_dark_candidates = np.mean(history['dark_candidates'])
    avg_dark_fraction = np.mean(history['dark_matter_fractions'])
    total_annihilated = sum(history['num_annihilated'])

    print("DARK MATTER SUMMARY:")
    print(f"  Total vortices annihilated: {total_annihilated}")
    print(f"  Average dark candidates per cycle: {avg_dark_candidates:.1f}")
    print(f"  Average dark matter fraction: {avg_dark_fraction:.6f} ({avg_dark_fraction*100:.4f}%)")
    print(f"  Dark matter production rate: {total_annihilated / args.num_cycles:.2f} candidates/cycle")
    print()

    # Save results
    results = {
        'config': vars(args),
        'total_time_sec': total_time,
        'history': history,
        'dark_matter_summary': {
            'total_annihilated': total_annihilated,
            'avg_dark_candidates_per_cycle': avg_dark_candidates,
            'avg_dark_matter_fraction': avg_dark_fraction,
            'dark_production_rate': total_annihilated / args.num_cycles
        }
    }

    results_path = output_dir / f"dark_matter_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
