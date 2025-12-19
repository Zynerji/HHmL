#!/usr/bin/env python3
"""
Spatiotemporal Möbius Training - Basic Example
==============================================

Demonstrates (2+1)D spatiotemporal Möbius framework with:
- Spatial Möbius strip (θ dimension)
- Temporal Möbius loop (t dimension)
- Forward/backward evolution with retrocausal coupling
- RNN control of 32 parameters (23 spatial + 9 temporal)
- Temporal fixed point convergence

Goal: Achieve high temporal fixed point percentage (90-100%)
      while maintaining vortex quality on spatial manifold.

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.spatiotemporal import (
    SpatiotemporalMobiusStrip,
    TemporalEvolver,
    RetrocausalCoupler
)
from src.hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Spatiotemporal Möbius Training')

    parser.add_argument('--num-nodes', type=int, default=4000,
                       help='Number of spatial nodes')
    parser.add_argument('--num-time-steps', type=int, default=50,
                       help='Number of temporal steps')
    parser.add_argument('--num-cycles', type=int, default=100,
                       help='Training cycles')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def compute_reward(
    spacetime: SpatiotemporalMobiusStrip,
    temporal_fixed_pct: float,
    divergence: float
) -> float:
    """
    Compute reward for current spatiotemporal state.

    Components:
    1. Temporal fixed points: Reward high percentage of self-consistent time steps
    2. Convergence: Reward low divergence between forward/backward
    3. Field stability: Penalize wild oscillations

    Args:
        spacetime: Current spatiotemporal state
        temporal_fixed_pct: Percentage of temporal fixed points (0-100)
        divergence: Forward-backward divergence

    Returns:
        reward: Scalar reward value
    """
    # Target: 90%+ temporal fixed points
    fixed_point_reward = 100.0 * (temporal_fixed_pct / 100.0)

    # Convergence reward (exponential bonus for low divergence)
    convergence_reward = 50.0 * np.exp(-10 * divergence)

    # Field stability (penalize high field magnitudes)
    field_mag_forward = torch.mean(torch.abs(spacetime.field_forward)).item()
    field_mag_backward = torch.mean(torch.abs(spacetime.field_backward)).item()
    stability_penalty = -10.0 * max(field_mag_forward - 1.0, 0) \
                        -10.0 * max(field_mag_backward - 1.0, 0)

    reward = fixed_point_reward + convergence_reward + stability_penalty

    return reward


def temporal_loop_iteration(
    spacetime: SpatiotemporalMobiusStrip,
    evolver: TemporalEvolver,
    coupler: RetrocausalCoupler,
    params: dict,
    max_iterations: int = 50
) -> tuple:
    """
    Run one temporal loop iteration to convergence.

    Alternates forward/backward evolution with retrocausal coupling
    until temporal fixed points emerge.

    Args:
        spacetime: Spatiotemporal Möbius strip
        evolver: Temporal evolution dynamics
        coupler: Retrocausal coupling
        params: RNN parameters (32 total)
        max_iterations: Maximum convergence iterations

    Returns:
        final_divergence, num_fixed_points, pct_fixed: Convergence metrics
    """
    # Extract temporal parameters from RNN output
    spatial_coupling = float(params['kappa'])
    temporal_coupling = float(params['lambda'])

    for iter_idx in range(max_iterations):
        # Forward sweep: t=0 → t=T
        spacetime.field_forward = evolver.full_forward_sweep(
            spacetime.field_forward,
            spatial_coupling,
            temporal_coupling
        )

        # Backward sweep: t=T → t=0
        spacetime.field_backward = evolver.full_backward_sweep(
            spacetime.field_backward,
            spatial_coupling,
            temporal_coupling
        )

        # Apply retrocausal coupling
        spacetime.field_forward, spacetime.field_backward = coupler.apply_coupling(
            spacetime.field_forward,
            spacetime.field_backward,
            enable_mixing=True,
            enable_swapping=(iter_idx % 5 == 0),  # Swap every 5 iterations
            enable_anchoring=True
        )

        # Apply Möbius boundary conditions
        spacetime.field_forward = spacetime.apply_spatial_mobius_bc(spacetime.field_forward)
        spacetime.field_forward = spacetime.apply_temporal_mobius_bc(spacetime.field_forward)
        spacetime.field_backward = spacetime.apply_spatial_mobius_bc(spacetime.field_backward)
        spacetime.field_backward = spacetime.apply_temporal_mobius_bc(spacetime.field_backward)

        # Check convergence
        divergence = spacetime.compute_divergence()
        num_fixed, pct_fixed = spacetime.compute_temporal_fixed_points()

        if iter_idx % 10 == 0:
            print(f"    Iteration {iter_idx}: divergence={divergence:.6f}, fixed={pct_fixed:.1f}%")

        # Converged if divergence low and high fixed point percentage
        if divergence < 0.01 and pct_fixed > 90.0:
            print(f"    Converged at iteration {iter_idx}")
            break

    return divergence, num_fixed, pct_fixed


def main():
    """Run spatiotemporal training."""
    args = parse_args()

    print("="*80)
    print("SPATIOTEMPORAL MÖBIUS TRAINING")
    print("="*80)
    print()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize spatiotemporal Möbius strip
    print("Initializing (2+1)D Spatiotemporal Möbius...")
    spacetime = SpatiotemporalMobiusStrip(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        temporal_twist=np.pi,  # 180° temporal twist
        device=args.device
    )
    print()

    # Initialize temporal dynamics
    evolver = TemporalEvolver(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        relaxation_factor=0.3,
        device=args.device
    )
    print()

    # Initialize retrocausal coupling
    coupler = RetrocausalCoupler(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,
        prophetic_mixing=0.3,
        device=args.device
    )
    print()

    # Initialize RNN (32 parameters)
    print("Initializing Spatiotemporal RNN (32 parameters)...")
    rnn = SpatiotemporalRNN(
        state_dim=256,
        hidden_dim=4096,
        device=args.device
    )
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    print()

    # Training loop
    print("="*80)
    print("TRAINING LOOP")
    print("="*80)
    print()

    metrics_history = {
        'divergences': [],
        'fixed_point_percentages': [],
        'rewards': [],
        'temporal_parameters': []
    }

    for cycle in range(args.num_cycles):
        print(f"Cycle {cycle+1}/{args.num_cycles}")

        # Self-consistent initialization
        spacetime.initialize_self_consistent(seed=args.seed + cycle)

        # Get current state
        state_tensor = spacetime.get_state_tensor()
        state_input = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)

        # RNN forward pass
        params, _ = rnn(state_input)
        params_rescaled = rnn.rescale_parameters(params)

        # Run temporal loop to convergence
        divergence, num_fixed, pct_fixed = temporal_loop_iteration(
            spacetime, evolver, coupler, params_rescaled, max_iterations=50
        )

        # Compute reward
        reward = compute_reward(spacetime, pct_fixed, divergence)

        # Update RNN
        loss = -reward  # Maximize reward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        metrics_history['divergences'].append(divergence)
        metrics_history['fixed_point_percentages'].append(pct_fixed)
        metrics_history['rewards'].append(reward)
        metrics_history['temporal_parameters'].append({
            'temporal_twist': float(params_rescaled['temporal_twist']),
            'retrocausal_strength': float(params_rescaled['retrocausal_strength']),
            'temporal_relaxation': float(params_rescaled['temporal_relaxation']),
        })

        print(f"  Divergence: {divergence:.6f}")
        print(f"  Fixed points: {num_fixed}/{args.num_time_steps} ({pct_fixed:.1f}%)")
        print(f"  Reward: {reward:.2f}")
        print()

    # Save results
    output_dir = Path('results/spatiotemporal_basic')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'training_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics_history,
            'final': {
                'divergence': divergence,
                'fixed_points': num_fixed,
                'fixed_point_pct': pct_fixed,
                'reward': reward
            }
        }, f, indent=2)

    print(f"Results saved: {results_file}")

    # Save checkpoint
    checkpoint_file = output_dir / f'checkpoint_{timestamp}.pt'
    torch.save({
        'rnn_state_dict': rnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics_history,
        'config': vars(args)
    }, checkpoint_file)

    print(f"Checkpoint saved: {checkpoint_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
