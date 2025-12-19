#!/usr/bin/env python3
"""
Spatiotemporal Möbius Training - Basic Example with Emergent Verification
=========================================================================

Demonstrates (2+1)D spatiotemporal Möbius framework with:
- Spatial Möbius strip (θ dimension)
- Temporal Möbius loop (t dimension)
- Forward/backward evolution with retrocausal coupling
- RNN control of 32 parameters (23 spatial + 9 temporal)
- Temporal fixed point convergence
- LIGO/CMB/Particle verification
- Automated whitepaper generation

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
sys.path.insert(0, str(project_root / 'src'))

from hhml.core.spatiotemporal import (
    SpatiotemporalMobiusStrip,
    TemporalEvolver,
    RetrocausalCoupler
)
from hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN
from hhml.utils.emergent_verifier import EmergentVerifier
from hhml.utils.emergent_whitepaper import EmergentWhitepaperGenerator


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
) -> torch.Tensor:
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
        reward: Torch tensor reward value (for backprop)
    """
    # Target: 90%+ temporal fixed points
    fixed_point_reward = 100.0 * (temporal_fixed_pct / 100.0)

    # Convergence reward (handle NaN divergence)
    if np.isnan(divergence) or divergence > 1e6:
        convergence_reward = -1000.0  # Large penalty for numerical instability
    else:
        convergence_reward = 50.0 * np.exp(-0.1 * min(divergence, 100.0))

    # Field stability (penalize high field magnitudes)
    field_mag_forward = torch.mean(torch.abs(spacetime.field_forward))
    field_mag_backward = torch.mean(torch.abs(spacetime.field_backward))

    # Convert to tensor for gradient tracking
    stability_penalty = -10.0 * torch.clamp(field_mag_forward - 1.0, min=0) \
                        -10.0 * torch.clamp(field_mag_backward - 1.0, min=0)

    reward = torch.tensor(fixed_point_reward + convergence_reward, dtype=torch.float32, device=spacetime.device) + stability_penalty

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
    # Extract temporal parameters from RNN output with numerical stability
    spatial_coupling = float(params['kappa'].detach()) * 0.001  # Scale down for stability
    temporal_coupling = float(params['lambda'].detach()) * 0.001  # Scale down for stability

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

        # Compute reward (detached from graph for logging)
        reward_value = compute_reward(spacetime, pct_fixed, divergence).item()

        # Use RNN value prediction for training (has gradients)
        value_prediction = params['value']

        # Compute temporal fixed point target (what we want to achieve)
        target_value = torch.tensor(pct_fixed / 100.0, dtype=torch.float32, device=spacetime.device)

        # Loss: MSE between predicted value and target (maximize fixed points)
        loss = torch.nn.functional.mse_loss(value_prediction, target_value)

        # Update RNN
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics
        metrics_history['divergences'].append(divergence)
        metrics_history['fixed_point_percentages'].append(pct_fixed)
        metrics_history['rewards'].append(reward_value)
        metrics_history['temporal_parameters'].append({
            'temporal_twist': float(params_rescaled['temporal_twist'].detach()),
            'retrocausal_strength': float(params_rescaled['retrocausal_strength'].detach()),
            'temporal_relaxation': float(params_rescaled['temporal_relaxation'].detach()),
        })

        print(f"  Divergence: {divergence:.6f}")
        print(f"  Fixed points: {num_fixed}/{args.num_time_steps} ({pct_fixed:.1f}%)")
        print(f"  Reward: {reward_value:.2f}")
        print(f"  Loss: {loss.item():.6f}")
        print()

    # Save results
    output_dir = Path('results/spatiotemporal_basic')
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'training_{timestamp}.json'

    final_results = {
        'config': vars(args),
        'metrics': metrics_history,
        'final': {
            'divergence': divergence,
            'fixed_points': num_fixed,
            'fixed_point_pct': pct_fixed,
            'reward': reward_value
        }
    }

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"Results saved: {results_file}")

    # Save checkpoint
    checkpoint_file = output_dir / f'checkpoint_{timestamp}.pt'
    torch.save({
        'rnn_state_dict': rnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics_history,
        'config': vars(args),
        'final_field_forward': spacetime.field_forward.cpu(),
        'final_field_backward': spacetime.field_backward.cpu()
    }, checkpoint_file)

    print(f"Checkpoint saved: {checkpoint_file}")
    print()

    # =========================================================================
    # EMERGENT PHENOMENON VERIFICATION (MANDATORY)
    # =========================================================================

    print("="*80)
    print("EMERGENT PHENOMENON VERIFICATION")
    print("="*80)
    print()

    # Only verify if results meet quality threshold
    # Threshold: At least 50% temporal fixed points achieved
    if pct_fixed >= 50.0:
        print(f"Results meet threshold for emergent verification (fixed points: {pct_fixed:.1f}%)")
        print()

        # Prepare discovery data
        discovery_data = {
            'phenomenon_name': 'Spatiotemporal Fixed Point Convergence',
            'training_run': str(Path(__file__)),
            'discovery_cycle': args.num_cycles - 1,  # Final cycle
            'timestamp': timestamp,
            'random_seed': args.seed,
            'hardware': {
                'device': args.device,
                'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else 'CPU',
                'vram_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if args.device == 'cuda' else None,
                'hardware_tier': 'GPU' if args.device == 'cuda' else 'CPU',
                'auto_scaled': False
            },
            'system_size': {
                'nodes': args.num_nodes,
                'time_steps': args.num_time_steps,
                'total_dof': args.num_nodes * args.num_time_steps
            },
            'parameters': {
                'temporal_twist': float(params_rescaled['temporal_twist'].detach()),
                'retrocausal_strength': float(params_rescaled['retrocausal_strength'].detach()),
                'temporal_relaxation': float(params_rescaled['temporal_relaxation'].detach()),
                'num_time_steps': float(params_rescaled['num_time_steps'].detach()),
                'prophetic_coupling': float(params_rescaled['prophetic_coupling'].detach())
            },
            'key_metrics': {
                'final_divergence': divergence,
                'final_fixed_points': num_fixed,
                'final_fixed_point_pct': pct_fixed,
                'final_reward': reward_value,
                'peak_fixed_point_pct': max(metrics_history['fixed_point_percentages']),
                'mean_divergence': np.mean(metrics_history['divergences']),
                'min_divergence': min(metrics_history['divergences'])
            },
            'correlations': {
                # Parameter-observable correlations (can be computed post-hoc)
                'temporal_twist_vs_fixed_points': {
                    'r': 0.0,  # Placeholder - compute from metrics_history if needed
                    'p': 1.0
                }
            },
            'checkpoint': str(checkpoint_file),
            'configuration': vars(args)
        }

        # Initialize verifier
        print("Initializing EmergentVerifier...")
        verifier = EmergentVerifier(data_dir="data")
        print()

        # Combine forward and backward fields for verification
        # Shape: (num_time_steps, num_nodes) - temporal evolution
        combined_field = torch.stack([
            spacetime.field_forward.T,  # Transpose to (time, nodes)
            spacetime.field_backward.T
        ], dim=0).mean(dim=0)  # Average forward/backward

        # Run verification
        print("Running automated verification against real-world physics...")
        print("  - LIGO: Gravitational wave comparison")
        print("  - CMB: Cosmic microwave background comparison")
        print("  - Particles: Standard model mass comparison")
        print()

        verification_results = verifier.verify_phenomenon(
            field_tensor=combined_field,
            phenomenon_type='auto',  # Auto-detect from field properties
            save_results=True,
            output_dir=str(output_dir / "verification")
        )

        print(f"Verification complete:")
        print(f"  Novelty score: {verification_results['novelty_score']:.3f}")
        print(f"  Is novel: {verification_results['is_novel']}")
        print(f"  Interpretation: {verification_results['interpretation']}")
        print()

        # Print recommendations
        if verification_results.get('recommendations'):
            print("Recommendations:")
            for rec in verification_results['recommendations']:
                print(f"  {rec}")
            print()

        # Generate whitepaper
        print("Generating comprehensive whitepaper...")
        generator = EmergentWhitepaperGenerator()

        try:
            whitepaper_path = generator.generate(
                phenomenon_name=discovery_data['phenomenon_name'],
                discovery_data=discovery_data,
                verification_results=verification_results,
                output_dir=str(output_dir / "whitepapers" / "EMERGENTS"),
                compile_pdf=True
            )

            print(f"Whitepaper generated: {whitepaper_path}")
            print()

            if verification_results['is_novel']:
                print("="*80)
                print("NOVEL EMERGENT PHENOMENON DETECTED")
                print("="*80)
                print()
                print("ACTION REQUIRED:")
                print("1. Review whitepaper for scientific accuracy")
                print("2. Update EMERGENTS.md with full discovery template")
                print("3. Update README.md if this represents new capability")
                print("4. Commit results with detailed message")
                print()
            else:
                print("="*80)
                print("RESULTS DOCUMENTED")
                print("="*80)
                print()
                print("Phenomenon documented but does not meet novelty threshold.")
                print("Review whitepaper for detailed analysis.")
                print()

        except Exception as e:
            print(f"Warning: Whitepaper generation failed: {e}")
            print("Verification results saved, but PDF not generated.")
            print()

        # Add verification to results
        final_results['emergent_verification'] = {
            'novelty_score': verification_results['novelty_score'],
            'is_novel': verification_results['is_novel'],
            'interpretation': verification_results['interpretation'],
            'verification_file': verification_results.get('output_file', 'N/A')
        }

        # Re-save results with verification
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)

    else:
        print(f"Results do not meet threshold for verification (fixed points: {pct_fixed:.1f}% < 50%)")
        print("Skipping emergent verification.")
        print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
