#!/usr/bin/env python3
"""
Spatiotemporal Möbius Training - H200 Optimized
===============================================

H200-optimized version for large-scale spatiotemporal training:
- 4K-20K nodes (optimized for H200 memory)
- 50-100 time steps
- 100-1000 cycles
- Mixed precision training (FP16)
- Gradient accumulation
- Optimized batch processing

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


def parse_args():
    """Parse command-line arguments."""
    import argparse
    parser = argparse.ArgumentParser(description='Spatiotemporal Möbius Training (H200)')

    # Scale parameters
    parser.add_argument('--num-nodes', type=int, default=4000,
                       help='Number of spatial nodes (default: 4000, H200: up to 20000)')
    parser.add_argument('--num-time-steps', type=int, default=50,
                       help='Number of temporal steps (default: 50, H200: up to 100)')
    parser.add_argument('--num-cycles', type=int, default=100,
                       help='Training cycles (default: 100, H200: up to 1000)')

    # Optimization parameters
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                       help='Gradient accumulation steps (H200: 4-8)')
    parser.add_argument('--use-amp', action='store_true',
                       help='Use automatic mixed precision (FP16)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    # Checkpointing
    parser.add_argument('--checkpoint-every', type=int, default=50,
                       help='Save checkpoint every N cycles')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')

    return parser.parse_args()


def compute_reward(
    spacetime: SpatiotemporalMobiusStrip,
    temporal_fixed_pct: float,
    divergence: float
) -> torch.Tensor:
    """Compute reward with gradient tracking."""
    fixed_point_reward = 100.0 * (temporal_fixed_pct / 100.0)

    if np.isnan(divergence) or divergence > 1e6:
        convergence_reward = -1000.0
    else:
        convergence_reward = 50.0 * np.exp(-0.1 * min(divergence, 100.0))

    field_mag_forward = torch.mean(torch.abs(spacetime.field_forward))
    field_mag_backward = torch.mean(torch.abs(spacetime.field_backward))

    stability_penalty = -10.0 * torch.clamp(field_mag_forward - 1.0, min=0) \
                        -10.0 * torch.clamp(field_mag_backward - 1.0, min=0)

    reward = torch.tensor(fixed_point_reward + convergence_reward, dtype=torch.float32, device=spacetime.device) + stability_penalty

    return reward


def temporal_loop_iteration(
    spacetime: SpatiotemporalMobiusStrip,
    evolver: TemporalEvolver,
    coupler: RetrocausalCoupler,
    params: dict,
    max_iterations: int = 50,
    verbose: bool = False
) -> tuple:
    """Run temporal loop iteration to convergence."""
    spatial_coupling = float(params['kappa'].detach()) * 0.001
    temporal_coupling = float(params['lambda'].detach()) * 0.001

    for iter_idx in range(max_iterations):
        # Forward sweep
        spacetime.field_forward = evolver.full_forward_sweep(
            spacetime.field_forward,
            spatial_coupling,
            temporal_coupling
        )

        # Backward sweep
        spacetime.field_backward = evolver.full_backward_sweep(
            spacetime.field_backward,
            spatial_coupling,
            temporal_coupling
        )

        # Retrocausal coupling
        spacetime.field_forward, spacetime.field_backward = coupler.apply_coupling(
            spacetime.field_forward,
            spacetime.field_backward,
            enable_mixing=True,
            enable_swapping=(iter_idx % 5 == 0),
            enable_anchoring=True
        )

        # Möbius boundary conditions
        spacetime.field_forward = spacetime.apply_spatial_mobius_bc(spacetime.field_forward)
        spacetime.field_forward = spacetime.apply_temporal_mobius_bc(spacetime.field_forward)
        spacetime.field_backward = spacetime.apply_spatial_mobius_bc(spacetime.field_backward)
        spacetime.field_backward = spacetime.apply_temporal_mobius_bc(spacetime.field_backward)

        # Check convergence
        divergence = spacetime.compute_divergence()
        num_fixed, pct_fixed = spacetime.compute_temporal_fixed_points()

        if verbose and iter_idx % 10 == 0:
            print(f"    Iteration {iter_idx}: divergence={divergence:.6f}, fixed={pct_fixed:.1f}%")

        if divergence < 0.01 and pct_fixed > 90.0:
            if verbose:
                print(f"    Converged at iteration {iter_idx}")
            break

    return divergence, num_fixed, pct_fixed


def main():
    """Run H200-optimized spatiotemporal training."""
    args = parse_args()

    print("="*80)
    print("SPATIOTEMPORAL MOBIUS TRAINING (H200 OPTIMIZED)")
    print("="*80)
    print()

    # Check GPU
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # Initialize components
    print("Initializing (2+1)D Spatiotemporal Möbius...")
    spacetime = SpatiotemporalMobiusStrip(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        temporal_twist=np.pi,
        device=args.device
    )
    print()

    evolver = TemporalEvolver(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        relaxation_factor=0.3,
        device=args.device
    )
    print()

    coupler = RetrocausalCoupler(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,
        prophetic_mixing=0.3,
        device=args.device
    )
    print()

    print("Initializing Spatiotemporal RNN (32 parameters)...")
    rnn = SpatiotemporalRNN(
        state_dim=256,
        hidden_dim=4096,
        device=args.device
    )
    optimizer = torch.optim.AdamW(rnn.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_cycles)
    print()

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and args.device == 'cuda' else None
    if scaler:
        print("Mixed precision training enabled (FP16)")
        print()

    # Resume from checkpoint
    start_cycle = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        rnn.load_state_dict(checkpoint['rnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_cycle = checkpoint.get('cycle', 0) + 1
        print(f"Resuming from cycle {start_cycle}")
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
        'losses': [],
        'temporal_parameters': []
    }

    start_time = time.time()

    for cycle in range(start_cycle, args.num_cycles):
        cycle_start = time.time()

        print(f"Cycle {cycle+1}/{args.num_cycles}")

        # Self-consistent initialization
        spacetime.initialize_self_consistent(seed=args.seed + cycle)

        # Forward pass with optional AMP
        state_tensor = spacetime.get_state_tensor()
        state_input = state_tensor.unsqueeze(0).unsqueeze(0)

        with torch.cuda.amp.autocast() if scaler else torch.nullcontext():
            params, _ = rnn(state_input)
            params_rescaled = rnn.rescale_parameters(params)

        # Temporal loop iteration
        divergence, num_fixed, pct_fixed = temporal_loop_iteration(
            spacetime, evolver, coupler, params_rescaled,
            max_iterations=50,
            verbose=(cycle % 10 == 0)
        )

        # Compute reward and loss
        reward_value = compute_reward(spacetime, pct_fixed, divergence).item()
        value_prediction = params['value']
        target_value = torch.tensor(pct_fixed / 100.0, dtype=torch.float32, device=spacetime.device)
        loss = torch.nn.functional.mse_loss(value_prediction, target_value)

        # Backward pass with gradient accumulation
        if scaler:
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            if (cycle + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            (loss / args.gradient_accumulation_steps).backward()
            if (cycle + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()

        # Log metrics
        metrics_history['divergences'].append(divergence)
        metrics_history['fixed_point_percentages'].append(pct_fixed)
        metrics_history['rewards'].append(reward_value)
        metrics_history['losses'].append(loss.item())
        metrics_history['temporal_parameters'].append({
            'temporal_twist': float(params_rescaled['temporal_twist'].detach()),
            'retrocausal_strength': float(params_rescaled['retrocausal_strength'].detach()),
            'temporal_relaxation': float(params_rescaled['temporal_relaxation'].detach()),
        })

        cycle_time = time.time() - cycle_start

        print(f"  Divergence: {divergence:.6f}")
        print(f"  Fixed points: {num_fixed}/{args.num_time_steps} ({pct_fixed:.1f}%)")
        print(f"  Reward: {reward_value:.2f}")
        print(f"  Loss: {loss.item():.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        print(f"  Time: {cycle_time:.1f}s")
        print()

        # Checkpoint
        if (cycle + 1) % args.checkpoint_every == 0:
            checkpoint_dir = Path('checkpoints/spatiotemporal_h200')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / f'checkpoint_cycle_{cycle+1}.pt'
            torch.save({
                'cycle': cycle,
                'rnn_state_dict': rnn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'metrics': metrics_history,
                'config': vars(args)
            }, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
            print()

    elapsed_time = time.time() - start_time

    # Final summary
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {elapsed_time/60:.1f} minutes")
    print(f"Average cycle time: {elapsed_time/args.num_cycles:.1f}s")
    print(f"Final divergence: {divergence:.6f}")
    print(f"Final fixed points: {pct_fixed:.1f}%")
    print()

    # Save results
    output_dir = Path('results/spatiotemporal_h200')
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
                'reward': reward_value
            },
            'timing': {
                'total_seconds': elapsed_time,
                'average_cycle_seconds': elapsed_time / args.num_cycles
            }
        }, f, indent=2)

    print(f"Results saved: {results_file}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
