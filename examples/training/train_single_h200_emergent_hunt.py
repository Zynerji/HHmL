#!/usr/bin/env python3
"""
Single H200 Emergent Phenomenon Hunt - 30 Minute Optimized
===========================================================

Intensive training optimized for emergent discovery on single H200:
- Maximum scale for single GPU (140GB VRAM)
- 30 minute runtime target
- 50K nodes, 100 time steps
- ~800-1000 cycles in 30 min
- Mixed precision (FP16) optimization
- Automated emergent verification
- Whitepaper generation
- Auto-commit and push results

Hardware Requirements:
- 1x NVIDIA H200 (140GB VRAM)
- 32+ CPU cores recommended
- 256GB+ system RAM recommended

Strategy: Maximize single-GPU utilization with largest possible scale
          to discover emergent phenomena from complex spatiotemporal dynamics.

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
import subprocess
import argparse

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

from hhml.core.spatiotemporal import (
    SpatiotemporalMobiusStrip,
    TemporalEvolver,
    RetrocausalCoupler,
    TemporalVortexController
)
from hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN
from hhml.utils.emergent_verifier import EmergentVerifier
from hhml.utils.emergent_whitepaper import EmergentWhitepaperGenerator


def compute_reward(
    spacetime,
    vortex_controller,
    temporal_fixed_pct,
    divergence,
    vortex_stats
):
    """Compute reward for temporal vortex control."""
    # Fixed point reward (target: 90%+)
    fixed_point_reward = 100.0 * (temporal_fixed_pct / 100.0)

    # Convergence reward
    if np.isnan(divergence) or divergence > 1e6:
        convergence_reward = -1000.0
    else:
        convergence_reward = 50.0 * np.exp(-0.1 * min(divergence, 100.0))

    # Vortex tube density reward (optimal: 10-30%)
    tube_density = vortex_stats['vortex_tube_density']
    if 0.1 <= tube_density <= 0.3:
        tube_reward = 50.0
    else:
        deviation = abs(tube_density - 0.2)
        tube_reward = 50.0 * np.exp(-10 * deviation)

    # Topological charge conservation (lower = better)
    temporal_charge = abs(vortex_stats['total_topological_charge_temporal'])
    tube_charge = abs(vortex_stats['total_topological_charge_tubes'])
    total_charge = temporal_charge + tube_charge
    charge_reward = 30.0 * np.exp(-0.1 * total_charge)

    # Field stability penalty
    field_mag_forward = torch.mean(torch.abs(spacetime.field_forward))
    field_mag_backward = torch.mean(torch.abs(spacetime.field_backward))

    stability_penalty = -10.0 * torch.clamp(field_mag_forward - 1.0, min=0) \
                        -10.0 * torch.clamp(field_mag_backward - 1.0, min=0)

    reward = torch.tensor(
        fixed_point_reward + convergence_reward + tube_reward + charge_reward,
        dtype=torch.float32,
        device=spacetime.device
    ) + stability_penalty

    return reward


def temporal_loop_with_vortex_control(
    spacetime,
    evolver,
    coupler,
    vortex_controller,
    params,
    max_iterations=20,
    threshold=1e-3,
    debug=False
):
    """Run temporal loop with vortex control and return convergence metrics."""
    
    # DIAGNOSTIC: Check initial field state
    if debug:
        print(f"  [DEBUG] Initial field_forward: min={spacetime.field_forward.abs().min().item():.3e}, "
              f"max={spacetime.field_forward.abs().max().item():.3e}, "
              f"mean={spacetime.field_forward.abs().mean().item():.3e}, "
              f"has_nan={torch.isnan(spacetime.field_forward).any().item()}")
        print(f"  [DEBUG] Initial field_backward: min={spacetime.field_backward.abs().min().item():.3e}, "
              f"max={spacetime.field_backward.abs().max().item():.3e}, "
              f"mean={spacetime.field_backward.abs().mean().item():.3e}, "
              f"has_nan={torch.isnan(spacetime.field_backward).any().item()}")
        print(f"  [DEBUG] Params: kappa={params['kappa']:.3f}, lambda={params['lambda']:.3f}")
    
    # Probabilistic temporal vortex injection
    injection_rate = params['temporal_vortex_injection_rate']
    if torch.rand(1).item() < injection_rate:
        if debug:
            print(f"  [DEBUG] Injecting temporal vortex (rate={injection_rate:.3f})")
            print(f"  [DEBUG] Before vortex injection: has_nan={torch.isnan(spacetime.field_forward).any().item()}")
        
        t_idx = torch.randint(0, spacetime.num_time_steps, (1,)).item()
        winding = int(params['temporal_vortex_winding'])
        core_size = params['temporal_vortex_core_size']
        spacetime.field_forward = vortex_controller.inject_temporal_vortex(
            spacetime.field_forward, t_idx, winding, core_size
        )
        
        if debug:
            print(f"  [DEBUG] After vortex injection: has_nan={torch.isnan(spacetime.field_forward).any().item()}, "
                  f"max={spacetime.field_forward.abs().max().item():.3e}")

    # Probabilistic spatiotemporal vortex tube formation
    tube_prob = params['vortex_tube_probability']
    if torch.rand(1).item() < tube_prob:
        num_points = torch.randint(5, 15, (1,)).item()
        trajectory = []
        for _ in range(num_points):
            theta_idx = torch.randint(0, spacetime.num_nodes, (1,)).item()
            t_idx = torch.randint(0, spacetime.num_time_steps, (1,)).item()
            trajectory.append((theta_idx, t_idx))

        tube_winding = int(params['tube_winding_number'])
        tube_core_size = params['tube_core_size']
        spacetime.field_forward = vortex_controller.inject_spatiotemporal_vortex_tube(
            spacetime.field_forward, trajectory, tube_winding, tube_core_size
        )

    # Temporal loop iteration
    for iteration in range(max_iterations):
        prev_forward = spacetime.field_forward.clone()

        # Forward evolution
        if debug:
            print(f"  [DEBUG] Before forward_sweep: field_forward has_nan={torch.isnan(spacetime.field_forward).any().item()}")
        
        spacetime.field_forward = evolver.full_forward_sweep(
            spacetime.field_forward,
            spatial_coupling=float(params['kappa']),
            temporal_coupling=float(params['lambda'])
        )
        
        if debug:
            print(f"  [DEBUG] After forward_sweep: has_nan={torch.isnan(spacetime.field_forward).any().item()}, "
                  f"max={spacetime.field_forward.abs().max().item():.3e}")

        # Backward evolution
        if debug:
            print(f"  [DEBUG] Before backward_sweep: field_backward has_nan={torch.isnan(spacetime.field_backward).any().item()}")
        
        spacetime.field_backward = evolver.full_backward_sweep(
            spacetime.field_backward,
            spatial_coupling=float(params['kappa']),
            temporal_coupling=float(params['lambda'])
        )
        
        if debug:
            print(f"  [DEBUG] After backward_sweep: has_nan={torch.isnan(spacetime.field_backward).any().item()}, "
                  f"max={spacetime.field_backward.abs().max().item():.3e}")

        # Retrocausal coupling
        if debug:
            print(f"  [DEBUG] Before coupling: forward has_nan={torch.isnan(spacetime.field_forward).any().item()}, "
                  f"backward has_nan={torch.isnan(spacetime.field_backward).any().item()}")
        
        spacetime.field_forward, spacetime.field_backward = coupler.apply_coupling(
            spacetime.field_forward,
            spacetime.field_backward,
            enable_mixing=True,
            enable_swapping=True,
            enable_anchoring=True
        )
        
        if debug:
            print(f"  [DEBUG] After coupling: forward has_nan={torch.isnan(spacetime.field_forward).any().item()}, "
                  f"backward has_nan={torch.isnan(spacetime.field_backward).any().item()}, "
                  f"forward_max={spacetime.field_forward.abs().max().item():.3e}, "
                  f"backward_max={spacetime.field_backward.abs().max().item():.3e}")

        # Apply temporal Mobius boundary conditions
        if debug:
            print(f"  [DEBUG] Before Mobius BC: forward has_nan={torch.isnan(spacetime.field_forward).any().item()}")
        
        spacetime.field_forward = spacetime.apply_temporal_mobius_bc(
            spacetime.field_forward
        )
        spacetime.field_backward = spacetime.apply_temporal_mobius_bc(
            spacetime.field_backward
        )
        
        if debug:
            print(f"  [DEBUG] After Mobius BC: forward has_nan={torch.isnan(spacetime.field_forward).any().item()}, "
                  f"backward has_nan={torch.isnan(spacetime.field_backward).any().item()}, "
                  f"forward_max={spacetime.field_forward.abs().max().item():.3e}")

        # Check convergence
        divergence = torch.mean(torch.abs(spacetime.field_forward - prev_forward)).item()
        
        if debug:
            print(f"  [DEBUG] Iteration {iteration} divergence: {divergence:.3e}")
            has_nan_fwd = torch.isnan(spacetime.field_forward).any().item()
            has_nan_bwd = torch.isnan(spacetime.field_backward).any().item()
            print(f"  [DEBUG] After iteration: forward has_nan={has_nan_fwd}, backward has_nan={has_nan_bwd}")
            if has_nan_fwd or has_nan_bwd:
                print(f"  [DEBUG] *** NaN DETECTED at iteration {iteration}! Breaking loop. ***")
                break
        
        if divergence < threshold:
            if debug:
                print(f"  [DEBUG] Converged at iteration {iteration}")
            break

    # Probabilistic temporal vortex annihilation
    annihilation_rate = params['temporal_vortex_annihilation_rate']
    if torch.rand(1).item() < annihilation_rate:
        temporal_vortices, _ = vortex_controller.detect_temporal_vortices(
            spacetime.field_forward
        )
        if len(temporal_vortices) > 0:
            t_idx_to_remove = temporal_vortices[torch.randint(0, len(temporal_vortices), (1,)).item()]
            spacetime.field_forward = vortex_controller.annihilate_temporal_vortex(
                spacetime.field_forward, t_idx_to_remove
            )

    # Measure temporal fixed points
    fixed_agreement = torch.abs(spacetime.field_forward - spacetime.field_backward) < threshold
    num_fixed = torch.sum(fixed_agreement).item()
    total_points = spacetime.num_nodes * spacetime.num_time_steps
    pct_fixed = 100.0 * num_fixed / total_points

    # Get vortex statistics
    vortex_stats = vortex_controller.get_vortex_statistics(spacetime.field_forward)

    return divergence, num_fixed, pct_fixed, vortex_stats


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Single H200 Emergent Hunt')
    parser.add_argument('--num-nodes', type=int, default=50000,
                       help='Number of spatial nodes (default: 50K for H200)')
    parser.add_argument('--num-time-steps', type=int, default=100,
                       help='Number of time steps (default: 100)')
    parser.add_argument('--num-cycles', type=int, default=800,
                       help='Number of training cycles (default: 800 -> ~30 min)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='scratch/results/h200_emergent',
                       help='Output directory')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("SINGLE H200 EMERGENT PHENOMENON HUNT")
    print("Full Dimensional Control: 39 RNN Parameters")
    print("="*80)
    print()
    print(f"Device: {device}")
    print(f"Nodes: {args.num_nodes:,}")
    print(f"Time steps: {args.num_time_steps}")
    print(f"Target cycles: {args.num_cycles}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {output_dir}")
    print()

    # Initialize components
    print("Initializing spatiotemporal system...")
    spacetime = SpatiotemporalMobiusStrip(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        temporal_twist=np.pi,  # Full Mobius twist
        device=device
    )

    evolver = TemporalEvolver(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        device=device
    )
    coupler = RetrocausalCoupler(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        device=device
    )
    vortex_controller = TemporalVortexController(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        device=device
    )

    print("Initializing RNN (39 parameters)...")
    rnn = SpatiotemporalRNN(state_dim=256, hidden_dim=4096, device=device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

    # Use mixed precision for H200 optimization
    scaler = torch.cuda.amp.GradScaler() if device == 'cuda' else None

    print()
    print("Starting training...")
    print()

    # Training metrics
    metrics = {
        'rewards': [],
        'fixed_point_percentages': [],
        'divergences': [],
        'temporal_vortex_counts': [],
        'vortex_tube_counts': [],
        'vortex_tube_densities': [],
        'total_charges_temporal': [],
        'total_charges_tubes': []
    }

    param_history = []
    best_reward = -float('inf')
    best_checkpoint = None

    # Initialize with self-consistent boundary conditions (ONCE before training)
    print()
    print("Initializing fields with self-consistent boundary conditions...")
    spacetime.initialize_self_consistent()
    print()

    start_time = time.time()

    for cycle in range(args.num_cycles):

        # Prepare state input for RNN
        # field_forward shape: (num_nodes, num_time_steps)
        # Take mean over nodes to get (num_time_steps,) then add batch/feature dims
        field_features = torch.cat([
            spacetime.field_forward.real.mean(dim=0, keepdim=False),  # (num_time_steps,)
            spacetime.field_forward.imag.mean(dim=0, keepdim=False)   # (num_time_steps,)
        ], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2*num_time_steps, 1)

        # Reshape to (1, num_time_steps, 2)
        field_features = field_features.view(1, args.num_time_steps, 2)

        # Pad to state_dim=256
        padding = torch.zeros(1, args.num_time_steps, 254, device=device)
        state_input = torch.cat([field_features, padding], dim=2)  # (1, num_time_steps, 256)

        # RNN forward pass (with mixed precision if available)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                params, hidden = rnn(state_input)
                value = params['value']  # Extract value from params dict
                params_rescaled = rnn.rescale_parameters(params)
        else:
            params, hidden = rnn(state_input)
            value = params['value']  # Extract value from params dict
            params_rescaled = rnn.rescale_parameters(params)

        # Run temporal loop with vortex control (detach params - no grad through dynamics)
        with torch.no_grad():
            divergence, num_fixed, pct_fixed, vortex_stats = temporal_loop_with_vortex_control(
                spacetime, evolver, coupler, vortex_controller,
                {k: v.detach() if torch.is_tensor(v) else v for k, v in params_rescaled.items()},
                max_iterations=20,
                threshold=1e-3,
                debug=(cycle == 0)  # Enable debug logging for first cycle
            )

            # Check for NaN in fields after evolution
            if torch.isnan(spacetime.field_forward).any():
                print(f"WARNING: NaN detected in field_forward after temporal loop (cycle {cycle})")
                # Replace NaN with zeros to prevent crash
                spacetime.field_forward = torch.nan_to_num(spacetime.field_forward, nan=0.0)
            if torch.isnan(spacetime.field_backward).any():
                print(f"WARNING: NaN detected in field_backward after temporal loop (cycle {cycle})")
                spacetime.field_backward = torch.nan_to_num(spacetime.field_backward, nan=0.0)

        # Compute reward (detach - policy gradient doesn't need environment grads)
        with torch.no_grad():
            reward_value = compute_reward(
                spacetime, vortex_controller,
                pct_fixed, divergence, vortex_stats
            )

        # Create differentiable loss for policy gradient
        # Loss = -value_prediction * reward (actor-critic style)
        loss = -value * reward_value.item()  # Maximize value prediction accuracy

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()

            # Gradient clipping to prevent NaN in long training runs
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            # Check for NaN in gradients
            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in rnn.parameters()
            )

            if has_nan_grad:
                print(f"WARNING: NaN detected in RNN gradients (cycle {cycle}), skipping update")
                optimizer.zero_grad()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()

            # Gradient clipping to prevent NaN in long training runs
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            # Check for NaN in gradients
            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in rnn.parameters()
            )

            if has_nan_grad:
                print(f"WARNING: NaN detected in RNN gradients (cycle {cycle}), skipping update")
                optimizer.zero_grad()
            else:
                optimizer.step()

        # Track metrics
        metrics['rewards'].append(float(reward_value.item()))
        metrics['fixed_point_percentages'].append(float(pct_fixed))
        metrics['divergences'].append(float(divergence))
        metrics['temporal_vortex_counts'].append(int(vortex_stats['temporal_vortex_count']))
        metrics['vortex_tube_counts'].append(int(vortex_stats['vortex_tube_count']))
        metrics['vortex_tube_densities'].append(float(vortex_stats['vortex_tube_density']))
        metrics['total_charges_temporal'].append(float(vortex_stats['total_topological_charge_temporal']))
        metrics['total_charges_tubes'].append(float(vortex_stats['total_topological_charge_tubes']))

        param_history.append({k: float(v.item() if torch.is_tensor(v) else v)
                             for k, v in params_rescaled.items() if k != 'value'})

        # Track best checkpoint
        if reward_value.item() > best_reward:
            best_reward = reward_value.item()
            best_checkpoint = {
                'cycle': cycle,
                'reward': float(reward_value.item()),
                'fixed_pct': float(pct_fixed),
                'params': params_rescaled,
                'field_forward': spacetime.field_forward.cpu().clone(),
                'field_backward': spacetime.field_backward.cpu().clone(),
                'rnn_state_dict': rnn.state_dict()
            }

        # Progress logging every 50 cycles
        if (cycle + 1) % 50 == 0 or cycle == 0:
            elapsed = time.time() - start_time
            cycles_per_sec = (cycle + 1) / elapsed
            eta_sec = (args.num_cycles - cycle - 1) / cycles_per_sec if cycles_per_sec > 0 else 0
            eta_min = eta_sec / 60

            print(f"Cycle {cycle+1}/{args.num_cycles}")
            print(f"  Reward: {reward_value.item():.2f}")
            print(f"  Fixed points: {num_fixed}/{args.num_nodes * args.num_time_steps} ({pct_fixed:.1f}%)")
            print(f"  Divergence: {divergence:.6f}")
            print(f"  Temporal vortices: {vortex_stats['temporal_vortex_count']} ({vortex_stats['temporal_vortex_density']*100:.1f}%)")
            print(f"  Vortex tubes: {vortex_stats['vortex_tube_count']} (density: {vortex_stats['vortex_tube_density']*100:.1f}%)")
            print(f"  Topological charge: {vortex_stats['total_topological_charge_temporal']:.2f} (temporal) + {vortex_stats['total_topological_charge_tubes']:.2f} (tubes)")
            print(f"  Speed: {cycles_per_sec:.3f} cycles/sec, ETA: {eta_min:.1f} min")
            print()

    total_time = time.time() - start_time

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {total_time/60:.1f} minutes")
    if best_checkpoint is not None:
        print(f"Best reward: {best_reward:.2f} (cycle {best_checkpoint['cycle']})")
        print(f"Best fixed points: {best_checkpoint['fixed_pct']:.1f}%")
    else:
        print(f"Best reward: {best_reward:.2f} (no valid checkpoint saved)")
    print()

    # Save results
    print("Saving results...")
    
    # Extract JSON-serializable parts of best_checkpoint (exclude tensors)
    best_checkpoint_json = None
    if best_checkpoint is not None:
        best_checkpoint_json = {
            'cycle': best_checkpoint['cycle'],
            'reward': best_checkpoint['reward'],
            'fixed_pct': best_checkpoint['fixed_pct']
            # Tensors (field_forward, field_backward, params, rnn_state_dict) saved to .pt file only
        }
    
    results = {
        'config': vars(args),
        'device': device,
        'total_time_sec': total_time,
        'metrics': metrics,
        'param_history': param_history,
        'best_checkpoint': best_checkpoint_json
    }

    results_file = output_dir / f'training_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save PyTorch checkpoint
    checkpoint_file = output_dir / f'best_checkpoint_{timestamp}.pt'
    torch.save(best_checkpoint, checkpoint_file)

    print(f"Results: {results_file}")
    print(f"Checkpoint: {checkpoint_file}")
    print()

    # Emergent verification (if sufficient quality)
    if best_checkpoint is not None and best_checkpoint['fixed_pct'] >= 50.0:
        print("="*80)
        print("EMERGENT PHENOMENON VERIFICATION")
        print("="*80)
        print()
        print("Results meet threshold for verification...")

        discovery_data = {
            'phenomenon_name': 'Single H200 Temporal Vortex Control',
            'training_run': str(Path(__file__)),
            'timestamp': timestamp,
            'random_seed': args.seed,
            'hardware': {
                'device': device,
                'num_nodes': args.num_nodes,
                'num_time_steps': args.num_time_steps,
                'num_cycles': args.num_cycles
            },
            'parameters': param_history[-1],  # Final parameters
            'key_metrics': {
                'best_reward': float(best_reward),
                'best_fixed_pct': float(best_checkpoint['fixed_pct']),
                'max_fixed_pct': float(max(metrics['fixed_point_percentages'])),
                'final_vortex_tube_density': float(metrics['vortex_tube_densities'][-1])
            },
            'checkpoint': str(checkpoint_file)
        }

        # Run verification
        verifier = EmergentVerifier(data_dir="data")
        verification_results = verifier.verify_phenomenon(
            field_tensor=best_checkpoint['field_forward'],
            phenomenon_type='auto',
            save_results=True,
            output_dir=str(output_dir / "verification")
        )

        print(f"Novelty score: {verification_results['novelty_score']:.3f}")
        print(f"Is novel: {verification_results['is_novel']}")
        print()

        # Generate whitepaper
        print("Generating whitepaper...")
        generator = EmergentWhitepaperGenerator()
        whitepaper_path = generator.generate(
            phenomenon_name="Single H200 Temporal Vortex Hunt Results",
            discovery_data=discovery_data,
            verification_results=verification_results,
            output_dir=str(output_dir / "whitepapers")
        )

        print(f"Whitepaper: {whitepaper_path}")
        print()

        if verification_results['is_novel']:
            print("[NOVEL] Update EMERGENTS.md with this discovery")
        else:
            print("[DOCUMENTED] Does not meet novelty threshold")
    else:
        if best_checkpoint is not None:
            print(f"Fixed point percentage ({best_checkpoint['fixed_pct']:.1f}%) below threshold (50%), skipping verification")
        else:
            print("No valid checkpoint (all rewards were invalid), skipping verification")

    print()

    # Git commit and push
    print("="*80)
    print("GIT OPERATIONS")
    print("="*80)
    print()

    try:
        # Add results
        subprocess.run(['git', 'add', str(output_dir)], check=True, cwd=project_root)

        # Commit
        best_fixed_pct = best_checkpoint['fixed_pct'] if best_checkpoint is not None else 0.0
        commit_msg = f"""feat: single H200 emergent hunt - 30 min intensive training

Training completed:
- Nodes: {args.num_nodes:,}
- Time steps: {args.num_time_steps}
- Cycles: {args.num_cycles}
- Best reward: {best_reward:.2f}
- Best fixed points: {best_fixed_pct:.1f}%
- Runtime: {total_time/60:.1f} minutes

Generated with Claude Code (tHHmL v0.1.0)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"""

        subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd=project_root)

        # Push
        subprocess.run(['git', 'push'], check=True, cwd=project_root)

        print("[SUCCESS] Results committed and pushed")
    except subprocess.CalledProcessError as e:
        print(f"[WARNING] Git operation failed: {e}")
        print("Please manually commit and push results")

    print()
    print("="*80)
    print("HUNT COMPLETE")
    print("="*80)
    print()
    print(f"Results: {output_dir}")
    print(f"Best checkpoint: {checkpoint_file}")
    if best_checkpoint is not None and best_checkpoint['fixed_pct'] >= 50.0:
        print(f"Whitepaper: {whitepaper_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
