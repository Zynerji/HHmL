#!/usr/bin/env python3
"""
Surgical Parameter Reset Training
==================================
Keep learned RNN weights, but OVERRIDE annihilation parameters to force quality generation.

Strategy:
1. Load checkpoint with all learned vortex generation knowledge
2. FREEZE annihilation controls at near-zero (disable the crutch)
3. Force RNN to learn quality generation without cleanup
4. Curriculum: Gradually unlock annihilation after 300 cycles (if needed)

This surgical approach preserves good learned behaviors while breaking bad dependencies.

Usage:
    python train_surgical_reset.py --cycles 500 --resume checkpoints/annihilation/checkpoint_final_cycle_499.pt

Author: HHmL Framework
Date: 2025-12-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import from original scripts
from train_multi_strip import (
    MultiStripRNNAgent, vortex_quality_score, controlled_annihilation,
    save_checkpoint, load_checkpoint, torch, nn, np, time, datetime,
    argparse, json, HardwareConfig, SparseTokamakMobiusStrips,
    helical_vortex_reset
)
from train_quality_generator import compute_quality_generator_reward
from hhml.utils.live_dashboard import TrainingDashboard


def surgical_parameter_override(control_params: dict, cycle: int, start_cycle: int,
                                unlock_threshold: int = 300) -> dict:
    """
    Surgically override specific control parameters to force quality generation.

    Phase 1 (relative cycles 0-300): DISABLE annihilation completely
    Phase 2 (relative cycles 300+): Gradually unlock annihilation (if quality is stable)

    Args:
        control_params: Original RNN-generated parameters
        cycle: Current absolute training cycle
        start_cycle: Cycle number training started from (for relative counting)
        unlock_threshold: Relative cycle at which to start unlocking annihilation

    Returns:
        Modified control_params with overrides applied
    """
    # Create a copy to avoid modifying original
    overridden = control_params.copy()

    # CRITICAL FIX: Use RELATIVE cycles (not absolute)
    # If resuming from cycle 500, relative cycles are 0, 1, 2, ...
    relative_cycle = cycle - start_cycle

    # PHASE 1: Disable annihilation completely (force quality generation)
    if relative_cycle < unlock_threshold:
        # Force annihilation parameters to near-zero (but not exactly zero to avoid NaN)
        overridden['antivortex_strength'] = torch.tensor(0.01, device=control_params['kappa'].device)
        overridden['annihilation_radius'] = torch.tensor(0.1, device=control_params['kappa'].device)  # Minimum
        overridden['pruning_threshold'] = torch.tensor(0.95, device=control_params['kappa'].device)  # Only worst quality
        overridden['preserve_ratio'] = torch.tensor(0.95, device=control_params['kappa'].device)  # Preserve almost all

        # BOOST vortex generation parameters to encourage quality
        # Let RNN control these, but ensure they're in productive ranges
        # (No override needed - RNN will learn)

    # PHASE 2: Gradual unlocking (after unlock_threshold)
    else:
        # Gradually increase annihilation access over 200 cycles
        unlock_progress = min(1.0, (relative_cycle - unlock_threshold) / 200.0)

        # Soft unlock: interpolate between disabled and RNN-controlled
        rnn_strength = control_params['antivortex_strength'].item()
        overridden['antivortex_strength'] = torch.tensor(
            0.01 + unlock_progress * (rnn_strength - 0.01),
            device=control_params['kappa'].device
        )

        rnn_threshold = control_params['pruning_threshold'].item()
        overridden['pruning_threshold'] = torch.tensor(
            0.95 + unlock_progress * (rnn_threshold - 0.95),
            device=control_params['kappa'].device
        )

        rnn_preserve = control_params['preserve_ratio'].item()
        overridden['preserve_ratio'] = torch.tensor(
            0.95 + unlock_progress * (rnn_preserve - 0.95),
            device=control_params['kappa'].device
        )

    return overridden


def train_surgical_reset(args):
    """
    Main training loop with surgical parameter override

    Keeps learned RNN weights, but forces quality generation by disabling annihilation
    """

    # Hardware detection
    hw_config = HardwareConfig()
    hw_config.print_info()

    # Get parameters
    if args.strips and args.nodes:
        num_strips = args.strips
        nodes_per_strip = args.nodes
        hidden_dim = args.hidden_dim
    else:
        params = hw_config.get_optimal_params(args.mode)
        num_strips = params.num_strips
        nodes_per_strip = params.nodes_per_strip
        hidden_dim = params.hidden_dim

    device = hw_config.device_type
    num_cycles = args.cycles

    print(f"\n{'='*80}")
    print("SURGICAL PARAMETER RESET TRAINING")
    print(f"{'='*80}")
    print(f"\nStrategy:")
    print(f"  1. Load checkpoint with learned vortex generation")
    print(f"  2. OVERRIDE annihilation controls -> near-zero")
    print(f"  3. Force quality generation without cleanup crutch")
    print(f"  4. Unlock annihilation gradually after cycle {args.unlock_at}")
    print(f"\n{'='*80}")

    print(f"\nConfiguration:")
    print(f"  Strips: {num_strips}")
    print(f"  Nodes per strip: {nodes_per_strip:,}")
    print(f"  Total nodes: {num_strips * nodes_per_strip:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Cycles: {num_cycles}")
    print(f"  Device: {device.upper()}")
    print(f"  Annihilation unlock at cycle: {args.unlock_at}")

    # Initialize
    print("\nInitializing multi-strip system...")
    strips = SparseTokamakMobiusStrips(
        num_strips=num_strips,
        nodes_per_strip=nodes_per_strip,
        device=device
    )

    agent = MultiStripRNNAgent(
        num_strips=num_strips,
        nodes_per_strip=nodes_per_strip,
        hidden_dim=hidden_dim,
        device=device
    )

    # Checkpoint management
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_cycle = 0
    param_history = []
    cycles_at_target = 0

    # MUST resume from checkpoint for surgical approach
    if not args.resume:
        raise ValueError("Surgical reset REQUIRES --resume checkpoint (need learned weights)")

    print("\n" + "=" * 80)
    print("LOADING CHECKPOINT FOR SURGICAL RESET")
    print("=" * 80)
    start_cycle, loaded_metrics = load_checkpoint(args.resume, agent)

    print(f"\n[SURGICAL] Loaded learned weights from {args.resume}")
    print(f"[SURGICAL] Will OVERRIDE annihilation parameters to force quality generation")
    print(f"[SURGICAL] Annihilation locked at near-zero until cycle {args.unlock_at}")

    # Initialize metrics
    metrics = {
        'cycle_times': [],
        'rewards': loaded_metrics['rewards'],
        'vortex_densities': loaded_metrics['vortex_densities'],
        'vortex_qualities': [],
        'vortex_std': [],
        'global_params': [],
        'annihilation_counts': [],
        'cycles_at_target_history': [],
        'override_active_history': [],  # Track when overrides are active
        'mode': 'sparse' if strips.use_sparse else 'dense'
    }
    param_history = loaded_metrics['param_history']

    print("\n" + "=" * 80)
    print("STARTING SURGICAL RESET TRAINING")
    print("=" * 80)

    # Initialize live dashboard
    dashboard = TrainingDashboard(port=8000, auto_open=True)
    dashboard.start()
    print("[Dashboard] Live monitoring at http://localhost:8000")

    start_time = time.time()
    total_cycles = start_cycle + num_cycles
    prev_density = None

    try:
        for cycle in range(start_cycle, total_cycles):
            cycle_start = time.time()

            # Encode state
            state = agent.encode_state(strips)

            # RNN forward (generates control parameters)
            actions, control_params, value, hidden = agent.forward(state)

            # Add exploration noise (reduced for fine-tuning)
            exploration_noise = 0.05 * (1.0 - (cycle - start_cycle) / num_cycles)  # 0.05 -> 0
            if exploration_noise > 0:
                for key in control_params:
                    control_params[key] = control_params[key] + torch.randn_like(control_params[key]) * exploration_noise * 0.3

            # ========================================
            # SURGICAL OVERRIDE: Force specific parameters
            # ========================================
            original_params = {k: v.item() for k, v in control_params.items()}

            # Apply surgical overrides
            control_params = surgical_parameter_override(
                control_params,
                cycle,
                start_cycle=start_cycle,  # FIXED: Pass start_cycle for relative counting
                unlock_threshold=args.unlock_at
            )

            # Track override status (using relative cycles)
            relative_cycle = cycle - start_cycle
            override_active = (relative_cycle < args.unlock_at)

            # Extract control parameters (now with overrides)
            cp = {k: v.item() for k, v in control_params.items()}

            # Apply actions to strips
            for strip_idx in range(num_strips):
                strip_mask = strips.strip_indices == strip_idx
                strip_node_indices = torch.where(strip_mask)[0]

                action = actions[strip_idx].detach()
                strips.amplitudes[strip_node_indices] += action * 0.05 * cp['amp_variance']
                strips.amplitudes[strip_node_indices] = torch.clamp(
                    strips.amplitudes[strip_node_indices], 0.1, 5.0
                )

            # Apply vortex seeding
            if cp['vortex_seed_strength'] > 0.3:
                num_seeds = int(cp['vortex_seed_strength'] * 100)
                seed_indices = torch.randperm(strips.total_nodes, device=device)[:num_seeds]
                strips.field[seed_indices] *= 0.1

            # Evolve field
            field_updates, sample_indices = strips.evolve_field(
                t=float(cycle),
                sample_ratio=cp['sample_ratio'],
                damping=cp['damping'],
                nonlinearity=cp['nonlinearity'],
                omega=cp['omega'],
                diffusion_dt=cp['diffusion_dt'],
                spectral_weight=cp['spectral_weight']
            )

            # NaN protection
            field_updates = torch.nan_to_num(field_updates, nan=0.0, posinf=1.0, neginf=-1.0)
            strips.field[sample_indices] = field_updates

            # Helical vortex reset if needed
            current_vortex_density = (torch.abs(strips.field) < cp['sparsity_threshold']).float().mean().item()
            if current_vortex_density < 0.05 and cycle % 20 == 10:
                strips.field = helical_vortex_reset(
                    positions=strips.positions,
                    field=strips.field,
                    omega=cp['omega'],
                    vortex_target=cp['vortex_target'],
                    reset_strength=cp['reset_strength']
                )

            # Annihilation (with overridden parameters - should be minimal in Phase 1)
            annihilation_stats = {'num_removed': 0, 'avg_quality_removed': 0.0, 'num_preserved': 0}
            if cycle % 5 == 0:
                annihilation_stats = controlled_annihilation(strips, cp, device)

            # Track parameters
            param_history.append(cp.copy())

            # Compute reward (quality-focused)
            reward, reward_breakdown = compute_quality_generator_reward(
                strips,
                param_history=param_history,
                annihilation_stats=annihilation_stats,
                prev_density=prev_density,
                cycles_at_target=cycles_at_target
            )

            # Update tracking
            prev_density = reward_breakdown['vortex_density_mean']
            cycles_at_target = reward_breakdown['cycles_at_target']

            # RL update
            reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
            loss = -value * reward_tensor

            # NaN protection
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f"  [WARNING] NaN/Inf loss at cycle {cycle+1}, skipping")
            else:
                agent.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                agent.optimizer.step()

            # Track metrics
            cycle_time = time.time() - cycle_start
            metrics['cycle_times'].append(cycle_time)
            metrics['rewards'].append(reward)
            metrics['vortex_densities'].append(reward_breakdown['vortex_density_mean'])
            metrics['vortex_qualities'].append(reward_breakdown['avg_vortex_quality'])
            metrics['vortex_std'].append(reward_breakdown['vortex_density_std'])
            metrics['global_params'].append(cp.copy())
            metrics['annihilation_counts'].append(annihilation_stats['num_removed'])
            metrics['cycles_at_target_history'].append(cycles_at_target)
            metrics['override_active_history'].append(override_active)

            # Update live dashboard
            dashboard.update({
                'cycle': cycle,
                'density': reward_breakdown['vortex_density_mean'],
                'quality': reward_breakdown['avg_vortex_quality'],
                'reward': reward,
                'annihilations': annihilation_stats['num_removed'],
                'cycles_at_target': cycles_at_target
            })

            # Print progress
            if (cycle + 1) % max(1, num_cycles // 20) == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (cycle + 1 - start_cycle)) * (total_cycles - cycle - 1)

                # Status indicators
                target_marker = " [AT TARGET!]" if 0.95 <= prev_density <= 1.0 else ""
                override_marker = " [OVERRIDE ACTIVE]" if override_active else " [UNLOCKING]"
                annihilation_marker = f" [Removed: {annihilation_stats['num_removed']}]" if annihilation_stats['num_removed'] > 0 else ""

                print(f"Cycle {cycle+1:4d}/{total_cycles} | "
                      f"Density: {prev_density:.1%}{target_marker} | "
                      f"Quality: {reward_breakdown['avg_vortex_quality']:.2f} | "
                      f"Reward: {reward:7.1f} | "
                      f"Stable: {cycles_at_target:2d}{override_marker}{annihilation_marker} | "
                      f"ETA: {eta:.0f}s")

            # Periodic checkpoint
            if args.save_every > 0 and (cycle + 1) % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_surgical_cycle_{cycle+1}.pt"
                save_checkpoint(agent, cycle, {
                    'rewards': metrics['rewards'],
                    'vortex_densities': metrics['vortex_densities'],
                    'param_history': param_history
                }, args, checkpoint_path)

    finally:
        # Always stop dashboard
        dashboard.stop()

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("SURGICAL RESET TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Average cycle time: {np.mean(metrics['cycle_times']):.3f}s")
    print(f"Cycles per second: {num_cycles / total_time:.2f}")
    print(f"\nResults:")
    print(f"  Final vortex density: {metrics['vortex_densities'][-1]:.1%}")
    print(f"  Final vortex quality: {metrics['vortex_qualities'][-1]:.2f}")
    print(f"  Final reward: {metrics['rewards'][-1]:.1f}")
    print(f"  Peak density: {max(metrics['vortex_densities']):.1%}")
    print(f"  Peak quality: {max(metrics['vortex_qualities']):.2f}")
    print(f"  Max consecutive at target: {max(metrics['cycles_at_target_history'])}")
    print(f"  Total annihilations: {sum(metrics['annihilation_counts'])} (avg {sum(metrics['annihilation_counts'])/num_cycles:.1f}/cycle)")
    print(f"  Cycles with override active: {sum(metrics['override_active_history'])}/{num_cycles}")
    print("=" * 80)

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f"checkpoint_surgical_final_cycle_{total_cycles-1}.pt"
    save_checkpoint(agent, total_cycles - 1, {
        'rewards': metrics['rewards'],
        'vortex_densities': metrics['vortex_densities'],
        'param_history': param_history
    }, args, final_checkpoint_path)

    # Save results
    output_dir = Path("results/surgical_reset_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"training_{timestamp}.json"

    results = {
        'configuration': {
            'num_strips': num_strips,
            'nodes_per_strip': nodes_per_strip,
            'total_nodes': num_strips * nodes_per_strip,
            'hidden_dim': hidden_dim,
            'num_cycles': num_cycles,
            'mode': metrics['mode'],
            'device': device,
            'training_type': 'surgical_reset',
            'unlock_threshold': args.unlock_at
        },
        'performance': {
            'total_time': total_time,
            'avg_cycle_time': float(np.mean(metrics['cycle_times'])),
            'cycles_per_second': num_cycles / total_time
        },
        'final_state': {
            'vortex_density': float(metrics['vortex_densities'][-1]),
            'vortex_quality': float(metrics['vortex_qualities'][-1]),
            'vortex_std': float(metrics['vortex_std'][-1]),
            'reward': float(metrics['rewards'][-1]),
            'cycles_at_target': int(metrics['cycles_at_target_history'][-1]),
            'global_params': metrics['global_params'][-1]
        },
        'metrics': {
            'vortex_densities': [float(x) for x in metrics['vortex_densities']],
            'vortex_qualities': [float(x) for x in metrics['vortex_qualities']],
            'rewards': [float(x) for x in metrics['rewards']],
            'annihilation_counts': [int(x) for x in metrics['annihilation_counts']],
            'cycles_at_target_history': [int(x) for x in metrics['cycles_at_target_history']],
            'override_active_history': [bool(x) for x in metrics['override_active_history']],
            'cycle_times': [float(x) for x in metrics['cycle_times'][-100:]]
        },
        'summary': {
            'peak_density': float(max(metrics['vortex_densities'])),
            'peak_quality': float(max(metrics['vortex_qualities'])),
            'max_consecutive_at_target': int(max(metrics['cycles_at_target_history'])),
            'total_annihilations': int(sum(metrics['annihilation_counts'])),
            'avg_annihilations_per_cycle': float(sum(metrics['annihilation_counts']) / num_cycles),
            'override_cycles': int(sum(metrics['override_active_history']))
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Surgical Parameter Reset Training (Override Annihilation Controls)'
    )

    # Mode selection
    parser.add_argument('--mode', choices=['benchmark', 'training', 'production'],
                        default='training',
                        help='Training mode')

    # Custom parameters
    parser.add_argument('--strips', type=int, default=None)
    parser.add_argument('--nodes', type=int, default=None)
    parser.add_argument('--cycles', type=int, default=500,
                        help='Number of training cycles (default: 500)')
    parser.add_argument('--hidden-dim', type=int, default=512)

    # Surgical reset specific
    parser.add_argument('--unlock-at', type=int, default=300,
                        help='Cycle at which to start unlocking annihilation (default: 300)')

    # Checkpoint management
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to checkpoint (REQUIRED for surgical reset)')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/surgical_reset',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=0)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("SURGICAL PARAMETER RESET TRAINING")
    print("=" * 80)
    print("\nStrategy: Keep learned weights, override annihilation controls")
    print(f"Phase 1 (0-{args.unlock_at}): Annihilation DISABLED - force quality generation")
    print(f"Phase 2 ({args.unlock_at}+): Gradual unlock - if quality stabilizes")
    print("=" * 80)

    results = train_surgical_reset(args)


if __name__ == "__main__":
    main()
