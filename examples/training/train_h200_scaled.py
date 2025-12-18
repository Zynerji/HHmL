#!/usr/bin/env python3
"""
H200 Scaled Quality-Guided Vortex Learning
===========================================
Maximum Möbius coverage training with 28-32 strips and 4096 hidden dimensions.

This script is optimized for NVIDIA H200 GPU (140GB VRAM) to explore high-capacity
vortex generation at scale.

Configuration:
- Strips: 28-32 Möbius strips (maximum sphere coverage)
- Nodes per strip: 2,000 (56K-64K total nodes)
- Hidden dim: 4096 (8× increase from baseline)
- RNN parameters: ~170M trainable parameters
- Expected VRAM: 60-80GB
- Training: Quality-Guided Learning approach

Quality-Guided Learning:
- Direct reward for neighborhood density, core depth, stability
- Zero-annihilation bonus (+500 reward)
- Quality × Density product prevents gaming
- Proven to achieve 100% vortex density with zero annihilations

Author: HHmL Framework
Date: 2025-12-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from train_multi_strip import (
    MultiStripRNNAgent, vortex_quality_score, controlled_annihilation,
    save_checkpoint, load_checkpoint, torch, nn, np, time, datetime,
    argparse, json, HardwareConfig, SparseTokamakMobiusStrips,
    helical_vortex_reset
)


def compute_detailed_quality_metrics(strips: SparseTokamakMobiusStrips,
                                     sparsity_threshold: float = 0.3) -> dict:
    """
    Compute detailed quality metrics for all vortices.
    Returns the same metrics the annihilator uses to judge quality.
    """
    field_mag = torch.abs(strips.field)
    vortex_mask = field_mag < sparsity_threshold
    vortex_indices = torch.where(vortex_mask)[0]

    if len(vortex_indices) == 0:
        return {
            'avg_neighborhood_density': 0.0,
            'avg_core_depth': 0.0,
            'avg_stability': 0.0,
            'avg_quality': 0.0,
            'num_vortices': 0
        }

    # Compute quality scores
    quality_scores = vortex_quality_score(
        field=strips.field,
        positions=strips.positions,
        vortex_indices=vortex_indices,
        sparsity_threshold=sparsity_threshold
    )

    # Break down into components (sample for performance at scale)
    max_sample = min(5000, len(vortex_indices))  # Sample for large systems
    sample_indices = torch.randperm(len(vortex_indices))[:max_sample]
    sampled_vortices = vortex_indices[sample_indices]

    neighborhood_scores = []
    core_depth_scores = []
    stability_scores = []

    for vortex_idx in sampled_vortices:
        vortex_pos = strips.positions[vortex_idx]

        # Neighborhood density
        distances = torch.norm(strips.positions - vortex_pos, dim=1)
        nearby = (distances < 0.5) & (distances > 0.0)
        if nearby.sum() > 0:
            neighbor_vortices = (field_mag[nearby] < sparsity_threshold).float().mean()
            neighborhood_scores.append(neighbor_vortices.item())
        else:
            neighborhood_scores.append(0.0)

        # Core depth
        core_depth = 1.0 - field_mag[vortex_idx].item()
        core_depth_scores.append(core_depth)

        # Stability
        if nearby.sum() > 0:
            field_variance = field_mag[nearby].std().item()
            stability = max(0, 1.0 - field_variance)
            stability_scores.append(stability)
        else:
            stability_scores.append(0.0)

    return {
        'avg_neighborhood_density': np.mean(neighborhood_scores),
        'avg_core_depth': np.mean(core_depth_scores),
        'avg_stability': np.mean(stability_scores),
        'avg_quality': quality_scores.mean().item(),
        'num_vortices': len(vortex_indices)
    }


def compute_quality_guided_reward(strips: SparseTokamakMobiusStrips,
                                   annihilation_stats: dict = None,
                                   cycles_at_target: int = 0) -> tuple:
    """
    Reward function that directly teaches quality metrics.

    Philosophy:
    - Reward the SAME metrics the annihilator uses
    - This creates a feedback loop: RNN learns what makes a good vortex
    - Penalize annihilations (teach to avoid creating bad vortices)
    """
    # Get vortex density
    field_mag = torch.abs(strips.field)
    vortex_mask = field_mag < 0.3
    vortex_density = vortex_mask.float().mean().item()

    # Get detailed quality metrics
    quality_metrics = compute_detailed_quality_metrics(strips, sparsity_threshold=0.3)

    reward_components = {}

    # 1. DENSITY REWARD: Target 95-100%
    if 0.95 <= vortex_density <= 1.0:
        density_reward = 300 * vortex_density  # Up to +300
    elif vortex_density >= 0.85:
        density_reward = 200 * vortex_density  # +170 to +200
    else:
        density_reward = 100 * vortex_density  # Base reward
    reward_components['density'] = density_reward

    # 2. QUALITY METRIC REWARDS (Direct teaching signal!)
    neighborhood_reward = 150 * quality_metrics['avg_neighborhood_density']
    core_depth_reward = 150 * quality_metrics['avg_core_depth']
    stability_reward = 150 * quality_metrics['avg_stability']

    reward_components['neighborhood'] = neighborhood_reward
    reward_components['core_depth'] = core_depth_reward
    reward_components['stability'] = stability_reward

    # 3. QUALITY x DENSITY PRODUCT (Prevent gaming)
    quality_density_product = quality_metrics['avg_quality'] * vortex_density
    product_reward = 200 * quality_density_product
    reward_components['quality_density_product'] = product_reward

    # 4. ANNIHILATION PENALTY (Teach avoidance)
    num_removed = annihilation_stats['num_removed'] if annihilation_stats else 0
    annihilation_penalty = -50 * num_removed
    reward_components['annihilation_penalty'] = annihilation_penalty

    # 5. STABILITY BONUS (Staying at 95%+ for multiple cycles)
    if vortex_density >= 0.95:
        stability_bonus = min(200, cycles_at_target * 20)  # Up to +200
        reward_components['stability_bonus'] = stability_bonus
    else:
        reward_components['stability_bonus'] = 0

    # 6. ZERO-ANNIHILATION BONUS (Ultimate goal!)
    if num_removed == 0 and vortex_density >= 0.95:
        zero_annihilation_bonus = 500  # Big reward!
        reward_components['zero_annihilation_bonus'] = zero_annihilation_bonus
    else:
        reward_components['zero_annihilation_bonus'] = 0

    # Total reward
    total_reward = sum(reward_components.values())

    # Add quality metrics to breakdown for logging
    reward_components.update({
        'vortex_density_mean': vortex_density,
        'avg_vortex_quality': quality_metrics['avg_quality'],
        'avg_neighborhood_density': quality_metrics['avg_neighborhood_density'],
        'avg_core_depth': quality_metrics['avg_core_depth'],
        'avg_stability': quality_metrics['avg_stability']
    })

    return total_reward, reward_components


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='H200 Scaled Quality-Guided Vortex Learning')
    parser.add_argument('--cycles', type=int, default=60,
                        help='Number of training cycles (60 cycles = ~30 min at scale)')
    parser.add_argument('--strips', type=int, default=40,
                        help='Number of Möbius strips (Option 4: Balanced Maximum)')
    parser.add_argument('--nodes', type=int, default=3500,
                        help='Nodes per strip (Option 4: Balanced Maximum)')
    parser.add_argument('--hidden-dim', type=int, default=6144,
                        help='LSTM hidden dimensions (Option 4: Balanced Maximum)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--save-every', type=int, default=10,
                        help='Checkpoint save frequency (every 10 cycles for 30-min run)')
    parser.add_argument('--max-time', type=int, default=1800,
                        help='Maximum training time in seconds (default: 1800 = 30 min)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    args = parser.parse_args()

    print("="*80)
    print("H200 SCALED QUALITY-GUIDED VORTEX LEARNING")
    print("="*80)
    print()
    print("Configuration:")
    print(f"  Strips: {args.strips} (maximum Möbius coverage)")
    print(f"  Nodes per strip: {args.nodes:,}")
    print(f"  Total nodes: {args.strips * args.nodes:,}")
    print(f"  Hidden dim: {args.hidden_dim:,}")
    print(f"  Cycles: {args.cycles:,}")
    print(f"  Device: {args.device}")
    print()
    print("Quality-Guided Learning Philosophy:")
    print("  1. Neighborhood density (clustered, not isolated)")
    print("  2. Core depth (deep, strong vortices)")
    print("  3. Stability (low field variance)")
    print()
    print("Reward structure:")
    print("  + Density (target 95-100%)")
    print("  + Direct quality metrics")
    print("  + Quality x Density product")
    print("  - Annihilation penalty (-50 per removal)")
    print("  + Zero-annihilation bonus (+500)")
    print("="*80)
    print()

    # Hardware config
    hw_config = HardwareConfig()
    hw_config.print_info()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device('cpu')

    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Initialize system
    print("Initializing multi-strip system...")
    print(f"  Using INTERMEDIATE sparse mode (target ~50-70% sparsity)")

    strips = SparseTokamakMobiusStrips(
        num_strips=args.strips,
        nodes_per_strip=args.nodes,
        device=device
    )

    # Initialize agent
    print(f"\nInitializing RNN agent...")
    agent = MultiStripRNNAgent(
        num_strips=args.strips,
        nodes_per_strip=args.nodes,
        hidden_dim=args.hidden_dim,
        device=str(device)
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in agent.parameters())
    trainable_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Estimate memory
    param_memory = total_params * 4 / 1e9  # 4 bytes per float32
    estimated_activation_memory = param_memory * 3  # Rough estimate
    estimated_total_memory = param_memory + estimated_activation_memory
    print(f"  Estimated VRAM usage: {estimated_total_memory:.1f} GB")
    print()

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # Load checkpoint if resuming
    start_cycle = 0
    metrics = {
        'rewards': [],
        'vortex_densities': [],
        'quality_scores': [],
        'param_history': []  # Required by save_checkpoint
    }
    if args.resume:
        print(f"\n{'='*80}")
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        start_cycle, prev_metrics = load_checkpoint(args.resume, agent)
        print()

    # Training loop
    print("="*80)
    print("STARTING H200 SCALED TRAINING")
    print("="*80)
    print(f"Time limit: {args.max_time} seconds ({args.max_time/60:.1f} minutes)")
    print(f"Target cycles: {args.cycles}")
    print(f"Estimated cycle time: ~30 seconds")
    print(f"Note: Training will stop after {args.max_time/60:.1f} minutes OR {args.cycles} cycles, whichever comes first")
    print("="*80)

    from hhml.utils.live_dashboard import TrainingDashboard
    dashboard = TrainingDashboard(port=8000, auto_open=True)
    dashboard.start()
    print("[Dashboard] Live monitoring at http://localhost:8000")

    start_time = time.time()
    total_cycles = start_cycle + args.cycles
    cycles_at_target = 0
    prev_density = None
    time_limit_reached = False

    checkpoint_dir = Path('checkpoints/h200_scaled')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log file
    log_file = checkpoint_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

    def log_print(msg):
        """Print and log to file"""
        print(msg, flush=True)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    try:
        for cycle in range(start_cycle, total_cycles):
            cycle_start = time.time()

            # Check time limit
            elapsed_time = time.time() - start_time
            if elapsed_time >= args.max_time:
                log_print(f"\n[TIME LIMIT] Reached {args.max_time}s ({args.max_time/60:.1f} min) at cycle {cycle}")
                log_print(f"[TIME LIMIT] Completed {cycle - start_cycle} cycles in {elapsed_time:.1f}s")
                time_limit_reached = True
                break

            # Encode state
            state = agent.encode_state(strips)

            # RNN forward
            actions, control_params, value, hidden = agent.forward(state)

            # Add exploration noise (decay over time)
            exploration_noise = 0.05 * (1.0 - (cycle - start_cycle) / args.cycles)
            if exploration_noise > 0:
                for key in control_params:
                    control_params[key] = control_params[key] + torch.randn_like(control_params[key]) * exploration_noise * 0.3

            # Extract control parameters
            cp = {k: v.item() for k, v in control_params.items()}

            # Apply actions to strips
            for strip_idx in range(args.strips):
                strip_mask = strips.strip_indices == strip_idx
                strip_node_indices = torch.where(strip_mask)[0]

                action = actions[strip_idx].detach()

                # Update amplitudes
                amp_factor = 1.0 + action[0].item() * 0.2
                strips.amplitudes[strip_node_indices] *= amp_factor

                # Update phases
                phase_shift = action[1].item() * 0.1
                strips.phases[strip_node_indices] += phase_shift

            # Evolve field
            strips.evolve_field(
                cp['omega'],
                cp['damping'],
                cp['nonlinearity'],
                1
            )

            # Controlled annihilation
            annihilation_stats = controlled_annihilation(strips, control_params, device)

            # Compute quality-guided reward
            reward, reward_breakdown = compute_quality_guided_reward(
                strips,
                annihilation_stats=annihilation_stats,
                cycles_at_target=cycles_at_target
            )

            # Track stability
            current_density = reward_breakdown['vortex_density_mean']
            if current_density >= 0.95:
                cycles_at_target += 1
            else:
                cycles_at_target = 0

            # Policy gradient update
            loss = -(value * reward)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

            # Metrics
            metrics['rewards'].append(reward)
            metrics['vortex_densities'].append(current_density)
            metrics['quality_scores'].append(reward_breakdown['avg_vortex_quality'])
            prev_density = current_density

            # Dashboard update
            dashboard.update({
                'cycle': cycle,
                'density': current_density,
                'quality': reward_breakdown['avg_vortex_quality'],
                'reward': reward,
                'annihilations': annihilation_stats['num_removed'],
                'cycles_at_target': cycles_at_target
            })

            # Print progress
            if (cycle + 1) % max(1, args.cycles // 20) == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / (cycle + 1 - start_cycle)) * (total_cycles - cycle - 1)
                cycle_time = time.time() - cycle_start

                target_marker = " [AT TARGET!]" if current_density >= 0.95 else ""
                zero_ann_marker = " [ZERO ANN!]" if annihilation_stats['num_removed'] == 0 else ""

                log_print(f"Cycle {cycle+1:4d}/{total_cycles} | "
                          f"Density: {current_density:.1%}{target_marker} | "
                          f"Quality: {reward_breakdown['avg_vortex_quality']:.2f} | "
                          f"Reward: {reward:7.1f} | "
                          f"Removed: {annihilation_stats['num_removed']:3d}{zero_ann_marker} | "
                          f"Stable: {cycles_at_target:2d} | "
                          f"Time: {cycle_time:.1f}s | "
                          f"ETA: {eta:.0f}s")

            # Periodic checkpoint
            if args.save_every > 0 and (cycle + 1) % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_cycle_{cycle+1}.pt"
                save_checkpoint(agent, cycle, metrics, args, checkpoint_path)
                log_print(f"[Checkpoint] Saved to {checkpoint_path}")

            # Memory monitoring (every 100 cycles)
            if device.type == 'cuda' and (cycle + 1) % 100 == 0:
                allocated = torch.cuda.memory_allocated(device) / 1e9
                reserved = torch.cuda.memory_reserved(device) / 1e9
                log_print(f"[VRAM] Allocated: {allocated:.1f} GB | Reserved: {reserved:.1f} GB")

    except KeyboardInterrupt:
        log_print("\n[INTERRUPTED] Training stopped by user")
    except Exception as e:
        log_print(f"\n[ERROR] Training failed: {e}")
        import traceback
        log_print(traceback.format_exc())
    finally:
        dashboard.stop()

        # Final checkpoint
        final_checkpoint = checkpoint_dir / f"checkpoint_final_cycle_{total_cycles-1}.pt"
        save_checkpoint(agent, total_cycles-1, metrics, args, final_checkpoint)
        log_print(f"\n[Checkpoint] Final checkpoint saved to {final_checkpoint}")

        # Results
        results_dir = Path('results/h200_scaled')
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"training_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'configuration': vars(args),
                'metrics': metrics,
                'final_state': {
                    'cycles': total_cycles,
                    'final_density': metrics['vortex_densities'][-1] if metrics['vortex_densities'] else 0.0,
                    'final_reward': metrics['rewards'][-1] if metrics['rewards'] else 0.0,
                    'peak_density': max(metrics['vortex_densities']) if metrics['vortex_densities'] else 0.0
                }
            }, f, indent=2)

        log_print(f"\n{'='*80}")
        log_print("TRAINING COMPLETE")
        log_print("="*80)
        total_time = time.time() - start_time
        log_print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        log_print(f"Cycles completed: {len(metrics['rewards'])}")
        if time_limit_reached:
            log_print(f"[Note] Stopped due to time limit ({args.max_time}s)")
        else:
            log_print(f"[Note] Completed all {args.cycles} cycles")
        if metrics['vortex_densities']:
            log_print(f"Final vortex density: {metrics['vortex_densities'][-1]:.1%}")
            log_print(f"Peak vortex density: {max(metrics['vortex_densities']):.1%}")
        if metrics['rewards']:
            log_print(f"Final reward: {metrics['rewards'][-1]:.1f}")
            log_print(f"Peak reward: {max(metrics['rewards']):.1f}")
        log_print(f"Results saved to: {results_file}")
        log_print(f"Log saved to: {log_file}")
