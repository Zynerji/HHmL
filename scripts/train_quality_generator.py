#!/usr/bin/env python3
"""
Quality Vortex Generator Training (No Annihilation Dependency)
================================================================
Train RNN to generate HIGH-QUALITY vortices from the start,
WITHOUT relying on annihilation as a crutch.

Goal: Reach 100% vortex density and HOLD IT STABLE
Approach: Heavily penalize need for annihilation, reward quality generation

Key Differences from train_multi_strip.py:
- Target density: 95-100% (not 50-80%)
- Annihilation PENALTY (not bonus): -100 per vortex removed
- Stability bonus: +200 for maintaining 95-100% density
- Quality generation reward: +150 * avg_quality for high-quality vortices

This trains the generator to produce only vortices that don't need pruning.

Usage:
    python train_quality_generator.py --cycles 500

Author: HHmL Framework
Date: 2025-12-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import everything from the original train_multi_strip.py
from train_multi_strip import (
    MultiStripRNNAgent, vortex_quality_score, controlled_annihilation,
    save_checkpoint, load_checkpoint, torch, nn, np, time, datetime,
    argparse, json, HardwareConfig, SparseTokamakMobiusStrips,
    helical_vortex_reset
)


def compute_quality_generator_reward(strips: SparseTokamakMobiusStrips,
                                    param_history: list = None,
                                    annihilation_stats: dict = None,
                                    prev_density: float = None,
                                    cycles_at_target: int = 0) -> tuple:
    """
    Reward function optimized for training high-quality vortex generation

    NEW PHILOSOPHY:
    - Generate good vortices from the start (don't rely on cleanup)
    - Reach 100% density and HOLD it stable
    - Heavily penalize need for annihilation

    Rewards:
    - Target density: 95-100% (NEW: much higher than 50-80%)
    - Stability bonus: Maintaining target density across cycles (+200)
    - Quality generation: High avg vortex quality (+150 * quality)
    - Annihilation PENALTY: Heavy penalty for needing cleanup (-100 * removed)
    - Uniformity: Consistent across strips
    """
    # Get field for each strip
    vortex_densities = []
    vortex_qualities = []
    field_magnitudes = []

    sparsity_threshold = 0.3  # Consistent vortex detection

    for strip_idx in range(strips.num_strips):
        field = strips.get_strip_field(strip_idx)
        field_mag = torch.abs(field)

        # Vortex density (low field regions)
        vortex_mask = field_mag < sparsity_threshold
        vortex_density = vortex_mask.float().mean().item()
        vortex_densities.append(vortex_density)

        # Measure QUALITY of generated vortices (BEFORE annihilation)
        vortex_indices = torch.where(vortex_mask)[0]
        if len(vortex_indices) > 0:
            # Get positions for this strip
            strip_mask = strips.strip_indices == strip_idx
            strip_positions = strips.positions[strip_mask]

            qualities = vortex_quality_score(
                field=field,
                positions=strip_positions,
                vortex_indices=vortex_indices,
                sparsity_threshold=sparsity_threshold
            )
            avg_quality = qualities.mean().item() if len(qualities) > 0 else 0.0
        else:
            avg_quality = 0.0

        vortex_qualities.append(avg_quality)
        field_magnitudes.append(field_mag.mean().item())

    # Aggregate metrics
    avg_vortex_density = np.mean(vortex_densities)
    std_vortex_density = np.std(vortex_densities)
    avg_vortex_quality = np.mean(vortex_qualities)

    # ==================================================================
    # NEW REWARD STRUCTURE: Train quality generation, not annihilation
    # ==================================================================

    # 1. DENSITY REWARD: Target 95-100% (not 50-80%)
    if 0.95 <= avg_vortex_density <= 1.0:
        # Perfect range: huge reward
        density_reward = 300 * avg_vortex_density  # Up to +300
    elif 0.8 <= avg_vortex_density < 0.95:
        # Getting close: encourage progress
        density_reward = 200 * avg_vortex_density - 50
    elif avg_vortex_density > 1.0:
        # Shouldn't happen, but handle gracefully
        density_reward = 250
    else:
        # Too low: strong penalty
        density_reward = 150 * avg_vortex_density - 200 * (0.8 - avg_vortex_density)**2

    # 2. STABILITY BONUS: Reward maintaining target density
    stability_bonus = 0
    if prev_density is not None and 0.95 <= avg_vortex_density <= 1.0:
        # Currently at target
        if 0.95 <= prev_density <= 1.0:
            # Was at target last cycle too: STABLE!
            stability_bonus = 200  # Large bonus for stability
        else:
            # Just reached target: good progress
            stability_bonus = 50

    # Track consecutive cycles at target (passed back to training loop)
    if 0.95 <= avg_vortex_density <= 1.0:
        cycles_at_target += 1
    else:
        cycles_at_target = 0

    # Extra bonus for sustained stability
    sustained_stability_bonus = 0
    if cycles_at_target >= 10:
        sustained_stability_bonus = 100 + min(cycles_at_target * 5, 200)  # Up to +300

    # 3. QUALITY GENERATION REWARD: Reward high-quality vortices BEFORE annihilation
    # This teaches: "Generate good vortices from the start"
    quality_generation_reward = 150 * avg_vortex_quality  # Up to +150 for perfect quality

    # 4. ANNIHILATION PENALTY: Heavily penalize need for cleanup
    # This is the KEY change: annihilation = failure, not success
    annihilation_penalty = 0
    if annihilation_stats is not None:
        num_removed = annihilation_stats['num_removed']
        if num_removed > 0:
            # HEAVY PENALTY for needing to remove ANY vortices
            # Message to RNN: "Don't generate vortices that need removal!"
            annihilation_penalty = -100 * num_removed  # -100 per removed vortex

            # Extra penalty if removing HIGH-quality vortices (means we're confused)
            if annihilation_stats['avg_quality_removed'] > 0.5:
                annihilation_penalty -= 50 * num_removed  # Total -150 per vortex

    # 5. UNIFORMITY REWARD: Consistent across strips
    uniformity_reward = -50 * std_vortex_density

    # 6. COLLAPSE PREVENTION: Extra penalty if density too low
    collapse_penalty = 0
    if avg_vortex_density < 0.1:
        collapse_penalty = -300 * (0.1 - avg_vortex_density)

    # 7. Spectral connectivity (keep from original)
    spectral_bonus = 0
    try:
        if hasattr(strips, 'edge_index') and strips.edge_index.shape[1] > 0:
            avg_degree = strips.edge_index.shape[1] / strips.total_nodes
            target_degree = 500  # Optimal for 4K nodes
            connectivity_score = 1.0 - abs(avg_degree - target_degree) / target_degree
            spectral_bonus = 20 * max(0, connectivity_score)
    except:
        pass

    # ==================================================================
    # TOTAL REWARD
    # ==================================================================
    total_reward = (
        density_reward +
        stability_bonus +
        sustained_stability_bonus +
        quality_generation_reward +
        annihilation_penalty +  # NOTE: This is negative!
        uniformity_reward +
        collapse_penalty +
        spectral_bonus
    )

    # NaN protection
    if np.isnan(total_reward) or np.isinf(total_reward):
        print(f"  [WARNING] NaN/Inf reward detected, using penalty")
        total_reward = -1000.0
        density_reward = -1000.0

    return total_reward, {
        'vortex_density_mean': avg_vortex_density,
        'vortex_density_std': std_vortex_density,
        'avg_vortex_quality': avg_vortex_quality,
        'density_reward': density_reward,
        'stability_bonus': stability_bonus + sustained_stability_bonus,
        'quality_generation_reward': quality_generation_reward,
        'annihilation_penalty': annihilation_penalty,  # Will be negative or zero
        'uniformity_reward': uniformity_reward,
        'collapse_penalty': collapse_penalty,
        'spectral_bonus': spectral_bonus,
        'cycles_at_target': cycles_at_target  # Pass back to tracking
    }


def train_quality_generator(args):
    """
    Main training loop for quality vortex generation

    Goal: Train RNN to generate high-quality vortices that don't need annihilation
    """

    # Hardware detection
    hw_config = HardwareConfig()
    hw_config.print_info()

    # Get optimal parameters for mode
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

    print(f"\nQuality Generator Training Configuration:")
    print(f"  Strips: {num_strips}")
    print(f"  Nodes per strip: {nodes_per_strip:,}")
    print(f"  Total nodes: {num_strips * nodes_per_strip:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Cycles: {num_cycles}")
    print(f"  Device: {device.upper()}")
    print(f"\n  GOAL: Reach 95-100% density and HOLD IT STABLE")
    print(f"  METHOD: Generate high-quality vortices, penalize annihilation need")

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
    cycles_at_target = 0  # Track consecutive cycles at 95-100% density

    if args.resume:
        print("\n" + "=" * 80)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 80)
        start_cycle, loaded_metrics = load_checkpoint(args.resume, agent)

        metrics = {
            'cycle_times': [],
            'rewards': loaded_metrics['rewards'],
            'vortex_densities': loaded_metrics['vortex_densities'],
            'vortex_qualities': [],
            'vortex_std': [],
            'global_params': [],
            'annihilation_counts': [],
            'cycles_at_target_history': [],
            'mode': 'sparse' if strips.use_sparse else 'dense'
        }
        param_history = loaded_metrics['param_history']
    else:
        metrics = {
            'cycle_times': [],
            'rewards': [],
            'vortex_densities': [],
            'vortex_qualities': [],
            'vortex_std': [],
            'global_params': [],
            'annihilation_counts': [],
            'cycles_at_target_history': [],
            'mode': 'sparse' if strips.use_sparse else 'dense'
        }

    print("\n" + "=" * 80)
    print("STARTING QUALITY GENERATOR TRAINING")
    print("=" * 80)

    start_time = time.time()
    total_cycles = start_cycle + num_cycles
    prev_density = None

    for cycle in range(start_cycle, total_cycles):
        cycle_start = time.time()

        # Encode state
        state = agent.encode_state(strips)

        # RNN forward
        actions, control_params, value, hidden = agent.forward(state)

        # Add exploration noise (anneal over time)
        exploration_noise = 0.1 * (1.0 - cycle / total_cycles)
        if exploration_noise > 0:
            for key in control_params:
                control_params[key] = control_params[key] + torch.randn_like(control_params[key]) * exploration_noise * 0.3

        # Extract control parameters
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

        # Apply vortex seeding (RNN-controlled)
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

        # RNN-CONTROLLED VORTEX ANNIHILATION
        # Still available, but now HEAVILY PENALIZED in reward
        # RNN should learn to avoid triggering it
        annihilation_stats = {'num_removed': 0, 'avg_quality_removed': 0.0, 'num_preserved': 0}
        if cycle % 5 == 0:
            annihilation_stats = controlled_annihilation(strips, cp, device)

        # Track parameters
        param_history.append(cp.copy())

        # Compute reward with NEW quality-focused reward function
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
            print(f"  [WARNING] NaN/Inf loss detected at cycle {cycle+1}, skipping optimizer step")
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

        # Print progress
        if (cycle + 1) % max(1, num_cycles // 20) == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (cycle + 1 - start_cycle)) * (total_cycles - cycle - 1)

            # Highlight if at target density
            target_marker = " [TARGET!]" if 0.95 <= prev_density <= 1.0 else ""
            annihilation_marker = f" [ANNIHILATED: {annihilation_stats['num_removed']}]" if annihilation_stats['num_removed'] > 0 else ""

            print(f"Cycle {cycle+1:4d}/{total_cycles} | "
                  f"Density: {prev_density:.1%}{target_marker} | "
                  f"Quality: {reward_breakdown['avg_vortex_quality']:.2f} | "
                  f"Reward: {reward:7.1f} | "
                  f"Stable: {cycles_at_target} cyc{annihilation_marker} | "
                  f"Time: {cycle_time:.3f}s | "
                  f"ETA: {eta:.0f}s")

        # Periodic checkpoint saving
        if args.save_every > 0 and (cycle + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_quality_gen_cycle_{cycle+1}.pt"
            save_checkpoint(agent, cycle, {
                'rewards': metrics['rewards'],
                'vortex_densities': metrics['vortex_densities'],
                'param_history': param_history
            }, args, checkpoint_path)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("QUALITY GENERATOR TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Average cycle time: {np.mean(metrics['cycle_times']):.3f}s")
    print(f"Cycles per second: {num_cycles / total_time:.2f}")
    print(f"Final vortex density: {metrics['vortex_densities'][-1]:.1%}")
    print(f"Final vortex quality: {metrics['vortex_qualities'][-1]:.2f}")
    print(f"Final reward: {metrics['rewards'][-1]:.1f}")
    print(f"Peak vortex density: {max(metrics['vortex_densities']):.1%}")
    print(f"Max consecutive cycles at target: {max(metrics['cycles_at_target_history'])}")
    print(f"Total annihilations in training: {sum(metrics['annihilation_counts'])}")
    print("=" * 80)

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f"checkpoint_quality_gen_final_cycle_{total_cycles-1}.pt"
    save_checkpoint(agent, total_cycles - 1, {
        'rewards': metrics['rewards'],
        'vortex_densities': metrics['vortex_densities'],
        'param_history': param_history
    }, args, final_checkpoint_path)

    # Save results
    output_dir = Path("results/quality_generator_training")
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
            'training_type': 'quality_generator'
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
            'cycle_times': [float(x) for x in metrics['cycle_times'][-100:]]
        },
        'summary': {
            'peak_density': float(max(metrics['vortex_densities'])),
            'peak_quality': float(max(metrics['vortex_qualities'])),
            'max_consecutive_at_target': int(max(metrics['cycles_at_target_history'])),
            'total_annihilations': int(sum(metrics['annihilation_counts']))
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Quality Vortex Generator Training (No Annihilation Dependency)'
    )

    # Mode selection
    parser.add_argument('--mode', choices=['benchmark', 'training', 'production'],
                        default='training',
                        help='Training mode (auto-scales parameters)')

    # Custom parameters
    parser.add_argument('--strips', type=int, default=None,
                        help='Number of Möbius strips')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Nodes per strip')
    parser.add_argument('--cycles', type=int, default=500,
                        help='Number of training cycles (default: 500)')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='RNN hidden dimension')

    # Checkpoint management
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/quality_generator',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N cycles (0 = only at end)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("QUALITY VORTEX GENERATOR TRAINING")
    print("=" * 80)
    print("\nGoal: Generate high-quality vortices WITHOUT relying on annihilation")
    print("Target: 95-100% density, held stable")
    print("Method: Penalize annihilation need, reward quality generation")
    print("=" * 80)

    results = train_quality_generator(args)


if __name__ == "__main__":
    main()
