#!/usr/bin/env python3
"""
Quality-Guided Vortex Learning
================================
Train RNN to generate vortices that match the annihilator's quality criteria.

KEY INSIGHT:
The annihilator removes vortices with low:
  1. Neighborhood density (isolated vortices)
  2. Core depth (shallow/weak vortices)
  3. Stability (high field variance)

This script DIRECTLY REWARDS these quality metrics, teaching the RNN
to create vortices that naturally pass the annihilator's standards.

Reward Structure:
  - Density reward: Target 95-100%
  - Quality rewards: Direct metrics (neighborhood, core, stability)
  - Annihilation penalty: -50 per removal (teach avoidance)
  - Quality x Density product: Cannot game one vs the other

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

    # Break down into components
    neighborhood_scores = []
    core_depth_scores = []
    stability_scores = []

    for vortex_idx in vortex_indices:
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
    # High density with low quality = bad
    # Low density with high quality = bad
    # Both high = good!
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
    parser = argparse.ArgumentParser(description='Quality-Guided Vortex Learning')
    parser.add_argument('--cycles', type=int, default=500)
    parser.add_argument('--strips', type=int, default=2)
    parser.add_argument('--nodes', type=int, default=2000)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save-every', type=int, default=100)
    args = parser.parse_args()

    print("="*80)
    print("QUALITY-GUIDED VORTEX LEARNING")
    print("="*80)
    print()
    print("Teaching RNN the annihilator's quality criteria:")
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
    device = hw_config.device

    # Initialize system
    print(f"\nConfiguration:")
    print(f"  Strips: {args.strips}")
    print(f"  Nodes per strip: {args.nodes:,}")
    print(f"  Total nodes: {args.strips * args.nodes:,}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Device: {device}")
    print()

    print("Initializing multi-strip system...")
    print(f"  Using INTERMEDIATE sparse mode (target ~50-70% sparsity, 1000 neighbors, r=0.8)")
    strips = SparseTokamakMobiusStrips(
        num_strips=args.strips,
        nodes_per_strip=args.nodes,
        device=device
    )

    # Initialize agent
    agent = MultiStripRNNAgent(
        num_strips=args.strips,
        nodes_per_strip=args.nodes,
        hidden_dim=args.hidden_dim
    ).to(device)

    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # Load checkpoint if resuming
    start_cycle = 0
    metrics = {'rewards': [], 'vortex_densities': [], 'quality_scores': []}
    if args.resume:
        print(f"\n{'='*80}")
        print("RESUMING FROM CHECKPOINT")
        print("="*80)
        start_cycle, prev_metrics = load_checkpoint(args.resume, agent)
        # Note: load_checkpoint handles printing
        print()

    # Training loop
    print("="*80)
    print("STARTING QUALITY-GUIDED TRAINING")
    print("="*80)

    from hhml.utils.live_dashboard import TrainingDashboard
    dashboard = TrainingDashboard(port=8000, auto_open=True)
    dashboard.start()
    print("[Dashboard] Live monitoring at http://localhost:8000")

    start_time = time.time()
    total_cycles = start_cycle + args.cycles
    cycles_at_target = 0
    prev_density = None

    checkpoint_dir = Path('checkpoints/quality_guided')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    try:
        for cycle in range(start_cycle, total_cycles):
            cycle_start = time.time()

            # Encode state
            state = agent.encode_state(strips)

            # RNN forward
            actions, control_params, value, hidden = agent.forward(state)

            # Add exploration noise
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

                target_marker = " [AT TARGET!]" if current_density >= 0.95 else ""
                zero_ann_marker = " [ZERO ANN!]" if annihilation_stats['num_removed'] == 0 else ""

                print(f"Cycle {cycle+1:4d}/{total_cycles} | "
                      f"Density: {current_density:.1%}{target_marker} | "
                      f"Quality: {reward_breakdown['avg_vortex_quality']:.2f} | "
                      f"Reward: {reward:7.1f} | "
                      f"Removed: {annihilation_stats['num_removed']:3d}{zero_ann_marker} | "
                      f"Stable: {cycles_at_target:2d} | "
                      f"ETA: {eta:.0f}s", flush=True)

            # Periodic checkpoint
            if args.save_every > 0 and (cycle + 1) % args.save_every == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_cycle_{cycle+1}.pt"
                save_checkpoint(agent, cycle, metrics, args, checkpoint_path)

    finally:
        dashboard.stop()

        # Final checkpoint
        final_checkpoint = checkpoint_dir / f"checkpoint_final_cycle_{total_cycles-1}.pt"
        save_checkpoint(agent, total_cycles-1, metrics, args, final_checkpoint)

        # Results
        results_dir = Path('results/quality_guided')
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"training_{timestamp}.json"

        with open(results_file, 'w') as f:
            json.dump({
                'configuration': vars(args),
                'metrics': metrics,
                'final_state': {
                    'cycles': total_cycles,
                    'final_density': metrics['vortex_densities'][-1],
                    'final_reward': metrics['rewards'][-1],
                    'peak_density': max(metrics['vortex_densities'])
                }
            }, f, indent=2)

        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Final vortex density: {metrics['vortex_densities'][-1]:.1%}")
        print(f"Peak vortex density: {max(metrics['vortex_densities']):.1%}")
        print(f"Final reward: {metrics['rewards'][-1]:.1f}")
        print(f"Results saved to: {results_file}")
