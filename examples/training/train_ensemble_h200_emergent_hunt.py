#!/usr/bin/env python3
"""
Ensemble Emergent Phenomenon Hunt - 8x H200 Optimized
======================================================

Massively parallel training optimized for emergent discovery:
- 8 independent H200 training sessions (1 per GPU)
- 30 minute runtime target
- 20K nodes, 100 time steps per instance
- ~500-1000 cycles per GPU in 30 min
- Automated emergent verification
- Consolidated whitepaper generation
- Auto-commit and push results

Hardware Requirements:
- 8x NVIDIA H200 (140GB VRAM each)
- 128 CPU cores
- 1600GB system RAM

Strategy: Ensemble exploration with varied initializations to maximize
          probability of discovering novel emergent phenomena.

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import json
import time
from datetime import datetime
import subprocess

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
    fixed_point_reward = 100.0 * (temporal_fixed_pct / 100.0)

    if np.isnan(divergence) or divergence > 1e6:
        convergence_reward = -1000.0
    else:
        convergence_reward = 50.0 * np.exp(-0.1 * min(divergence, 100.0))

    tube_density = vortex_stats['vortex_tube_density']
    if 0.1 <= tube_density <= 0.3:
        tube_reward = 50.0
    else:
        deviation = abs(tube_density - 0.2)
        tube_reward = 50.0 * np.exp(-10 * deviation)

    temporal_charge = abs(vortex_stats['total_topological_charge_temporal'])
    tube_charge = abs(vortex_stats['total_topological_charge_tubes'])
    total_charge = temporal_charge + tube_charge
    charge_reward = 30.0 * np.exp(-0.1 * total_charge)

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
    max_iterations=20  # Reduced for speed
):
    """Run temporal loop iteration with vortex control (optimized)."""
    spatial_coupling = float(params['kappa'].detach()) * 0.001
    temporal_coupling = float(params['lambda'].detach()) * 0.001

    injection_rate = float(params['temporal_vortex_injection_rate'].detach())
    vortex_winding = int(params['temporal_vortex_winding'].detach())
    vortex_core_size = float(params['temporal_vortex_core_size'].detach())
    tube_probability = float(params['vortex_tube_probability'].detach())
    tube_winding = int(params['tube_winding_number'].detach())
    tube_core_size = float(params['tube_core_size'].detach())
    annihilation_rate = float(params['temporal_vortex_annihilation_rate'].detach())

    for iter_idx in range(max_iterations):
        # Temporal vortex injection
        if torch.rand(1).item() < injection_rate:
            t_idx = torch.randint(0, spacetime.num_time_steps, (1,)).item()
            spacetime.field_forward = vortex_controller.inject_temporal_vortex(
                spacetime.field_forward,
                t_idx=t_idx,
                winding_number=vortex_winding,
                core_size=vortex_core_size
            )

        # Vortex tube injection
        if torch.rand(1).item() < tube_probability:
            trajectory_length = torch.randint(5, 15, (1,)).item()
            trajectory = []
            theta_start = torch.randint(0, spacetime.num_nodes, (1,)).item()
            t_start = torch.randint(0, spacetime.num_time_steps, (1,)).item()

            for _ in range(trajectory_length):
                theta_start = (theta_start + torch.randint(-2, 3, (1,)).item()) % spacetime.num_nodes
                t_start = (t_start + torch.randint(-1, 2, (1,)).item()) % spacetime.num_time_steps
                trajectory.append((theta_start, t_start))

            spacetime.field_forward = vortex_controller.inject_spatiotemporal_vortex_tube(
                spacetime.field_forward,
                trajectory=trajectory,
                winding_number=tube_winding,
                core_size=tube_core_size
            )

        # Evolution
        spacetime.field_forward = evolver.full_forward_sweep(
            spacetime.field_forward, spatial_coupling, temporal_coupling
        )
        spacetime.field_backward = evolver.full_backward_sweep(
            spacetime.field_backward, spatial_coupling, temporal_coupling
        )

        # Retrocausal coupling
        spacetime.field_forward, spacetime.field_backward = coupler.apply_coupling(
            spacetime.field_forward,
            spacetime.field_backward,
            enable_mixing=True,
            enable_swapping=(iter_idx % 5 == 0),
            enable_anchoring=True
        )

        # Vortex annihilation
        if torch.rand(1).item() < annihilation_rate:
            temporal_vortices, _ = vortex_controller.detect_temporal_vortices(spacetime.field_forward)
            if temporal_vortices:
                t_idx = temporal_vortices[torch.randint(0, len(temporal_vortices), (1,)).item()]
                spacetime.field_forward = vortex_controller.annihilate_temporal_vortex(
                    spacetime.field_forward, t_idx=t_idx
                )

        # Möbius BCs
        spacetime.field_forward = spacetime.apply_spatial_mobius_bc(spacetime.field_forward)
        spacetime.field_forward = spacetime.apply_temporal_mobius_bc(spacetime.field_forward)
        spacetime.field_backward = spacetime.apply_spatial_mobius_bc(spacetime.field_backward)
        spacetime.field_backward = spacetime.apply_temporal_mobius_bc(spacetime.field_backward)

        # Check convergence
        divergence = spacetime.compute_divergence()
        num_fixed, pct_fixed = spacetime.compute_temporal_fixed_points()

        if divergence < 0.01 and pct_fixed > 90.0:
            break

    vortex_stats = vortex_controller.get_vortex_statistics(spacetime.field_forward)
    return divergence, num_fixed, pct_fixed, vortex_stats


def train_single_gpu(
    gpu_id,
    num_nodes,
    num_time_steps,
    num_cycles,
    seed_offset,
    output_base_dir,
    exploration_config
):
    """
    Train on a single GPU with specific exploration configuration.

    Args:
        gpu_id: GPU device ID (0-7)
        num_nodes: Number of spatial nodes
        num_time_steps: Number of temporal steps
        num_cycles: Training cycles
        seed_offset: Seed offset for this GPU
        output_base_dir: Base output directory
        exploration_config: Dictionary with exploration parameters
    """
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    seed = 42 + seed_offset
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)

    print(f"[GPU {gpu_id}] Starting training with config: {exploration_config['name']}")
    print(f"[GPU {gpu_id}] Nodes: {num_nodes}, Time steps: {num_time_steps}, Cycles: {num_cycles}")
    print(f"[GPU {gpu_id}] Device: {device}, Seed: {seed}")

    # Initialize components
    spacetime = SpatiotemporalMobiusStrip(
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        temporal_twist=exploration_config['temporal_twist'],
        device=device
    )

    evolver = TemporalEvolver(
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        relaxation_factor=exploration_config['relaxation_factor'],
        device=device
    )

    coupler = RetrocausalCoupler(
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        retrocausal_strength=exploration_config['retrocausal_strength'],
        prophetic_mixing=exploration_config['prophetic_mixing'],
        device=device
    )

    vortex_controller = TemporalVortexController(
        num_nodes=num_nodes,
        num_time_steps=num_time_steps,
        device=device
    )

    rnn = SpatiotemporalRNN(state_dim=256, hidden_dim=4096, device=device)
    optimizer = torch.optim.AdamW(
        rnn.parameters(),
        lr=exploration_config['learning_rate'],
        weight_decay=1e-5
    )

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training metrics
    metrics_history = {
        'divergences': [],
        'fixed_point_percentages': [],
        'rewards': [],
        'vortex_statistics': []
    }

    start_time = time.time()

    # Training loop
    for cycle in range(num_cycles):
        spacetime.initialize_self_consistent(seed=seed + cycle)

        state_tensor = spacetime.get_state_tensor()
        state_input = state_tensor.unsqueeze(0).unsqueeze(0)

        # Forward pass with AMP
        with torch.cuda.amp.autocast():
            params, _ = rnn(state_input)
            params_rescaled = rnn.rescale_parameters(params)

        # Temporal loop
        divergence, num_fixed, pct_fixed, vortex_stats = temporal_loop_with_vortex_control(
            spacetime, evolver, coupler, vortex_controller, params_rescaled, max_iterations=20
        )

        # Compute reward and loss
        reward = compute_reward(spacetime, vortex_controller, pct_fixed, divergence, vortex_stats)
        reward_value = reward.item()

        value_prediction = params['value']
        target_value = torch.tensor(pct_fixed / 100.0, dtype=torch.float32, device=device)
        loss = torch.nn.functional.mse_loss(value_prediction, target_value)

        # Backward pass with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Log metrics
        metrics_history['divergences'].append(divergence)
        metrics_history['fixed_point_percentages'].append(pct_fixed)
        metrics_history['rewards'].append(reward_value)
        metrics_history['vortex_statistics'].append({
            'temporal_vortex_count': vortex_stats['temporal_vortex_count'],
            'temporal_vortex_density': vortex_stats['temporal_vortex_density'],
            'vortex_tube_count': vortex_stats['vortex_tube_count'],
            'vortex_tube_density': vortex_stats['vortex_tube_density'],
            'avg_tube_length': vortex_stats['avg_tube_length']
        })

        # Print progress every 50 cycles
        if (cycle + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[GPU {gpu_id}] Cycle {cycle+1}/{num_cycles} ({elapsed/60:.1f}min) - "
                  f"Fixed: {pct_fixed:.1f}%, Tubes: {vortex_stats['vortex_tube_count']}, "
                  f"Reward: {reward_value:.2f}")

    total_time = time.time() - start_time

    # Save results
    output_dir = Path(output_base_dir) / f"gpu_{gpu_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f'training_{timestamp}.json'

    final_results = {
        'gpu_id': gpu_id,
        'exploration_config': exploration_config,
        'config': {
            'num_nodes': num_nodes,
            'num_time_steps': num_time_steps,
            'num_cycles': num_cycles,
            'seed': seed
        },
        'metrics': metrics_history,
        'final': {
            'divergence': divergence,
            'fixed_points': num_fixed,
            'fixed_point_pct': pct_fixed,
            'reward': reward_value,
            'vortex_statistics': vortex_stats
        },
        'timing': {
            'total_seconds': total_time,
            'cycles_per_second': num_cycles / total_time
        }
    }

    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Save checkpoint
    checkpoint_file = output_dir / f'checkpoint_{timestamp}.pt'
    torch.save({
        'rnn_state_dict': rnn.state_dict(),
        'final_field_forward': spacetime.field_forward.cpu(),
        'final_field_backward': spacetime.field_backward.cpu()
    }, checkpoint_file)

    print(f"[GPU {gpu_id}] Training complete - {total_time/60:.1f} minutes")
    print(f"[GPU {gpu_id}] Results: {results_file}")

    return results_file, final_results


def run_emergent_verification_ensemble(results_dir):
    """Run emergent verification on all ensemble results."""
    print("\n" + "="*80)
    print("ENSEMBLE EMERGENT VERIFICATION")
    print("="*80)
    print()

    results_dir = Path(results_dir)
    all_results = []
    verification_results = []

    # Load all GPU results
    for gpu_dir in sorted(results_dir.glob("gpu_*")):
        result_files = list(gpu_dir.glob("training_*.json"))
        if result_files:
            with open(result_files[0]) as f:
                result = json.load(f)
                all_results.append(result)
                print(f"Loaded results from {gpu_dir.name}")

    if not all_results:
        print("No results found for verification")
        return

    # Find best result (highest peak fixed point percentage)
    best_result = max(
        all_results,
        key=lambda r: max(r['metrics']['fixed_point_percentages'])
    )

    print(f"\nBest result: GPU {best_result['gpu_id']}")
    print(f"  Peak fixed points: {max(best_result['metrics']['fixed_point_percentages']):.1f}%")
    print(f"  Final reward: {best_result['final']['reward']:.2f}")
    print()

    # Verify top 3 results
    top_results = sorted(
        all_results,
        key=lambda r: max(r['metrics']['fixed_point_percentages']),
        reverse=True
    )[:3]

    for idx, result in enumerate(top_results):
        print(f"Verifying result {idx+1}/3 (GPU {result['gpu_id']})...")

        # Load checkpoint
        gpu_dir = results_dir / f"gpu_{result['gpu_id']}"
        checkpoint_files = list(gpu_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            print(f"  No checkpoint found, skipping")
            continue

        checkpoint = torch.load(checkpoint_files[0], map_location='cpu')

        # Combine fields for verification
        combined_field = torch.stack([
            checkpoint['final_field_forward'].T,
            checkpoint['final_field_backward'].T
        ], dim=0).mean(dim=0)

        # Run verification
        verifier = EmergentVerifier(data_dir="data")

        discovery_data = {
            'phenomenon_name': f'Ensemble Discovery - GPU {result["gpu_id"]}',
            'training_run': str(gpu_dir),
            'discovery_cycle': result['config']['num_cycles'] - 1,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'random_seed': result['config']['seed'],
            'hardware': {
                'device': f'H200 GPU {result["gpu_id"]}',
                'gpu_name': 'NVIDIA H200',
                'vram_gb': 140,
                'hardware_tier': 'H200',
                'ensemble_id': result['gpu_id']
            },
            'system_size': {
                'nodes': result['config']['num_nodes'],
                'time_steps': result['config']['num_time_steps'],
                'total_dof': result['config']['num_nodes'] * result['config']['num_time_steps']
            },
            'parameters': result['exploration_config'],
            'key_metrics': result['final']
        }

        verification = verifier.verify_phenomenon(
            field_tensor=combined_field,
            phenomenon_type='auto',
            save_results=True,
            output_dir=str(gpu_dir / "verification")
        )

        verification_results.append({
            'gpu_id': result['gpu_id'],
            'novelty_score': verification['novelty_score'],
            'is_novel': verification['is_novel'],
            'discovery_data': discovery_data,
            'verification': verification
        })

        print(f"  Novelty score: {verification['novelty_score']:.3f}")
        print(f"  Is novel: {verification['is_novel']}")
        print()

    return verification_results, best_result


def generate_consolidated_whitepaper(verification_results, results_dir):
    """Generate consolidated whitepaper for ensemble discoveries."""
    print("="*80)
    print("GENERATING CONSOLIDATED WHITEPAPER")
    print("="*80)
    print()

    # Find most novel result
    most_novel = max(verification_results, key=lambda r: r['novelty_score'])

    print(f"Most novel discovery: GPU {most_novel['gpu_id']}")
    print(f"  Novelty score: {most_novel['novelty_score']:.3f}")
    print()

    # Generate whitepaper for most novel
    generator = EmergentWhitepaperGenerator()

    try:
        whitepaper_path = generator.generate(
            phenomenon_name=most_novel['discovery_data']['phenomenon_name'],
            discovery_data=most_novel['discovery_data'],
            verification_results=most_novel['verification'],
            output_dir=str(Path(results_dir) / "whitepapers" / "EMERGENTS"),
            compile_pdf=True
        )

        print(f"Whitepaper generated: {whitepaper_path}")
        print()

        return whitepaper_path
    except Exception as e:
        print(f"Whitepaper generation failed: {e}")
        return None


def main():
    """Run ensemble emergent hunt."""
    print("="*80)
    print("ENSEMBLE EMERGENT PHENOMENON HUNT")
    print("8x H200 Massively Parallel Training")
    print("="*80)
    print()

    # Configuration
    num_gpus = 8
    num_nodes = 20000  # 20K nodes per GPU
    num_time_steps = 100
    target_runtime_minutes = 30
    estimated_cycles = 800  # ~500-1000 cycles in 30 min with optimization

    output_base_dir = Path('results/ensemble_emergent_hunt')
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Exploration configurations (varied across GPUs)
    exploration_configs = [
        {
            'name': 'Standard',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.7,
            'relaxation_factor': 0.3,
            'prophetic_mixing': 0.3,
            'learning_rate': 1e-4
        },
        {
            'name': 'High Retrocausal',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.9,
            'relaxation_factor': 0.3,
            'prophetic_mixing': 0.5,
            'learning_rate': 1e-4
        },
        {
            'name': 'Low Twist',
            'temporal_twist': np.pi * 0.5,
            'retrocausal_strength': 0.7,
            'relaxation_factor': 0.3,
            'prophetic_mixing': 0.3,
            'learning_rate': 1e-4
        },
        {
            'name': 'High Mixing',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.7,
            'relaxation_factor': 0.3,
            'prophetic_mixing': 0.7,
            'learning_rate': 1e-4
        },
        {
            'name': 'Low Relaxation',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.7,
            'relaxation_factor': 0.1,
            'prophetic_mixing': 0.3,
            'learning_rate': 1e-4
        },
        {
            'name': 'High Learning Rate',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.7,
            'relaxation_factor': 0.3,
            'prophetic_mixing': 0.3,
            'learning_rate': 5e-4
        },
        {
            'name': 'Extreme Retrocausal',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.95,
            'relaxation_factor': 0.2,
            'prophetic_mixing': 0.8,
            'learning_rate': 1e-4
        },
        {
            'name': 'Conservative',
            'temporal_twist': np.pi,
            'retrocausal_strength': 0.5,
            'relaxation_factor': 0.5,
            'prophetic_mixing': 0.2,
            'learning_rate': 5e-5
        }
    ]

    print("Ensemble Configuration:")
    for i, config in enumerate(exploration_configs):
        print(f"  GPU {i}: {config['name']}")
    print()
    print(f"Per-GPU: {num_nodes} nodes, {num_time_steps} time steps, {estimated_cycles} cycles")
    print(f"Target runtime: {target_runtime_minutes} minutes")
    print()

    # Launch parallel training
    print("="*80)
    print("LAUNCHING PARALLEL TRAINING")
    print("="*80)
    print()

    mp.set_start_method('spawn', force=True)

    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=train_single_gpu,
            args=(
                gpu_id,
                num_nodes,
                num_time_steps,
                estimated_cycles,
                gpu_id * 1000,
                output_base_dir,
                exploration_configs[gpu_id]
            )
        )
        p.start()
        processes.append(p)

    # Wait for all to complete
    for p in processes:
        p.join()

    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE")
    print("="*80)
    print()

    # Run verification
    verification_results, best_result = run_emergent_verification_ensemble(output_base_dir)

    # Generate whitepaper
    if verification_results:
        whitepaper_path = generate_consolidated_whitepaper(verification_results, output_base_dir)

    # Git commit and push
    print("="*80)
    print("COMMITTING RESULTS TO GIT")
    print("="*80)
    print()

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Add results
        subprocess.run(['git', 'add', str(output_base_dir)], check=True)

        # Commit message
        commit_msg = f"""feat: ensemble emergent hunt results - 8x H200 30min run

Massively parallel training across 8 H200 GPUs:
- {num_nodes} nodes × {num_time_steps} time steps per GPU
- {estimated_cycles} cycles per GPU
- {num_gpus} exploration configurations

Best result:
- GPU {best_result['gpu_id']} ({exploration_configs[best_result['gpu_id']]['name']})
- Peak fixed points: {max(best_result['metrics']['fixed_point_percentages']):.1f}%
- Final reward: {best_result['final']['reward']:.2f}

Emergent verification:
- {len([v for v in verification_results if v['is_novel']])} novel discoveries
- Top novelty score: {max(v['novelty_score'] for v in verification_results):.3f}

Results: {output_base_dir}
Timestamp: {timestamp}
"""

        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)

        # Push
        subprocess.run(['git', 'push'], check=True)

        print("Results committed and pushed successfully!")
        print()

    except subprocess.CalledProcessError as e:
        print(f"Git operation failed: {e}")
        print("Results saved locally but not pushed to remote")
        print()

    print("="*80)
    print("ENSEMBLE EMERGENT HUNT COMPLETE")
    print("="*80)
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
