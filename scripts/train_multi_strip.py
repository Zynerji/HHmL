#!/usr/bin/env python3
"""
Multi-Strip Flux Tube Training with RNN Control
================================================
Train RNN to control multi-scale tokamak-style Möbius strips.

Auto-scales from CPU (2 strips, 2K nodes) to H200 (20 strips, 500K nodes).

Key Features:
- Hardware auto-detection and parameter scaling
- RNN controls per-strip parameters (amplitudes, phases, frequencies)
- Cross-strip coupling learned via RL
- Sparse/dense mode auto-switching
- Vortex collision tracking
- Real-time performance monitoring

Usage:
    # Auto-detect hardware and run
    python train_multi_strip.py

    # Force specific mode
    python train_multi_strip.py --mode benchmark  # quick test
    python train_multi_strip.py --mode training   # full training
    python train_multi_strip.py --mode production  # H200 max scale

    # Custom parameters
    python train_multi_strip.py --strips 4 --nodes 10000 --cycles 500

Author: HHmL Framework
Date: 2025-12-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
import argparse
import json

from hhml.utils.hardware_config import HardwareConfig
from hhml.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
from hhml.mobius.helical_vortex_optimizer import helical_vortex_reset, compute_vortex_stability_score


class MultiStripRNNAgent(nn.Module):
    """
    RNN agent for controlling multi-strip flux tube system

    Learns to optimize:
    - Per-strip wave amplitudes
    - Per-strip phases
    - Per-strip frequencies
    - Cross-strip coupling strengths
    - Global geometry parameters (kappa, delta)
    """

    def __init__(self, num_strips: int, nodes_per_strip: int, hidden_dim: int = 512, device: str = 'cpu'):
        super().__init__()

        self.num_strips = num_strips
        self.nodes_per_strip = nodes_per_strip
        self.total_nodes = num_strips * nodes_per_strip
        self.hidden_dim = hidden_dim
        self.device = device

        # State encoding: field magnitude + phase per strip
        state_dim_per_strip = 64  # Sample 64 nodes per strip
        self.state_dim = num_strips * state_dim_per_strip * 2  # mag + phase

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=self.state_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True
        ).to(device)

        # Action heads (one per strip)
        self.amplitude_heads = nn.ModuleList([
            nn.Linear(hidden_dim, nodes_per_strip) for _ in range(num_strips)
        ]).to(device)

        # UNIFIED CONTROL HEAD - RNN controls ALL parameters for full glass-box system
        # This enables tracking correlations between any RNN output and emergent phenomena
        self.control_head = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 19)  # ALL control parameters (see mapping below)
        ).to(device)

        # Control parameter indices:
        # [0-3]   Geometry: kappa, delta, vortex_target, num_qec_layers
        # [4-7]   Physics: damping, nonlinearity, amp_variance, vortex_seed_strength
        # [8-10]  Spectral: omega, diffusion_dt, reset_strength
        # [11-13] Sampling: sample_ratio, max_neighbors_factor, sparsity_threshold
        # [14-15] Mode: sparse_density, spectral_weight
        # [16-18] Geometry2: winding_density, twist_rate, cross_coupling

        # Value head (for RL)
        self.value_head = nn.Linear(hidden_dim, 1).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"\nMulti-Strip RNN Agent:")
        print(f"  Strips: {num_strips}")
        print(f"  Nodes per strip: {nodes_per_strip:,}")
        print(f"  State dim: {self.state_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Parameters: {num_params:,}")

    def encode_state(self, strips: SparseTokamakMobiusStrips) -> torch.Tensor:
        """
        Encode multi-strip field state

        Returns: [batch=1, state_dim]
        """
        states = []

        for strip_idx in range(self.num_strips):
            # Get field for this strip
            field = strips.get_strip_field(strip_idx)

            # Sample 64 nodes
            sample_size = min(64, len(field))
            sample_indices = torch.randperm(len(field))[:sample_size]
            field_sample = field[sample_indices]

            # Pad if needed
            if len(field_sample) < 64:
                padding = torch.zeros(64 - len(field_sample), dtype=torch.complex64, device=self.device)
                field_sample = torch.cat([field_sample, padding])

            # Extract magnitude and phase
            mag = torch.abs(field_sample)
            phase = torch.angle(field_sample)

            states.append(mag)
            states.append(phase)

        # Concatenate all strips
        state = torch.cat(states)  # [state_dim]
        return state.unsqueeze(0)  # [1, state_dim]

    def map_control_params(self, raw_params: torch.Tensor) -> dict:
        """
        Map raw RNN outputs to proper parameter ranges

        Args:
            raw_params: [19] tensor of raw outputs

        Returns:
            Dictionary with all control parameters properly scaled
        """
        # Geometry parameters [0-3]
        kappa = 1.0 + 1.0 * torch.sigmoid(raw_params[0])  # 1.0 to 2.0
        delta = 0.0 + 0.5 * torch.sigmoid(raw_params[1])  # 0.0 to 0.5
        vortex_target = 0.5 + 0.3 * torch.sigmoid(raw_params[2])  # 0.5 to 0.8
        num_qec_layers = 1 + 9 * torch.sigmoid(raw_params[3])  # 1 to 10

        # Physics parameters [4-7]
        damping = 0.01 + 0.19 * torch.sigmoid(raw_params[4])  # 0.01 to 0.2
        nonlinearity = 2.0 * torch.tanh(raw_params[5])  # -2.0 to 2.0
        amp_variance = 0.1 + 2.9 * torch.sigmoid(raw_params[6])  # 0.1 to 3.0
        vortex_seed_strength = torch.sigmoid(raw_params[7])  # 0.0 to 1.0

        # Spectral parameters [8-10]
        omega = 0.1 + 0.9 * torch.sigmoid(raw_params[8])  # 0.1 to 1.0
        diffusion_dt = 0.01 + 0.49 * torch.sigmoid(raw_params[9])  # 0.01 to 0.5
        reset_strength = torch.sigmoid(raw_params[10])  # 0.0 to 1.0

        # Sampling parameters [11-13]
        sample_ratio = 0.01 + 0.49 * torch.sigmoid(raw_params[11])  # 0.01 to 0.5
        max_neighbors_factor = 0.1 + 1.9 * torch.sigmoid(raw_params[12])  # 0.1 to 2.0 (multiply by base)
        sparsity_threshold = 0.1 + 0.4 * torch.sigmoid(raw_params[13])  # 0.1 to 0.5

        # Mode selection [14-15]
        sparse_density = torch.sigmoid(raw_params[14])  # 0.0 to 1.0 (0=dense, 1=sparse)
        spectral_weight = torch.sigmoid(raw_params[15])  # 0.0 to 1.0 (0=spatial, 1=spectral)

        # Geometry extended [16-18]
        winding_density = 0.5 + 2.0 * torch.sigmoid(raw_params[16])  # 0.5 to 2.5
        twist_rate = 0.5 + 1.5 * torch.sigmoid(raw_params[17])  # 0.5 to 2.0
        cross_coupling = torch.sigmoid(raw_params[18])  # 0.0 to 1.0

        return {
            'kappa': kappa,
            'delta': delta,
            'vortex_target': vortex_target,
            'num_qec_layers': num_qec_layers,
            'damping': damping,
            'nonlinearity': nonlinearity,
            'amp_variance': amp_variance,
            'vortex_seed_strength': vortex_seed_strength,
            'omega': omega,
            'diffusion_dt': diffusion_dt,
            'reset_strength': reset_strength,
            'sample_ratio': sample_ratio,
            'max_neighbors_factor': max_neighbors_factor,
            'sparsity_threshold': sparsity_threshold,
            'sparse_density': sparse_density,
            'spectral_weight': spectral_weight,
            'winding_density': winding_density,
            'twist_rate': twist_rate,
            'cross_coupling': cross_coupling
        }

    def forward(self, state: torch.Tensor):
        """
        Forward pass

        Args:
            state: [batch, state_dim]

        Returns:
            actions: List of [nodes_per_strip] tensors (one per strip)
            control_params: Dictionary with all 19 control parameters
            value: [1] scalar
            hidden: LSTM hidden state
        """
        # LSTM
        lstm_out, hidden = self.lstm(state.unsqueeze(1))  # [batch, 1, hidden]
        features = lstm_out.squeeze(1)  # [batch, hidden]

        # Per-strip actions (amplitude adjustments)
        actions = []
        for i in range(self.num_strips):
            action = torch.tanh(self.amplitude_heads[i](features))  # [-1, 1]
            actions.append(action.squeeze(0))  # [nodes_per_strip]

        # UNIFIED CONTROL - RNN controls ALL parameters
        control_raw = self.control_head(features).squeeze(0)  # [19]
        control_params = self.map_control_params(control_raw)

        # Value
        value = self.value_head(features).squeeze()

        return actions, control_params, value, hidden


def compute_multi_strip_reward(strips: SparseTokamakMobiusStrips, global_params: torch.Tensor = None,
                              param_history: list = None) -> float:
    """
    Compute reward for multi-strip system

    Rewards:
    - High vortex density (target ~50-80%)
    - Uniform distribution across strips
    - Cross-strip coherence
    - Stability (low variance)
    - Exploration bonus (parameter variation)
    """
    # Get field for each strip
    vortex_densities = []
    field_magnitudes = []

    for strip_idx in range(strips.num_strips):
        field = strips.get_strip_field(strip_idx)
        field_mag = torch.abs(field)

        # Vortex density (low field regions)
        vortex_mask = field_mag < 0.3
        vortex_density = vortex_mask.float().mean().item()
        vortex_densities.append(vortex_density)

        # Average field magnitude
        avg_mag = field_mag.mean().item()
        field_magnitudes.append(avg_mag)

    # Reward components
    avg_vortex_density = np.mean(vortex_densities)
    std_vortex_density = np.std(vortex_densities)

    # Target vortex density: 50-80% (ENHANCED: smoother reward curve)
    if 0.5 <= avg_vortex_density <= 0.8:
        density_reward = 200 * avg_vortex_density  # Higher reward in target range
    elif avg_vortex_density > 0.8:
        # Soft penalty for exceeding target
        density_reward = 160 - 50 * (avg_vortex_density - 0.8)
    else:
        # Strong penalty for low vortex density (prevent collapse)
        density_reward = 100 * avg_vortex_density - 150 * (0.5 - avg_vortex_density)**2

    # Uniformity reward (penalize large variance between strips)
    uniformity_reward = -50 * std_vortex_density

    # Stability reward (ENHANCED: gradual penalty for collapse)
    if avg_vortex_density < 0.1:
        stability_penalty = -200 * (0.1 - avg_vortex_density)  # Stronger penalty
    else:
        stability_penalty = 0

    # Exploration bonus (encourage parameter variation)
    exploration_bonus = 0
    if global_params is not None and param_history is not None and len(param_history) > 5:
        # Calculate variance of recent kappa and delta values
        recent_params = np.array(param_history[-10:])  # Last 10 cycles
        kappa_std = np.std(recent_params[:, 0])
        delta_std = np.std(recent_params[:, 1])
        exploration_bonus = 10 * (kappa_std + delta_std)  # Reward exploration

    # SPECTRAL GRAPH REWARD: Connectivity and clustering quality
    # Use graph Laplacian properties to measure vortex configuration quality
    spectral_bonus = 0
    try:
        # Get edge connectivity (proxy for spectral gap)
        if hasattr(strips, 'edge_index') and strips.edge_index.shape[1] > 0:
            # Measure graph connectivity: avg degree / total_nodes
            avg_degree = strips.edge_index.shape[1] / strips.total_nodes
            # Reward intermediate connectivity (not too sparse, not too dense)
            target_degree = 500  # Optimal for 4K nodes
            connectivity_score = 1.0 - abs(avg_degree - target_degree) / target_degree
            spectral_bonus = 20 * max(0, connectivity_score)
    except:
        pass  # Skip if graph not available

    total_reward = density_reward + uniformity_reward + stability_penalty + exploration_bonus + spectral_bonus

    # NaN protection: return large negative reward if NaN detected
    if np.isnan(total_reward) or np.isinf(total_reward):
        print(f"  [WARNING] NaN/Inf reward detected, using penalty")
        total_reward = -1000.0
        density_reward = -1000.0

    return total_reward, {
        'vortex_density_mean': avg_vortex_density,
        'vortex_density_std': std_vortex_density,
        'density_reward': density_reward,
        'uniformity_reward': uniformity_reward,
        'stability_penalty': stability_penalty,
        'exploration_bonus': exploration_bonus,
        'spectral_bonus': spectral_bonus
    }


def save_checkpoint(agent, cycle, metrics, args, checkpoint_path):
    """
    Save training checkpoint for sequential learning

    Args:
        agent: MultiStripRNNAgent instance
        cycle: Current cycle number
        metrics: Training metrics dictionary
        args: Command-line arguments
        checkpoint_path: Path to save checkpoint
    """
    checkpoint = {
        'cycle': cycle,
        'agent_state_dict': agent.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'metrics': {
            'rewards': metrics['rewards'],
            'vortex_densities': metrics['vortex_densities'],
            'param_history': metrics['param_history'],
        },
        'config': {
            'num_strips': agent.num_strips,
            'nodes_per_strip': agent.nodes_per_strip,
            'hidden_dim': agent.hidden_dim,
            'state_dim': agent.state_dim,
        },
        'timestamp': datetime.now().isoformat(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, agent):
    """
    Load training checkpoint for sequential learning

    Args:
        checkpoint_path: Path to checkpoint file
        agent: MultiStripRNNAgent instance

    Returns:
        start_cycle: Cycle number to resume from
        metrics: Previous training metrics
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=agent.device, weights_only=False)

    # Verify configuration matches
    config = checkpoint['config']
    if (config['num_strips'] != agent.num_strips or
        config['nodes_per_strip'] != agent.nodes_per_strip):
        raise ValueError(
            f"Checkpoint config mismatch: "
            f"expected ({agent.num_strips} strips, {agent.nodes_per_strip} nodes/strip), "
            f"got ({config['num_strips']} strips, {config['nodes_per_strip']} nodes/strip)"
        )

    # Load model and optimizer state
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load metrics
    metrics = {
        'rewards': checkpoint['metrics']['rewards'],
        'vortex_densities': checkpoint['metrics']['vortex_densities'],
        'param_history': checkpoint['metrics']['param_history'],
    }

    start_cycle = checkpoint['cycle'] + 1

    print(f"  Checkpoint loaded: {checkpoint_path}")
    print(f"  Resuming from cycle: {start_cycle}")
    print(f"  Previous training: {checkpoint['cycle']} cycles")
    print(f"  Checkpoint created: {checkpoint['timestamp']}")

    return start_cycle, metrics


def train_multi_strip(args):
    """Main training loop"""

    # Hardware detection
    hw_config = HardwareConfig()
    hw_config.print_info()

    # Get optimal parameters for mode
    if args.strips and args.nodes:
        # Custom parameters
        num_strips = args.strips
        nodes_per_strip = args.nodes
        hidden_dim = args.hidden_dim
    else:
        # Auto-detect optimal parameters
        params = hw_config.get_optimal_params(args.mode)
        num_strips = params.num_strips
        nodes_per_strip = params.nodes_per_strip
        hidden_dim = params.hidden_dim

    device = hw_config.device_type
    num_cycles = args.cycles

    print(f"\nTraining Configuration:")
    print(f"  Strips: {num_strips}")
    print(f"  Nodes per strip: {nodes_per_strip:,}")
    print(f"  Total nodes: {num_strips * nodes_per_strip:,}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Cycles: {num_cycles}")
    print(f"  Device: {device.upper()}")

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

    if args.resume:
        # Resume from checkpoint
        print("\n" + "=" * 80)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 80)
        start_cycle, loaded_metrics = load_checkpoint(args.resume, agent)

        # Initialize metrics from checkpoint
        metrics = {
            'cycle_times': [],
            'rewards': loaded_metrics['rewards'],
            'vortex_densities': loaded_metrics['vortex_densities'],
            'vortex_std': [],
            'global_params': [],
            'mode': 'sparse' if strips.use_sparse else 'dense'
        }
        param_history = loaded_metrics['param_history']
    else:
        # Start fresh
        metrics = {
            'cycle_times': [],
            'rewards': [],
            'vortex_densities': [],
            'vortex_std': [],
            'global_params': [],
            'mode': 'sparse' if strips.use_sparse else 'dense'
        }

    print("\n" + "=" * 80)
    print("STARTING MULTI-STRIP TRAINING")
    print("=" * 80)

    start_time = time.time()
    total_cycles = start_cycle + num_cycles

    for cycle in range(start_cycle, total_cycles):
        cycle_start = time.time()

        # Encode state
        state = agent.encode_state(strips)

        # RNN forward (UNIFIED: all 19 control parameters)
        actions, control_params, value, hidden = agent.forward(state)

        # Add exploration noise (anneal over time)
        exploration_noise = 0.1 * (1.0 - cycle / total_cycles)  # Decay from 0.1 to 0
        if exploration_noise > 0:
            # Add noise to all control parameters
            for key in control_params:
                control_params[key] = control_params[key] + torch.randn_like(control_params[key]) * exploration_noise * 0.3

        # Extract all control parameters (convert to Python floats/ints)
        cp = {k: v.item() for k, v in control_params.items()}  # Detach and convert to scalars

        # Apply actions to strips (detach to avoid backprop through environment)
        for strip_idx in range(num_strips):
            # Get strip node indices
            strip_mask = strips.strip_indices == strip_idx
            strip_node_indices = torch.where(strip_mask)[0]

            # Apply amplitude adjustments (RNN-controlled variance)
            action = actions[strip_idx].detach()
            strips.amplitudes[strip_node_indices] += action * 0.05 * cp['amp_variance']
            strips.amplitudes[strip_node_indices] = torch.clamp(
                strips.amplitudes[strip_node_indices], 0.1, 5.0
            )

        # Apply vortex seeding (RNN-controlled)
        if cp['vortex_seed_strength'] > 0.3:
            num_seeds = int(cp['vortex_seed_strength'] * 100)
            seed_indices = torch.randperm(strips.total_nodes, device=device)[:num_seeds]
            strips.field[seed_indices] *= 0.1  # Create vortex cores

        # Evolve field with RNN-controlled physics and spectral parameters
        field_updates, sample_indices = strips.evolve_field(
            t=float(cycle),
            sample_ratio=cp['sample_ratio'],  # RNN-controlled
            damping=cp['damping'],  # RNN-controlled
            nonlinearity=cp['nonlinearity'],  # RNN-controlled
            omega=cp['omega'],  # RNN-controlled spectral frequency
            diffusion_dt=cp['diffusion_dt'],  # RNN-controlled diffusion timestep
            spectral_weight=cp['spectral_weight']  # RNN-controlled blend (0=spatial, 1=spectral)
        )

        # NaN protection: sanitize field updates
        field_updates = torch.nan_to_num(field_updates, nan=0.0, posinf=1.0, neginf=-1.0)
        strips.field[sample_indices] = field_updates

        # Helical vortex reset (RNN-controlled, if density too low)
        current_vortex_density = (torch.abs(strips.field) < cp['sparsity_threshold']).float().mean().item()
        if current_vortex_density < 0.05 and cycle % 20 == 10:
            # Apply RNN-controlled helical spectral reset
            strips.field = helical_vortex_reset(
                positions=strips.positions,
                field=strips.field,
                omega=cp['omega'],  # RNN-controlled
                vortex_target=cp['vortex_target'],  # RNN-controlled
                reset_strength=cp['reset_strength']  # RNN-controlled
            )

        # Track ALL control parameters for correlation analysis
        param_history.append(cp.copy())  # Store full parameter dict

        # Compute reward
        reward, reward_breakdown = compute_multi_strip_reward(
            strips,
            global_params=None,  # No longer needed, using full control_params
            param_history=param_history
        )

        # RL update (convert reward to tensor for gradient computation)
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device)
        loss = -value * reward_tensor  # Policy gradient

        # NaN protection: skip optimizer step if NaN/Inf detected
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"  [WARNING] NaN/Inf loss detected at cycle {cycle+1}, skipping optimizer step")
        else:
            agent.optimizer.zero_grad()
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            agent.optimizer.step()

        # Track metrics
        cycle_time = time.time() - cycle_start
        metrics['cycle_times'].append(cycle_time)
        metrics['rewards'].append(reward)
        metrics['vortex_densities'].append(reward_breakdown['vortex_density_mean'])
        metrics['vortex_std'].append(reward_breakdown['vortex_density_std'])
        # Track ALL 19 control parameters for full correlation analysis
        metrics['global_params'].append(cp.copy())

        # Print progress
        if (cycle + 1) % max(1, num_cycles // 20) == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (cycle + 1)) * (num_cycles - cycle - 1)

            # Display key control parameters (subset of 19)
            print(f"Cycle {cycle+1:4d}/{total_cycles} | "
                  f"Vortex: {reward_breakdown['vortex_density_mean']:.1%} +/- {reward_breakdown['vortex_density_std']:.1%} | "
                  f"Reward: {reward:7.1f} | "
                  f"om: {cp['omega']:.2f} spec: {cp['spectral_weight']:.2f} | "
                  f"QEC: {cp['num_qec_layers']:.1f} | "
                  f"Time: {cycle_time:.3f}s | "
                  f"ETA: {eta:.0f}s")

        # Periodic checkpoint saving
        if args.save_every > 0 and (cycle + 1) % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_cycle_{cycle+1}.pt"
            save_checkpoint(agent, cycle, {'rewards': metrics['rewards'],
                                          'vortex_densities': metrics['vortex_densities'],
                                          'param_history': param_history},
                          args, checkpoint_path)

    total_time = time.time() - start_time

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f"Average cycle time: {np.mean(metrics['cycle_times']):.3f}s")
    print(f"Cycles per second: {num_cycles / total_time:.2f}")
    print(f"Final vortex density: {metrics['vortex_densities'][-1]:.1%}")
    print(f"Final reward: {metrics['rewards'][-1]:.1f}")
    print(f"Peak vortex density: {max(metrics['vortex_densities']):.1%}")
    print("=" * 80)

    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / f"checkpoint_final_cycle_{total_cycles-1}.pt"
    save_checkpoint(agent, total_cycles - 1,
                   {'rewards': metrics['rewards'],
                    'vortex_densities': metrics['vortex_densities'],
                    'param_history': param_history},
                   args, final_checkpoint_path)

    # Save results
    output_dir = Path("results/multi_strip_training")
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
            'device': device
        },
        'performance': {
            'total_time': total_time,
            'avg_cycle_time': float(np.mean(metrics['cycle_times'])),
            'cycles_per_second': num_cycles / total_time
        },
        'final_state': {
            'vortex_density': float(metrics['vortex_densities'][-1]),
            'vortex_std': float(metrics['vortex_std'][-1]),
            'reward': float(metrics['rewards'][-1]),
            'global_params': metrics['global_params'][-1]
        },
        'metrics': {
            'vortex_densities': [float(x) for x in metrics['vortex_densities']],
            'rewards': [float(x) for x in metrics['rewards']],
            'cycle_times': [float(x) for x in metrics['cycle_times'][-100:]]  # Last 100 only
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Strip Flux Tube Training')

    # Mode selection
    parser.add_argument('--mode', choices=['benchmark', 'training', 'production'],
                        default='training',
                        help='Training mode (auto-scales parameters)')

    # Custom parameters (override mode defaults)
    parser.add_argument('--strips', type=int, default=None,
                        help='Number of Möbius strips')
    parser.add_argument('--nodes', type=int, default=None,
                        help='Nodes per strip')
    parser.add_argument('--cycles', type=int, default=200,
                        help='Number of training cycles')
    parser.add_argument('--hidden-dim', type=int, default=512,
                        help='RNN hidden dimension')

    # Checkpoint management
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-every', type=int, default=0,
                        help='Save checkpoint every N cycles (0 = only at end)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MULTI-STRIP FLUX TUBE TRAINING")
    print("=" * 80)

    results = train_multi_strip(args)


if __name__ == "__main__":
    main()
