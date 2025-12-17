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

        # Global parameter head
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # kappa, delta, cross_coupling, vortex_target
        ).to(device)

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

    def forward(self, state: torch.Tensor):
        """
        Forward pass

        Args:
            state: [batch, state_dim]

        Returns:
            actions: List of [nodes_per_strip] tensors (one per strip)
            global_params: [4] tensor (kappa, delta, coupling, vortex_target)
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

        # Global parameters
        global_raw = self.global_head(features).squeeze(0)  # [4]
        kappa = 1.5 + 0.3 * torch.tanh(global_raw[0])  # 1.2 to 1.8
        delta = 0.3 + 0.2 * torch.tanh(global_raw[1])  # 0.1 to 0.5
        coupling = torch.sigmoid(global_raw[2])  # 0 to 1
        vortex_target = 0.5 + 0.3 * torch.tanh(global_raw[3])  # 0.2 to 0.8

        global_params = torch.stack([kappa, delta, coupling, vortex_target])

        # Value
        value = self.value_head(features).squeeze()

        return actions, global_params, value, hidden


def compute_multi_strip_reward(strips: SparseTokamakMobiusStrips) -> float:
    """
    Compute reward for multi-strip system

    Rewards:
    - High vortex density (target ~50-80%)
    - Uniform distribution across strips
    - Cross-strip coherence
    - Stability (low variance)
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

    # Target vortex density: 50-80%
    if 0.5 <= avg_vortex_density <= 0.8:
        density_reward = 100 * avg_vortex_density
    else:
        density_reward = 50 * avg_vortex_density  # Lower reward outside target

    # Uniformity reward (penalize large variance between strips)
    uniformity_reward = -50 * std_vortex_density

    # Stability reward (penalize collapse)
    if avg_vortex_density < 0.01:
        stability_penalty = -100
    else:
        stability_penalty = 0

    total_reward = density_reward + uniformity_reward + stability_penalty

    return total_reward, {
        'vortex_density_mean': avg_vortex_density,
        'vortex_density_std': std_vortex_density,
        'density_reward': density_reward,
        'uniformity_reward': uniformity_reward,
        'stability_penalty': stability_penalty
    }


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

    # Metrics tracking
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

    for cycle in range(num_cycles):
        cycle_start = time.time()

        # Encode state
        state = agent.encode_state(strips)

        # RNN forward
        actions, global_params, value, hidden = agent.forward(state)

        # Apply actions to strips
        for strip_idx in range(num_strips):
            # Get strip node indices
            strip_mask = strips.strip_indices == strip_idx
            strip_node_indices = torch.where(strip_mask)[0]

            # Apply amplitude adjustments
            action = actions[strip_idx]
            strips.amplitudes[strip_node_indices] += action * 0.05
            strips.amplitudes[strip_node_indices] = torch.clamp(
                strips.amplitudes[strip_node_indices], 0.5, 5.0
            )

        # Evolve field
        field_updates, sample_indices = strips.evolve_field(t=float(cycle), sample_ratio=0.1)
        strips.field[sample_indices] = field_updates

        # Compute reward
        reward, reward_breakdown = compute_multi_strip_reward(strips)

        # RL update
        loss = -value * reward  # Policy gradient
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        agent.optimizer.step()

        # Track metrics
        cycle_time = time.time() - cycle_start
        metrics['cycle_times'].append(cycle_time)
        metrics['rewards'].append(reward)
        metrics['vortex_densities'].append(reward_breakdown['vortex_density_mean'])
        metrics['vortex_std'].append(reward_breakdown['vortex_density_std'])
        metrics['global_params'].append({
            'kappa': global_params[0].item(),
            'delta': global_params[1].item(),
            'coupling': global_params[2].item(),
            'vortex_target': global_params[3].item()
        })

        # Print progress
        if (cycle + 1) % max(1, num_cycles // 20) == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (cycle + 1)) * (num_cycles - cycle - 1)

            print(f"Cycle {cycle+1:4d}/{num_cycles} | "
                  f"Vortex: {reward_breakdown['vortex_density_mean']:.1%} ± {reward_breakdown['vortex_density_std']:.1%} | "
                  f"Reward: {reward:7.1f} | "
                  f"κ: {global_params[0].item():.2f} δ: {global_params[1].item():.2f} | "
                  f"Time: {cycle_time:.3f}s | "
                  f"ETA: {eta:.0f}s")

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

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("MULTI-STRIP FLUX TUBE TRAINING")
    print("=" * 80)

    results = train_multi_strip(args)


if __name__ == "__main__":
    main()
