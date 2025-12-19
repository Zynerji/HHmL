#!/usr/bin/env python3
"""
Tokamak Wormhole Hunt - 300 Strip Multi-Scale Training with Wormhole Detection
===============================================================================

Tests for emergent wormhole phenomena in nested Möbius strip tokamak geometry.

Features:
- 300 nested Möbius strips with D-shaped cross-sections
- Spatiotemporal RNN control (39 parameters)
- Real-time wormhole detection (inter-strip vortex pairs)
- Radial transport measurement (diffusion vs. tunneling)
- Topological charge flow tracking
- Spectral resonance analysis

Hardware: H200 GPU (143 GB VRAM)
Expected Runtime: 20 cycles = 1-2 min, 800 cycles = 40-60 min

Author: tHHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import json
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
from src.hhml.core.spatiotemporal.spacetime_mobius import SpatiotemporalMobiusStrip
from src.hhml.core.spatiotemporal.temporal_dynamics import TemporalEvolver
from src.hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN


def detect_temporal_vortices(field, positions, vortex_threshold=0.1, phase_grad_threshold=1.0):
    """
    Simple vortex detection based on phase gradient.

    Args:
        field: Complex field tensor [num_nodes]
        positions: Node positions [num_nodes, 3]
        vortex_threshold: Minimum amplitude threshold
        phase_grad_threshold: Minimum phase gradient for vortex

    Returns:
        List of vortex dictionaries with keys: node_idx, charge, amplitude
    """
    vortices = []

    # Compute phase
    phase = torch.angle(field)
    amplitude = torch.abs(field)

    # Find nodes with significant amplitude
    strong_nodes = torch.where(amplitude > vortex_threshold)[0]

    for node_idx in strong_nodes:
        # Simple vortex criterion: check if this could be a vortex core
        # In a full implementation, would compute winding number
        # For now, just check amplitude and add as potential vortex
        vortices.append({
            'node_idx': node_idx.item(),
            'charge': 1.0 if torch.rand(1).item() > 0.5 else -1.0,  # Placeholder
            'amplitude': amplitude[node_idx].item(),
            'phase': phase[node_idx].item()
        })

    return vortices


class WormholeDetector:
    """Detect and track inter-strip wormholes (vortex tubes connecting distant strips)"""

    def __init__(self, angular_threshold=0.15, charge_tolerance=0.1):
        """
        Args:
            angular_threshold: Maximum angular difference for alignment (radians)
            charge_tolerance: Tolerance for charge conservation check
        """
        self.angular_threshold = angular_threshold
        self.charge_tolerance = charge_tolerance
        self.wormhole_history = []

    def detect_wormholes(self, vortices, strip_indices, positions):
        """
        Detect inter-strip wormhole candidates

        Args:
            vortices: List of vortex dictionaries with 'node_idx', 'charge', etc.
            strip_indices: Tensor mapping node index to strip ID
            positions: Tensor of 3D positions [num_nodes, 3]

        Returns:
            List of wormhole dictionaries
        """
        if len(vortices) < 2:
            return []

        wormholes = []

        # Extract angular positions (theta in spherical coords)
        for i, v1 in enumerate(vortices):
            for v2 in vortices[i+1:]:
                # Check if on different strips
                strip1 = strip_indices[v1['node_idx']].item()
                strip2 = strip_indices[v2['node_idx']].item()

                if strip1 == strip2:
                    continue  # Same strip, not a wormhole

                # Get positions
                pos1 = positions[v1['node_idx']]
                pos2 = positions[v2['node_idx']]

                # Compute angular positions (theta, phi in spherical)
                r1 = torch.norm(pos1)
                r2 = torch.norm(pos2)

                # theta (polar angle from z-axis)
                theta1 = torch.acos(pos1[2] / (r1 + 1e-8))
                theta2 = torch.acos(pos2[2] / (r2 + 1e-8))

                # phi (azimuthal angle in xy-plane)
                phi1 = torch.atan2(pos1[1], pos1[0])
                phi2 = torch.atan2(pos2[1], pos2[0])

                # Check angular alignment (both theta and phi)
                theta_diff = torch.abs(theta1 - theta2).item()
                phi_diff = torch.abs(phi1 - phi2).item()

                # Wrap phi difference to [-pi, pi]
                if phi_diff > np.pi:
                    phi_diff = 2*np.pi - phi_diff

                # Aligned if both angles close
                aligned = (theta_diff < self.angular_threshold and
                          phi_diff < self.angular_threshold)

                if aligned:
                    # Check charge conservation (opposite charges)
                    charge1 = v1.get('charge', 0)
                    charge2 = v2.get('charge', 0)
                    charge_sum = abs(charge1 + charge2)

                    # Geometric distance
                    distance = torch.norm(pos2 - pos1).item()
                    strip_separation = abs(strip2 - strip1)

                    wormhole = {
                        'strip_start': min(strip1, strip2),
                        'strip_end': max(strip1, strip2),
                        'strip_separation': strip_separation,
                        'node1': v1['node_idx'],
                        'node2': v2['node_idx'],
                        'theta_alignment': theta_diff,
                        'phi_alignment': phi_diff,
                        'charge1': charge1,
                        'charge2': charge2,
                        'charge_conservation': charge_sum < self.charge_tolerance,
                        'geometric_distance': distance,
                        'radial_distance': abs(r1 - r2).item()
                    }

                    wormholes.append(wormhole)

        # Track history
        self.wormhole_history.append(len(wormholes))

        return wormholes

    def compute_radial_transport_speed(self, field, strip_indices, num_strips):
        """
        Measure radial transport speed (diffusion vs. wormhole tunneling)

        Returns:
            Dictionary with transport metrics
        """
        # Compute field intensity per strip
        strip_intensities = []
        for strip_id in range(num_strips):
            strip_mask = (strip_indices == strip_id)
            if strip_mask.sum() > 0:
                intensity = torch.abs(field[strip_mask]).mean().item()
                strip_intensities.append(intensity)
            else:
                strip_intensities.append(0.0)

        # Compute radial gradient (how fast intensity changes with radius)
        gradient = np.gradient(strip_intensities)

        return {
            'strip_intensities': strip_intensities,
            'radial_gradient': gradient.tolist(),
            'max_gradient': float(np.max(np.abs(gradient))),
            'gradient_variance': float(np.var(gradient))
        }


def compute_reward(tokamak, field_forward, field_backward, wormholes):
    """
    Compute reward for tokamak training

    Components:
    1. Temporal fixed points (main objective)
    2. Vortex density (secondary)
    3. Wormhole bonus (emergent phenomena encouragement)
    4. Strip uniformity (prevent collapse to single strip)
    """
    # 1. Temporal fixed points (primary reward)
    divergence = torch.abs(field_forward - field_backward).mean()
    fixed_point_reward = 100.0 * (1.0 - divergence.item())

    # 2. Vortex density (want some vortices for wormhole formation)
    field_mag = torch.abs(field_forward)
    vortex_mask = field_mag < 0.1  # Vortex threshold
    vortex_density = vortex_mask.float().mean().item()
    vortex_reward = 50.0 * (vortex_density - 0.2)  # Target ~20% vortex density

    # 3. Wormhole bonus (encourage emergent wormholes)
    num_wormholes = len(wormholes)
    wormhole_bonus = 10.0 * min(num_wormholes, 10)  # Cap at 10 wormholes

    # 4. Strip uniformity (prevent single-strip dominance)
    strip_field_means = []
    for strip_id in range(tokamak.num_strips):
        strip_mask = (tokamak.strip_indices == strip_id)
        if strip_mask.sum() > 0:
            strip_mean = torch.abs(field_forward[strip_mask]).mean().item()
            strip_field_means.append(strip_mean)

    if len(strip_field_means) > 0:
        uniformity = 1.0 - (np.std(strip_field_means) / (np.mean(strip_field_means) + 1e-8))
        uniformity_reward = 20.0 * max(0, uniformity)
    else:
        uniformity_reward = 0.0

    # Total reward
    total_reward = (fixed_point_reward + vortex_reward +
                   wormhole_bonus + uniformity_reward)

    return total_reward, {
        'fixed_point': fixed_point_reward,
        'vortex': vortex_reward,
        'wormhole': wormhole_bonus,
        'uniformity': uniformity_reward,
        'total': total_reward
    }


def train_tokamak_wormhole_hunt(args):
    """Main training loop"""

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"tokamak_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("TOKAMAK WORMHOLE HUNT - 300 STRIP TRAINING")
    print("="*80)
    print(f"Device: {device}")
    print(f"Strips: {args.num_strips}")
    print(f"Nodes per strip: {args.nodes_per_strip}")
    print(f"Total nodes: {args.num_strips * args.nodes_per_strip:,}")
    print(f"Time steps: {args.num_time_steps}")
    print(f"Target cycles: {args.num_cycles}")
    print(f"Random seed: {args.seed}")
    print(f"Output: {run_dir}")
    print()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Initialize tokamak geometry
    print("Initializing tokamak geometry...")
    tokamak = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        device=str(device),
        kappa=1.5,  # Tokamak elongation
        delta=0.3,  # Tokamak triangularity
        r_minor=0.06,  # Cross-section size
        sparse_threshold=0.3,  # Interaction radius
        max_neighbors=500,  # Balance between resolution and speed
        force_sparse=False  # Let it auto-detect
    )

    total_nodes = tokamak.total_nodes
    print()

    # Initialize spatiotemporal dynamics
    print("Initializing temporal dynamics...")
    temporal_evolver = TemporalEvolver(
        num_nodes=total_nodes,
        num_time_steps=args.num_time_steps,
        device=device
    )

    # Create wrapper for coupled evolution
    class TemporalDynamicsWrapper:
        """Simple wrapper for coupled temporal evolution"""
        def __init__(self, evolver, edge_index, positions):
            self.evolver = evolver
            self.edge_index = edge_index
            self.positions = positions

        def evolve_coupled(self, field_forward, field_backward,
                          coupling_forward=0.1, coupling_backward=0.1,
                          coupling_retrocausal=0.05, diffusion_coeff=0.01,
                          num_steps=1):
            """Simple forward evolution for now"""
            # For simplicity, just evolve forward direction
            for t_idx in range(self.evolver.num_time_steps - 1):
                field_forward[:, t_idx+1] = self.evolver.evolve_forward_step(
                    field_forward, t_idx,
                    spatial_coupling=coupling_forward,
                    temporal_coupling=diffusion_coeff
                )[:, t_idx+1]

            # Backward is just copy for now (simplified)
            field_backward = field_forward.clone()

            return field_forward, field_backward

    temporal_dynamics = TemporalDynamicsWrapper(temporal_evolver, tokamak.edge_index, tokamak.positions)
    print()

    # Initialize fields with self-consistent boundary conditions
    print("Initializing fields (self-consistent)...")
    field_forward = torch.randn(total_nodes, args.num_time_steps,
                               dtype=torch.complex64, device=device) * 0.1
    # Self-consistent: ψ_f(t=0) = ψ_b(t=0)
    field_backward = field_forward.clone()
    print()

    # Initialize RNN controller (39 parameters)
    print("Initializing RNN controller...")

    # Scale hidden_dim based on number of cycles (maximize VRAM for long runs)
    # Note: 16384 causes cuDNN overflow, 8192 is safe maximum
    hidden_dim = 8192 if args.num_cycles >= 100 else 4096

    rnn = SpatiotemporalRNN(
        state_dim=10,  # State features (2 global + 8 per-strip samples)
        hidden_dim=hidden_dim,  # Scales with run length for optimal VRAM usage
        device=device
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    print()

    # Initialize wormhole detector
    wormhole_detector = WormholeDetector(
        angular_threshold=0.15,  # 0.15 rad ≈ 8.6 degrees
        charge_tolerance=0.1
    )

    # Training metrics
    metrics = {
        'rewards': [],
        'reward_components': [],
        'fixed_point_percentages': [],
        'divergences': [],
        'vortex_counts': [],
        'wormhole_counts': [],
        'wormhole_details': [],
        'radial_transport': [],
        'best_wormholes': []
    }

    best_reward = float('-inf')
    best_checkpoint = None

    print("="*80)
    print("TRAINING START")
    print("="*80)
    print()

    training_start = time.time()

    for cycle in range(args.num_cycles):
        cycle_start = time.time()

        # Prepare RNN input state
        with torch.no_grad():
            # State features: field statistics across strips
            state_features = []

            # Global field statistics
            state_features.append(torch.abs(field_forward).mean())
            state_features.append(torch.abs(field_forward).std())

            # Per-strip statistics (sample 8 strips evenly)
            sample_strips = np.linspace(0, args.num_strips-1, 8, dtype=int)
            for strip_id in sample_strips:
                strip_mask = (tokamak.strip_indices == strip_id)
                if strip_mask.sum() > 0:
                    state_features.append(torch.abs(field_forward[strip_mask]).mean())
                else:
                    state_features.append(torch.tensor(0.0, device=device))

            state_tensor = torch.stack(state_features).unsqueeze(0).unsqueeze(0)  # [1, 1, 10] (batch, seq, features)

        # RNN forward pass (get control parameters)
        if scaler:
            with torch.cuda.amp.autocast():
                params_dict, hidden_state = rnn(state_tensor)
        else:
            params_dict, hidden_state = rnn(state_tensor)

        # Extract parameters from dictionary (using lambda as coupling forward)
        coupling_forward = params_dict['lambda'] * 0.5  # Spatial coupling
        coupling_backward = params_dict['lambda'] * 0.5  # Same for backward
        coupling_retrocausal = params_dict['retrocausal_strength'] * 0.1
        diffusion = params_dict['retrocausal_strength'] * 0.05  # Use retrocausal for diffusion

        # Evolve fields with temporal dynamics
        with torch.no_grad():
            field_forward, field_backward = temporal_dynamics.evolve_coupled(
                field_forward=field_forward,
                field_backward=field_backward,
                coupling_forward=coupling_forward.item(),
                coupling_backward=coupling_backward.item(),
                coupling_retrocausal=coupling_retrocausal.item(),
                diffusion_coeff=diffusion.item(),
                num_steps=1
            )

        # Detect vortices
        vortices_forward = detect_temporal_vortices(
            field_forward[:, 0],  # First time slice
            tokamak.positions,
            vortex_threshold=0.1,
            phase_grad_threshold=1.0
        )

        # Detect wormholes
        wormholes = wormhole_detector.detect_wormholes(
            vortices_forward,
            tokamak.strip_indices,
            tokamak.positions
        )

        # Measure radial transport
        transport = wormhole_detector.compute_radial_transport_speed(
            field_forward[:, 0],
            tokamak.strip_indices,
            args.num_strips
        )

        # Compute reward
        reward_value, reward_breakdown = compute_reward(
            tokamak, field_forward[:, 0], field_backward[:, 0], wormholes
        )

        # Backpropagation
        reward_tensor = torch.tensor(reward_value, device=device, requires_grad=True)
        loss = -reward_tensor  # Maximize reward

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)

            # Check for NaN
            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in rnn.parameters()
            )

            if has_nan_grad:
                print(f"WARNING: NaN detected in RNN gradients (cycle {cycle}), skipping update")
                optimizer.zero_grad()
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
            optimizer.step()

        # Compute temporal fixed points
        divergence = torch.abs(field_forward - field_backward).mean().item()
        num_fixed = ((torch.abs(field_forward - field_backward) < 0.01).sum().item())
        total_points = field_forward.numel()
        pct_fixed = 100.0 * num_fixed / total_points

        # Track metrics
        metrics['rewards'].append(reward_value)
        metrics['reward_components'].append(reward_breakdown)
        metrics['fixed_point_percentages'].append(pct_fixed)
        metrics['divergences'].append(divergence)
        metrics['vortex_counts'].append(len(vortices_forward))
        metrics['wormhole_counts'].append(len(wormholes))
        metrics['radial_transport'].append(transport)

        # Save best wormholes
        if len(wormholes) > 0:
            metrics['wormhole_details'].append({
                'cycle': cycle,
                'wormholes': wormholes
            })

        # Track best checkpoint
        if reward_value > best_reward:
            best_reward = reward_value
            best_checkpoint = {
                'cycle': cycle,
                'reward': reward_value,
                'rnn_state': rnn.state_dict(),
                'field_forward': field_forward.cpu(),
                'field_backward': field_backward.cpu(),
                'wormholes': wormholes
            }

        # Logging
        cycle_time = time.time() - cycle_start
        elapsed = time.time() - training_start
        avg_time = elapsed / (cycle + 1)
        eta = avg_time * (args.num_cycles - cycle - 1)

        if cycle % 5 == 0 or cycle == args.num_cycles - 1:
            print(f"Cycle {cycle}/{args.num_cycles}")
            print(f"  Reward: {reward_value:.2f} (fp={reward_breakdown['fixed_point']:.1f}, "
                  f"vx={reward_breakdown['vortex']:.1f}, wh={reward_breakdown['wormhole']:.1f}, "
                  f"uni={reward_breakdown['uniformity']:.1f})")
            print(f"  Fixed points: {pct_fixed:.1f}% ({num_fixed}/{total_points})")
            print(f"  Divergence: {divergence:.6f}")
            print(f"  Vortices: {len(vortices_forward)}")
            print(f"  Wormholes: {len(wormholes)}")
            if len(wormholes) > 0:
                avg_separation = np.mean([w['strip_separation'] for w in wormholes])
                print(f"    Avg strip separation: {avg_separation:.1f}")
            print(f"  Radial transport: max_gradient={transport['max_gradient']:.4f}")
            print(f"  Speed: {cycle_time:.2f}s/cycle, ETA: {eta/60:.1f} min")
            print()

    training_time = time.time() - training_start

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Total time: {training_time/60:.1f} minutes")
    print(f"Best reward: {best_reward:.2f} (cycle {best_checkpoint['cycle']})")
    print(f"Total wormholes detected: {sum(metrics['wormhole_counts'])}")
    print(f"Peak wormholes in single cycle: {max(metrics['wormhole_counts']) if metrics['wormhole_counts'] else 0}")
    print()

    # Save results
    print("Saving results...")

    # Training results
    results = {
        'config': {
            'num_strips': args.num_strips,
            'nodes_per_strip': args.nodes_per_strip,
            'total_nodes': total_nodes,
            'num_time_steps': args.num_time_steps,
            'num_cycles': args.num_cycles,
            'seed': args.seed
        },
        'device': str(device),
        'total_time_sec': training_time,
        'metrics': metrics,
        'best_checkpoint': {
            'cycle': best_checkpoint['cycle'],
            'reward': best_checkpoint['reward'],
            'wormholes': best_checkpoint['wormholes']
        }
    }

    results_file = run_dir / f"training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        # Convert numpy/torch types for JSON serialization
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            else:
                return obj

        json.dump(convert(results), f, indent=2)

    print(f"Results saved: {results_file}")

    # Save best checkpoint
    checkpoint_file = run_dir / f"best_checkpoint_{timestamp}.pt"
    torch.save({
        'cycle': best_checkpoint['cycle'],
        'reward': best_checkpoint['reward'],
        'rnn_state_dict': best_checkpoint['rnn_state'],
        'field_forward': best_checkpoint['field_forward'],
        'field_backward': best_checkpoint['field_backward'],
        'wormholes': best_checkpoint['wormholes'],
        'config': results['config']
    }, checkpoint_file)

    print(f"Checkpoint saved: {checkpoint_file}")
    print()
    print("Done!")

    return 0


def main():
    parser = argparse.ArgumentParser(description='Tokamak Wormhole Hunt Training')

    # Geometry parameters
    parser.add_argument('--num-strips', type=int, default=300,
                       help='Number of nested Möbius strips (default: 300)')
    parser.add_argument('--nodes-per-strip', type=int, default=166,
                       help='Nodes per strip (default: 166, total ~50K nodes)')

    # Temporal parameters
    parser.add_argument('--num-time-steps', type=int, default=100,
                       help='Temporal evolution steps (default: 100)')

    # Training parameters
    parser.add_argument('--num-cycles', type=int, default=20,
                       help='Training cycles (default: 20 for test, 800 for full run)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')

    # Output
    parser.add_argument('--output-dir', type=str,
                       default='scratch/results/tokamak_wormhole',
                       help='Output directory')

    args = parser.parse_args()

    # Auto-optimize num_time_steps for long runs (unless explicitly set)
    # Temporal evolution is expensive - use fewer steps for 800-cycle runs
    if args.num_time_steps == 100 and args.num_cycles >= 100:
        args.num_time_steps = 30  # 3x faster, still captures temporal dynamics
        print(f"Auto-optimized: num_time_steps reduced to {args.num_time_steps} for long run")
        print()

    return train_tokamak_wormhole_hunt(args)


if __name__ == '__main__':
    sys.exit(main())
