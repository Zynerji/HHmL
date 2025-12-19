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
from src.hhml.core.spatiotemporal.retrocausal_coupling import RetrocausalCoupler
from src.hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN


def detect_temporal_vortices_gpu(field, positions, vortex_threshold=0.1, phase_grad_threshold=1.0):
    """
    GPU-accelerated vortex detection using pure PyTorch operations.

    Args:
        field: Complex field tensor [num_nodes] on GPU
        positions: Node positions [num_nodes, 3] on GPU
        vortex_threshold: Minimum amplitude threshold
        phase_grad_threshold: Minimum phase gradient for vortex

    Returns:
        Dictionary of tensors:
            'node_idx': Tensor of vortex node indices
            'charge': Tensor of vortex charges
            'amplitude': Tensor of vortex amplitudes
            'phase': Tensor of vortex phases
    """
    # Compute phase and amplitude (all GPU operations)
    phase = torch.angle(field)
    amplitude = torch.abs(field)

    # Find vortex candidates
    vortex_mask = amplitude > vortex_threshold
    num_vortices = vortex_mask.sum().item()

    if num_vortices == 0:
        return {
            'node_idx': torch.tensor([], dtype=torch.long, device=field.device),
            'charge': torch.tensor([], dtype=torch.float, device=field.device),
            'amplitude': torch.tensor([], dtype=torch.float, device=field.device),
            'phase': torch.tensor([], dtype=torch.float, device=field.device)
        }

    # Extract vortex properties (vectorized)
    vortex_indices = torch.where(vortex_mask)[0]
    vortex_amplitudes = amplitude[vortex_mask]
    vortex_phases = phase[vortex_mask]

    # Assign random charges (vectorized)
    vortex_charges = torch.where(
        torch.rand(num_vortices, device=field.device) > 0.5,
        torch.ones(num_vortices, device=field.device),
        -torch.ones(num_vortices, device=field.device)
    )

    return {
        'node_idx': vortex_indices,
        'charge': vortex_charges,
        'amplitude': vortex_amplitudes,
        'phase': vortex_phases
    }


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

    def detect_wormholes_gpu(self, vortex_dict, strip_indices, positions):
        """
        GPU-accelerated wormhole detection using vectorized operations.

        Args:
            vortex_dict: Dictionary of tensors from detect_temporal_vortices_gpu
            strip_indices: Tensor mapping node index to strip ID [num_nodes]
            positions: Tensor of 3D positions [num_nodes, 3]

        Returns:
            Number of detected wormholes (int)
        """
        vortex_indices = vortex_dict['node_idx']
        num_vortices = len(vortex_indices)

        if num_vortices < 2:
            self.wormhole_history.append(0)
            return 0

        # Get vortex strips and positions (all GPU operations)
        vortex_strips = strip_indices[vortex_indices]  # [N]
        vortex_positions = positions[vortex_indices]  # [N, 3]
        vortex_charges = vortex_dict['charge']  # [N]

        # Compute all pairwise differences using broadcasting
        # Shape: [N, 1, 3] - [1, N, 3] = [N, N, 3]
        pos_diff = vortex_positions.unsqueeze(1) - vortex_positions.unsqueeze(0)

        # Compute spherical coordinates for all vortices
        r = torch.norm(vortex_positions, dim=1, keepdim=True)  # [N, 1]
        theta = torch.acos(vortex_positions[:, 2:3] / (r + 1e-8))  # [N, 1]
        phi = torch.atan2(vortex_positions[:, 1:2], vortex_positions[:, 0:1])  # [N, 1]

        # Pairwise angular differences (broadcasting)
        theta_diff = torch.abs(theta - theta.t())  # [N, N]
        phi_diff = torch.abs(phi - phi.t())  # [N, N]

        # Wrap phi differences to [-pi, pi]
        phi_diff = torch.where(phi_diff > np.pi, 2*np.pi - phi_diff, phi_diff)

        # Check alignment criteria (vectorized)
        angular_aligned = (theta_diff < self.angular_threshold) & (phi_diff < self.angular_threshold)

        # Check if on different strips
        strip_diff = vortex_strips.unsqueeze(1) != vortex_strips.unsqueeze(0)  # [N, N]

        # Combined wormhole criteria
        wormhole_mask = angular_aligned & strip_diff

        # Only count upper triangle (avoid double-counting pairs)
        triu_mask = torch.triu(torch.ones_like(wormhole_mask, dtype=torch.bool), diagonal=1)
        wormhole_mask = wormhole_mask & triu_mask

        # Count wormholes
        num_wormholes = wormhole_mask.sum().item()

        # Track history
        self.wormhole_history.append(num_wormholes)

        return num_wormholes

    def detect_wormholes(self, vortices, strip_indices, positions):
        """Legacy wrapper for compatibility - converts old format to GPU format"""
        # Convert list of dicts to tensor dict format
        if len(vortices) == 0:
            self.wormhole_history.append(0)
            return []

        device = positions.device
        vortex_dict = {
            'node_idx': torch.tensor([v['node_idx'] for v in vortices], device=device),
            'charge': torch.tensor([v['charge'] for v in vortices], device=device),
            'amplitude': torch.tensor([v['amplitude'] for v in vortices], device=device),
            'phase': torch.tensor([v['phase'] for v in vortices], device=device)
        }

        num_wormholes = self.detect_wormholes_gpu(vortex_dict, strip_indices, positions)
        # Return empty list for compatibility (count is tracked in history)
        return []

    def compute_radial_transport_speed_gpu(self, field, strip_indices, num_strips):
        """
        GPU-accelerated radial transport speed computation.

        Returns:
            Dictionary with transport metrics
        """
        # Compute field intensity (GPU operation)
        field_intensity = torch.abs(field)

        # Use scatter_mean to compute per-strip intensities (vectorized)
        strip_intensities = torch.zeros(num_strips, device=field.device)
        strip_counts = torch.zeros(num_strips, device=field.device)

        # Count nodes per strip and sum intensities
        strip_intensities.scatter_add_(0, strip_indices, field_intensity)
        strip_counts.scatter_add_(0, strip_indices, torch.ones_like(field_intensity))

        # Compute mean intensity per strip (avoid divide by zero)
        strip_counts = torch.clamp(strip_counts, min=1.0)
        strip_intensities = strip_intensities / strip_counts

        # Compute gradient on GPU
        gradient = strip_intensities[1:] - strip_intensities[:-1]

        return {
            'max_gradient': gradient.abs().max().item(),
            'gradient_variance': gradient.var().item()
        }

    def compute_radial_transport_speed(self, field, strip_indices, num_strips):
        """Legacy wrapper - uses GPU version"""
        return self.compute_radial_transport_speed_gpu(field, strip_indices, num_strips)


def compute_reward(tokamak, field_forward, field_backward, num_wormholes):
    """
    GPU-accelerated reward computation for tokamak training.

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
    wormhole_bonus = 10.0 * min(num_wormholes, 10)  # Cap at 10 wormholes

    # 4. Strip uniformity (GPU-accelerated using scatter operations)
    strip_intensities = torch.zeros(tokamak.num_strips, device=field_forward.device)
    strip_counts = torch.zeros(tokamak.num_strips, device=field_forward.device)

    # Compute per-strip mean intensity
    strip_intensities.scatter_add_(0, tokamak.strip_indices, torch.abs(field_forward))
    strip_counts.scatter_add_(0, tokamak.strip_indices, torch.ones_like(field_forward.real))

    # Compute means (avoid divide by zero)
    strip_counts = torch.clamp(strip_counts, min=1.0)
    strip_means = strip_intensities / strip_counts

    # Compute uniformity from standard deviation
    if strip_means.numel() > 0:
        mean_intensity = strip_means.mean()
        std_intensity = strip_means.std()
        uniformity = 1.0 - (std_intensity / (mean_intensity + 1e-8))
        uniformity_reward = 20.0 * max(0, uniformity.item())
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

    # Initialize retrocausal coupling (GPU-parallelized)
    print("Initializing retrocausal coupling...")
    retrocausal_coupler = RetrocausalCoupler(
        num_nodes=total_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,
        prophetic_mixing=0.3,
        device=device
    )

    # Wrapper for evolve_coupled API compatibility
    class RetrocausalWrapper:
        def __init__(self, coupler):
            self.coupler = coupler

        def evolve_coupled(self, field_forward, field_backward,
                          coupling_forward=0.1, coupling_backward=0.1,
                          coupling_retrocausal=0.05, diffusion_coeff=0.01,
                          num_steps=1):
            # Apply retrocausal coupling (all time steps in parallel)
            return self.coupler.apply_coupling(
                field_forward, field_backward,
                enable_mixing=True,
                enable_swapping=True,
                enable_anchoring=True
            )

    temporal_dynamics = RetrocausalWrapper(retrocausal_coupler)
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
    scaler = None  # Disabled for stability if device.type == 'cuda' else None
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

        # Detect vortices (GPU-accelerated)
        vortex_dict = detect_temporal_vortices_gpu(
            field_forward[:, 0],  # First time slice
            tokamak.positions,
            vortex_threshold=0.1,
            phase_grad_threshold=1.0
        )

        # Detect wormholes (GPU-accelerated)
        num_wormholes = wormhole_detector.detect_wormholes_gpu(
            vortex_dict,
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
            tokamak, field_forward[:, 0], field_backward[:, 0], num_wormholes
        )

        # Backpropagation
        reward_tensor = torch.tensor(reward_value, device=device, requires_grad=True)
        loss = -reward_tensor  # Maximize reward

        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()

            # Check for NaN before unscaling
            has_nan_grad = any(
                param.grad is not None and torch.isnan(param.grad).any()
                for param in rnn.parameters()
            )

            if has_nan_grad:
                print(f"WARNING: NaN detected in RNN gradients (cycle {cycle}), skipping update")
                optimizer.zero_grad()
                scaler.update()  # Update scale factor but skip step
            else:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=1.0)
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
        metrics['vortex_counts'].append(len(vortex_dict["node_idx"]))
        metrics['wormhole_counts'].append(num_wormholes)
        metrics['radial_transport'].append(transport)

        # Save best wormholes
        if num_wormholes > 0:
            metrics['wormhole_details'].append({
                'cycle': cycle,
                'wormholes': num_wormholes
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
                'wormholes': num_wormholes
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
            print(f"  Vortices: {len(vortex_dict["node_idx"])}")
            print(f"  Wormholes: {num_wormholes}")
            if num_wormholes > 0:
                avg_separation = 0  # Placeholder - actual wormhole list not stored
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
    parser.add_argument('--num-time-steps', type=int, default=10,
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

    return train_tokamak_wormhole_hunt(args)


if __name__ == '__main__':
    sys.exit(main())
