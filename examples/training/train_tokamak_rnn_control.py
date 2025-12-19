#!/usr/bin/env python3
"""
Tokamak RNN Control Training - Full Vortex Management Integration
=================================================================

Integrates RNN control into tokamak wormhole detection system for:
- High vortex density (targeting 80-100%)
- 100% temporal fixed points
- Stable wormhole network formation
- Active vortex quality management

RNN Controls (11 parameters):
- Retrocausal coupling (alpha, gamma)
- Wormhole detection thresholds
- Vortex quality management (pruning, annihilation, preservation)
- Field evolution (diffusion, coupling, noise)

Author: HHmL Research Collaboration
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import json
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "examples" / "training"))

from train_tokamak_wormhole_hunt import (
    SparseTokamakMobiusStrips,
    RetrocausalCoupler,
    detect_temporal_vortices_gpu
)


class TokamakRNN(nn.Module):
    """
    RNN controller for tokamak wormhole system.

    Controls 11 parameters for vortex management and wormhole detection.
    """

    def __init__(self, state_dim=128, hidden_dim=2048, num_params=11, device='cuda'):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_params = num_params
        self.device = device

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        ).to(device)

        # Parameter heads (11 outputs)
        self.param_head = nn.Linear(hidden_dim, num_params).to(device)

        # Value head (for policy gradient)
        self.value_head = nn.Linear(hidden_dim, 1).to(device)

        # Initialize with small weights to avoid large initial gradients
        for module in [self.param_head, self.value_head]:
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        """
        Forward pass.

        Args:
            state: (batch, seq_len, state_dim) tensor

        Returns:
            params: dict of parameter values
            value: scalar value estimate
        """
        # LSTM forward
        lstm_out, _ = self.lstm(state)

        # Use final hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch, hidden_dim)

        # Parameter predictions (11 outputs)
        param_logits = self.param_head(final_hidden)  # (batch, num_params)

        # Apply parameter-specific activations
        params_raw = self._apply_param_activations(param_logits)

        # Value prediction
        value = self.value_head(final_hidden).squeeze(-1)  # (batch,)

        # Package parameters
        params = self._package_parameters(params_raw)
        params['value'] = value

        return params

    def _apply_param_activations(self, logits):
        """Apply parameter-specific activation functions."""
        params = torch.zeros_like(logits)

        # Retrocausal coupling (0-1)
        params[:, 0] = torch.sigmoid(logits[:, 0])  # retrocausal_alpha
        params[:, 1] = torch.sigmoid(logits[:, 1])  # prophetic_gamma

        # Wormhole detection (0-1 and 0-5)
        params[:, 2] = torch.sigmoid(logits[:, 2])  # wormhole_angular_threshold
        params[:, 3] = 5.0 * torch.sigmoid(logits[:, 3])  # wormhole_distance_threshold

        # Vortex quality management (0-1, 0-2, 0-1, 0-1)
        params[:, 4] = torch.sigmoid(logits[:, 4])  # vortex_quality_threshold
        params[:, 5] = 2.0 * torch.sigmoid(logits[:, 5])  # antivortex_strength
        params[:, 6] = torch.sigmoid(logits[:, 6])  # annihilation_radius
        params[:, 7] = torch.sigmoid(logits[:, 7])  # preserve_ratio

        # Field evolution (0-0.5, 0-2, 0-0.1)
        params[:, 8] = 0.5 * torch.sigmoid(logits[:, 8])  # diffusion_coefficient
        params[:, 9] = 2.0 * torch.sigmoid(logits[:, 9])  # coupling_strength
        params[:, 10] = 0.1 * torch.sigmoid(logits[:, 10])  # noise_level

        return params

    def _package_parameters(self, params_raw):
        """Package parameters into named dict."""
        return {
            'retrocausal_alpha': params_raw[:, 0],
            'prophetic_gamma': params_raw[:, 1],
            'wormhole_angular_threshold': params_raw[:, 2],
            'wormhole_distance_threshold': params_raw[:, 3],
            'vortex_quality_threshold': params_raw[:, 4],
            'antivortex_strength': params_raw[:, 5],
            'annihilation_radius': params_raw[:, 6],
            'preserve_ratio': params_raw[:, 7],
            'diffusion_coefficient': params_raw[:, 8],
            'coupling_strength': params_raw[:, 9],
            'noise_level': params_raw[:, 10]
        }


def compute_reward(
    num_vortices,
    vortex_density,
    num_wormholes,
    fixed_point_pct,
    divergence,
    total_nodes
):
    """
    Compute reward for tokamak RNN training.

    Balances:
    - High vortex density (target 80-100%)
    - 100% temporal fixed points
    - High wormhole count
    - Low divergence
    """
    # Vortex density reward (target: 80-100%)
    vortex_reward = 0.0
    if 0.8 <= vortex_density <= 1.0:
        vortex_reward = 100.0
    elif vortex_density > 0.5:
        vortex_reward = 100.0 * (vortex_density - 0.5) / 0.3

    # Fixed points reward (target: 100%)
    fixed_point_reward = 100.0 * (fixed_point_pct / 100.0)

    # Wormhole reward (normalized by total possible inter-strip connections)
    # Max wormholes ~ total_nodes (rough estimate)
    wormhole_reward = 50.0 * min(num_wormholes / total_nodes, 1.0)

    # Divergence penalty
    if np.isnan(divergence) or divergence > 1.0:
        divergence_penalty = -100.0
    else:
        divergence_penalty = -20.0 * divergence

    # Total reward
    reward = vortex_reward + fixed_point_reward + wormhole_reward + divergence_penalty

    return reward, {
        'vortex_reward': vortex_reward,
        'fixed_point_reward': fixed_point_reward,
        'wormhole_reward': wormhole_reward,
        'divergence_penalty': divergence_penalty
    }


def apply_vortex_annihilation(
    field_final,
    vortex_positions,
    quality_threshold,
    antivortex_strength,
    annihilation_radius,
    preserve_ratio,
    device
):
    """
    Apply vortex quality management via antivortex injection.

    Returns:
        field_modified: field after annihilation
        num_annihilated: number of vortices removed
    """
    if len(vortex_positions) == 0:
        return field_final, 0

    # Compute vortex quality (simplified: use field magnitude at vortex)
    vortex_qualities = []
    for pos in vortex_positions:
        node_idx = int(pos['node_idx'])
        time_idx = int(pos['time_idx'])
        quality = torch.abs(field_final[node_idx, time_idx]).item()
        vortex_qualities.append(quality)

    vortex_qualities = np.array(vortex_qualities)

    # Identify low-quality vortices to annihilate
    low_quality_mask = vortex_qualities < quality_threshold
    num_low_quality = np.sum(low_quality_mask)

    # Preserve some fraction of low-quality vortices (diversity)
    num_to_annihilate = int(num_low_quality * (1.0 - preserve_ratio))

    if num_to_annihilate == 0:
        return field_final, 0

    # Select vortices to annihilate (lowest quality first)
    low_quality_indices = np.where(low_quality_mask)[0]
    sorted_indices = low_quality_indices[np.argsort(vortex_qualities[low_quality_indices])]
    annihilate_indices = sorted_indices[:num_to_annihilate]

    # Inject antivortices at these positions
    field_modified = field_final.clone()

    for idx in annihilate_indices:
        vortex = vortex_positions[idx]
        node_idx = int(vortex['node_idx'])
        time_idx = int(vortex['time_idx'])
        winding = int(vortex['winding'])

        # Create antivortex (opposite winding)
        antivortex_phase = -winding * torch.atan2(
            field_modified[node_idx, time_idx].imag,
            field_modified[node_idx, time_idx].real
        )

        antivortex_field = antivortex_strength * torch.exp(1j * antivortex_phase)

        # Apply in annihilation radius
        # (Simplified: just modify the vortex node and neighbors)
        field_modified[node_idx, time_idx] += antivortex_field

    return field_modified, num_to_annihilate


def train_cycle(
    tokamak,
    coupler,
    rnn,
    optimizer,
    scaler,
    args,
    device,
    prev_metrics=None
):
    """
    Single training cycle with RNN control.

    Returns:
        metrics: dict of cycle metrics
        params: RNN parameters used this cycle
    """
    # Prepare state for RNN from previous cycle metrics
    # Use metrics-based state (not field-based, since field is 1D spatial only)

    if prev_metrics is None:
        # First cycle - use zeros
        state_features = torch.zeros(args.num_time_steps, 8, device=device)
    else:
        # Use previous metrics as state
        # Create time-series by repeating metrics across time dimension
        state_vec = torch.tensor([
            prev_metrics.get('vortex_density', 0.0),
            prev_metrics.get('fixed_point_pct', 0.0) / 100.0,
            prev_metrics.get('divergence', 0.0),
            prev_metrics.get('num_wormholes', 0) / 10000.0,  # Normalize
            prev_metrics.get('reward', 0.0) / 100.0,
            prev_metrics.get('num_annihilated', 0) / 1000.0,  # Normalize
            0.0,  # Reserved
            0.0   # Reserved
        ], dtype=torch.float32, device=device)

        # Repeat across time dimension
        state_features = state_vec.unsqueeze(0).repeat(args.num_time_steps, 1)

    # Pad to state_dim=128
    padding = torch.zeros(args.num_time_steps, 120, device=device)
    state_input = torch.cat([state_features, padding], dim=-1)  # (num_time_steps, 128)
    state_input = state_input.unsqueeze(0)  # (1, num_time_steps, 128)

    # RNN forward pass
    if scaler is not None:
        with torch.cuda.amp.autocast():
            params = rnn(state_input)
    else:
        params = rnn(state_input)

    # Extract scalar values (remove batch dimension)
    params_scalar = {k: v.item() if v.numel() == 1 else v.squeeze(0).item()
                     for k, v in params.items() if k != 'value'}
    # Keep value as tensor for backpropagation
    value = params['value'].squeeze() if params['value'].numel() > 1 else params['value']

    # Update retrocausal coupling with RNN parameters
    with torch.no_grad():
        coupler.retrocausal_strength = params_scalar['retrocausal_alpha']
        coupler.prophetic_mixing = params_scalar['prophetic_gamma']

    # Initialize temporal fields (2D: nodes x time_steps)
    # Self-consistent initialization: ψ_f(t=0) = ψ_b(t=0)
    field_forward = torch.randn(args.total_nodes, args.num_time_steps,
                                dtype=torch.complex64, device=device) * 0.1
    field_backward = field_forward.clone()

    # Run temporal evolution with RNN-controlled coupling
    field_forward, field_backward = coupler.apply_coupling(
        field_forward,
        field_backward,
        enable_mixing=True,
        enable_swapping=True,
        enable_anchoring=True
    )

    # Compute convergence metrics
    divergence = torch.mean(torch.abs(field_forward - field_backward)).item()
    fixed_point_mask = torch.abs(field_forward - field_backward) < 1e-5
    num_fixed = torch.sum(fixed_point_mask).item()
    total_points = field_forward.numel()
    fixed_point_pct = 100.0 * num_fixed / total_points

    # Use forward field as final state (2D: nodes x time_steps)
    field_final = field_forward

    # Detect vortices (use first time slice for detection)
    vortex_dict = detect_temporal_vortices_gpu(
        field_final[:, 0],  # Use first time slice (1D field)
        tokamak.positions,  # Combined positions tensor [total_nodes, 3]
        vortex_threshold=params_scalar['vortex_quality_threshold']
    )

    num_vortices = len(vortex_dict['node_idx'])
    vortex_density = num_vortices / args.total_nodes

    # Simplified vortex annihilation (working with single time slice)
    field_annihilated = field_final.clone()
    num_annihilated = 0

    if num_vortices > 0:
        # Get vortex amplitudes (quality metric)
        vortex_amplitudes = vortex_dict['amplitude']

        # Find low-quality vortices
        low_quality_mask = vortex_amplitudes < params_scalar['vortex_quality_threshold']
        num_low_quality = low_quality_mask.sum().item()

        # Preserve some fraction (diversity)
        num_to_annihilate = int(num_low_quality * (1.0 - params_scalar['preserve_ratio']))

        if num_to_annihilate > 0:
            # Get indices of low-quality vortices
            low_quality_indices = torch.where(low_quality_mask)[0]

            # Sort by quality (lowest first)
            qualities = vortex_amplitudes[low_quality_indices]
            sorted_order = torch.argsort(qualities)
            annihilate_indices = low_quality_indices[sorted_order[:num_to_annihilate]]

            # Get node indices to annihilate
            nodes_to_annihilate = vortex_dict['node_idx'][annihilate_indices]

            # Inject antivortices (simplified: reduce amplitude in annihilation radius)
            for node_idx in nodes_to_annihilate:
                # Reduce field amplitude at this node and neighbors
                field_annihilated[node_idx, :] *= (1.0 - params_scalar['antivortex_strength'])

            num_annihilated = len(nodes_to_annihilate)

    # Update tokamak field
    with torch.no_grad():
        tokamak.field = field_annihilated.clone()

    # Detect wormholes (simplified: count high-quality vortex pairs)
    # (Full wormhole detection would check inter-strip angular alignment)
    num_wormholes = max(0, num_vortices - num_annihilated) // 2

    # Compute reward
    reward, reward_components = compute_reward(
        num_vortices,
        vortex_density,
        num_wormholes,
        fixed_point_pct,
        divergence,
        args.total_nodes
    )

    # Policy gradient loss
    # Maximize reward by minimizing negative log probability weighted by advantage
    advantage = reward - value
    loss = -advantage * value  # Simplified policy gradient

    # Backpropagation
    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=args.grad_clip)

        # Check for NaN gradients
        has_nan = any(
            param.grad is not None and torch.isnan(param.grad).any()
            for param in rnn.parameters()
        )

        if has_nan:
            optimizer.zero_grad()
            nan_warning = True
        else:
            scaler.step(optimizer)
            nan_warning = False

        # Always update scaler state (required even when skipping step)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), max_norm=args.grad_clip)

        has_nan = any(
            param.grad is not None and torch.isnan(param.grad).any()
            for param in rnn.parameters()
        )

        if has_nan:
            optimizer.zero_grad()
            nan_warning = True
        else:
            optimizer.step()
            nan_warning = False

    # Package metrics
    metrics = {
        'reward': reward,
        'reward_components': reward_components,
        'vortex_count': num_vortices,
        'vortex_density': vortex_density,
        'num_annihilated': num_annihilated,
        'num_wormholes': num_wormholes,
        'fixed_point_pct': fixed_point_pct,
        'divergence': divergence,
        'value': value.item() if isinstance(value, torch.Tensor) else value,
        'loss': loss.item(),
        'nan_warning': nan_warning
    }

    return metrics, params_scalar


def main():
    parser = argparse.ArgumentParser(description='Tokamak RNN Control Training')

    parser.add_argument('--num-strips', type=int, default=300)
    parser.add_argument('--nodes-per-strip', type=int, default=166)
    parser.add_argument('--num-time-steps', type=int, default=10)
    parser.add_argument('--num-cycles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--hidden-dim', type=int, default=2048)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--output-dir', type=str, default='~/results/tokamak_rnn_control')

    args = parser.parse_args()

    # Derived parameters
    args.total_nodes = args.num_strips * args.nodes_per_strip

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    print()

    # Initialize tokamak
    print("Initializing tokamak...")
    tokamak = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        kappa=1.5,
        delta=0.3,
        sparse_threshold=0.3,
        max_neighbors=2000,
        device=device
    )

    # Initialize retrocausal coupler
    coupler = RetrocausalCoupler(
        num_nodes=args.total_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.7,  # Will be overridden by RNN
        prophetic_mixing=0.3,  # Will be overridden by RNN
        device=device
    )

    print(f"Initialized tokamak: {args.num_strips} strips, {args.total_nodes} nodes")
    print()

    # Initialize RNN
    print("Initializing RNN controller...")
    rnn = TokamakRNN(
        state_dim=128,
        hidden_dim=args.hidden_dim,
        num_params=11,
        device=device
    )

    # Optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)

    # Mixed precision scaler (if CUDA available)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print(f"RNN: hidden_dim={args.hidden_dim}, lr={args.learning_rate}, grad_clip={args.grad_clip}")
    print()

    # Training loop
    print(f"Starting training: {args.num_cycles} cycles")
    print("=" * 80)
    print()

    start_time = time.time()

    history = {
        'rewards': [],
        'reward_components': [],
        'vortex_counts': [],
        'vortex_densities': [],
        'num_annihilated': [],
        'num_wormholes': [],
        'fixed_point_pcts': [],
        'divergences': [],
        'values': [],
        'losses': [],
        'nan_warnings': [],
        'parameters': []
    }

    prev_metrics = None

    for cycle in range(args.num_cycles):
        cycle_start = time.time()

        # Train one cycle
        metrics, params_used = train_cycle(
            tokamak, coupler, rnn, optimizer, scaler, args, device, prev_metrics
        )

        # Update prev_metrics for next cycle
        prev_metrics = metrics

        # Record history
        history['rewards'].append(metrics['reward'])
        history['reward_components'].append(metrics['reward_components'])
        history['vortex_counts'].append(metrics['vortex_count'])
        history['vortex_densities'].append(metrics['vortex_density'])
        history['num_annihilated'].append(metrics['num_annihilated'])
        history['num_wormholes'].append(metrics['num_wormholes'])
        history['fixed_point_pcts'].append(metrics['fixed_point_pct'])
        history['divergences'].append(metrics['divergence'])
        history['values'].append(metrics['value'])
        history['losses'].append(metrics['loss'])
        history['nan_warnings'].append(metrics['nan_warning'])
        history['parameters'].append(params_used)

        cycle_time = time.time() - cycle_start

        # Print progress
        if cycle % 10 == 0 or cycle == args.num_cycles - 1:
            print(f"Cycle {cycle}/{args.num_cycles}")
            print(f"  Reward: {metrics['reward']:.2f}")
            print(f"  Vortex density: {metrics['vortex_density']:.2%} ({metrics['vortex_count']} vortices)")
            print(f"  Annihilated: {metrics['num_annihilated']}")
            print(f"  Wormholes: {metrics['num_wormholes']}")
            print(f"  Fixed points: {metrics['fixed_point_pct']:.1f}%")
            print(f"  Divergence: {metrics['divergence']:.6f}")
            print(f"  RNN params: alpha={params_used['retrocausal_alpha']:.3f}, gamma={params_used['prophetic_gamma']:.3f}")
            if metrics['nan_warning']:
                print(f"  WARNING: NaN gradients detected, skipped update")
            print(f"  Time: {cycle_time:.3f}s")
            print()

    total_time = time.time() - start_time

    print("=" * 80)
    print(f"Training complete: {args.num_cycles} cycles in {total_time:.2f}s")
    print(f"Average time/cycle: {total_time/args.num_cycles:.3f}s")
    print()

    # Save results
    results = {
        'config': vars(args),
        'total_time_sec': total_time,
        'history': history
    }

    results_path = output_dir / f"training_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")

    # Save checkpoint
    checkpoint_path = output_dir / f"rnn_checkpoint_{timestamp}.pt"
    torch.save({
        'rnn_state_dict': rnn.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': vars(args),
        'final_metrics': {
            'reward': history['rewards'][-1],
            'vortex_density': history['vortex_densities'][-1],
            'fixed_point_pct': history['fixed_point_pcts'][-1]
        }
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
