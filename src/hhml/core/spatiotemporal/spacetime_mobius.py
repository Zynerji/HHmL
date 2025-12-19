#!/usr/bin/env python3
"""
Spatiotemporal Möbius Strip - (2+1)D Spacetime Boundary
=======================================================

Extends HHmL spatial Möbius to include temporal dimension with Möbius twist.

Key Features:
- θ ∈ [0, 2π): Spatial Möbius coordinate (inherited from HHmL)
- t ∈ [0, 2π): Temporal Möbius coordinate (NEW)
- Field ψ(θ, t): Complex field on (2+1)D boundary
- Forward/backward time evolution with retrocausal coupling
- Temporal twist parameter τ: Controls temporal Möbius geometry

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class SpatiotemporalMobiusStrip(nn.Module):
    """
    (2+1)D Spatiotemporal Möbius Strip

    Topology:
    - Spatial: Möbius strip (θ ∈ [0, 2π) with 180° twist)
    - Temporal: Möbius loop (t ∈ [0, 2π) with temporal twist)
    - Combined: (2+1)D manifold with both space AND time as topological

    Fields:
    - ψ_forward(θ, t): Forward time evolution
    - ψ_backward(θ, t): Backward time evolution
    - Prophetic coupling: ψ_f ↔ ψ_b via retrocausal feedback

    Parameters:
    - num_nodes: Number of spatial nodes (θ discretization)
    - num_time_steps: Number of temporal nodes (t discretization)
    - temporal_twist: τ parameter (temporal Möbius twist angle)
    - device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        num_nodes: int = 4000,
        num_time_steps: int = 50,
        temporal_twist: float = np.pi,  # Default: 180° temporal twist
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.temporal_twist = temporal_twist
        self.device = device

        # Spatial Möbius coordinates (inherited from HHmL)
        self.theta = torch.linspace(0, 2*np.pi, num_nodes, device=device)

        # Temporal Möbius coordinates (NEW)
        self.t = torch.linspace(0, 2*np.pi, num_time_steps, device=device)

        # Create (2+1)D coordinate mesh
        self.theta_grid, self.t_grid = torch.meshgrid(self.theta, self.t, indexing='ij')

        # Initialize forward and backward fields
        # Shape: (num_nodes, num_time_steps)
        self.field_forward = torch.zeros(
            num_nodes, num_time_steps,
            dtype=torch.complex64,
            device=device
        )

        self.field_backward = torch.zeros(
            num_nodes, num_time_steps,
            dtype=torch.complex64,
            device=device
        )

        # Temporal divergence tracking
        self.divergence_history = []

        print(f"Initialized (2+1)D Spatiotemporal Möbius:")
        print(f"  Spatial nodes: {num_nodes}")
        print(f"  Temporal steps: {num_time_steps}")
        print(f"  Temporal twist: {temporal_twist:.3f} rad ({np.degrees(temporal_twist):.1f} deg)")
        print(f"  Total DOF: {num_nodes * num_time_steps}")
        print(f"  Device: {device}")

    def initialize_self_consistent(self, seed: Optional[int] = None):
        """
        Self-consistent initialization: ψ_f(θ, t=0) = ψ_b(θ, t=0)

        This is CRITICAL for temporal fixed point convergence.
        Random forward/backward initialization causes immediate divergence.

        Based on Perfect Temporal Loop discovery (HHmL, 2025-12-18).
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Generate initial spatial field at t=0
        initial_state = torch.randn(
            self.num_nodes,
            dtype=torch.complex64,
            device=self.device
        )

        # CRITICAL: Both forward and backward start from SAME initial state
        self.field_forward[:, 0] = initial_state.clone()
        self.field_backward[:, 0] = initial_state.clone()

        # Remaining time steps initialized to zero (will evolve)
        self.field_forward[:, 1:] = 0
        self.field_backward[:, 1:] = 0

        print(f"Self-consistent initialization: psi_f(t=0) = psi_b(t=0)")

    def apply_spatial_mobius_bc(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial Möbius boundary condition.

        At θ = 2π, field reconnects to θ = 0 with 180° phase twist:
        ψ(2π, t) = exp(iπ) * ψ(0, t) = -ψ(0, t)

        Args:
            field: Shape (num_nodes, num_time_steps)

        Returns:
            field: With spatial Möbius BC enforced
        """
        # Enforce spatial twist at boundary
        field[-1, :] = -field[0, :]
        return field

    def apply_temporal_mobius_bc(self, field: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal Möbius boundary condition.

        At t = 2π, field reconnects to t = 0 with temporal twist:
        ψ(θ, 2π) = exp(i*τ) * ψ(θ, 0)

        where τ is the temporal twist parameter.

        Args:
            field: Shape (num_nodes, num_time_steps)

        Returns:
            field: With temporal Möbius BC enforced
        """
        # Enforce temporal twist at boundary
        # Create complex phase: exp(i*τ)
        twist_tensor = torch.tensor(self.temporal_twist, device=self.device, dtype=torch.float32)
        twist_phase = torch.exp(1j * twist_tensor)
        field[:, -1] = twist_phase * field[:, 0]
        return field

    def compute_divergence(self) -> float:
        """
        Compute temporal divergence D = |ψ_f - ψ_b|

        This measures how far forward/backward evolutions have diverged.
        Perfect temporal loop: D → 0 (self-consistent fixed point).

        Returns:
            divergence: RMS divergence across all (θ, t) points
        """
        diff = self.field_forward - self.field_backward
        divergence = torch.sqrt(torch.mean(torch.abs(diff)**2))
        return divergence.item()

    def compute_temporal_fixed_points(self, threshold: float = 0.01) -> Tuple[int, float]:
        """
        Count temporal fixed points where ψ_f(θ, t) ≈ ψ_b(θ, t).

        Fixed point at time t if: |ψ_f(:, t) - ψ_b(:, t)| < threshold

        Args:
            threshold: Divergence threshold for fixed point detection

        Returns:
            num_fixed_points: Count of time steps with fixed points
            percentage: Percentage of time steps (0-100)
        """
        fixed_points = 0

        for t_idx in range(self.num_time_steps):
            diff_t = self.field_forward[:, t_idx] - self.field_backward[:, t_idx]
            divergence_t = torch.sqrt(torch.mean(torch.abs(diff_t)**2))

            if divergence_t < threshold:
                fixed_points += 1

        percentage = 100 * fixed_points / self.num_time_steps

        return fixed_points, percentage

    def get_state_tensor(self, target_dim: int = 256) -> torch.Tensor:
        """
        Get current state as tensor for RNN encoding.

        Args:
            target_dim: Target state dimension (default: 256)

        Returns:
            state: Shape (target_dim,) encoding current spatiotemporal state

        Features (core 10, then expanded):
        - Global divergence
        - Temporal fixed point ratio
        - Forward/backward field statistics
        - Spatial/temporal coherence
        - Expanded with spatial/temporal field samples
        """
        # Core metrics (10 features)
        divergence = self.compute_divergence()
        num_fixed, pct_fixed = self.compute_temporal_fixed_points()

        forward_mag = torch.abs(self.field_forward)
        backward_mag = torch.abs(self.field_backward)

        forward_mean = torch.mean(forward_mag).item()
        forward_std = torch.std(forward_mag).item()
        backward_mean = torch.mean(backward_mag).item()
        backward_std = torch.std(backward_mag).item()

        spatial_coherence_f = torch.mean(torch.abs(
            torch.diff(self.field_forward, dim=0)
        )).item()
        spatial_coherence_b = torch.mean(torch.abs(
            torch.diff(self.field_backward, dim=0)
        )).item()

        temporal_coherence_f = torch.mean(torch.abs(
            torch.diff(self.field_forward, dim=1)
        )).item()
        temporal_coherence_b = torch.mean(torch.abs(
            torch.diff(self.field_backward, dim=1)
        )).item()

        # Core state vector (10 features)
        core_state = torch.tensor([
            divergence,
            pct_fixed / 100.0,
            forward_mean,
            forward_std,
            backward_mean,
            backward_std,
            spatial_coherence_f,
            spatial_coherence_b,
            temporal_coherence_f,
            temporal_coherence_b,
        ], device=self.device)

        # Expand to target_dim by sampling field values
        if target_dim > 10:
            # Sample spatial-temporal field points
            num_samples = target_dim - 10

            # Sample forward field
            forward_flat = self.field_forward.flatten()
            backward_flat = self.field_backward.flatten()

            # Uniform sampling indices
            indices = torch.linspace(0, len(forward_flat) - 1, num_samples // 2, dtype=torch.long)

            forward_samples = torch.abs(forward_flat[indices]).real
            backward_samples = torch.abs(backward_flat[indices]).real

            # Pad if necessary
            if len(forward_samples) + len(backward_samples) < num_samples:
                padding = torch.zeros(num_samples - len(forward_samples) - len(backward_samples), device=self.device)
                field_samples = torch.cat([forward_samples, backward_samples, padding])
            else:
                field_samples = torch.cat([forward_samples, backward_samples])[:num_samples]

            # Concatenate core state + field samples
            state = torch.cat([core_state, field_samples])
        else:
            state = core_state[:target_dim]

        return state

    def save_state(self, filepath: str):
        """Save current spatiotemporal state to file."""
        state_dict = {
            'num_nodes': self.num_nodes,
            'num_time_steps': self.num_time_steps,
            'temporal_twist': self.temporal_twist,
            'field_forward': self.field_forward.cpu(),
            'field_backward': self.field_backward.cpu(),
            'divergence_history': self.divergence_history,
            'theta': self.theta.cpu(),
            't': self.t.cpu(),
        }
        torch.save(state_dict, filepath)
        print(f"Saved spatiotemporal state: {filepath}")

    def load_state(self, filepath: str):
        """Load spatiotemporal state from file."""
        state_dict = torch.load(filepath, map_location=self.device)

        self.field_forward = state_dict['field_forward'].to(self.device)
        self.field_backward = state_dict['field_backward'].to(self.device)
        self.divergence_history = state_dict['divergence_history']

        print(f"Loaded spatiotemporal state: {filepath}")
        print(f"  Current divergence: {self.compute_divergence():.6f}")

        num_fixed, pct_fixed = self.compute_temporal_fixed_points()
        print(f"  Temporal fixed points: {num_fixed}/{self.num_time_steps} ({pct_fixed:.1f}%)")
