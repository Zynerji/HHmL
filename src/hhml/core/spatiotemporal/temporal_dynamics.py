#!/usr/bin/env python3
"""
Temporal Evolution Dynamics
============================

Forward and backward time evolution for (2+1)D spatiotemporal Möbius strip.

Based on Perfect Temporal Loop discovery (HHmL, 2025-12-18):
- Self-consistent initialization: ψ_f(0) = ψ_b(0)
- Relaxation factor β prevents temporal oscillations
- Convergence to fixed points: ψ_f(t) = ψ_b(t) for all t

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class TemporalEvolver(nn.Module):
    """
    Temporal evolution dynamics for spatiotemporal Möbius strip.

    Implements forward and backward time evolution:
    - Forward: t=0 → t=T (causal evolution)
    - Backward: t=T → t=0 (retrocausal evolution)

    Temporal relaxation prevents oscillations during convergence.
    """

    def __init__(
        self,
        num_nodes: int,
        num_time_steps: int,
        relaxation_factor: float = 0.3,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.beta = relaxation_factor  # Temporal relaxation
        self.device = device

        print(f"Initialized Temporal Evolver:")
        print(f"  Relaxation factor (beta): {self.beta:.2f}")

    def evolve_forward_step(
        self,
        field: torch.Tensor,
        t_idx: int,
        spatial_coupling: float = 0.1,
        temporal_coupling: float = 0.05
    ) -> torch.Tensor:
        """
        Evolve field forward one time step: t → t+1.

        Dynamics:
        ψ(θ, t+1) = ψ(θ, t) + Δψ_spatial + Δψ_temporal

        where:
        - Δψ_spatial: Spatial coupling (Laplacian along θ)
        - Δψ_temporal: Temporal propagation

        Args:
            field: Current field state (num_nodes, num_time_steps)
            t_idx: Current time index
            spatial_coupling: κ parameter (spatial diffusion)
            temporal_coupling: λ parameter (temporal propagation)

        Returns:
            updated_field: Field at next time step
        """
        if t_idx >= self.num_time_steps - 1:
            return field  # At boundary, no evolution

        # Current field at time t
        psi_t = field[:, t_idx]

        # Spatial Laplacian (discrete approximation)
        psi_left = torch.roll(psi_t, shifts=1, dims=0)
        psi_right = torch.roll(psi_t, shifts=-1, dims=0)
        laplacian = psi_left + psi_right - 2 * psi_t

        # Spatial diffusion
        delta_spatial = spatial_coupling * laplacian

        # Temporal propagation (simple forward euler)
        delta_temporal = temporal_coupling * psi_t

        # Update field at t+1
        field[:, t_idx + 1] = psi_t + delta_spatial + delta_temporal

        return field

    def evolve_backward_step(
        self,
        field: torch.Tensor,
        t_idx: int,
        spatial_coupling: float = 0.1,
        temporal_coupling: float = 0.05
    ) -> torch.Tensor:
        """
        Evolve field backward one time step: t → t-1.

        Retrocausal dynamics (time-reversed evolution).

        Args:
            field: Current field state (num_nodes, num_time_steps)
            t_idx: Current time index
            spatial_coupling: κ parameter
            temporal_coupling: λ parameter

        Returns:
            updated_field: Field at previous time step
        """
        if t_idx <= 0:
            return field  # At boundary, no evolution

        # Current field at time t
        psi_t = field[:, t_idx]

        # Spatial Laplacian
        psi_left = torch.roll(psi_t, shifts=1, dims=0)
        psi_right = torch.roll(psi_t, shifts=-1, dims=0)
        laplacian = psi_left + psi_right - 2 * psi_t

        # Backward spatial diffusion
        delta_spatial = spatial_coupling * laplacian

        # Backward temporal propagation
        delta_temporal = -temporal_coupling * psi_t  # Negative for backward

        # Update field at t-1
        field[:, t_idx - 1] = psi_t + delta_spatial + delta_temporal

        return field

    def full_forward_sweep(
        self,
        field: torch.Tensor,
        spatial_coupling: float,
        temporal_coupling: float
    ) -> torch.Tensor:
        """
        Complete forward evolution: t=0 → t=T.

        Evolves field from initial condition at t=0 to final state at t=T.

        Args:
            field: Field with initial condition at t=0
            spatial_coupling: κ parameter
            temporal_coupling: λ parameter

        Returns:
            field: Fully evolved forward trajectory
        """
        for t_idx in range(self.num_time_steps - 1):
            field = self.evolve_forward_step(
                field, t_idx, spatial_coupling, temporal_coupling
            )

        return field

    def full_backward_sweep(
        self,
        field: torch.Tensor,
        spatial_coupling: float,
        temporal_coupling: float
    ) -> torch.Tensor:
        """
        Complete backward evolution: t=T → t=0.

        Evolves field from final condition at t=T to initial state at t=0.

        Args:
            field: Field with final condition at t=T
            spatial_coupling: κ parameter
            temporal_coupling: λ parameter

        Returns:
            field: Fully evolved backward trajectory
        """
        for t_idx in range(self.num_time_steps - 1, 0, -1):
            field = self.evolve_backward_step(
                field, t_idx, spatial_coupling, temporal_coupling
            )

        return field

    def relaxed_update(
        self,
        field_old: torch.Tensor,
        field_new: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal relaxation to prevent oscillations.

        Relaxed update:
        ψ_relaxed = β * ψ_new + (1 - β) * ψ_old

        This prevents wild oscillations during temporal loop convergence.

        Args:
            field_old: Previous iteration field
            field_new: Current iteration field

        Returns:
            field_relaxed: Relaxed field update
        """
        return self.beta * field_new + (1 - self.beta) * field_old
