#!/usr/bin/env python3
"""
Temporal Vortex Dynamics - Topological Defects in Time
=======================================================

Implements temporal vortex generators, annihilators, and detection for
topological defects in the temporal dimension t ∈ [0, 2π).

Key Concepts:
- Temporal vortices: Phase singularities at specific time slices
- Spatiotemporal vortex tubes: Vortex lines threading through (θ, t)
- Topological protection via temporal Möbius twist
- Persistence through temporal loop iterations

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict


class TemporalVortexController(nn.Module):
    """
    Control temporal vortices and spatiotemporal vortex tubes.

    Temporal vortices are phase singularities in the temporal dimension,
    analogous to spatial vortices but occurring at specific time slices.

    Spatiotemporal vortex tubes are vortex lines that thread through
    both spatial (θ) and temporal (t) dimensions, creating the most
    interesting topological structures in (2+1)D spacetime.
    """

    def __init__(
        self,
        num_nodes: int,
        num_time_steps: int,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps
        self.device = device

        # Vortex tracking
        self.temporal_vortex_positions = []  # Time slices with temporal vortices
        self.vortex_tube_trajectories = []   # (θ, t) paths of vortex tubes

        print(f"Initialized Temporal Vortex Controller:")
        print(f"  Temporal dimension: t ∈ [0, 2π), {num_time_steps} steps")
        print(f"  Spatial dimension: θ ∈ [0, 2π), {num_nodes} nodes")
        print(f"  Spacetime manifold: (2+1)D Möbius × Möbius")

    def detect_temporal_vortices(
        self,
        field: torch.Tensor,
        threshold: float = 0.1
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Detect temporal vortices (phase singularities in time dimension).

        A temporal vortex exists at time slice t* if the field magnitude
        drops below threshold across all spatial positions:

            ψ(θ, t*) → 0 for all θ

        And the phase winds around the temporal loop:

            ∮ dt ∂ψ/∂t = 2πn  (winding number n)

        Args:
            field: Complex field (num_nodes, num_time_steps)
            threshold: Magnitude threshold for vortex core

        Returns:
            temporal_vortex_times: List of time indices with temporal vortices
            winding_numbers: Winding number at each temporal vortex
        """
        temporal_vortices = []
        winding_numbers = []

        # Check each time slice
        for t_idx in range(self.num_time_steps):
            field_t = field[:, t_idx]

            # Check if field magnitude is small (vortex core)
            mean_magnitude = torch.mean(torch.abs(field_t))

            if mean_magnitude < threshold:
                # Potential temporal vortex - compute winding number
                phases = torch.angle(field_t)

                # Compute phase gradient in spatial direction
                phase_diff = torch.diff(phases)

                # Unwrap phase discontinuities
                phase_diff = torch.where(
                    phase_diff > np.pi,
                    phase_diff - 2*np.pi,
                    phase_diff
                )
                phase_diff = torch.where(
                    phase_diff < -np.pi,
                    phase_diff + 2*np.pi,
                    phase_diff
                )

                # Winding number = total phase change / 2π
                winding = torch.sum(phase_diff) / (2 * np.pi)

                # If winding number is significant, it's a temporal vortex
                if torch.abs(winding) > 0.5:
                    temporal_vortices.append(t_idx)
                    winding_numbers.append(winding.item())

        return temporal_vortices, torch.tensor(winding_numbers, device=self.device)

    def detect_spatiotemporal_vortex_tubes(
        self,
        field: torch.Tensor,
        threshold: float = 0.1,
        min_length: int = 3
    ) -> List[Dict]:
        """
        Detect spatiotemporal vortex tubes (vortex lines through (θ, t)).

        A vortex tube is a connected path of vortex cores that extends
        through both spatial and temporal dimensions. These are the most
        topologically interesting structures.

        Algorithm:
        1. Find all vortex cores (|ψ| < threshold)
        2. Connect nearby cores into trajectories
        3. Filter trajectories by minimum length

        Args:
            field: Complex field (num_nodes, num_time_steps)
            threshold: Magnitude threshold for vortex core
            min_length: Minimum tube length (number of connected cores)

        Returns:
            vortex_tubes: List of tube dictionaries with:
                - trajectory: List of (theta_idx, t_idx) positions
                - length: Number of points in tube
                - winding_number: Topological charge
                - spatial_extent: Range in θ dimension
                - temporal_extent: Range in t dimension
        """
        # Find all vortex cores
        magnitude = torch.abs(field)
        is_core = magnitude < threshold

        # Get coordinates of all cores
        core_coords = torch.nonzero(is_core)  # Shape: (N_cores, 2)

        if len(core_coords) == 0:
            return []

        # Build tubes by connecting nearby cores
        tubes = []
        visited = torch.zeros(len(core_coords), dtype=torch.bool, device=self.device)

        for start_idx in range(len(core_coords)):
            if visited[start_idx]:
                continue

            # Start new tube
            tube_trajectory = [core_coords[start_idx].tolist()]
            visited[start_idx] = True

            # Grow tube by finding connected cores
            current_idx = start_idx

            while True:
                current_pos = core_coords[current_idx]

                # Find nearest unvisited core
                distances = torch.sum(
                    (core_coords - current_pos.unsqueeze(0))**2,
                    dim=1
                ).float()  # Ensure float dtype for inf assignment
                distances[visited] = float('inf')

                nearest_idx = torch.argmin(distances)
                nearest_dist = distances[nearest_idx]

                # If nearest core is close enough, add to tube
                if nearest_dist < 9:  # Within 3 units in (θ, t) space
                    tube_trajectory.append(core_coords[nearest_idx].tolist())
                    visited[nearest_idx] = True
                    current_idx = nearest_idx
                else:
                    break

            # Save tube if long enough
            if len(tube_trajectory) >= min_length:
                # Compute tube properties
                theta_coords = [pos[0] for pos in tube_trajectory]
                t_coords = [pos[1] for pos in tube_trajectory]

                tube = {
                    'trajectory': tube_trajectory,
                    'length': len(tube_trajectory),
                    'spatial_extent': (min(theta_coords), max(theta_coords)),
                    'temporal_extent': (min(t_coords), max(t_coords)),
                    'winding_number': self._compute_tube_winding(field, tube_trajectory)
                }

                tubes.append(tube)

        return tubes

    def _compute_tube_winding(
        self,
        field: torch.Tensor,
        trajectory: List[List[int]]
    ) -> float:
        """Compute topological winding number along tube trajectory."""
        if len(trajectory) < 2:
            return 0.0

        total_winding = 0.0

        for i in range(len(trajectory) - 1):
            theta_i, t_i = trajectory[i]
            theta_j, t_j = trajectory[i + 1]

            # Phase difference along trajectory segment
            phase_i = torch.angle(field[theta_i, t_i])
            phase_j = torch.angle(field[theta_j, t_j])

            phase_diff = phase_j - phase_i

            # Unwrap
            if phase_diff > np.pi:
                phase_diff -= 2*np.pi
            elif phase_diff < -np.pi:
                phase_diff += 2*np.pi

            total_winding += phase_diff

        return float(total_winding / (2*np.pi))

    def inject_temporal_vortex(
        self,
        field: torch.Tensor,
        t_idx: int,
        winding_number: int = 1,
        core_size: float = 0.1
    ) -> torch.Tensor:
        """
        Inject temporal vortex at specific time slice.

        Creates a phase singularity in the temporal dimension by imposing
        a winding phase pattern across spatial positions at time t_idx:

            ψ(θ, t_idx) *= exp(i * n * θ)

        where n is the winding number.

        Args:
            field: Complex field (num_nodes, num_time_steps)
            t_idx: Time slice index for vortex placement
            winding_number: Topological charge (±1, ±2, ...)
            core_size: Magnitude suppression at vortex core

        Returns:
            field: Updated field with temporal vortex
        """
        # Create phase winding pattern
        theta_coords = torch.linspace(0, 2*np.pi, self.num_nodes, device=self.device)
        phase_winding = torch.exp(1j * winding_number * theta_coords)

        # Apply phase winding to time slice
        field[:, t_idx] = field[:, t_idx] * phase_winding

        # Suppress magnitude at core
        field[:, t_idx] = field[:, t_idx] * core_size

        # Track vortex position
        if t_idx not in self.temporal_vortex_positions:
            self.temporal_vortex_positions.append(t_idx)

        return field

    def inject_spatiotemporal_vortex_tube(
        self,
        field: torch.Tensor,
        trajectory: List[Tuple[int, int]],
        winding_number: int = 1,
        core_size: float = 0.1
    ) -> torch.Tensor:
        """
        Inject spatiotemporal vortex tube along (θ, t) trajectory.

        Creates a vortex line that threads through spacetime, with
        phase winding that evolves smoothly along the trajectory.

        Args:
            field: Complex field (num_nodes, num_time_steps)
            trajectory: List of (theta_idx, t_idx) positions defining tube path
            winding_number: Topological charge
            core_size: Magnitude suppression at tube core

        Returns:
            field: Updated field with vortex tube
        """
        for theta_idx, t_idx in trajectory:
            # Apply phase winding at this point
            phase = winding_number * 2*np.pi * (theta_idx / self.num_nodes)
            field[theta_idx, t_idx] *= torch.exp(torch.tensor(1j * phase, device=self.device))

            # Suppress magnitude at tube core
            field[theta_idx, t_idx] *= core_size

        # Track tube
        self.vortex_tube_trajectories.append(trajectory)

        return field

    def annihilate_temporal_vortex(
        self,
        field: torch.Tensor,
        t_idx: int,
        smoothing_radius: int = 2
    ) -> torch.Tensor:
        """
        Remove temporal vortex at time slice by interpolating from neighbors.

        Args:
            field: Complex field (num_nodes, num_time_steps)
            t_idx: Time slice index
            smoothing_radius: Number of neighboring time slices to average

        Returns:
            field: Updated field without temporal vortex
        """
        # Get neighboring time slices
        t_prev = max(0, t_idx - smoothing_radius)
        t_next = min(self.num_time_steps - 1, t_idx + smoothing_radius)

        # Interpolate field from neighbors
        if t_prev != t_next:
            field[:, t_idx] = 0.5 * (field[:, t_prev] + field[:, t_next])

        # Remove from tracking
        if t_idx in self.temporal_vortex_positions:
            self.temporal_vortex_positions.remove(t_idx)

        return field

    def compute_temporal_vortex_density(self, field: torch.Tensor) -> float:
        """
        Compute temporal vortex density (fraction of time slices with vortices).

        Args:
            field: Complex field (num_nodes, num_time_steps)

        Returns:
            density: Temporal vortex density (0-1)
        """
        temporal_vortices, _ = self.detect_temporal_vortices(field)
        return len(temporal_vortices) / self.num_time_steps

    def compute_vortex_tube_density(self, field: torch.Tensor) -> float:
        """
        Compute spatiotemporal vortex tube density.

        Args:
            field: Complex field (num_nodes, num_time_steps)

        Returns:
            density: Fraction of spacetime volume occupied by vortex tubes
        """
        tubes = self.detect_spatiotemporal_vortex_tubes(field)

        if not tubes:
            return 0.0

        total_length = sum(tube['length'] for tube in tubes)
        total_spacetime_points = self.num_nodes * self.num_time_steps

        return total_length / total_spacetime_points

    def get_vortex_statistics(self, field: torch.Tensor) -> Dict:
        """
        Compute comprehensive vortex statistics.

        Returns:
            stats: Dictionary with:
                - temporal_vortex_count: Number of temporal vortices
                - temporal_vortex_density: Fraction of time slices
                - vortex_tube_count: Number of spatiotemporal tubes
                - vortex_tube_density: Fraction of spacetime volume
                - avg_tube_length: Average tube length
                - total_topological_charge: Sum of winding numbers
        """
        temporal_vortices, temporal_windings = self.detect_temporal_vortices(field)
        vortex_tubes = self.detect_spatiotemporal_vortex_tubes(field)

        stats = {
            'temporal_vortex_count': len(temporal_vortices),
            'temporal_vortex_density': len(temporal_vortices) / self.num_time_steps,
            'vortex_tube_count': len(vortex_tubes),
            'vortex_tube_density': self.compute_vortex_tube_density(field),
            'avg_tube_length': np.mean([tube['length'] for tube in vortex_tubes]) if vortex_tubes else 0.0,
            'total_topological_charge_temporal': torch.sum(temporal_windings).item() if len(temporal_windings) > 0 else 0.0,
            'total_topological_charge_tubes': sum(tube['winding_number'] for tube in vortex_tubes)
        }

        return stats
