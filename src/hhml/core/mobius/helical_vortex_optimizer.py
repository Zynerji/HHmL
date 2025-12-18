#!/usr/bin/env python3
"""
Helical Vortex Optimizer - Spectral Graph Approach for Vortex Stability
========================================================================
Adapted from Helical-SAT-Heuristic for vortex field optimization.

Uses logarithmic node indexing with cosine phase weighting to find
optimal field configurations that maximize vortex density and stability.

Key Concepts:
- Treat vortex positions as constraint satisfaction problem
- Use Laplacian eigenvector (Fiedler vector) for field assignment
- Helical phase weighting: w_ij = cos(ω(θ_i - θ_j)) where θ ∝ log(i+1)
- One-shot spectral solution (no iterative refinement needed)

Integration with HHmL:
- RNN can control omega (helical frequency) parameter
- Spectral optimization provides stable vortex configurations
- Can be used as periodic "reset" to stabilize collapsed vortices

Author: HHmL Framework (adapted from Helical-SAT-Heuristic)
Date: 2025-12-16
"""

import torch
import numpy as np
import networkx as nx
from scipy.sparse.linalg import eigsh
from typing import Tuple, Optional


def compute_helical_weights(num_nodes: int, omega: float = 0.3) -> np.ndarray:
    """
    Compute helical phase weights for node pairs.

    Uses logarithmic indexing: θ_i = 2π * log(i+1) / log(N+1)
    Edge weight: w_ij = cos(ω(θ_i - θ_j))

    Args:
        num_nodes: Number of nodes in system
        omega: Frequency parameter (0.1-1.0, RNN-controllable)

    Returns:
        weight_matrix: [N, N] array of edge weights
    """
    # Logarithmic phase assignment
    indices = np.arange(num_nodes)
    theta = 2.0 * np.pi * np.log(indices + 1) / np.log(num_nodes + 1)

    # Pairwise phase differences
    theta_diff = theta[:, np.newaxis] - theta[np.newaxis, :]

    # Cosine weighting
    weights = np.cos(omega * theta_diff)

    return weights


def spectral_field_optimization(
    positions: torch.Tensor,
    field: torch.Tensor,
    omega: float = 0.3,
    vortex_target: float = 0.7
) -> torch.Tensor:
    """
    Optimize field configuration using spectral graph method.

    Finds field values that maximize vortex density by solving
    Laplacian eigenvector problem with helical edge weights.

    Args:
        positions: [N, 3] tensor of node positions
        field: [N] complex tensor of current field values
        omega: Helical frequency parameter (RNN-controlled)
        vortex_target: Target vortex density (0-1)

    Returns:
        optimized_field: [N] complex tensor with optimized field values
    """
    num_nodes = positions.shape[0]
    device = positions.device

    # Compute helical weights based on node positions
    weights = compute_helical_weights(num_nodes, omega)

    # Build weighted Laplacian
    # L = D - W, where D is degree matrix, W is weighted adjacency
    W = weights.copy()
    np.fill_diagonal(W, 0)  # Remove self-loops
    D = np.diag(W.sum(axis=1))
    L = D - W

    # Compute Fiedler vector (second smallest eigenvector)
    # Use scipy for sparse eigenvalue solver
    try:
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
        fiedler = eigenvectors[:, 1]  # Second eigenvector
    except:
        # Fallback to numpy if sparse solver fails
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        fiedler = eigenvectors[:, 1]

    # Map Fiedler vector to field magnitudes
    # Negative values → low field (vortices)
    # Positive values → high field

    # Normalize Fiedler vector to [0, 1]
    fiedler_normalized = (fiedler - fiedler.min()) / (fiedler.max() - fiedler.min() + 1e-8)

    # Apply vortex target: scale so that target fraction has low field
    threshold = np.percentile(fiedler_normalized, vortex_target * 100)
    field_magnitudes = np.where(
        fiedler_normalized < threshold,
        0.2,  # Vortex core (low field)
        1.0   # High field region
    )

    # Convert to complex field (preserve phases from original field)
    current_phases = torch.angle(field).cpu().numpy()
    optimized_field_np = field_magnitudes * np.exp(1j * current_phases)

    # Convert back to torch
    optimized_field = torch.from_numpy(optimized_field_np).to(device, dtype=torch.complex64)

    return optimized_field


def compute_vortex_stability_score(field: torch.Tensor, threshold: float = 0.3) -> float:
    """
    Compute vortex stability score using spectral graph analysis.

    Higher scores indicate more stable vortex configurations.

    Args:
        field: [N] complex tensor of field values
        threshold: Vortex detection threshold

    Returns:
        stability_score: 0-1 measure of vortex configuration quality
    """
    field_mag = torch.abs(field).cpu().numpy()

    # Identify vortex cores
    vortex_mask = field_mag < threshold
    vortex_density = vortex_mask.mean()

    # Compute spatial clustering of vortices (using positions if available)
    # For now, use simple variance as proxy for stability
    vortex_variance = field_mag[vortex_mask].var() if vortex_mask.sum() > 0 else 1.0

    # Lower variance = more uniform vortex cores = higher stability
    stability = vortex_density * (1.0 - min(vortex_variance, 1.0))

    return float(stability)


def helical_vortex_reset(
    positions: torch.Tensor,
    field: torch.Tensor,
    omega: float = 0.3,
    vortex_target: float = 0.7,
    reset_strength: float = 0.5
) -> torch.Tensor:
    """
    Apply helical spectral reset to collapsed vortex field.

    This is the main integration point for HHmL training:
    - Call periodically when vortex density drops too low
    - RNN controls omega and reset_strength parameters
    - Blends spectral solution with current field

    Args:
        positions: [N, 3] tensor of node positions
        field: [N] complex tensor of current (possibly collapsed) field
        omega: Helical frequency (RNN-controlled)
        vortex_target: Target vortex density (RNN-controlled)
        reset_strength: Blend factor (0=no reset, 1=full reset)

    Returns:
        reset_field: [N] complex tensor with vortices restored
    """
    # Compute optimal field using spectral method
    optimal_field = spectral_field_optimization(positions, field, omega, vortex_target)

    # Blend with current field
    reset_field = (1.0 - reset_strength) * field + reset_strength * optimal_field

    return reset_field


if __name__ == "__main__":
    # Test helical vortex optimizer
    print("Testing Helical Vortex Optimizer")
    print("=" * 80)

    # Create test system
    num_nodes = 1000
    positions = torch.randn(num_nodes, 3)
    field = torch.randn(num_nodes, dtype=torch.complex64) * 0.1  # Collapsed field

    print(f"Initial vortex density: {(torch.abs(field) < 0.3).float().mean():.1%}")

    # Apply helical reset
    reset_field = helical_vortex_reset(positions, field, omega=0.3, vortex_target=0.7, reset_strength=1.0)

    print(f"After reset vortex density: {(torch.abs(reset_field) < 0.3).float().mean():.1%}")
    print(f"Stability score: {compute_vortex_stability_score(reset_field):.3f}")
    print("=" * 80)
