#!/usr/bin/env python3
"""
Multiverse Branch Generator for Dark Matter Theory Testing
===========================================================

Generates multiple Möbius strip configurations representing alternate
quantum timelines (multiverse branches) with controlled perturbations.

Each branch:
- Shares base topology (Möbius twist, tokamak parameters)
- Has unique initial field configuration
- Represents alternate quantum path diverged at t=0

Theory mapping:
- Each branch = possible universe timeline
- Field perturbations = quantum decoherence points
- Ensemble = multiverse before holographic projection

Author: HHmL Project
Date: 2025-12-17
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy


@dataclass
class MultiverseConfig:
    """Configuration for multiverse branch generation."""

    num_branches: int = 10
    """Number of parallel universe branches to generate."""

    perturbation_scale: float = 0.1
    """Amplitude of random perturbations (0.0 = identical, 1.0 = fully random)."""

    base_strips: int = 10
    """Number of Möbius strips in base configuration."""

    base_nodes: int = 2000
    """Nodes per strip in base configuration."""

    coherence_seed: int = 42
    """Random seed for reproducibility."""

    perturbation_type: str = 'gaussian'
    """Type of perturbation: 'gaussian', 'uniform', 'quantum_noise'."""

    topology_variance: bool = False
    """If True, allow topology variations (strip count, twist angle)."""

    quantum_decoherence: float = 0.0
    """Strength of quantum decoherence effects (0.0 = pure state)."""


class MultiverseBranch:
    """
    Represents a single branch of the multiverse.

    A branch is a complete Möbius strip lattice configuration
    with unique field state representing alternate timeline.
    """

    def __init__(self,
                 strips_geometry,
                 branch_id: int,
                 divergence_time: float = 0.0,
                 parent_branch: Optional['MultiverseBranch'] = None):
        """
        Initialize a multiverse branch.

        Args:
            strips_geometry: SparseTokamakMobiusStrips instance
            branch_id: Unique identifier for this branch
            divergence_time: When this branch diverged from parent
            parent_branch: Branch this diverged from (None = original)
        """
        self.geometry = strips_geometry
        self.branch_id = branch_id
        self.divergence_time = divergence_time
        self.parent_branch = parent_branch

        # Metadata
        self.is_pruned = False
        self.coherence_score = 1.0
        self.entropy = 0.0

    def clone(self, new_id: int) -> 'MultiverseBranch':
        """Create independent copy of this branch."""
        new_geometry = copy.deepcopy(self.geometry)
        return MultiverseBranch(
            new_geometry,
            new_id,
            self.divergence_time,
            self
        )

    def compute_mass(self) -> float:
        """Total field energy (proxy for mass)."""
        return torch.sum(torch.abs(self.geometry.field)**2).item()

    def compute_entropy(self) -> float:
        """Von Neumann entropy of field state."""
        # Approximate via field variance (true entropy requires density matrix)
        field_probs = torch.abs(self.geometry.field)**2
        field_probs = field_probs / field_probs.sum()
        # Remove zeros to avoid log(0)
        field_probs = field_probs[field_probs > 1e-12]
        entropy = -torch.sum(field_probs * torch.log(field_probs))
        self.entropy = entropy.item()
        return self.entropy


def generate_multiverse_branches(
    base_strips_geometry,
    config: MultiverseConfig,
    device: str = 'cuda'
) -> List[MultiverseBranch]:
    """
    Generate ensemble of multiverse branches from base configuration.

    Core algorithm:
    1. Clone base geometry N times
    2. Apply controlled perturbations to each clone
    3. Optionally add quantum decoherence noise
    4. Evolve each branch independently for stabilization

    Args:
        base_strips_geometry: Reference Möbius strip configuration
        config: Multiverse generation parameters
        device: 'cuda' or 'cpu'

    Returns:
        List of MultiverseBranch instances

    Example:
        >>> from hhml.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
        >>> base = SparseTokamakMobiusStrips(num_strips=10, nodes_per_strip=2000)
        >>> config = MultiverseConfig(num_branches=20, perturbation_scale=0.15)
        >>> branches = generate_multiverse_branches(base, config)
        >>> print(f"Generated {len(branches)} universe branches")
    """

    torch.manual_seed(config.coherence_seed)
    np.random.seed(config.coherence_seed)

    branches = []

    for i in range(config.num_branches):
        # Create independent copy
        branch_geometry = copy.deepcopy(base_strips_geometry)

        # Apply perturbations based on type
        if config.perturbation_type == 'gaussian':
            _apply_gaussian_perturbations(branch_geometry, config.perturbation_scale)
        elif config.perturbation_type == 'uniform':
            _apply_uniform_perturbations(branch_geometry, config.perturbation_scale)
        elif config.perturbation_type == 'quantum_noise':
            _apply_quantum_noise(branch_geometry, config.perturbation_scale, config.quantum_decoherence)
        else:
            raise ValueError(f"Unknown perturbation type: {config.perturbation_type}")

        # Create branch instance
        branch = MultiverseBranch(branch_geometry, branch_id=i)

        # Compute initial entropy
        branch.compute_entropy()

        branches.append(branch)

    return branches


def _apply_gaussian_perturbations(geometry, scale: float):
    """
    Apply Gaussian random perturbations to field configuration.

    Perturbs:
    - Amplitudes: A → A + N(0, scale*A)
    - Phases: φ → φ + N(0, scale*2π)
    - Field: ψ → ψ + N(0, scale*|ψ|)
    """
    # Amplitude perturbations
    amp_noise = torch.randn_like(geometry.amplitudes) * scale
    geometry.amplitudes = geometry.amplitudes * (1.0 + amp_noise)
    geometry.amplitudes = torch.clamp(geometry.amplitudes, min=0.1, max=10.0)

    # Phase perturbations
    phase_noise = torch.randn_like(geometry.phases) * scale * 2 * np.pi
    geometry.phases = (geometry.phases + phase_noise) % (2 * np.pi)

    # Direct field perturbations (complex)
    field_noise_real = torch.randn_like(geometry.field.real) * scale
    field_noise_imag = torch.randn_like(geometry.field.imag) * scale
    field_noise = torch.complex(field_noise_real, field_noise_imag)

    geometry.field = geometry.field + field_noise * torch.abs(geometry.field)


def _apply_uniform_perturbations(geometry, scale: float):
    """
    Apply uniform random perturbations.

    Uses uniform distribution U(-scale, +scale) instead of Gaussian.
    More extreme perturbations, less concentrated around mean.
    """
    # Amplitude perturbations (uniform)
    amp_noise = (torch.rand_like(geometry.amplitudes) * 2 - 1) * scale
    geometry.amplitudes = geometry.amplitudes * (1.0 + amp_noise)
    geometry.amplitudes = torch.clamp(geometry.amplitudes, min=0.1, max=10.0)

    # Phase perturbations (uniform)
    phase_noise = (torch.rand_like(geometry.phases) * 2 - 1) * scale * 2 * np.pi
    geometry.phases = (geometry.phases + phase_noise) % (2 * np.pi)

    # Field perturbations
    field_noise_real = (torch.rand_like(geometry.field.real) * 2 - 1) * scale
    field_noise_imag = (torch.rand_like(geometry.field.imag) * 2 - 1) * scale
    field_noise = torch.complex(field_noise_real, field_noise_imag)

    geometry.field = geometry.field + field_noise * torch.abs(geometry.field)


def _apply_quantum_noise(geometry, scale: float, decoherence: float):
    """
    Apply quantum decoherence noise.

    Models quantum measurement-induced decoherence:
    - Pure state: |ψ⟩ → ρ (density matrix)
    - Decoherence: Off-diagonal elements decay
    - Effective: Random phase kicks + amplitude damping

    Args:
        geometry: Möbius strip geometry to perturb
        scale: Overall noise strength
        decoherence: Decoherence rate (0 = no decoherence, 1 = full collapse)
    """
    # Random phase kicks (measurement back-action)
    phase_kicks = torch.randn_like(geometry.phases) * scale * decoherence * 2 * np.pi
    geometry.phases = (geometry.phases + phase_kicks) % (2 * np.pi)

    # Amplitude damping (environment interaction)
    damping = 1.0 - decoherence * scale
    geometry.amplitudes = geometry.amplitudes * damping

    # Add thermal noise to field
    thermal_noise_real = torch.randn_like(geometry.field.real) * scale * np.sqrt(decoherence)
    thermal_noise_imag = torch.randn_like(geometry.field.imag) * scale * np.sqrt(decoherence)
    thermal_noise = torch.complex(thermal_noise_real, thermal_noise_imag)

    geometry.field = geometry.field + thermal_noise


def compute_branch_divergence(branch1: MultiverseBranch,
                               branch2: MultiverseBranch) -> float:
    """
    Measure divergence between two multiverse branches.

    Uses field distance metric:
    D(ψ₁, ψ₂) = ||ψ₁ - ψ₂||₂ / (||ψ₁||₂ + ||ψ₂||₂)

    Returns:
        Divergence in [0, 1], where 0 = identical, 1 = maximally different
    """
    field1 = branch1.geometry.field
    field2 = branch2.geometry.field

    diff_norm = torch.norm(field1 - field2)
    sum_norm = torch.norm(field1) + torch.norm(field2)

    if sum_norm < 1e-12:
        return 0.0

    divergence = diff_norm / sum_norm
    return divergence.item()


def visualize_multiverse(branches: List[MultiverseBranch],
                        output_path: str = 'multiverse_visualization.png'):
    """
    Create visualization of multiverse branch ensemble.

    Plots:
    - 2D projection of field configurations
    - Divergence matrix (pairwise distances)
    - Entropy distribution
    - Mass distribution

    Args:
        branches: List of multiverse branches
        output_path: Where to save visualization
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    N = len(branches)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)

    # 1. Divergence matrix
    ax1 = fig.add_subplot(gs[0, :2])
    divergence_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            divergence_matrix[i, j] = compute_branch_divergence(branches[i], branches[j])

    im1 = ax1.imshow(divergence_matrix, cmap='viridis', aspect='auto')
    ax1.set_title('Branch Divergence Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Branch ID')
    ax1.set_ylabel('Branch ID')
    plt.colorbar(im1, ax=ax1, label='Divergence')

    # 2. Entropy distribution
    ax2 = fig.add_subplot(gs[0, 2])
    entropies = [b.entropy for b in branches]
    ax2.hist(entropies, bins=20, alpha=0.7, edgecolor='black')
    ax2.set_title('Branch Entropy Distribution')
    ax2.set_xlabel('Entropy (nats)')
    ax2.set_ylabel('Count')
    ax2.axvline(np.mean(entropies), color='red', linestyle='--', label='Mean')
    ax2.legend()

    # 3. Mass distribution
    ax3 = fig.add_subplot(gs[1, 2])
    masses = [b.compute_mass() for b in branches]
    ax3.hist(masses, bins=20, alpha=0.7, color='orange', edgecolor='black')
    ax3.set_title('Branch Mass Distribution')
    ax3.set_xlabel('Total Mass (field energy)')
    ax3.set_ylabel('Count')
    ax3.axvline(np.mean(masses), color='red', linestyle='--', label='Mean')
    ax3.legend()

    # 4. Field magnitude comparison (first 4 branches)
    for idx in range(min(4, N)):
        ax = fig.add_subplot(gs[1 + idx//2, idx%2])
        field_mag = torch.abs(branches[idx].geometry.field).cpu().numpy()
        ax.plot(field_mag, alpha=0.6, linewidth=0.5)
        ax.set_title(f'Branch {idx} Field Magnitude')
        ax.set_xlabel('Node Index')
        ax.set_ylabel('|ψ|')
        ax.set_ylim(0, max(field_mag) * 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Multiverse visualization saved to {output_path}")


def export_branches(branches: List[MultiverseBranch],
                   output_file: str = 'multiverse_branches.pt'):
    """
    Export multiverse branches to file for later analysis.

    Args:
        branches: List of branches to export
        output_file: Path to output file (.pt format)
    """
    export_data = {
        'num_branches': len(branches),
        'branches': [
            {
                'branch_id': b.branch_id,
                'field': b.geometry.field.cpu(),
                'amplitudes': b.geometry.amplitudes.cpu(),
                'phases': b.geometry.phases.cpu(),
                'positions': b.geometry.positions.cpu(),
                'entropy': b.entropy,
                'mass': b.compute_mass(),
                'is_pruned': b.is_pruned,
                'coherence_score': b.coherence_score
            }
            for b in branches
        ],
        'metadata': {
            'num_strips': branches[0].geometry.num_strips,
            'nodes_per_strip': branches[0].geometry.nodes_per_strip,
            'total_nodes': branches[0].geometry.positions.shape[0]
        }
    }

    torch.save(export_data, output_file)
    print(f"Exported {len(branches)} branches to {output_file}")


def load_branches(input_file: str, device: str = 'cuda') -> List[Dict]:
    """
    Load previously exported multiverse branches.

    Args:
        input_file: Path to .pt file
        device: Device to load tensors to

    Returns:
        List of branch data dictionaries
    """
    data = torch.load(input_file, map_location=device)
    print(f"Loaded {data['num_branches']} branches from {input_file}")
    return data['branches']


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("MULTIVERSE BRANCH GENERATOR TEST")
    print("="*80)
    print()

    # This is a standalone test - won't work without full HHmL setup
    # Provided for documentation purposes

    print("Configuration:")
    config = MultiverseConfig(
        num_branches=20,
        perturbation_scale=0.15,
        base_strips=10,
        base_nodes=2000,
        perturbation_type='quantum_noise',
        quantum_decoherence=0.05
    )
    print(f"  Branches: {config.num_branches}")
    print(f"  Perturbation: {config.perturbation_scale}")
    print(f"  Type: {config.perturbation_type}")
    print(f"  Decoherence: {config.quantum_decoherence}")
    print()

    print("Note: Full test requires SparseTokamakMobiusStrips instance")
    print("Run via: python simulations/dark_matter/test_multiverse_generation.py")
