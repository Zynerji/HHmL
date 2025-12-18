#!/usr/bin/env python3
"""
Dark Matter Residue Analysis Module
====================================

Analyzes pruned multiverse branches to measure dark matter signatures.

Metrics:
1. Density anomaly: Excess mass in pruned sectors
2. Entropy contribution: Informational residue (von Neumann entropy)
3. Gravitational signature: Field curvature from residue
4. Fractal dimension: Spatial distribution pattern (box-counting)
5. Rotation curve test: Does residue explain flat galactic rotation?

Theory:
- Pruned branches = dark matter candidates
- Must be gravitationally active (contributes to mass)
- Must be electromagnetically inert (no vortex interactions)
- Target: 27% of total mass in residue

Author: HHmL Project
Date: 2025-12-17
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .multiverse_generator import MultiverseBranch
from .pruning_simulator import PruningResult


@dataclass
class DarkMatterMetrics:
    """Comprehensive dark matter signature measurements."""

    density_anomaly: float
    """Excess vortex density in pruned branches vs hologram."""

    residue_entropy: float
    """Total von Neumann entropy in pruned branches."""

    hologram_entropy: float
    """Von Neumann entropy of kept branches hologram."""

    entropy_ratio: float
    """residue_entropy / total_entropy (should be ~0.27)."""

    curvature_residue: float
    """RMS field curvature from pruned branches."""

    fractal_dimension: float
    """Box-counting dimension of residue distribution (target ~2.6)."""

    rotation_curve_match: float
    """Goodness-of-fit to flat rotation curve (0=bad, 1=perfect)."""

    mass_fraction: float
    """Pruned mass / total mass (should match dark_fraction from pruning)."""

    spatial_clustering: float
    """Hopkins statistic for clustering (0=random, 1=clustered)."""

    field_coherence_residue: float
    """Average field correlation within pruned branches."""


def measure_dark_residue(pruning_result: PruningResult,
                         device: str = 'cuda') -> DarkMatterMetrics:
    """
    Quantify dark matter signatures from pruned multiverse branches.

    Analyzes:
    - Mass distribution (pruned vs kept)
    - Entropy contribution (information content)
    - Spatial distribution (clustering, fractality)
    - Gravitational effects (field curvature, rotation curves)

    Args:
        pruning_result: Output from prune_discordant()
        device: 'cuda' or 'cpu'

    Returns:
        DarkMatterMetrics with comprehensive measurements

    Example:
        >>> result = prune_discordant(branches, threshold=0.82)
        >>> metrics = measure_dark_residue(result)
        >>> print(f"Dark fraction: {metrics.mass_fraction:.2%}")
        >>> print(f"Fractal dimension: {metrics.fractal_dimension:.2f}")
    """
    pruned = pruning_result.pruned_branches
    kept = pruning_result.kept_branches

    if len(pruned) == 0:
        # No dark matter if nothing pruned
        return DarkMatterMetrics(
            density_anomaly=0.0,
            residue_entropy=0.0,
            hologram_entropy=pruning_result.entropy_after,
            entropy_ratio=0.0,
            curvature_residue=0.0,
            fractal_dimension=0.0,
            rotation_curve_match=0.0,
            mass_fraction=0.0,
            spatial_clustering=0.0,
            field_coherence_residue=0.0
        )

    # 1. Density anomaly
    density_anomaly = _compute_density_anomaly(kept, pruned)

    # 2. Entropy measurements
    residue_entropy = sum(b.entropy for b in pruned)
    hologram_entropy = pruning_result.entropy_after - residue_entropy
    total_entropy = pruning_result.entropy_before
    entropy_ratio = residue_entropy / total_entropy if total_entropy > 0 else 0.0

    # 3. Gravitational signature (field curvature)
    curvature_residue = _compute_field_curvature(pruned, device)

    # 4. Fractal dimension (box-counting)
    fractal_dimension = _compute_fractal_dimension(pruned)

    # 5. Rotation curve test
    rotation_curve_match = _test_rotation_curve(kept, pruned, device)

    # 6. Mass fraction (should match pruning_result.dark_fraction)
    total_mass = sum(b.compute_mass() for b in kept + pruned)
    pruned_mass = sum(b.compute_mass() for b in pruned)
    mass_fraction = pruned_mass / total_mass if total_mass > 0 else 0.0

    # 7. Spatial clustering (Hopkins statistic)
    spatial_clustering = _compute_hopkins_statistic(pruned)

    # 8. Field coherence within residue
    field_coherence_residue = _compute_intra_coherence(pruned)

    return DarkMatterMetrics(
        density_anomaly=density_anomaly,
        residue_entropy=residue_entropy,
        hologram_entropy=hologram_entropy,
        entropy_ratio=entropy_ratio,
        curvature_residue=curvature_residue,
        fractal_dimension=fractal_dimension,
        rotation_curve_match=rotation_curve_match,
        mass_fraction=mass_fraction,
        spatial_clustering=spatial_clustering,
        field_coherence_residue=field_coherence_residue
    )


def _compute_density_anomaly(kept: List[MultiverseBranch],
                              pruned: List[MultiverseBranch]) -> float:
    """
    Measure excess vortex density in pruned branches vs hologram.

    Returns:
        Positive = pruned branches more dense (unexpected for low-quality)
        Negative = pruned branches less dense (expected)
    """
    if len(kept) == 0 or len(pruned) == 0:
        return 0.0

    # Approximate vortex density via field magnitude variance
    kept_densities = [torch.std(torch.abs(b.geometry.field)).item() for b in kept]
    pruned_densities = [torch.std(torch.abs(b.geometry.field)).item() for b in pruned]

    mean_kept = np.mean(kept_densities)
    mean_pruned = np.mean(pruned_densities)

    anomaly = mean_pruned - mean_kept
    return anomaly


def _compute_field_curvature(branches: List[MultiverseBranch],
                              device: str = 'cuda') -> float:
    """
    Compute RMS field curvature (gravitational signature).

    Curvature ≈ Laplacian of field:
    ∇²ψ ≈ (ψ_neighbors - ψ_center) / distance²

    Returns:
        RMS curvature across all pruned branches
    """
    if len(branches) == 0:
        return 0.0

    curvatures = []

    for branch in branches:
        field = branch.geometry.field
        positions = branch.geometry.positions

        # Compute approximate Laplacian via finite differences
        # Use KNN to find neighbors
        from scipy.spatial import cKDTree

        tree = cKDTree(positions.cpu().numpy())
        k_neighbors = 8  # Use 8 nearest neighbors

        laplacian = torch.zeros_like(field.real)

        for i in range(len(positions)):
            # Find k nearest neighbors
            dists, indices = tree.query(positions[i].cpu().numpy(), k=k_neighbors+1)
            indices = indices[1:]  # Exclude self
            dists = dists[1:]

            if len(indices) > 0:
                # Approximate Laplacian
                neighbor_vals = field[indices]
                center_val = field[i]

                # Weighted average based on distance
                weights = 1.0 / (dists**2 + 1e-6)
                weights = weights / weights.sum()

                laplacian_val = torch.sum(
                    torch.abs(neighbor_vals - center_val) *
                    torch.tensor(weights, device=field.device, dtype=field.dtype)
                )
                laplacian[i] = laplacian_val.real

        # RMS curvature for this branch
        rms = torch.sqrt(torch.mean(laplacian**2))
        curvatures.append(rms.item())

    # Average across all pruned branches
    return np.mean(curvatures)


def _compute_fractal_dimension(branches: List[MultiverseBranch],
                                min_box_size: float = 0.01,
                                max_box_size: float = 1.0,
                                num_sizes: int = 10) -> float:
    """
    Compute fractal dimension via box-counting method.

    Theory:
    - Cosmic web has fractal dimension D ≈ 2.6
    - If dark matter = pruned residue, should match this structure

    Algorithm:
    1. Overlay grid of boxes at different scales
    2. Count boxes containing vortices
    3. Plot log(count) vs log(1/box_size)
    4. Slope = fractal dimension

    Returns:
        Fractal dimension D (2.0 = surface, 3.0 = volume)
    """
    if len(branches) == 0:
        return 0.0

    # Collect all vortex positions from pruned branches
    all_positions = []
    for branch in branches:
        # Identify vortices (local field maxima)
        field_mag = torch.abs(branch.geometry.field)
        vortex_mask = field_mag > field_mag.mean() + field_mag.std()
        vortex_positions = branch.geometry.positions[vortex_mask]
        all_positions.append(vortex_positions.cpu().numpy())

    if len(all_positions) == 0:
        return 0.0

    all_positions = np.vstack(all_positions)

    # Normalize positions to [0, 1]³
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    pos_norm = (all_positions - pos_min) / (pos_max - pos_min + 1e-6)

    # Box-counting at different scales
    box_sizes = np.logspace(np.log10(min_box_size), np.log10(max_box_size), num_sizes)
    counts = []

    for box_size in box_sizes:
        # Create grid
        num_boxes = int(1.0 / box_size) + 1
        grid = np.zeros((num_boxes, num_boxes, num_boxes), dtype=bool)

        # Mark boxes containing points
        for pos in pos_norm:
            i = int(pos[0] / box_size)
            j = int(pos[1] / box_size)
            k = int(pos[2] / box_size)

            # Clamp to grid bounds
            i = min(i, num_boxes - 1)
            j = min(j, num_boxes - 1)
            k = min(k, num_boxes - 1)

            grid[i, j, k] = True

        # Count occupied boxes
        count = np.sum(grid)
        counts.append(count)

    # Linear regression: log(count) = D * log(1/box_size) + const
    # D = slope
    log_box_sizes = np.log(1.0 / box_sizes)
    log_counts = np.log(np.array(counts) + 1)  # +1 to avoid log(0)

    # Fit line
    coeffs = np.polyfit(log_box_sizes, log_counts, 1)
    fractal_dim = coeffs[0]

    # Clamp to reasonable range [1.0, 3.0]
    fractal_dim = max(1.0, min(3.0, fractal_dim))

    return fractal_dim


def _test_rotation_curve(kept: List[MultiverseBranch],
                          pruned: List[MultiverseBranch],
                          device: str = 'cuda',
                          num_radii: int = 20) -> float:
    """
    Test if residue mass distribution produces flat rotation curve.

    Rotation curve: v(r) = velocity at radius r
    - Keplerian (no dark matter): v ∝ 1/√r (decaying)
    - Flat (with dark matter): v ≈ constant

    Algorithm:
    1. Compute mass enclosed at various radii
    2. Calculate v(r) = √(GM(r)/r)
    3. Measure flatness (std of v vs r)
    4. Return goodness-of-fit (0 = Keplerian, 1 = flat)

    Returns:
        Match score in [0, 1], where 1 = perfectly flat rotation
    """
    if len(kept) == 0 or len(pruned) == 0:
        return 0.0

    # Combine all positions and masses
    all_positions = []
    all_masses = []

    for branch in kept + pruned:
        positions = branch.geometry.positions.cpu().numpy()
        field_mag = torch.abs(branch.geometry.field).cpu().numpy()
        masses = field_mag**2  # Mass ∝ |ψ|²

        all_positions.append(positions)
        all_masses.append(masses)

    all_positions = np.vstack(all_positions)
    all_masses = np.concatenate(all_masses)

    # Compute radii from center
    center = np.mean(all_positions, axis=0)
    radii = np.linalg.norm(all_positions - center, axis=1)

    # Define radial bins
    r_min = radii.min() + 1e-6
    r_max = radii.max()
    r_bins = np.linspace(r_min, r_max, num_radii)

    # Compute enclosed mass and velocity at each radius
    velocities = []

    for r in r_bins:
        # Mass enclosed within radius r
        enclosed_mask = radii <= r
        M_enclosed = all_masses[enclosed_mask].sum()

        if M_enclosed < 1e-12 or r < 1e-12:
            continue

        # Circular velocity: v = √(GM/r)
        # Normalize G=1 for simplicity
        v = np.sqrt(M_enclosed / r)
        velocities.append(v)

    if len(velocities) < 3:
        return 0.0

    velocities = np.array(velocities)

    # Measure flatness: low variance = flat rotation curve
    v_mean = np.mean(velocities)
    v_std = np.std(velocities)

    if v_mean < 1e-12:
        return 0.0

    # Coefficient of variation: CV = std/mean
    # Low CV → flat curve
    cv = v_std / v_mean

    # Map CV to [0, 1] score
    # CV = 0 → perfect flatness → score = 1
    # CV > 0.5 → highly variable → score ≈ 0
    flatness_score = np.exp(-cv * 2)  # Exponential decay

    return float(flatness_score)


def _compute_hopkins_statistic(branches: List[MultiverseBranch],
                                sample_size: int = 100) -> float:
    """
    Compute Hopkins statistic for spatial clustering.

    Hopkins statistic H:
    - H ≈ 0.5: Random distribution
    - H → 1.0: Highly clustered
    - H → 0.0: Highly uniform

    Algorithm:
    1. Sample n random points from data
    2. Measure distance to nearest neighbor: u_i
    3. Sample n random points from uniform distribution
    4. Measure distance to nearest data point: w_i
    5. H = Σw_i / (Σu_i + Σw_i)

    Returns:
        Hopkins statistic in [0, 1]
    """
    if len(branches) == 0:
        return 0.5

    # Collect all vortex positions
    all_positions = []
    for branch in branches:
        field_mag = torch.abs(branch.geometry.field)
        vortex_mask = field_mag > field_mag.mean() + field_mag.std()
        vortex_positions = branch.geometry.positions[vortex_mask]
        all_positions.append(vortex_positions.cpu().numpy())

    if len(all_positions) == 0:
        return 0.5

    all_positions = np.vstack(all_positions)

    if len(all_positions) < sample_size:
        sample_size = len(all_positions) // 2

    if sample_size < 2:
        return 0.5

    from scipy.spatial import cKDTree

    tree = cKDTree(all_positions)

    # 1. Sample from data, measure u_i (distance to nearest neighbor)
    sample_indices = np.random.choice(len(all_positions), sample_size, replace=False)
    u_distances = []

    for idx in sample_indices:
        pos = all_positions[idx]
        # Find 2 nearest (1st is self, 2nd is true nearest neighbor)
        dists, _ = tree.query(pos, k=2)
        u_distances.append(dists[1])

    # 2. Sample from uniform distribution, measure w_i (distance to nearest data point)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)

    random_points = np.random.uniform(pos_min, pos_max, size=(sample_size, 3))
    w_distances = []

    for pos in random_points:
        dists, _ = tree.query(pos, k=1)
        w_distances.append(dists)

    # 3. Compute Hopkins statistic
    u_sum = np.sum(u_distances)
    w_sum = np.sum(w_distances)

    if u_sum + w_sum < 1e-12:
        return 0.5

    H = w_sum / (u_sum + w_sum)

    return float(H)


def _compute_intra_coherence(branches: List[MultiverseBranch]) -> float:
    """
    Measure average coherence between pruned branches (internal structure).

    High coherence → pruned branches are similar to each other
    Low coherence → pruned branches are diverse

    Returns:
        Average pairwise coherence among pruned branches
    """
    if len(branches) < 2:
        return 0.0

    from .pruning_simulator import compute_coherence

    coherences = []

    # Compute all pairwise coherences
    for i in range(len(branches)):
        for j in range(i+1, len(branches)):
            c = compute_coherence(branches[i], branches[j])
            coherences.append(c)

    if len(coherences) == 0:
        return 0.0

    return np.mean(coherences)


def visualize_dark_matter_signatures(metrics: DarkMatterMetrics,
                                     pruning_result: PruningResult,
                                     output_path: str = 'dark_matter_signatures.png'):
    """
    Create comprehensive visualization of dark matter signatures.

    Plots:
    - Mass distribution (physical vs dark)
    - Entropy distribution
    - Fractal dimension comparison
    - Rotation curve flatness
    - Spatial clustering
    - Curvature signature
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)

    # 1. Mass distribution pie chart
    ax1 = fig.add_subplot(gs[0, 0])
    mass_physical = (1 - metrics.mass_fraction) * 100
    mass_dark = metrics.mass_fraction * 100

    ax1.pie([mass_physical, mass_dark],
            labels=[f'Physical\n{mass_physical:.1f}%', f'Dark Matter\n{mass_dark:.1f}%'],
            colors=['green', 'purple'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax1.set_title(f'Mass Distribution\n(Target: 27% Dark)', fontweight='bold')

    # Add distance from target
    error = abs(mass_dark - 27.0)
    ax1.text(0, -1.4, f'Error: ±{error:.1f}%',
             ha='center', fontsize=10,
             color='red' if error > 5 else 'green')

    # 2. Entropy distribution
    ax2 = fig.add_subplot(gs[0, 1])
    entropy_vals = [metrics.hologram_entropy, metrics.residue_entropy]
    labels = ['Hologram\n(Physical)', 'Residue\n(Dark Matter)']
    colors = ['green', 'purple']

    ax2.bar(labels, entropy_vals, color=colors, alpha=0.7, edgecolor='black', width=0.6)
    ax2.set_ylabel('Von Neumann Entropy (nats)', fontweight='bold')
    ax2.set_title(f'Entropy Distribution\n(Ratio: {metrics.entropy_ratio:.2f})', fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')

    # 3. Fractal dimension gauge
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')

    # Target range: 2.4 - 2.8 (cosmic web)
    fractal_text = f"""
    Fractal Dimension
    {'='*25}

    Measured: {metrics.fractal_dimension:.2f}
    Target: 2.6 ± 0.2

    Cosmic Web: 2.4 - 2.8
    Surface: 2.0
    Volume: 3.0

    Match: {'✓ YES' if 2.4 <= metrics.fractal_dimension <= 2.8 else '✗ NO'}
    """
    ax3.text(0.1, 0.5, fractal_text, fontsize=11, family='monospace',
             verticalalignment='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # 4. Rotation curve match
    ax4 = fig.add_subplot(gs[1, 0])
    match_score = metrics.rotation_curve_match
    bar = ax4.barh(['Rotation\nCurve Match'], [match_score],
                   color='green' if match_score > 0.7 else 'orange', alpha=0.7, edgecolor='black')
    ax4.set_xlim(0, 1)
    ax4.set_xlabel('Flatness Score (1.0 = Perfect)', fontweight='bold')
    ax4.set_title('Galactic Rotation Curve Test', fontweight='bold')
    ax4.axvline(0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Threshold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='x')

    # Add score label
    ax4.text(match_score/2, 0, f'{match_score:.2f}',
             ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # 5. Spatial clustering (Hopkins statistic)
    ax5 = fig.add_subplot(gs[1, 1])
    hopkins = metrics.spatial_clustering

    colors_hopkins = ['blue', 'gray', 'red']
    labels_hopkins = ['Uniform\n(H<0.3)', 'Random\n(0.3-0.7)', 'Clustered\n(H>0.7)']
    values_hopkins = [
        1.0 if hopkins < 0.3 else 0.0,
        1.0 if 0.3 <= hopkins <= 0.7 else 0.0,
        1.0 if hopkins > 0.7 else 0.0
    ]

    ax5.bar(labels_hopkins, values_hopkins, color=colors_hopkins, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Active State', fontweight='bold')
    ax5.set_title(f'Spatial Clustering\n(Hopkins H={hopkins:.2f})', fontweight='bold')
    ax5.set_ylim(0, 1.2)
    ax5.grid(alpha=0.3, axis='y')

    # 6. Curvature residue
    ax6 = fig.add_subplot(gs[1, 2])
    curvature = metrics.curvature_residue

    ax6.barh(['Field\nCurvature'], [curvature],
             color='purple', alpha=0.7, edgecolor='black')
    ax6.set_xlabel('RMS Curvature (Gravitational Signature)', fontweight='bold')
    ax6.set_title('Gravitational Field Perturbations', fontweight='bold')
    ax6.grid(alpha=0.3, axis='x')

    # 7. Summary statistics table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    summary = f"""
    Dark Matter Signature Summary
    {'='*80}

    MASS DISTRIBUTION:
      Physical (Hologram): {(1-metrics.mass_fraction)*100:.1f}%
      Dark Matter (Residue): {metrics.mass_fraction*100:.1f}%
      Target Dark Fraction: 27.0%
      Error: ±{abs(metrics.mass_fraction*100 - 27.0):.1f}%

    ENTROPY ANALYSIS:
      Hologram Entropy: {metrics.hologram_entropy:.2f} nats
      Residue Entropy: {metrics.residue_entropy:.2f} nats
      Residue Fraction: {metrics.entropy_ratio:.2%}
      Coherence (Pruning): {pruning_result.entropy_conservation:.3f}

    SPATIAL STRUCTURE:
      Fractal Dimension: {metrics.fractal_dimension:.2f} (target: 2.6 ± 0.2)
      Hopkins Clustering: {metrics.spatial_clustering:.2f} ({'Clustered' if hopkins > 0.7 else 'Random' if hopkins > 0.3 else 'Uniform'})
      Density Anomaly: {metrics.density_anomaly:+.3f}

    GRAVITATIONAL SIGNATURES:
      Rotation Curve Match: {metrics.rotation_curve_match:.2f} ({'Flat' if match_score > 0.7 else 'Keplerian'})
      Field Curvature (RMS): {metrics.curvature_residue:.3f}
      Intra-Residue Coherence: {metrics.field_coherence_residue:.3f}

    THEORY VALIDATION:
      Mass Fraction Match: {'✓ PASS' if abs(metrics.mass_fraction - 0.27) < 0.05 else '✗ FAIL'} (±5% tolerance)
      Fractal Structure Match: {'✓ PASS' if 2.4 <= metrics.fractal_dimension <= 2.8 else '✗ FAIL'} (cosmic web range)
      Rotation Curve Match: {'✓ PASS' if match_score > 0.7 else '✗ FAIL'} (>0.7 threshold)
      Entropy Conservation: {'✓ PASS' if 0.95 <= pruning_result.entropy_conservation <= 1.05 else '✗ FAIL'} (±5% tolerance)
    """

    ax7.text(0.05, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax7.transAxes,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Dark matter signature visualization saved to {output_path}")


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("DARK MATTER RESIDUE ANALYZER TEST")
    print("="*80)
    print()

    print("Metrics computed:")
    print("  1. Density anomaly: Excess mass in pruned sectors")
    print("  2. Entropy contribution: Informational residue")
    print("  3. Gravitational signature: Field curvature")
    print("  4. Fractal dimension: Box-counting (target ~2.6)")
    print("  5. Rotation curve: Flatness test")
    print()

    print("Note: Full test requires PruningResult instance")
    print("Run via: python simulations/dark_matter/full_dark_matter_test.py")
