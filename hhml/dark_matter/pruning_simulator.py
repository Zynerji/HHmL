#!/usr/bin/env python3
"""
Coherence-Based Pruning Simulator for Dark Matter Theory
==========================================================

Implements the holographic pruning mechanism that removes discordant
multiverse branches based on coherence thresholds.

Core Algorithm:
1. Compute mean hologram from all branches
2. Measure coherence of each branch to hologram
3. Prune branches below threshold
4. Measure dark matter fraction (pruned mass / total mass)

Theory Mapping:
- High coherence = physical universe (kept)
- Low coherence = pruned timeline (becomes dark matter)
- Coherence threshold = holographic projection cutoff

Author: HHmL Project
Date: 2025-12-17
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .multiverse_generator import MultiverseBranch


@dataclass
class PruningResult:
    """Results from coherence-based pruning operation."""

    hologram_field: torch.Tensor
    """Mean field configuration of kept branches (the physical universe)."""

    kept_branches: List[MultiverseBranch]
    """Branches above coherence threshold (physical timelines)."""

    pruned_branches: List[MultiverseBranch]
    """Branches below coherence threshold (dark matter residue)."""

    coherence_scores: List[float]
    """Coherence score for each original branch."""

    dark_fraction: float
    """Fraction of total mass in pruned branches (target: 0.27)."""

    hologram_quality: float
    """Quality score of resulting hologram (vortex density)."""

    threshold_used: float
    """Coherence threshold that was applied."""

    entropy_before: float
    """Total von Neumann entropy before pruning."""

    entropy_after: float
    """Total entropy after pruning (hologram + residue)."""

    entropy_conservation: float
    """Ratio of after/before entropy (should be ≈1.0)."""


def compute_coherence(branch1: MultiverseBranch,
                      branch2: MultiverseBranch) -> float:
    """
    Measure coherence between two branches via normalized field correlation.

    Coherence metric:
    C(ψ₁, ψ₂) = 1 - ||ψ₁ - ψ₂||₂ / (||ψ₁||₂ + ||ψ₂||₂)

    Returns:
        Coherence in [0, 1], where 1 = identical, 0 = maximally different

    Theory:
        - High coherence → branches evolved similarly → both physical
        - Low coherence → timeline diverged → incompatible with hologram
    """
    field1 = branch1.geometry.field
    field2 = branch2.geometry.field

    # Ensure fields are same shape
    if field1.shape != field2.shape:
        raise ValueError(f"Field shapes don't match: {field1.shape} vs {field2.shape}")

    # Compute normalized difference
    diff_norm = torch.norm(field1 - field2)
    sum_norm = torch.norm(field1) + torch.norm(field2)

    if sum_norm < 1e-12:
        return 1.0  # Both fields are zero → perfectly coherent

    coherence = 1.0 - (diff_norm / sum_norm).item()

    # Clamp to [0, 1] for numerical stability
    return max(0.0, min(1.0, coherence))


def compute_mean_field(branches: List[MultiverseBranch]) -> torch.Tensor:
    """
    Compute mean field configuration across multiple branches.

    This represents the "hologram" - the average quantum state
    across all coherent timelines.

    Args:
        branches: List of multiverse branches

    Returns:
        Mean complex field tensor
    """
    if len(branches) == 0:
        raise ValueError("Cannot compute mean of empty branch list")

    # Stack all fields and compute mean
    fields = torch.stack([b.geometry.field for b in branches])
    mean_field = torch.mean(fields, dim=0)

    return mean_field


def prune_discordant(branches: List[MultiverseBranch],
                     threshold: float = 0.8,
                     device: str = 'cuda') -> PruningResult:
    """
    Prune branches with low coherence to mean hologram.

    Algorithm:
    1. Compute mean hologram from all branches
    2. Measure each branch's coherence to hologram
    3. Keep branches with coherence ≥ threshold
    4. Prune branches with coherence < threshold
    5. Measure dark matter fraction

    Args:
        branches: List of multiverse branches to prune
        threshold: Coherence cutoff (0.7-0.9 typical, 0.82 targets 27% dark)
        device: 'cuda' or 'cpu'

    Returns:
        PruningResult with hologram, kept/pruned branches, metrics

    Example:
        >>> result = prune_discordant(branches, threshold=0.82)
        >>> print(f"Dark fraction: {result.dark_fraction:.2%}")
        >>> print(f"Kept {len(result.kept_branches)}/{len(branches)} branches")
    """
    if len(branches) == 0:
        raise ValueError("Cannot prune empty branch list")

    # Compute initial entropy
    entropy_before = sum(b.compute_entropy() for b in branches)

    # Compute mean hologram from ALL branches initially
    hologram_field = compute_mean_field(branches)

    # Create temporary branch for coherence comparison
    from copy import deepcopy
    hologram_branch = deepcopy(branches[0])
    hologram_branch.geometry.field = hologram_field
    hologram_branch.branch_id = -1  # Special ID for hologram

    # Measure coherence of each branch to hologram
    coherence_scores = []
    for branch in branches:
        coherence = compute_coherence(branch, hologram_branch)
        coherence_scores.append(coherence)
        branch.coherence_score = coherence

    # Separate kept vs pruned based on threshold
    kept_branches = []
    pruned_branches = []

    for branch, coherence in zip(branches, coherence_scores):
        if coherence >= threshold:
            kept_branches.append(branch)
        else:
            pruned_branches.append(branch)
            branch.is_pruned = True

    # Recompute hologram using only KEPT branches
    if len(kept_branches) > 0:
        hologram_field = compute_mean_field(kept_branches)
    else:
        # Fallback: if all pruned, use original hologram
        hologram_field = compute_mean_field(branches)
        kept_branches = branches[:1]  # Keep at least one
        pruned_branches = branches[1:]

    # Compute dark matter fraction
    total_mass = sum(b.compute_mass() for b in branches)
    pruned_mass = sum(b.compute_mass() for b in pruned_branches)

    if total_mass > 0:
        dark_fraction = pruned_mass / total_mass
    else:
        dark_fraction = 0.0

    # Compute hologram quality (vortex density proxy)
    hologram_quality = _compute_hologram_quality(hologram_field, device)

    # Compute final entropy (hologram + residue)
    hologram_entropy = _compute_field_entropy(hologram_field)
    residue_entropy = sum(b.entropy for b in pruned_branches)
    entropy_after = hologram_entropy + residue_entropy

    # Entropy conservation check
    if entropy_before > 1e-12:
        entropy_conservation = entropy_after / entropy_before
    else:
        entropy_conservation = 1.0

    return PruningResult(
        hologram_field=hologram_field,
        kept_branches=kept_branches,
        pruned_branches=pruned_branches,
        coherence_scores=coherence_scores,
        dark_fraction=dark_fraction,
        hologram_quality=hologram_quality,
        threshold_used=threshold,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        entropy_conservation=entropy_conservation
    )


def sweep_pruning_thresholds(branches: List[MultiverseBranch],
                              thresholds: Optional[List[float]] = None,
                              device: str = 'cuda') -> List[PruningResult]:
    """
    Sweep across multiple coherence thresholds to find optimal dark fraction.

    Target: Find threshold yielding dark_fraction ≈ 0.27 (ΛCDM value)

    Args:
        branches: Multiverse branches to prune
        thresholds: List of coherence thresholds to test (default: 0.5 to 0.95)
        device: 'cuda' or 'cpu'

    Returns:
        List of PruningResult for each threshold, sorted by threshold

    Example:
        >>> results = sweep_pruning_thresholds(branches)
        >>> for r in results:
        >>>     if abs(r.dark_fraction - 0.27) < 0.05:
        >>>         print(f"Threshold {r.threshold_used:.2f} → {r.dark_fraction:.2%} dark")
    """
    if thresholds is None:
        # Default: sweep from 0.5 to 0.95 in 0.05 increments
        thresholds = np.arange(0.50, 0.96, 0.05).tolist()

    results = []
    for threshold in thresholds:
        result = prune_discordant(branches, threshold, device)
        results.append(result)
        print(f"Threshold {threshold:.2f}: "
              f"Dark={result.dark_fraction:.2%}, "
              f"Kept={len(result.kept_branches)}/{len(branches)}, "
              f"Quality={result.hologram_quality:.3f}")

    return results


def find_optimal_threshold(branches: List[MultiverseBranch],
                           target_dark_fraction: float = 0.27,
                           tolerance: float = 0.02,
                           device: str = 'cuda') -> Tuple[float, PruningResult]:
    """
    Binary search for threshold yielding target dark matter fraction.

    Uses bisection to find threshold where:
    |dark_fraction - target| < tolerance

    Args:
        branches: Multiverse branches to prune
        target_dark_fraction: Target dark matter fraction (0.27 for ΛCDM)
        tolerance: Acceptable error (default: ±2%)
        device: 'cuda' or 'cpu'

    Returns:
        (optimal_threshold, PruningResult)

    Raises:
        ValueError: If no threshold in [0, 1] yields target fraction
    """
    low, high = 0.0, 1.0
    max_iterations = 20
    best_result = None
    best_threshold = None
    best_error = float('inf')

    for iteration in range(max_iterations):
        threshold = (low + high) / 2.0
        result = prune_discordant(branches, threshold, device)
        error = abs(result.dark_fraction - target_dark_fraction)

        print(f"Iteration {iteration+1}: threshold={threshold:.4f}, "
              f"dark={result.dark_fraction:.4f}, error={error:.4f}")

        # Track best result
        if error < best_error:
            best_error = error
            best_result = result
            best_threshold = threshold

        # Check convergence
        if error < tolerance:
            print(f"✓ Converged: threshold={threshold:.4f} yields {result.dark_fraction:.2%} dark matter")
            return threshold, result

        # Binary search update
        if result.dark_fraction < target_dark_fraction:
            # Too little dark matter → lower threshold (prune more)
            high = threshold
        else:
            # Too much dark matter → raise threshold (prune less)
            low = threshold

    # Return best result even if didn't meet tolerance
    if best_result is not None:
        print(f"⚠ Did not converge to tolerance, returning best: "
              f"threshold={best_threshold:.4f}, error={best_error:.4f}")
        return best_threshold, best_result
    else:
        raise ValueError(f"Could not find threshold yielding dark fraction near {target_dark_fraction}")


def _compute_hologram_quality(field: torch.Tensor, device: str = 'cuda') -> float:
    """
    Compute quality score of hologram field.

    Proxy metrics:
    - Field uniformity (low variance = high quality)
    - Energy concentration (high peak density = organized vortices)
    - Smoothness (low gradient = stable configuration)

    Returns quality score in [0, 1]
    """
    # Energy distribution
    energy = torch.abs(field)**2
    mean_energy = torch.mean(energy)

    if mean_energy < 1e-12:
        return 0.0

    # Uniformity: 1 / (1 + variance/mean^2)
    variance = torch.var(energy)
    uniformity = 1.0 / (1.0 + (variance / mean_energy**2))

    # Normalized mean energy (higher = better organized)
    max_possible_energy = torch.max(energy)
    if max_possible_energy > 1e-12:
        concentration = mean_energy / max_possible_energy
    else:
        concentration = 0.0

    # Combine metrics
    quality = 0.7 * uniformity + 0.3 * concentration

    return quality.item()


def _compute_field_entropy(field: torch.Tensor) -> float:
    """
    Compute von Neumann entropy of field configuration.

    Approximation: Treat |ψ|² as probability distribution
    S = -Σ p_i log(p_i)
    """
    probs = torch.abs(field)**2
    probs = probs / probs.sum()

    # Remove zeros to avoid log(0)
    probs = probs[probs > 1e-12]

    if len(probs) == 0:
        return 0.0

    entropy = -torch.sum(probs * torch.log(probs))
    return entropy.item()


def analyze_pruning_statistics(result: PruningResult) -> Dict[str, float]:
    """
    Compute statistical analysis of pruning result.

    Returns dictionary with:
    - mean_kept_coherence: Average coherence of kept branches
    - mean_pruned_coherence: Average coherence of pruned branches
    - coherence_gap: Difference between kept and pruned means
    - pruning_efficiency: How well threshold separates branches
    """
    kept_coherences = [b.coherence_score for b in result.kept_branches]
    pruned_coherences = [b.coherence_score for b in result.pruned_branches]

    stats = {
        'num_kept': len(result.kept_branches),
        'num_pruned': len(result.pruned_branches),
        'mean_kept_coherence': np.mean(kept_coherences) if kept_coherences else 0.0,
        'mean_pruned_coherence': np.mean(pruned_coherences) if pruned_coherences else 0.0,
        'std_kept_coherence': np.std(kept_coherences) if kept_coherences else 0.0,
        'std_pruned_coherence': np.std(pruned_coherences) if pruned_coherences else 0.0,
        'dark_fraction': result.dark_fraction,
        'hologram_quality': result.hologram_quality,
        'entropy_conservation': result.entropy_conservation,
    }

    # Coherence gap (larger = better separation)
    if kept_coherences and pruned_coherences:
        stats['coherence_gap'] = stats['mean_kept_coherence'] - stats['mean_pruned_coherence']
    else:
        stats['coherence_gap'] = 0.0

    # Pruning efficiency: how cleanly threshold separates distributions
    all_coherences = kept_coherences + pruned_coherences
    if len(all_coherences) > 1:
        total_variance = np.var(all_coherences)
        within_variance = (stats['std_kept_coherence']**2 * len(kept_coherences) +
                          stats['std_pruned_coherence']**2 * len(pruned_coherences)) / len(all_coherences)

        if total_variance > 1e-12:
            stats['pruning_efficiency'] = 1.0 - (within_variance / total_variance)
        else:
            stats['pruning_efficiency'] = 1.0
    else:
        stats['pruning_efficiency'] = 0.0

    return stats


def visualize_pruning(result: PruningResult,
                      output_path: str = 'pruning_analysis.png'):
    """
    Create visualization of pruning results.

    Plots:
    - Coherence distribution (kept vs pruned)
    - Mass distribution (hologram vs residue)
    - Entropy before/after
    - Threshold location
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)

    # 1. Coherence histogram
    ax1 = fig.add_subplot(gs[0, 0])
    kept_coherences = [b.coherence_score for b in result.kept_branches]
    pruned_coherences = [b.coherence_score for b in result.pruned_branches]

    ax1.hist(kept_coherences, bins=20, alpha=0.7, color='green',
             edgecolor='black', label='Kept (Physical)')
    ax1.hist(pruned_coherences, bins=20, alpha=0.7, color='red',
             edgecolor='black', label='Pruned (Dark Matter)')
    ax1.axvline(result.threshold_used, color='black', linestyle='--',
                linewidth=2, label=f'Threshold={result.threshold_used:.2f}')
    ax1.set_xlabel('Coherence Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Coherence Distribution')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. Mass distribution
    ax2 = fig.add_subplot(gs[0, 1])
    kept_masses = [b.compute_mass() for b in result.kept_branches]
    pruned_masses = [b.compute_mass() for b in result.pruned_branches]

    total_kept = sum(kept_masses)
    total_pruned = sum(pruned_masses)

    ax2.bar(['Hologram\n(Physical)', 'Residue\n(Dark Matter)'],
            [total_kept, total_pruned],
            color=['green', 'red'], alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Total Mass (Field Energy)')
    ax2.set_title(f'Mass Distribution (Dark Fraction: {result.dark_fraction:.2%})')
    ax2.grid(alpha=0.3, axis='y')

    # Add percentage labels
    for i, (val, label) in enumerate([(total_kept, 'Physical'), (total_pruned, 'Dark')]):
        pct = val / (total_kept + total_pruned) * 100
        ax2.text(i, val, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 3. Entropy conservation
    ax3 = fig.add_subplot(gs[0, 2])
    entropies = [result.entropy_before, result.entropy_after]
    labels = ['Before\nPruning', 'After\nPruning']
    colors = ['blue', 'purple']

    ax3.bar(labels, entropies, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Von Neumann Entropy (nats)')
    ax3.set_title(f'Entropy Conservation (Ratio: {result.entropy_conservation:.3f})')
    ax3.grid(alpha=0.3, axis='y')

    # 4. Coherence vs Mass scatter
    ax4 = fig.add_subplot(gs[1, 0])
    all_coherences = [b.coherence_score for b in result.kept_branches + result.pruned_branches]
    all_masses = [b.compute_mass() for b in result.kept_branches + result.pruned_branches]
    colors_scatter = ['green' if c >= result.threshold_used else 'red' for c in all_coherences]

    ax4.scatter(all_coherences, all_masses, c=colors_scatter, alpha=0.6, s=50)
    ax4.axvline(result.threshold_used, color='black', linestyle='--', linewidth=2)
    ax4.set_xlabel('Coherence Score')
    ax4.set_ylabel('Branch Mass')
    ax4.set_title('Coherence vs Mass')
    ax4.grid(alpha=0.3)

    # 5. Quality metrics
    ax5 = fig.add_subplot(gs[1, 1])
    metrics = {
        'Hologram\nQuality': result.hologram_quality,
        'Entropy\nConserv.': min(result.entropy_conservation, 1.2),  # Cap at 1.2 for viz
        'Dark\nFraction': result.dark_fraction / 0.4,  # Normalize to ~1.0 at target
    }

    ax5.bar(metrics.keys(), metrics.values(), color=['blue', 'purple', 'orange'],
            alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Score')
    ax5.set_title('Quality Metrics')
    ax5.axhline(1.0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax5.set_ylim(0, 1.3)
    ax5.grid(alpha=0.3, axis='y')

    # 6. Statistics table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    stats = analyze_pruning_statistics(result)
    stats_text = f"""
    Pruning Statistics
    {'='*30}
    Threshold: {result.threshold_used:.3f}

    Branches:
      Kept: {stats['num_kept']}
      Pruned: {stats['num_pruned']}

    Coherence:
      Kept: {stats['mean_kept_coherence']:.3f} ± {stats['std_kept_coherence']:.3f}
      Pruned: {stats['mean_pruned_coherence']:.3f} ± {stats['std_pruned_coherence']:.3f}
      Gap: {stats['coherence_gap']:.3f}

    Dark Matter:
      Fraction: {stats['dark_fraction']:.2%}
      Target: 27.0%
      Error: {abs(stats['dark_fraction'] - 0.27):.2%}

    Quality:
      Hologram: {stats['hologram_quality']:.3f}
      Entropy Ratio: {stats['entropy_conservation']:.3f}
      Pruning Efficiency: {stats['pruning_efficiency']:.3f}
    """

    ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax6.transAxes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Pruning visualization saved to {output_path}")


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("PRUNING SIMULATOR TEST")
    print("="*80)
    print()

    print("Configuration:")
    print("  Coherence threshold: 0.82 (targeting 27% dark matter)")
    print("  Expected: ~27% of mass in pruned branches")
    print()

    print("Note: Full test requires MultiverseBranch instances")
    print("Run via: python simulations/dark_matter/test_multiverse_generation.py")
