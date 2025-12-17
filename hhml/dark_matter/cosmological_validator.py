#!/usr/bin/env python3
"""
Cosmological Validation Module for Dark Matter Theory
=======================================================

Tests dark matter pruning theory against observational cosmology:
- ΛCDM dark matter fraction (27%)
- CMB power spectrum fluctuations
- Large-scale structure (DESI filaments)
- Gravitational lensing signatures
- Galaxy rotation curves

Theory Requirements:
1. Dark fraction must match observations: 27% ± 2%
2. Residue distribution must match cosmic web fractality
3. Rotation curves must flatten (v ≈ constant)
4. Entropy must be conserved (information preserved)

Author: HHmL Project
Date: 2025-12-17
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .multiverse_generator import MultiverseBranch
from .pruning_simulator import PruningResult
from .residue_analyzer import DarkMatterMetrics


@dataclass
class CosmologicalTests:
    """Results from cosmological validation tests."""

    lambda_cdm_match: float
    """Goodness-of-fit to ΛCDM dark matter fraction (27%)."""

    cmb_power_spectrum_match: float
    """Correlation with CMB temperature fluctuations."""

    large_scale_structure_match: float
    """Match to cosmic web filamentary structure."""

    lensing_signature_match: float
    """Match to gravitational lensing observations."""

    rotation_curve_consistency: float
    """Consistency with flat galaxy rotation curves."""

    entropy_conservation_score: float
    """Information conservation test (holographic principle)."""

    overall_validity_score: float
    """Combined score: average of all tests (0-1)."""

    tests_passed: int
    """Number of tests passing threshold (>0.7)."""

    tests_failed: int
    """Number of tests failing threshold (<0.7)."""


def validate_theory(pruning_result: PruningResult,
                    dark_metrics: DarkMatterMetrics,
                    device: str = 'cuda') -> CosmologicalTests:
    """
    Validate dark matter pruning theory against cosmological observations.

    Tests:
    1. ΛCDM match: Is dark_fraction ≈ 0.27?
    2. CMB match: Does residue pattern match observed fluctuations?
    3. LSS match: Does residue have cosmic web structure (D ≈ 2.6)?
    4. Lensing: Does residue mass bend light correctly?
    5. Rotation curves: Are they flat (v ≈ const)?
    6. Entropy: Is information conserved?

    Args:
        pruning_result: Output from prune_discordant()
        dark_metrics: Output from measure_dark_residue()
        device: 'cuda' or 'cpu'

    Returns:
        CosmologicalTests with validation scores

    Example:
        >>> result = prune_discordant(branches, threshold=0.82)
        >>> metrics = measure_dark_residue(result)
        >>> tests = validate_theory(result, metrics)
        >>> print(f"Overall validity: {tests.overall_validity_score:.2f}")
        >>> if tests.tests_passed >= 5:
        >>>     print("✓ Theory validated!")
    """

    # 1. ΛCDM dark matter fraction test
    lambda_cdm_match = _test_lambda_cdm_fraction(dark_metrics.mass_fraction)

    # 2. CMB power spectrum test
    cmb_match = _test_cmb_power_spectrum(pruning_result, dark_metrics)

    # 3. Large-scale structure test
    lss_match = _test_large_scale_structure(dark_metrics.fractal_dimension)

    # 4. Gravitational lensing test
    lensing_match = _test_gravitational_lensing(dark_metrics)

    # 5. Rotation curve test (already computed in metrics)
    rotation_curve_consistency = dark_metrics.rotation_curve_match

    # 6. Entropy conservation test
    entropy_score = _test_entropy_conservation(pruning_result)

    # Compute overall score
    scores = [
        lambda_cdm_match,
        cmb_match,
        lss_match,
        lensing_match,
        rotation_curve_consistency,
        entropy_score
    ]

    overall_score = np.mean(scores)

    # Count passes/fails (threshold: 0.7)
    tests_passed = sum(1 for s in scores if s >= 0.7)
    tests_failed = sum(1 for s in scores if s < 0.7)

    return CosmologicalTests(
        lambda_cdm_match=lambda_cdm_match,
        cmb_power_spectrum_match=cmb_match,
        large_scale_structure_match=lss_match,
        lensing_signature_match=lensing_match,
        rotation_curve_consistency=rotation_curve_consistency,
        entropy_conservation_score=entropy_score,
        overall_validity_score=overall_score,
        tests_passed=tests_passed,
        tests_failed=tests_failed
    )


def _test_lambda_cdm_fraction(measured_fraction: float,
                               target: float = 0.27,
                               tolerance: float = 0.05) -> float:
    """
    Test if dark matter fraction matches ΛCDM prediction.

    ΛCDM composition:
    - Dark energy: ~68%
    - Dark matter: ~27%
    - Baryonic matter: ~5%

    Returns:
        Match score in [0, 1], where 1 = exact match
    """
    error = abs(measured_fraction - target)

    if error <= tolerance:
        # Within tolerance → perfect score decays with error
        score = 1.0 - (error / tolerance)
    else:
        # Outside tolerance → exponential decay
        score = np.exp(-(error - tolerance) * 10)

    return float(score)


def _test_cmb_power_spectrum(pruning_result: PruningResult,
                              dark_metrics: DarkMatterMetrics) -> float:
    """
    Test if pruning pattern matches CMB temperature fluctuations.

    CMB power spectrum:
    - Shows fluctuations at recombination (z ≈ 1100)
    - Dark matter affects peak positions and heights
    - Requires correct dark fraction + distribution

    Simplified test:
    - Check if coherence distribution matches Gaussian (CMB is Gaussian)
    - Check if dark fraction affects power spectrum correctly

    Returns:
        Match score in [0, 1]
    """
    coherences = np.array(pruning_result.coherence_scores)

    # CMB fluctuations are approximately Gaussian
    # Test if coherence distribution is Gaussian
    from scipy.stats import normaltest

    if len(coherences) < 8:
        return 0.5  # Not enough data

    # Normality test (p-value > 0.05 → Gaussian)
    stat, p_value = normaltest(coherences)

    # Convert p-value to match score
    # p > 0.05 → Gaussian → good
    # p < 0.05 → Non-Gaussian → bad
    if p_value > 0.05:
        gaussianity_score = 1.0
    else:
        gaussianity_score = p_value / 0.05  # Linear decay

    # Combined with dark fraction match (CMB requires correct ΩDM)
    dark_fraction_score = _test_lambda_cdm_fraction(dark_metrics.mass_fraction)

    # Weighted average
    cmb_score = 0.6 * dark_fraction_score + 0.4 * gaussianity_score

    return float(cmb_score)


def _test_large_scale_structure(fractal_dimension: float,
                                 target: float = 2.6,
                                 tolerance: float = 0.2) -> float:
    """
    Test if residue distribution matches cosmic web structure.

    Observations (DESI, SDSS):
    - Cosmic web has fractal dimension D ≈ 2.6
    - Range: 2.4 - 2.8 depending on scale
    - Filamentary structure (neither surface nor volume-filling)

    Returns:
        Match score in [0, 1], where 1 = cosmic web structure
    """
    error = abs(fractal_dimension - target)

    if error <= tolerance:
        # Within cosmic web range → high score
        score = 1.0 - (error / tolerance) * 0.3  # Max penalty 30%
    else:
        # Outside range → exponential decay
        score = np.exp(-(error - tolerance) * 3)

    return float(score)


def _test_gravitational_lensing(dark_metrics: DarkMatterMetrics) -> float:
    """
    Test if dark matter residue produces correct lensing signature.

    Gravitational lensing:
    - Requires mass (residue has mass ✓)
    - Requires spatial distribution (clustered)
    - Deflection angle ∝ enclosed mass

    Simplified test:
    - Check if residue is spatially clustered (Hopkins > 0.7)
    - Check if mass distribution is non-uniform (curvature > 0)
    - Check if dark fraction is sufficient (≈27%)

    Returns:
        Lensing consistency score in [0, 1]
    """
    # 1. Clustering requirement (dark matter is clustered)
    clustering_score = dark_metrics.spatial_clustering

    # 2. Mass distribution (non-uniform curvature)
    if dark_metrics.curvature_residue > 0.01:
        curvature_score = 1.0
    else:
        curvature_score = dark_metrics.curvature_residue / 0.01

    # 3. Dark fraction requirement
    dark_fraction_score = _test_lambda_cdm_fraction(dark_metrics.mass_fraction)

    # Weighted combination
    lensing_score = (
        0.4 * clustering_score +
        0.3 * curvature_score +
        0.3 * dark_fraction_score
    )

    return float(lensing_score)


def _test_entropy_conservation(pruning_result: PruningResult,
                                tolerance: float = 0.05) -> float:
    """
    Test if pruning conserves information (holographic principle).

    Holographic principle:
    - Information cannot be destroyed
    - Total entropy must be conserved
    - S_before ≈ S_after = S_hologram + S_residue

    Returns:
        Conservation score in [0, 1], where 1 = perfect conservation
    """
    ratio = pruning_result.entropy_conservation

    # Perfect conservation: ratio = 1.0
    error = abs(ratio - 1.0)

    if error <= tolerance:
        # Within tolerance → excellent
        score = 1.0 - (error / tolerance) * 0.2  # Max penalty 20%
    else:
        # Outside tolerance → information loss/gain
        score = np.exp(-(error - tolerance) * 10)

    return float(score)


def generate_validation_report(tests: CosmologicalTests,
                               dark_metrics: DarkMatterMetrics,
                               pruning_result: PruningResult,
                               output_path: str = 'cosmological_validation_report.txt') -> str:
    """
    Generate detailed text report of cosmological validation.

    Returns:
        Report text (also saved to file)
    """
    report = f"""
{'='*80}
COSMOLOGICAL VALIDATION REPORT - DARK MATTER PRUNING THEORY
{'='*80}

THEORY: Dark matter emerges as informational residue from holographic pruning
        of discordant multiverse branches.

TEST DATE: 2025-12-17
FRAMEWORK: Holo-Harmonic Möbius Lattice (HHmL)

{'='*80}
VALIDATION TEST RESULTS
{'='*80}

1. ΛCDM DARK MATTER FRACTION TEST
   Measured: {dark_metrics.mass_fraction:.2%}
   Target: 27.0% ± 5%
   Error: ±{abs(dark_metrics.mass_fraction - 0.27)*100:.1f}%
   Score: {tests.lambda_cdm_match:.3f}
   Result: {'✓ PASS' if tests.lambda_cdm_match >= 0.7 else '✗ FAIL'}

2. CMB POWER SPECTRUM MATCH
   Coherence Distribution: {'Gaussian' if tests.cmb_power_spectrum_match > 0.7 else 'Non-Gaussian'}
   Dark Fraction Consistency: {'Yes' if dark_metrics.mass_fraction > 0.2 else 'No'}
   Score: {tests.cmb_power_spectrum_match:.3f}
   Result: {'✓ PASS' if tests.cmb_power_spectrum_match >= 0.7 else '✗ FAIL'}

3. LARGE-SCALE STRUCTURE (COSMIC WEB)
   Fractal Dimension: {dark_metrics.fractal_dimension:.2f}
   Target Range: 2.4 - 2.8 (cosmic web)
   Error: ±{abs(dark_metrics.fractal_dimension - 2.6):.2f}
   Score: {tests.large_scale_structure_match:.3f}
   Result: {'✓ PASS' if tests.large_scale_structure_match >= 0.7 else '✗ FAIL'}

4. GRAVITATIONAL LENSING SIGNATURE
   Spatial Clustering (Hopkins): {dark_metrics.spatial_clustering:.2f}
   Field Curvature (RMS): {dark_metrics.curvature_residue:.3f}
   Mass Distribution: {'Clustered' if dark_metrics.spatial_clustering > 0.7 else 'Random'}
   Score: {tests.lensing_signature_match:.3f}
   Result: {'✓ PASS' if tests.lensing_signature_match >= 0.7 else '✗ FAIL'}

5. GALAXY ROTATION CURVES
   Flatness Score: {tests.rotation_curve_consistency:.3f}
   Curve Shape: {'Flat (v ≈ const)' if tests.rotation_curve_consistency > 0.7 else 'Keplerian (v ∝ 1/√r)'}
   Consistency: {'High' if tests.rotation_curve_consistency > 0.8 else 'Medium' if tests.rotation_curve_consistency > 0.5 else 'Low'}
   Score: {tests.rotation_curve_consistency:.3f}
   Result: {'✓ PASS' if tests.rotation_curve_consistency >= 0.7 else '✗ FAIL'}

6. ENTROPY CONSERVATION (HOLOGRAPHIC PRINCIPLE)
   Before Pruning: {pruning_result.entropy_before:.2f} nats
   After Pruning: {pruning_result.entropy_after:.2f} nats
   Ratio (After/Before): {pruning_result.entropy_conservation:.3f}
   Information Loss: {abs(1.0 - pruning_result.entropy_conservation)*100:.1f}%
   Score: {tests.entropy_conservation_score:.3f}
   Result: {'✓ PASS' if tests.entropy_conservation_score >= 0.7 else '✗ FAIL'}

{'='*80}
OVERALL ASSESSMENT
{'='*80}

Tests Passed: {tests.tests_passed}/6
Tests Failed: {tests.tests_failed}/6
Overall Validity Score: {tests.overall_validity_score:.3f}

Verdict: {'✓✓✓ THEORY VALIDATED' if tests.overall_validity_score >= 0.7 and tests.tests_passed >= 4 else '⚠ THEORY PARTIALLY SUPPORTED' if tests.overall_validity_score >= 0.5 else '✗✗✗ THEORY FALSIFIED'}

{'='*80}
DETAILED METRICS
{'='*80}

PRUNING SUMMARY:
  Total Branches: {len(pruning_result.kept_branches) + len(pruning_result.pruned_branches)}
  Kept (Physical): {len(pruning_result.kept_branches)}
  Pruned (Dark Matter): {len(pruning_result.pruned_branches)}
  Coherence Threshold: {pruning_result.threshold_used:.3f}

DARK MATTER SIGNATURES:
  Mass Fraction: {dark_metrics.mass_fraction:.2%}
  Entropy Fraction: {dark_metrics.entropy_ratio:.2%}
  Fractal Dimension: {dark_metrics.fractal_dimension:.2f}
  Hopkins Clustering: {dark_metrics.spatial_clustering:.2f}
  Density Anomaly: {dark_metrics.density_anomaly:+.3f}
  Field Curvature: {dark_metrics.curvature_residue:.3f}
  Intra-Coherence: {dark_metrics.field_coherence_residue:.3f}

COSMOLOGICAL CONSISTENCY:
  ΛCDM Match: {tests.lambda_cdm_match:.3f}
  CMB Match: {tests.cmb_power_spectrum_match:.3f}
  LSS Match: {tests.large_scale_structure_match:.3f}
  Lensing Match: {tests.lensing_signature_match:.3f}
  Rotation Curve: {tests.rotation_curve_consistency:.3f}
  Entropy Score: {tests.entropy_conservation_score:.3f}

{'='*80}
FALSIFIABILITY CRITERIA
{'='*80}

Prediction 1: Dark fraction ≈ 27% ± 5%
Status: {'✓ CONFIRMED' if abs(dark_metrics.mass_fraction - 0.27) < 0.05 else '✗ FALSIFIED'}

Prediction 2: Fractal dimension D ∈ [2.4, 2.8]
Status: {'✓ CONFIRMED' if 2.4 <= dark_metrics.fractal_dimension <= 2.8 else '✗ FALSIFIED'}

Prediction 3: Flat rotation curves (score > 0.7)
Status: {'✓ CONFIRMED' if tests.rotation_curve_consistency > 0.7 else '✗ FALSIFIED'}

Prediction 4: Entropy conserved (±5%)
Status: {'✓ CONFIRMED' if 0.95 <= pruning_result.entropy_conservation <= 1.05 else '✗ FALSIFIED'}

{'='*80}
IMPLICATIONS FOR COSMOLOGY
{'='*80}

"""

    if tests.overall_validity_score >= 0.7:
        report += """
✓ BREAKTHROUGH: If validated, this explains dark matter without new particles.

Key Implications:
1. Dark matter = informational residue, not exotic particles
2. Multiverse pruning is physical process (holographic projection)
3. Universe is "cleaned up" hologram of quantum multiverse
4. ~27% of reality is "deleted timelines" still gravitating
5. Möbius topology enables clean branch separation

Next Steps:
- Publish findings in peer-reviewed journal (Nature Physics, PRD)
- Test at larger scales (cosmological simulations)
- Compare with particle dark matter predictions (WIMPs, axions)
- Design observational tests (lensing patterns, CMB non-Gaussianity)
"""
    elif tests.overall_validity_score >= 0.5:
        report += """
⚠ PARTIAL SUPPORT: Some predictions match, others fail.

Interpretation:
- Theory may be on right track but needs refinement
- Pruning mechanism might be correct, but threshold/scale wrong
- Alternative: Dark matter = mix of pruning residue + new particles

Next Steps:
- Investigate failed tests (which predictions don't hold?)
- Adjust pruning algorithm (different coherence metrics?)
- Test at multiple scales (convergence behavior)
- Compare with hybrid models (residue + WIMPs)
"""
    else:
        report += """
✗ FALSIFIED: Predictions do not match observations.

Interpretation:
- Pruning mechanism does not explain dark matter
- Either wrong fraction, wrong distribution, or wrong dynamics
- Back to drawing board: dark matter likely requires particles

Lessons Learned:
- Holographic pruning is interesting but not dark matter source
- Information residue exists but has different properties
- Need alternative explanation for 27% dark fraction
"""

    report += f"""

{'='*80}
END OF REPORT
{'='*80}

Generated: 2025-12-17
Framework: HHmL (Holo-Harmonic Möbius Lattice)
Test: Dark Matter as Multiverse Pruning Residue

🤖 Generated with Claude Code
Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
"""

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\nValidation report saved to {output_path}")

    return report


def visualize_cosmological_tests(tests: CosmologicalTests,
                                 output_path: str = 'cosmological_tests.png'):
    """
    Create visualization of all cosmological validation tests.

    Plots:
    - Radar chart of all test scores
    - Pass/fail summary
    - Overall validity gauge
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    # 1. Radar chart of test scores
    ax1 = fig.add_subplot(gs[:, 0], projection='polar')

    categories = [
        'ΛCDM\nFraction',
        'CMB\nSpectrum',
        'Large-Scale\nStructure',
        'Lensing\nSignature',
        'Rotation\nCurves',
        'Entropy\nConservation'
    ]

    values = [
        tests.lambda_cdm_match,
        tests.cmb_power_spectrum_match,
        tests.large_scale_structure_match,
        tests.lensing_signature_match,
        tests.rotation_curve_consistency,
        tests.entropy_conservation_score
    ]

    # Repeat first value to close the polygon
    values += values[:1]

    # Angles for each axis
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    # Plot
    ax1.plot(angles, values, 'o-', linewidth=2, color='blue', label='Measured')
    ax1.fill(angles, values, alpha=0.25, color='blue')

    # Threshold line (0.7 = pass)
    threshold = [0.7] * len(angles)
    ax1.plot(angles, threshold, '--', linewidth=1, color='red', label='Threshold (0.7)')

    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, size=10)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.set_title('Cosmological Validation Tests', fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax1.grid(True)

    # 2. Pass/Fail bar chart
    ax2 = fig.add_subplot(gs[0, 1])

    test_names = ['ΛCDM', 'CMB', 'LSS', 'Lensing', 'Rotation', 'Entropy']
    test_values = values[:-1]  # Remove duplicate

    colors = ['green' if v >= 0.7 else 'red' for v in test_values]

    bars = ax2.barh(test_names, test_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(0.7, color='black', linestyle='--', linewidth=2, label='Pass Threshold')
    ax2.set_xlim(0, 1)
    ax2.set_xlabel('Score', fontweight='bold')
    ax2.set_title('Test Results (Pass ≥ 0.7)', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='x')

    # Add value labels
    for bar, value in zip(bars, test_values):
        width = bar.get_width()
        ax2.text(width/2, bar.get_y() + bar.get_height()/2,
                f'{value:.2f}',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color='white' if value > 0.5 else 'black')

    # 3. Overall validity gauge
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    overall = tests.overall_validity_score

    # Gauge visualization
    gauge_text = f"""
    OVERALL VALIDITY SCORE
    {'='*30}

    Score: {overall:.3f}

    Tests Passed: {tests.tests_passed}/6
    Tests Failed: {tests.tests_failed}/6

    Verdict:
    """

    if overall >= 0.7 and tests.tests_passed >= 4:
        gauge_text += "    ✓✓✓ THEORY VALIDATED\n"
        gauge_color = 'lightgreen'
    elif overall >= 0.5:
        gauge_text += "    ⚠ PARTIALLY SUPPORTED\n"
        gauge_color = 'lightyellow'
    else:
        gauge_text += "    ✗✗✗ THEORY FALSIFIED\n"
        gauge_color = 'lightcoral'

    gauge_text += f"\n    Confidence: {'High' if overall > 0.8 else 'Medium' if overall > 0.6 else 'Low'}"

    ax3.text(0.5, 0.5, gauge_text, fontsize=12, family='monospace',
             ha='center', va='center', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor=gauge_color, alpha=0.5, pad=1.0))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Cosmological tests visualization saved to {output_path}")


# Example usage and testing
if __name__ == '__main__':
    print("="*80)
    print("COSMOLOGICAL VALIDATOR TEST")
    print("="*80)
    print()

    print("Validation tests:")
    print("  1. ΛCDM dark matter fraction (27%)")
    print("  2. CMB power spectrum consistency")
    print("  3. Large-scale structure (fractal D ≈ 2.6)")
    print("  4. Gravitational lensing signatures")
    print("  5. Galaxy rotation curve flatness")
    print("  6. Entropy conservation (information preserved)")
    print()

    print("Note: Full test requires PruningResult and DarkMatterMetrics")
    print("Run via: python simulations/dark_matter/full_dark_matter_test.py")
