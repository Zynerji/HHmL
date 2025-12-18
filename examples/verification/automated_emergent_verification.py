"""
Automated Emergent Phenomenon Verification Example.

This script demonstrates the automated workflow for judging if discovered
emergent phenomena are truly novel by verifying against real-world physics data.

Usage:
    python examples/verification/automated_emergent_verification.py
"""

import torch
import numpy as np
from hhml.utils.emergent_verifier import EmergentVerifier, verify_emergent_phenomenon


def simulate_oscillatory_phenomenon():
    """Generate mock field exhibiting oscillatory behavior."""
    print("\n" + "="*70)
    print("Simulating Oscillatory Phenomenon (GW-like)")
    print("="*70)

    # Temporal evolution with chirp-like behavior
    t = torch.linspace(0, 4.0, 16384)  # 4 seconds @ 4096 Hz
    n_nodes = 100

    # Frequency chirp (30-250 Hz like GW150914)
    f0, f1 = 30, 250
    freq = f0 + (f1 - f0) * (t / 4.0) ** 2

    # Growing amplitude
    amp = 1 + 5 * (t / 4.0) ** 3

    # Field with spatial variation
    field = torch.zeros(len(t), n_nodes, 2)
    for i in range(n_nodes):
        phase_offset = 2 * np.pi * i / n_nodes
        field[:, i, 0] = amp * torch.cos(2 * np.pi * freq * t + phase_offset)
        field[:, i, 1] = amp * torch.sin(2 * np.pi * freq * t + phase_offset)

    print(f"Generated oscillatory field: {field.shape}")
    return field


def simulate_spatial_phenomenon():
    """Generate mock field exhibiting spatial fluctuations."""
    print("\n" + "="*70)
    print("Simulating Spatial Fluctuation Phenomenon (CMB-like)")
    print("="*70)

    # Spatial field with multipole structure
    n_nodes = 10000
    field = torch.randn(n_nodes, 1)

    # Add coherent multipole modes (mimics CMB acoustic peaks)
    theta = torch.linspace(0, 2 * np.pi, n_nodes)
    phi = torch.linspace(0, np.pi, n_nodes)

    for l in [10, 50, 100, 220, 500]:
        amplitude = 1.0 / (l ** 0.5)
        field[:, 0] += amplitude * torch.cos(l * theta) * torch.sin(l * phi)

    print(f"Generated spatial field: {field.shape}")
    return field


def simulate_energetic_phenomenon():
    """Generate mock vortex energies exhibiting discrete levels."""
    print("\n" + "="*70)
    print("Simulating Discrete Energy Phenomenon (Particle-like)")
    print("="*70)

    # Energy spectrum with peaks around SM masses
    peak_masses = [0.000511, 0.106, 1.777, 91.2, 125.0]  # GeV
    peak_widths = [0.001, 0.01, 0.1, 2.5, 2.0]

    energies = []
    for mass, width in zip(peak_masses, peak_widths):
        cluster = torch.randn(200) * width + mass
        energies.append(cluster)

    # Background continuum
    background = torch.rand(200) * 200
    energies.append(background)

    energies = torch.cat(energies)
    energies = torch.abs(energies)  # Physical energies are positive

    print(f"Generated energy spectrum: {len(energies)} excitations")
    return energies


def main():
    print("\n" + "="*70)
    print("AUTOMATED EMERGENT PHENOMENON VERIFICATION")
    print("="*70)
    print("\nThis script demonstrates the automated workflow for judging if")
    print("discovered emergent phenomena are truly novel by verifying against")
    print("real-world physics data (LIGO, CMB, particles).")
    print("\n" + "="*70)

    # Initialize verifier
    verifier = EmergentVerifier(data_dir="data")

    # Test 1: Oscillatory phenomenon (LIGO-like)
    field_osc = simulate_oscillatory_phenomenon()

    print("\nRunning automated verification (oscillatory)...")
    results_osc = verifier.verify_phenomenon(
        field_tensor=field_osc,
        phenomenon_type='oscillatory',
        save_results=True,
        output_dir="data/verification/oscillatory"
    )

    print("\n" + "-"*70)
    print("OSCILLATORY PHENOMENON RESULTS")
    print("-"*70)
    print(f"Novelty Score: {results_osc['novelty_score']:.3f}")
    print(f"Is Novel: {results_osc['is_novel']}")
    print(f"\nInterpretation:")
    print(f"  {results_osc['interpretation']}")

    if 'ligo' in results_osc['verification']:
        ligo_best = results_osc['verification']['ligo']['best_match']
        print(f"\nBest LIGO Match:")
        print(f"  Event: {ligo_best['event']}")
        print(f"  Overlap: {ligo_best['overlap']:.4f}")
        print(f"  Quality: {ligo_best['quality']}")

    print(f"\nRecommendations:")
    for rec in results_osc['recommendations']:
        print(f"  {rec}")

    # Test 2: Spatial phenomenon (CMB-like)
    field_spatial = simulate_spatial_phenomenon()

    print("\n" + "="*70)
    print("\nRunning automated verification (spatial)...")
    results_spatial = verifier.verify_phenomenon(
        field_tensor=field_spatial,
        phenomenon_type='spatial',
        save_results=True,
        output_dir="data/verification/spatial"
    )

    print("\n" + "-"*70)
    print("SPATIAL PHENOMENON RESULTS")
    print("-"*70)
    print(f"Novelty Score: {results_spatial['novelty_score']:.3f}")
    print(f"Is Novel: {results_spatial['is_novel']}")
    print(f"\nInterpretation:")
    print(f"  {results_spatial['interpretation']}")

    if 'cmb' in results_spatial['verification']:
        cmb_metrics = results_spatial['verification']['cmb']['metrics']
        print(f"\nCMB Comparison:")
        print(f"  χ²: {cmb_metrics['chi_squared']:.2f}")
        print(f"  χ²/DOF: {cmb_metrics['reduced_chi_squared']:.3f}")
        print(f"  p-value: {cmb_metrics['p_value']:.4f}")
        print(f"  Quality: {results_spatial['verification']['cmb']['quality']}")

    print(f"\nRecommendations:")
    for rec in results_spatial['recommendations']:
        print(f"  {rec}")

    # Test 3: Energetic phenomenon (Particle-like)
    energies = simulate_energetic_phenomenon()

    print("\n" + "="*70)
    print("\nRunning automated verification (energetic)...")
    results_energetic = verifier.verify_phenomenon(
        vortex_energies=energies,
        phenomenon_type='energetic',
        save_results=True,
        output_dir="data/verification/energetic"
    )

    print("\n" + "-"*70)
    print("ENERGETIC PHENOMENON RESULTS")
    print("-"*70)
    print(f"Novelty Score: {results_energetic['novelty_score']:.3f}")
    print(f"Is Novel: {results_energetic['is_novel']}")
    print(f"\nInterpretation:")
    print(f"  {results_energetic['interpretation']}")

    if 'particles' in results_energetic['verification']:
        pdg = results_energetic['verification']['particles']['pdg']
        print(f"\nPDG Mass Comparison:")
        print(f"  Matched: {pdg['matched_particles']}/{pdg['total_particles']}")
        print(f"  Match Fraction: {pdg['match_fraction']:.2%}")
        print(f"  Quality: {results_energetic['verification']['particles']['quality']}")

    print(f"\nRecommendations:")
    for rec in results_energetic['recommendations']:
        print(f"  {rec}")

    # Test 4: Auto-detection
    print("\n" + "="*70)
    print("AUTOMATIC PHENOMENON TYPE DETECTION")
    print("="*70)

    print("\nRunning auto-detection on oscillatory field...")
    results_auto = verifier.verify_phenomenon(
        field_tensor=field_osc,
        phenomenon_type='auto',
        save_results=False
    )

    print(f"Detected type: {results_auto['phenomenon_type']}")
    print(f"Novelty score: {results_auto['novelty_score']:.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nThe EmergentVerifier provides automated verification to judge if")
    print("emergent phenomena are novel by comparing against real-world physics:")
    print("\n  • LIGO waveforms for oscillatory phenomena")
    print("  • CMB spectra for spatial fluctuations")
    print("  • Particle masses for discrete energies")
    print("\nPhenomena with novelty_score ≥ 0.5 are considered NOVEL and exhibit")
    print("mathematical patterns similar to empirical physics, strengthening the")
    print("novelty claim in EMERGENTS.md.")
    print("\nAll results saved to data/verification/")
    print("="*70)


if __name__ == '__main__':
    main()
