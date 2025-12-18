"""
Example: Particle Physics Verification

This script demonstrates how to compare HHmL emergent excitations
against Standard Model particles and LHC data.

Usage:
    python examples/verification/verify_particles_example.py
"""

import torch
import numpy as np
from hhml.verification.particles import ParticleVerification


def generate_mock_excitation_spectrum(n_vortices=1000):
    """
    Generate mock HHmL vortex/excitation energies for demonstration.

    In real usage, this would be vortex energy levels from the simulation.
    """
    # Simulate energy spectrum with some clustering around "particle masses"

    # Create peaks around some SM masses (in GeV)
    # Electron, muon, tau, Z boson, Higgs
    peak_masses = [0.000511, 0.106, 1.777, 91.2, 125.0]
    peak_widths = [0.001, 0.01, 0.1, 2.5, 2.0]

    energies = []

    for mass, width in zip(peak_masses, peak_widths):
        # Generate cluster of energies around this mass
        n_cluster = n_vortices // len(peak_masses)
        cluster = torch.randn(n_cluster) * width + mass
        energies.append(cluster)

    # Add some background continuum
    n_background = n_vortices // 5
    background = torch.rand(n_background) * 200  # 0-200 GeV uniform

    energies.append(background)
    energies = torch.cat(energies)

    # Remove negative energies
    energies = torch.abs(energies)

    print(f"Generated mock energy spectrum: {len(energies)} excitations")
    print(f"Energy range: {energies.min():.4f} - {energies.max():.2f} GeV")

    return energies


def main():
    print("=" * 70)
    print("HHmL Particle Physics Verification Example")
    print("=" * 70)

    # Initialize verifier
    print("\n1. Initializing particle physics verification system...")
    verifier = ParticleVerification(data_dir="data/particles")

    print(f"\n   Known SM particles: {len(verifier.pdg_masses)}")
    print(f"   Sample masses (GeV):")
    for particle in ['electron', 'muon', 'Z_boson', 'Higgs']:
        print(f"     {particle}: {verifier.pdg_masses[particle]:.6f}")

    # Generate mock excitation spectrum
    print("\n2. Generating mock HHmL excitation spectrum...")
    sim_energies = generate_mock_excitation_spectrum()

    # Compare to PDG masses
    print("\n3. Comparing excitations to Standard Model particle masses...")
    pdg_results = verifier.compare_pdg_masses(
        sim_energies=sim_energies,
        particle_list=['electron', 'muon', 'tau', 'Z_boson', 'Higgs', 'W_boson'],
        tolerance=0.1  # ±10% match tolerance
    )

    print("\n" + "=" * 70)
    print("PDG MASS COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nTolerance: ±{pdg_results['tolerance']*100}%")
    print(f"Matched: {pdg_results['matched_particles']}/{pdg_results['total_particles']}")
    print(f"Match Fraction: {pdg_results['match_fraction']:.2%}")

    print("\nDetailed matches:")
    for match in pdg_results['matches']:
        status = "✓" if match['match'] == 'Yes' else "✗"
        print(f"  {status} {match['particle']:15s}: "
              f"PDG={match['pdg_mass']:.6f} GeV, "
              f"Sim={match['sim_energy']:.6f} GeV, "
              f"Error={match['relative_error']:.2%}")

    # Compare to LHC Higgs channel
    print("\n" + "=" * 70)
    print("LHC Channel Comparison")
    print("=" * 70)

    print("\n4. Comparing to Higgs → 4 lepton channel...")
    lhc_results = verifier.compare_lhc_channel(
        sim_energies=sim_energies,
        channel='higgs_4l',
        scale_factor=1.0,  # Already in GeV
        save_results=True
    )

    print(f"\nChannel: {lhc_results['channel']}")
    print(f"Mass range: {lhc_results['mass_range_GeV']} GeV")
    print(f"\nMetrics:")
    print(f"  χ²:       {lhc_results['metrics']['chi_squared']:.2f}")
    print(f"  χ²/DOF:   {lhc_results['metrics']['reduced_chi_squared']:.3f}")
    print(f"  KS stat:  {lhc_results['metrics']['ks_statistic']:.4f}")
    print(f"\nInterpretation: {lhc_results['metrics']['interpretation']}")

    # Additional channels
    print("\n" + "=" * 70)
    print("Comparing to additional LHC channels...")
    print("=" * 70)

    for channel in ['Z_ee', 'W_enu']:
        print(f"\n{channel}:")
        results = verifier.compare_lhc_channel(
            sim_energies, channel, scale_factor=1.0, save_results=False
        )
        print(f"  χ²/DOF: {results['metrics']['reduced_chi_squared']:.3f}")
        print(f"  {results['metrics']['interpretation']}")

    print("\n" + "=" * 70)
    print("Verification complete! Results saved to data/particles/")
    print("=" * 70)


if __name__ == '__main__':
    main()
