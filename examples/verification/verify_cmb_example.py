"""
Example: CMB Power Spectrum Verification

This script demonstrates how to compare HHmL field fluctuations
against Planck satellite CMB observations.

Usage:
    python examples/verification/verify_cmb_example.py
"""

import torch
import numpy as np
from hhml.verification.cmb import CMBVerification


def generate_mock_field_map(n_nodes=10000):
    """
    Generate mock HHmL field configuration for demonstration.

    In real usage, this would be the actual field tensor from a trained model.
    """
    # Simulate field with multipole structure
    # Real implementation would project from Möbius lattice to sphere

    # Random Gaussian field with some structure
    field = torch.randn(n_nodes, 1)  # [nodes, features]

    # Add some coherent structure (mimics CMB acoustic peaks)
    theta = torch.linspace(0, 2 * np.pi, n_nodes)
    phi = torch.linspace(0, np.pi, n_nodes)

    # Add harmonic modes (analogous to CMB multipoles)
    for l in [10, 50, 100, 220, 500, 800]:
        amplitude = 1.0 / (l ** 0.5)  # Power-law damping
        field[:, 0] += amplitude * torch.cos(l * theta) * torch.sin(l * phi)

    print(f"Generated mock field map: {field.shape}")
    return field


def main():
    print("=" * 70)
    print("HHmL CMB Power Spectrum Verification Example")
    print("=" * 70)

    # Initialize verifier
    print("\n1. Initializing CMB verification system...")
    verifier = CMBVerification(data_dir="data/cmb", nside=128)

    # Generate mock field
    print("\n2. Generating mock HHmL field configuration...")
    sim_field = generate_mock_field_map()

    # Compare to Planck TT spectrum
    print("\n3. Comparing to Planck 2018 TT power spectrum...")
    results = verifier.compare_planck(
        sim_field_tensor=sim_field,
        cl_type='TT',
        lmax=1000,
        save_results=True
    )

    # Display results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nSpectrum Type: {results['spectrum_type']}")
    print(f"Max Multipole (ℓ): {results['lmax']}")
    print(f"\nMetrics:")
    print(f"  χ²:       {results['metrics']['chi_squared']:.2f}")
    print(f"  DOF:      {results['metrics']['dof']}")
    print(f"  χ²/DOF:   {results['metrics']['reduced_chi_squared']:.3f}")
    print(f"  p-value:  {results['metrics']['p_value']:.4f}")
    print(f"\nInterpretation: {results['metrics']['interpretation']}")
    print(f"\nNotes: {results['notes']}")

    # Load and display Planck data for reference
    print("\n" + "=" * 70)
    print("Planck Reference Data")
    print("=" * 70)

    ells, cl, errors = verifier.load_planck_cl('TT', lmax=100)
    print(f"\nLoaded {len(cl)} multipoles from Planck TT spectrum")
    print(f"Sample Cℓ values (µK²):")
    print(f"  ℓ=2:   {cl[2]:.2f}")
    print(f"  ℓ=10:  {cl[10]:.2f}")
    print(f"  ℓ=100: {cl[100]:.2f}")

    print("\n" + "=" * 70)
    print("Verification complete! Results saved to data/cmb/")
    print("=" * 70)


if __name__ == '__main__':
    main()
