"""
Example: LIGO Gravitational Wave Verification

This script demonstrates how to compare HHmL boundary resonances
against real LIGO gravitational wave detections.

Usage:
    python examples/verification/verify_ligo_example.py
"""

import torch
import numpy as np
from hhml.verification.ligo import LIGOVerification


def generate_mock_simulation_data(duration=4.0, sample_rate=4096):
    """
    Generate mock HHmL field evolution for demonstration.

    In real usage, this would be the actual field tensor from training.
    """
    n_samples = int(duration * sample_rate)
    n_nodes = 100

    # Simulate field evolution with chirp-like behavior
    t = torch.linspace(0, duration, n_samples)

    # Chirping frequency
    f0 = 30
    f1 = 200
    freq = f0 + (f1 - f0) * (t / duration) ** 2

    # Growing amplitude (merger approach)
    amp = 1 + 5 * (t / duration) ** 3

    # Multiple nodes with phase variations
    field = torch.zeros(n_samples, n_nodes, 2)  # [time, nodes, (real, imag)]

    for i in range(n_nodes):
        phase_offset = 2 * np.pi * i / n_nodes
        field[:, i, 0] = amp * torch.cos(2 * np.pi * freq * t + phase_offset)
        field[:, i, 1] = amp * torch.sin(2 * np.pi * freq * t + phase_offset)

    print(f"Generated mock field: {field.shape}")
    return field


def main():
    print("=" * 70)
    print("HHmL LIGO Gravitational Wave Verification Example")
    print("=" * 70)

    # Initialize verifier
    print("\n1. Initializing LIGO verification system...")
    verifier = LIGOVerification(data_dir="data/ligo")

    print(f"\n   Known events: {list(verifier.events.keys())}")

    # Generate mock simulation data
    print("\n2. Generating mock HHmL simulation data...")
    sim_field = generate_mock_simulation_data()

    # Compare to GW150914 (first LIGO detection)
    print("\n3. Comparing to GW150914 (first black hole merger detection)...")
    event_name = 'GW150914'
    detector = 'H1'

    results = verifier.compare_event(
        event_name=event_name,
        sim_strain_tensor=sim_field,
        detector=detector,
        save_results=True
    )

    # Display results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"\nEvent: {results['event']}")
    print(f"Description: {results['description']}")
    print(f"Detector: {results['detector']}")
    print(f"\nMetrics:")
    print(f"  Overlap:   {results['metrics']['overlap']:.4f}")
    print(f"  SNR:       {results['metrics']['snr']:.2f}")
    print(f"  Mismatch:  {results['metrics']['mismatch']:.4f}")
    print(f"\nInterpretation: {results['interpretation']}")

    # Additional events
    print("\n" + "=" * 70)
    print("Comparing to additional events...")
    print("=" * 70)

    for event in ['GW151226', 'GW170817']:
        print(f"\n{event}:")
        results = verifier.compare_event(event, sim_field, detector='H1', save_results=False)
        print(f"  Overlap: {results['metrics']['overlap']:.4f}")
        print(f"  {results['interpretation']}")

    print("\n" + "=" * 70)
    print("Verification complete! Results saved to data/ligo/")
    print("=" * 70)


if __name__ == '__main__':
    main()
