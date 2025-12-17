#!/usr/bin/env python3
"""
Quick Möbius Test - 30 cycles, 1024 nodes
==========================================
Tests that the Möbius training can run from the HHmL repository.

This is a minimal test to validate:
1. Package imports work correctly
2. Möbius topology generates without errors
3. RNN training loop completes
4. Results are saved

Author: HHmL Testing
Date: 2025-12-16
"""

import sys
import os
from pathlib import Path

# Add hhml package to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time
from datetime import datetime

# Import from hhml package
from hhml.mobius.mobius_training import MobiusRNNAgent, MobiusHelixSphere, run_mobius_training

print("=" * 80)
print("QUICK MÖBIUS TEST")
print("=" * 80)
print("Configuration:")
print("  Cycles: 30")
print("  Nodes: 1024")
print("  Purpose: Validate HHmL repository structure")
print("=" * 80)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print("\nStarting test in 2 seconds...")
    time.sleep(2)

    # Run quick training
    results = run_mobius_training(
        num_cycles=30,           # Reduced from 1200
        num_nodes=1024,          # Reduced from 20M
        device=device,
        checkpoint_interval=15   # Checkpoint halfway
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETE!")
    print("=" * 80)
    print(f"Final vortex density: {results['final_vortex_density']:.2%}")
    print(f"Final w windings: {results['final_w_windings']:.2f}")
    print(f"RNN value: {results['final_rnn_value']:.2f}")
    print("\nHHmL repository structure validated successfully!")
    print("=" * 80)
