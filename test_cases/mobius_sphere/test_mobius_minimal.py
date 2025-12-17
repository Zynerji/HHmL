#!/usr/bin/env python3
"""Minimal Möbius test - just validate imports and structure"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("Testing HHmL package structure...")
print("=" * 60)

# Test 1: Import main module
print("1. Importing hhml.mobius.mobius_training... ", end="")
try:
    from hhml.mobius.mobius_training import MobiusRNNAgent, MobiusHelixSphere
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 2: Create sphere first to determine state dim
print("2. Creating Möbius Helix Sphere (1000 nodes)... ", end="")
try:
    import torch
    device = torch.device('cpu')
    sphere = MobiusHelixSphere(num_nodes=1000, device=device)  # Need 1000+ for 256-dim state
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 3: Create agent (small hidden dim for speed)
print("3. Creating Möbius RNN Agent (256 hidden)... ", end="")
try:
    agent = MobiusRNNAgent(state_dim=256, device=device, hidden_dim=256)  # Small for testing
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 4: Single forward pass
print("4. Running single RNN forward pass... ", end="")
try:
    state = sphere.get_state()
    action, value, features = agent.forward(state)
    structure_params = agent.compute_structure_params(features)
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

# Test 5: Single evolution step
print("5. Running single evolution step... ", end="")
try:
    sphere.evolve(structure_params=structure_params)
    # Vortex detection is part of compute_reward(), not a separate method
    reward = sphere.compute_reward()
    print("OK")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)

print("=" * 60)
print("ALL TESTS PASSED!")
print(f"Reward score: {reward:.4f}")
print(f"Final w: {structure_params['windings'].item():.2f}")
print(f"Final tau: {structure_params['tau_torsion'].item():.2f}")
print("\nHHmL repository structure is valid and functional!")
