#!/usr/bin/env python3
"""
Benchmark: Original vs Optimized Sphere Evolution
==================================================
Tests the 5-10× speedup from optimizations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import time
import numpy as np

from hhml.mobius.mobius_training import MobiusHelixSphere
from hhml.mobius.optimized_sphere import OptimizedMobiusHelixSphere

print("=" * 80)
print("SPHERE EVOLUTION BENCHMARK")
print("=" * 80)
print()

device = torch.device('cpu')
num_nodes = 8000
num_cycles = 20

print(f"Configuration:")
print(f"  Device: {device}")
print(f"  Nodes: {num_nodes:,}")
print(f"  Cycles: {num_cycles}")
print()

# Create fake structure params
dummy_params = {
    'windings': torch.tensor(50.0),
    'tau_torsion': torch.tensor(1.0),
    'num_sites': torch.tensor(8000.0),
    'uvhl_w': torch.tensor(3.2),
    'uvhl_L': torch.tensor(7.5),
    'uvhl_n': torch.tensor(3.8)
}

print("-" * 80)
print("ORIGINAL SPHERE (baseline)")
print("-" * 80)

sphere_original = MobiusHelixSphere(num_nodes=num_nodes, device=device)
times_original = []

for i in range(num_cycles):
    start = time.time()
    sphere_original.evolve(structure_params=dummy_params)
    elapsed = time.time() - start
    times_original.append(elapsed)
    if (i + 1) % 5 == 0:
        print(f"  Cycle {i+1}/{num_cycles}: {elapsed*1000:.1f}ms")

avg_original = np.mean(times_original)
print(f"\nAverage time per cycle: {avg_original*1000:.1f}ms")
print()

print("-" * 80)
print("OPTIMIZED SPHERE (with torch.compile + reduced sampling)")
print("-" * 80)

sphere_optimized = OptimizedMobiusHelixSphere(num_nodes=num_nodes, device=device)
times_optimized = []

# Warmup for compilation
print("Warming up JIT compiler...")
for i in range(3):
    sphere_optimized.evolve(structure_params=dummy_params)
print("✓ Warmup complete\n")

for i in range(num_cycles):
    start = time.time()
    sphere_optimized.evolve(structure_params=dummy_params)
    elapsed = time.time() - start
    times_optimized.append(elapsed)
    if (i + 1) % 5 == 0:
        print(f"  Cycle {i+1}/{num_cycles}: {elapsed*1000:.1f}ms")

avg_optimized = np.mean(times_optimized)
print(f"\nAverage time per cycle: {avg_optimized*1000:.1f}ms")
print()

print("=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Original sphere:   {avg_original*1000:6.1f}ms per cycle")
print(f"Optimized sphere:  {avg_optimized*1000:6.1f}ms per cycle")
print(f"Speedup:           {avg_original/avg_optimized:6.2f}×")
print()

if avg_original / avg_optimized > 5:
    print("✓ EXCELLENT: 5-10× speedup achieved!")
elif avg_original / avg_optimized > 2:
    print("✓ GOOD: 2-5× speedup achieved")
else:
    print("⚠ MARGINAL: <2× speedup (compilation overhead?)")

print()
print("Optimizations applied:")
print("  1. torch.compile() - JIT compilation for hot loop")
print("  2. Reduced sample size: 500→200 nodes")
print("  3. Skip evolution every N cycles")
print("  4. Single vectorized distance computation")
print("  5. No Python loops in evolution kernel")
print("=" * 80)
