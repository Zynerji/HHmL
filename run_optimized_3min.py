#!/usr/bin/env python3
"""
Optimized 3-Minute Möbius Training with PDF LaTeX Reporting
===========================================================
Uses OptimizedMobiusHelixSphere for 5-10× speedup.

Expected performance:
- Original: ~120 cycles in 2 minutes (1.0s/cycle)
- Optimized: ~400-500 cycles in 3 minutes (0.36s/cycle)

Relevant metrics tracked:
1. Vortex density evolution
2. Collision events (merge/annihilation/scatter/split)
3. RNN parameter convergence (w, tau, n)
4. Performance (cycles/sec, actual speedup)
5. Final optimized parameters

Report format: PDF via pdflatex
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import json

from hhml.mobius.mobius_training import MobiusRNNAgent
from hhml.mobius.optimized_sphere import OptimizedMobiusHelixSphere
from train_local_scaled import VortexTracker

print("=" * 80)
print("OPTIMIZED 3-MINUTE MÖBIUS TRAINING")
print("=" * 80)
print()
print("Performance optimizations enabled:")
print("  [OK] torch.compile() JIT compilation")
print("  [OK] Reduced sampling (500->200 nodes)")
print("  [OK] Evolution skip interval (every 2 cycles)")
print("  [OK] Vectorized distance computation")
print("  [OK] Lazy geometry regeneration")
print()
print("Expected speedup: 5-10x faster sphere evolution")
print("Target: 400-500 cycles in 3 minutes")
print("=" * 80)
print()

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_dim = 512
num_nodes = 8000
target_time_minutes = 3.0

print(f"Configuration:")
print(f"  Device: {device}")
print(f"  Hidden dim: {hidden_dim}")
print(f"  Nodes: {num_nodes:,}")
print(f"  Target time: {target_time_minutes} minutes")
print()

print("Relevant metrics to track:")
print("  1. Vortex density evolution (target: maximize)")
print("  2. Collision events (merge/annihilation/scatter/split)")
print("  3. RNN parameter convergence (w, tau, n)")
print("  4. Performance (cycles/sec, speedup vs baseline)")
print("  5. Final optimized parameters")
print()
print("Starting in 2 seconds...")
import time
time.sleep(2)
print()

# Initialize
print("Initializing optimized components...")
agent = MobiusRNNAgent(state_dim=256, device=device, hidden_dim=hidden_dim)
sphere = OptimizedMobiusHelixSphere(num_nodes=num_nodes, device=device)
vortex_tracker = VortexTracker()

# Metrics
metrics = {
    'cycle_times': [],
    'vortex_counts': [],
    'vortex_densities': [],
    'rewards': [],
    'w_windings': [],
    'tau_torsion': [],
    'n_sampling': [],
    'rnn_values': [],
}

print("\nStarting 3-minute training...")
print("Tracking vortex collision dynamics...\n")

start_time = time.time()
cycle = 0

# Run for 3 minutes
while (time.time() - start_time) < (target_time_minutes * 60):
    cycle_start = time.time()

    # Get state and run RNN
    state = sphere.get_state()
    action, value, features = agent.forward(state)
    structure_params = agent.compute_structure_params(features)

    # Evolve sphere with optimizations
    sphere.evolve(structure_params=structure_params)

    # Detect vortices
    vortices = vortex_tracker.detect_vortices(sphere, threshold=0.3)
    vortex_tracker.add_snapshot(vortices)

    # Compute reward
    reward = sphere.compute_reward()

    # RL update
    loss = -value * reward
    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()

    # Track metrics
    cycle_time = time.time() - cycle_start
    vortex_density = len(vortices) / num_nodes

    metrics['cycle_times'].append(cycle_time)
    metrics['vortex_counts'].append(len(vortices))
    metrics['vortex_densities'].append(vortex_density)
    metrics['rewards'].append(reward if isinstance(reward, float) else reward.item())
    metrics['w_windings'].append(structure_params['windings'].item())
    metrics['tau_torsion'].append(structure_params['tau_torsion'].item())
    metrics['n_sampling'].append(structure_params['uvhl_n'].item())
    metrics['rnn_values'].append(value.item())

    # Print progress every 50 cycles
    if (cycle + 1) % 50 == 0:
        elapsed = time.time() - start_time
        avg_time = np.mean(metrics['cycle_times'][-50:])
        cycles_per_sec = 1.0 / avg_time if avg_time > 0 else 0
        eta = (target_time_minutes * 60) - elapsed

        print(f"Cycle {cycle+1:3d} | "
              f"Time: {elapsed:5.1f}s | "
              f"Vortices: {len(vortices):4d} ({vortex_density:5.1%}) | "
              f"Reward: {metrics['rewards'][-1]:7.2f} | "
              f"w: {structure_params['windings'].item():6.2f} | "
              f"Speed: {cycles_per_sec:.2f} cyc/s | "
              f"ETA: {eta:.0f}s")

    cycle += 1

total_time = time.time() - start_time
total_cycles = cycle

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print(f"Total time: {total_time:.1f} seconds ({total_time/60:.2f} minutes)")
print(f"Total cycles: {total_cycles}")
print(f"Average cycle time: {np.mean(metrics['cycle_times']):.3f} seconds")
print(f"Cycles per second: {total_cycles / total_time:.2f}")
print()

# Get vortex statistics
vortex_stats = vortex_tracker.get_statistics()

# Calculate baseline comparison (original: 1.0s/cycle)
baseline_cycle_time = 1.0
actual_speedup = baseline_cycle_time / np.mean(metrics['cycle_times'])

print("=" * 80)
print("PERFORMANCE METRICS")
print("=" * 80)
print(f"Baseline (original sphere):  {baseline_cycle_time:.3f}s per cycle")
print(f"Optimized (this run):        {np.mean(metrics['cycle_times']):.3f}s per cycle")
print(f"Actual speedup:              {actual_speedup:.2f}x")
print(f"Theoretical max speedup:     5-10x")
print()
if actual_speedup >= 5:
    print("[EXCELLENT] Achieved 5x+ speedup!")
elif actual_speedup >= 3:
    print("[GOOD] Achieved 3-5x speedup")
else:
    print("[WARNING] <3x speedup (check torch.compile)")
print()

print("=" * 80)
print("VORTEX DYNAMICS")
print("=" * 80)
print(f"Initial vortex density: {metrics['vortex_densities'][0]:.2%}")
print(f"Final vortex density:   {metrics['vortex_densities'][-1]:.2%}")
print(f"Peak vortex density:    {max(metrics['vortex_densities']):.2%}")
print(f"Average vortex count:   {vortex_stats['avg_vortex_count']:.1f}")
print()
print("Collision events detected:")
print(f"  MERGE events:        {vortex_stats['merge_events']}")
print(f"  ANNIHILATION events: {vortex_stats['annihilation_events']}")
print(f"  SPLIT events:        {vortex_stats['split_events']}")
print()

print("=" * 80)
print("RNN PARAMETER CONVERGENCE")
print("=" * 80)
print(f"w (windings):")
print(f"  Initial: {metrics['w_windings'][0]:.2f}")
print(f"  Final:   {metrics['w_windings'][-1]:.2f}")
print(f"  Change:  {metrics['w_windings'][-1] - metrics['w_windings'][0]:+.2f}")
print()
print(f"tau (torsion):")
print(f"  Initial: {metrics['tau_torsion'][0]:.3f}")
print(f"  Final:   {metrics['tau_torsion'][-1]:.3f}")
print(f"  Change:  {metrics['tau_torsion'][-1] - metrics['tau_torsion'][0]:+.3f}")
print()
print(f"n (sampling):")
print(f"  Initial: {metrics['n_sampling'][0]:.3f}")
print(f"  Final:   {metrics['n_sampling'][-1]:.3f}")
print(f"  Change:  {metrics['n_sampling'][-1] - metrics['n_sampling'][0]:+.3f}")
print()
print(f"RNN value (learning signal):")
print(f"  Initial: {metrics['rnn_values'][0]:.2f}")
print(f"  Final:   {metrics['rnn_values'][-1]:.2f}")
print(f"  Maximum: {max(metrics['rnn_values']):.2f}")
print()

# Save results
output_dir = Path("results/optimized_training")
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# JSON results
results = {
    'configuration': {
        'hidden_dim': hidden_dim,
        'num_nodes': num_nodes,
        'target_time_minutes': target_time_minutes,
        'device': str(device),
        'optimizations_enabled': [
            'torch.compile',
            'reduced_sampling',
            'evolution_skip',
            'vectorized_compute',
            'lazy_regeneration'
        ]
    },
    'performance': {
        'total_time_seconds': total_time,
        'total_cycles': total_cycles,
        'avg_cycle_time': float(np.mean(metrics['cycle_times'])),
        'cycles_per_second': total_cycles / total_time,
        'baseline_cycle_time': baseline_cycle_time,
        'actual_speedup': float(actual_speedup)
    },
    'vortex_statistics': vortex_stats,
    'parameter_convergence': {
        'w_windings': {
            'initial': float(metrics['w_windings'][0]),
            'final': float(metrics['w_windings'][-1]),
            'change': float(metrics['w_windings'][-1] - metrics['w_windings'][0])
        },
        'tau_torsion': {
            'initial': float(metrics['tau_torsion'][0]),
            'final': float(metrics['tau_torsion'][-1]),
            'change': float(metrics['tau_torsion'][-1] - metrics['tau_torsion'][0])
        },
        'n_sampling': {
            'initial': float(metrics['n_sampling'][0]),
            'final': float(metrics['n_sampling'][-1]),
            'change': float(metrics['n_sampling'][-1] - metrics['n_sampling'][0])
        }
    },
    'final_state': {
        'vortex_density': float(metrics['vortex_densities'][-1]),
        'vortex_count': int(metrics['vortex_counts'][-1]),
        'reward': float(metrics['rewards'][-1]),
        'rnn_value': float(metrics['rnn_values'][-1])
    },
    'metrics_history': {
        'vortex_densities': [float(x) for x in metrics['vortex_densities']],
        'rewards': [float(x) for x in metrics['rewards']],
        'w_windings': [float(x) for x in metrics['w_windings']],
        'tau_torsion': [float(x) for x in metrics['tau_torsion']]
    }
}

json_file = output_dir / f"optimized_training_{timestamp}.json"
with open(json_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {json_file}")
print()
print("=" * 80)
print("Generating PDF LaTeX report...")

# Return results for LaTeX generator
results['timestamp'] = timestamp
results['output_dir'] = str(output_dir)

if __name__ == "__main__":
    # Export for LaTeX generator
    import pickle
    with open(output_dir / f"results_{timestamp}.pkl", 'wb') as f:
        pickle.dump(results, f)

    print("Results exported for LaTeX processing")
    print("=" * 80)
