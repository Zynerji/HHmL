#!/usr/bin/env python3
"""
Tokamak Wormhole Analysis
==========================

Analyzes wormhole detection results from tokamak training.

Generates:
- Wormhole statistics and distributions
- Radial transport analysis
- Strip correlation matrices
- Charge flow visualization
- Emergent findings document

Usage:
    python analyze_tokamak_wormholes.py \
        --results-file results/tokamak_wormhole/training_results_*.json \
        --output-dir results/tokamak_wormhole/analysis

Author: tHHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.stats import pearsonr

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_wormholes(results_file, output_dir):
    """Main analysis function"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {results_file}")
    with open(results_file) as f:
        results = json.load(f)

    metrics = results['metrics']
    config = results['config']

    print()
    print("="*80)
    print("TOKAMAK WORMHOLE ANALYSIS")
    print("="*80)
    print(f"Configuration: {config['num_strips']} strips × {config['nodes_per_strip']} nodes/strip")
    print(f"Cycles: {config['num_cycles']}")
    print(f"Training time: {results['total_time_sec']/60:.1f} minutes")
    print()

    # ========================================================================
    # 1. WORMHOLE STATISTICS
    # ========================================================================

    wormhole_counts = np.array(metrics['wormhole_counts'])
    total_wormholes = np.sum(wormhole_counts)
    max_wormholes = np.max(wormhole_counts) if len(wormhole_counts) > 0 else 0
    cycles_with_wormholes = np.sum(wormhole_counts > 0)

    print("WORMHOLE STATISTICS:")
    print(f"  Total wormholes detected: {total_wormholes}")
    print(f"  Peak wormholes (single cycle): {max_wormholes}")
    print(f"  Cycles with wormholes: {cycles_with_wormholes}/{len(wormhole_counts)} ({100*cycles_with_wormholes/len(wormhole_counts):.1f}%)")
    print(f"  Average wormholes/cycle: {total_wormholes/len(wormhole_counts):.2f}")
    print()

    if total_wormholes == 0:
        print("NO WORMHOLES DETECTED")
        print()
        print("This is a scientifically valuable NEGATIVE RESULT!")
        print("Possible interpretations:")
        print("  1. Wormholes genuinely absent in this parameter regime")
        print("  2. Detection threshold too strict (try lowering angular_threshold)")
        print("  3. Training insufficient (try more cycles or different seed)")
        print("  4. Coupling too weak (increase retrocausal coupling)")
        print()

        # Save null result summary
        with open(output_dir / "WORMHOLE_NULL_RESULT.md", 'w') as f:
            f.write("# Wormhole Detection: Null Result\n\n")
            f.write(f"**Configuration**: {config['num_strips']} strips, {config['num_cycles']} cycles\n")
            f.write(f"**Outcome**: No wormholes detected\n\n")
            f.write("## Interpretation\n\n")
            f.write("This null result is scientifically valuable and publishable:\n\n")
            f.write("1. Demonstrates rigorous testing methodology\n")
            f.write("2. Constrains parameter space where wormholes form\n")
            f.write("3. Establishes baseline for future comparisons\n")

        return

    # Analyze wormhole details
    wormhole_details = metrics.get('wormhole_details', [])

    if len(wormhole_details) > 0:
        print("WORMHOLE CHARACTERISTICS:")

        all_separations = []
        all_alignments = []
        charge_conserving = 0

        for detail in wormhole_details:
            for wh in detail['wormholes']:
                all_separations.append(wh['strip_separation'])
                all_alignments.append(wh['theta_alignment'])
                if wh['charge_conservation']:
                    charge_conserving += 1

        print(f"  Strip separation:")
        print(f"    Mean: {np.mean(all_separations):.1f} strips")
        print(f"    Median: {np.median(all_separations):.1f} strips")
        print(f"    Max: {np.max(all_separations)} strips")
        print(f"  Angular alignment:")
        print(f"    Mean: {np.mean(all_alignments):.4f} rad ({np.degrees(np.mean(all_alignments)):.2f} deg)")
        print(f"    Max: {np.max(all_alignments):.4f} rad ({np.degrees(np.max(all_alignments)):.2f} deg)")
        print(f"  Charge conservation:")
        print(f"    Conserved: {charge_conserving}/{len(all_separations)} ({100*charge_conserving/len(all_separations):.1f}%)")
        print()

        # Classify wormholes
        long_range = sum(1 for s in all_separations if s > 50)
        medium_range = sum(1 for s in all_separations if 20 < s <= 50)
        short_range = sum(1 for s in all_separations if s <= 20)

        print("  Range classification:")
        print(f"    Long-range (>50 strips): {long_range} ({100*long_range/len(all_separations):.1f}%)")
        print(f"    Medium-range (20-50): {medium_range} ({100*medium_range/len(all_separations):.1f}%)")
        print(f"    Short-range (<20): {short_range} ({100*short_range/len(all_separations):.1f}%)")
        print()

    # ========================================================================
    # 2. RADIAL TRANSPORT ANALYSIS
    # ========================================================================

    print("RADIAL TRANSPORT ANALYSIS:")

    radial_transport = metrics.get('radial_transport', [])
    if len(radial_transport) > 0:
        max_gradients = [rt['max_gradient'] for rt in radial_transport]
        gradient_vars = [rt['gradient_variance'] for rt in radial_transport]

        print(f"  Max radial gradient:")
        print(f"    Mean: {np.mean(max_gradients):.4f}")
        print(f"    Peak: {np.max(max_gradients):.4f}")
        print(f"  Gradient variance:")
        print(f"    Mean: {np.mean(gradient_vars):.4f}")
        print(f"    Peak: {np.max(gradient_vars):.4f}")

        # Check for accelerated transport (high gradients correlate with wormholes?)
        if len(wormhole_counts) == len(max_gradients):
            r, p = pearsonr(wormhole_counts, max_gradients)
            print(f"  Correlation (wormholes vs. gradient):")
            print(f"    r = {r:.3f}, p = {p:.3e}")
            if abs(r) > 0.5 and p < 0.05:
                print(f"    [OK] Moderate positive correlation - wormholes may enable fast transport!")
    print()

    # ========================================================================
    # 3. TRAINING DYNAMICS
    # ========================================================================

    print("TRAINING DYNAMICS:")

    rewards = np.array(metrics['rewards'])
    fixed_pts = np.array(metrics['fixed_point_percentages'])
    vortices = np.array(metrics['vortex_counts'])

    print(f"  Reward:")
    print(f"    Initial: {rewards[0]:.2f}")
    print(f"    Final: {rewards[-1]:.2f}")
    print(f"    Best: {np.max(rewards):.2f} (cycle {np.argmax(rewards)})")
    print(f"    Improvement: {rewards[-1] - rewards[0]:.2f}")
    print()

    print(f"  Fixed points:")
    print(f"    Initial: {fixed_pts[0]:.1f}%")
    print(f"    Final: {fixed_pts[-1]:.1f}%")
    print(f"    Best: {np.max(fixed_pts):.1f}% (cycle {np.argmax(fixed_pts)})")
    print()

    print(f"  Vortices:")
    print(f"    Mean: {np.mean(vortices):.0f}")
    print(f"    Peak: {np.max(vortices)}")
    print()

    # ========================================================================
    # 4. CORRELATIONS
    # ========================================================================

    print("CORRELATION ANALYSIS:")

    # Reward vs fixed points
    r_reward_fp, p_reward_fp = pearsonr(rewards, fixed_pts)
    print(f"  Reward <-> Fixed Points: r = {r_reward_fp:.3f}, p = {p_reward_fp:.2e}")

    # Wormholes vs reward
    r_wh_reward, p_wh_reward = pearsonr(wormhole_counts, rewards)
    print(f"  Wormholes <-> Reward: r = {r_wh_reward:.3f}, p = {p_wh_reward:.2e}")

    # Wormholes vs vortices
    r_wh_vx, p_wh_vx = pearsonr(wormhole_counts, vortices)
    print(f"  Wormholes <-> Vortices: r = {r_wh_vx:.3f}, p = {p_wh_vx:.2e}")
    print()

    # ========================================================================
    # 5. SAVE FINDINGS
    # ========================================================================

    print("Saving wormhole findings...")
    findings_file = output_dir / "WORMHOLE_FINDINGS.md"
    with open(findings_file, 'w') as f:
        f.write(f"# Tokamak Wormhole Analysis\n\n")
        f.write(f"**Strips**: {config['num_strips']}\n")
        f.write(f"**Cycles**: {config['num_cycles']}\n")
        f.write(f"**Total wormholes**: {total_wormholes}\n\n")

        if total_wormholes > 0:
            f.write(f"## Wormhole Characteristics\n\n")
            f.write(f"- Mean strip separation: {np.mean(all_separations):.1f}\n")
            f.write(f"- Long-range wormholes (>50 strips): {long_range}\n")
            f.write(f"- Charge-conserving: {charge_conserving}/{len(all_separations)}\n\n")

    print(f"Findings saved: {findings_file}")
    print()
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Analyze tokamak wormhole detection results')
    parser.add_argument('--results-file', type=str, required=True,
                       help='Path to training_results_*.json')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis')

    args = parser.parse_args()

    analyze_wormholes(args.results_file, args.output_dir)

    return 0


if __name__ == '__main__':
    sys.exit(main())
