#!/usr/bin/env python3
"""
Full Dark Matter Pruning Theory Test - H200 Optimized
=======================================================

Complete end-to-end test of dark matter as multiverse pruning residue.

Pipeline:
1. Generate multiverse branches (perturbed Möbius configurations)
2. Apply coherence-based pruning
3. Measure dark matter signatures
4. Validate against cosmological observations
5. Generate comprehensive report and visualizations

Target Hardware: NVIDIA H200 (150 GB VRAM)
Expected Duration: 10-30 minutes depending on scale

Author: HHmL Project
Date: 2025-12-17
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hhml.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
from hhml.dark_matter.multiverse_generator import (
    generate_multiverse_branches,
    MultiverseConfig,
    visualize_multiverse,
    export_branches
)
from hhml.dark_matter.tree_multiverse import generate_tree_multiverse
from hhml.dark_matter.pruning_simulator import (
    prune_discordant,
    sweep_pruning_thresholds,
    find_optimal_threshold,
    visualize_pruning
)
from hhml.dark_matter.residue_analyzer import (
    measure_dark_residue,
    visualize_dark_matter_signatures
)
from hhml.dark_matter.cosmological_validator import (
    validate_theory,
    generate_validation_report,
    visualize_cosmological_tests
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Full Dark Matter Pruning Theory Test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Multiverse generation
    parser.add_argument('--num-branches', type=int, default=20,
                       help='Number of multiverse branches to generate')
    parser.add_argument('--perturbation-scale', type=float, default=0.15,
                       help='Amplitude of random perturbations (0-1)')
    parser.add_argument('--perturbation-type', type=str, default='quantum_noise',
                       choices=['gaussian', 'uniform', 'quantum_noise', 'independent'],
                       help='Type of perturbation to apply')
    parser.add_argument('--quantum-decoherence', type=float, default=0.05,
                       help='Quantum decoherence strength (0-1)')
    parser.add_argument('--tree-mode', action='store_true',
                       help='Use tree-structured multiverse (sequential branching)')
    parser.add_argument('--branching-factor', type=int, default=2,
                       help='Branches per parent in tree mode (default: 2)')

    # Möbius strip configuration
    parser.add_argument('--num-strips', type=int, default=10,
                       help='Number of Möbius strips')
    parser.add_argument('--nodes-per-strip', type=int, default=2000,
                       help='Nodes per strip')

    # Pruning configuration
    parser.add_argument('--coherence-threshold', type=float, default=0.82,
                       help='Coherence threshold for pruning (0-1)')
    parser.add_argument('--sweep-thresholds', action='store_true',
                       help='Sweep multiple thresholds to find optimal')
    parser.add_argument('--find-optimal', action='store_true',
                       help='Binary search for 27%% dark fraction')

    # Target dark matter fraction
    parser.add_argument('--target-dark-fraction', type=float, default=0.27,
                       help='Target dark matter fraction (ΛCDM = 0.27)')

    # Hardware
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run on')

    # Output
    parser.add_argument('--output-dir', type=str, default='results/dark_matter_test',
                       help='Output directory for results')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    """Run full dark matter pruning test."""
    args = parse_args()

    print("="*80)
    print("DARK MATTER AS MULTIVERSE PRUNING RESIDUE - FULL TEST")
    print("="*80)
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create timestamp-based subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print(f"Output directory: {run_dir}")
    print()

    # Save configuration
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {config_path}")
    print()

    # Device setup
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, falling back to CPU")
            device = 'cpu'
        else:
            device = 'cuda'
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print()

    # =============================================================================
    # PHASE 1: Generate Multiverse Branches
    # =============================================================================

    print("="*80)
    print("PHASE 1: MULTIVERSE BRANCH GENERATION")
    print("="*80)
    print()

    print(f"Configuration:")
    print(f"  Möbius strips: {args.num_strips}")
    print(f"  Nodes per strip: {args.nodes_per_strip}")
    print(f"  Total nodes: {args.num_strips * args.nodes_per_strip:,}")
    print(f"  Multiverse branches: {args.num_branches}")
    print(f"  Perturbation scale: {args.perturbation_scale}")
    print(f"  Perturbation type: {args.perturbation_type}")
    print(f"  Quantum decoherence: {args.quantum_decoherence}")
    print()

    t0 = time.time()

    # Create base Möbius configuration
    print("Creating base Möbius strip configuration...")
    base_strips = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        device=device
    )

    # Generate multiverse branches
    multiverse_config = MultiverseConfig(
        num_branches=args.num_branches,
        perturbation_scale=args.perturbation_scale,
        base_strips=args.num_strips,
        base_nodes=args.num_strips * args.nodes_per_strip,
        coherence_seed=args.seed,
        perturbation_type=args.perturbation_type,
        quantum_decoherence=args.quantum_decoherence
    )

    if args.tree_mode:
        print(f"Generating {args.num_branches} multiverse branches (TREE MODE, branching_factor={args.branching_factor})...")
        branches = generate_tree_multiverse(base_strips, multiverse_config, args.branching_factor, device)
    else:
        print(f"Generating {args.num_branches} multiverse branches...")
        branches = generate_multiverse_branches(base_strips, multiverse_config, device)

    t1 = time.time()
    print(f"✓ Generated {len(branches)} branches in {t1-t0:.1f}s")
    print()

    # Visualize multiverse
    print("Visualizing multiverse ensemble...")
    multiverse_viz_path = run_dir / "multiverse_ensemble.png"
    visualize_multiverse(branches, str(multiverse_viz_path))
    print()

    # Export branches
    branches_export_path = run_dir / "multiverse_branches.pt"
    export_branches(branches, str(branches_export_path))
    print()

    # =============================================================================
    # PHASE 2: Coherence-Based Pruning
    # =============================================================================

    print("="*80)
    print("PHASE 2: COHERENCE-BASED PRUNING")
    print("="*80)
    print()

    t0 = time.time()

    if args.sweep_thresholds:
        # Sweep thresholds to find 27%
        print("Sweeping coherence thresholds...")
        print()
        sweep_results = sweep_pruning_thresholds(branches, device=device)

        # Find best match to target
        best_result = None
        best_error = float('inf')
        for result in sweep_results:
            error = abs(result.dark_fraction - args.target_dark_fraction)
            if error < best_error:
                best_error = error
                best_result = result

        pruning_result = best_result
        print()
        print(f"✓ Best threshold: {pruning_result.threshold_used:.3f}")
        print(f"  Dark fraction: {pruning_result.dark_fraction:.2%}")
        print(f"  Error from target: ±{best_error*100:.1f}%")

    elif args.find_optimal:
        # Binary search for optimal threshold
        print("Binary search for optimal threshold...")
        print()
        optimal_threshold, pruning_result = find_optimal_threshold(
            branches,
            target_dark_fraction=args.target_dark_fraction,
            device=device
        )

        print()
        print(f"✓ Optimal threshold: {optimal_threshold:.4f}")
        print(f"  Dark fraction: {pruning_result.dark_fraction:.2%}")

    else:
        # Single threshold
        print(f"Pruning with fixed threshold: {args.coherence_threshold:.3f}")
        print()
        pruning_result = prune_discordant(branches, args.coherence_threshold, device)

        print(f"✓ Pruning complete:")
        print(f"  Kept branches: {len(pruning_result.kept_branches)}")
        print(f"  Pruned branches: {len(pruning_result.pruned_branches)}")
        print(f"  Dark fraction: {pruning_result.dark_fraction:.2%}")
        print(f"  Hologram quality: {pruning_result.hologram_quality:.3f}")

    t1 = time.time()
    print(f"\n  Time: {t1-t0:.1f}s")
    print()

    # Visualize pruning
    print("Visualizing pruning results...")
    pruning_viz_path = run_dir / "pruning_analysis.png"
    visualize_pruning(pruning_result, str(pruning_viz_path))
    print()

    # =============================================================================
    # PHASE 3: Dark Matter Residue Measurement
    # =============================================================================

    print("="*80)
    print("PHASE 3: DARK MATTER RESIDUE MEASUREMENT")
    print("="*80)
    print()

    t0 = time.time()

    print("Measuring dark matter signatures...")
    dark_metrics = measure_dark_residue(pruning_result, device)

    t1 = time.time()

    print(f"✓ Measurements complete in {t1-t0:.1f}s")
    print()
    print("Dark Matter Metrics:")
    print(f"  Mass fraction: {dark_metrics.mass_fraction:.2%}")
    print(f"  Entropy ratio: {dark_metrics.entropy_ratio:.2%}")
    print(f"  Fractal dimension: {dark_metrics.fractal_dimension:.2f} (target: 2.6 ± 0.2)")
    print(f"  Hopkins clustering: {dark_metrics.spatial_clustering:.2f}")
    print(f"  Rotation curve match: {dark_metrics.rotation_curve_match:.3f}")
    print(f"  Field curvature (RMS): {dark_metrics.curvature_residue:.3f}")
    print()

    # Visualize signatures
    print("Visualizing dark matter signatures...")
    signatures_viz_path = run_dir / "dark_matter_signatures.png"
    visualize_dark_matter_signatures(dark_metrics, pruning_result, str(signatures_viz_path))
    print()

    # =============================================================================
    # PHASE 4: Cosmological Validation
    # =============================================================================

    print("="*80)
    print("PHASE 4: COSMOLOGICAL VALIDATION")
    print("="*80)
    print()

    t0 = time.time()

    print("Validating against cosmological observations...")
    cosmological_tests = validate_theory(pruning_result, dark_metrics, device)

    t1 = time.time()

    print(f"✓ Validation complete in {t1-t0:.1f}s")
    print()
    print("Cosmological Test Results:")
    print(f"  ΛCDM match: {cosmological_tests.lambda_cdm_match:.3f}")
    print(f"  CMB spectrum: {cosmological_tests.cmb_power_spectrum_match:.3f}")
    print(f"  Large-scale structure: {cosmological_tests.large_scale_structure_match:.3f}")
    print(f"  Lensing signature: {cosmological_tests.lensing_signature_match:.3f}")
    print(f"  Rotation curves: {cosmological_tests.rotation_curve_consistency:.3f}")
    print(f"  Entropy conservation: {cosmological_tests.entropy_conservation_score:.3f}")
    print()
    print(f"Overall Validity: {cosmological_tests.overall_validity_score:.3f}")
    print(f"Tests Passed: {cosmological_tests.tests_passed}/6")
    print(f"Tests Failed: {cosmological_tests.tests_failed}/6")
    print()

    # Generate validation report
    print("Generating validation report...")
    report_path = run_dir / "cosmological_validation_report.txt"
    report_text = generate_validation_report(cosmological_tests, dark_metrics, pruning_result, str(report_path))
    print()

    # Visualize tests
    print("Visualizing cosmological tests...")
    tests_viz_path = run_dir / "cosmological_tests.png"
    visualize_cosmological_tests(cosmological_tests, str(tests_viz_path))
    print()

    # =============================================================================
    # FINAL SUMMARY
    # =============================================================================

    print("="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print()

    if cosmological_tests.overall_validity_score >= 0.7 and cosmological_tests.tests_passed >= 4:
        print("✓✓✓ THEORY VALIDATED")
        print()
        print("The dark matter pruning hypothesis is supported by the data.")
        print("Key findings:")
        print(f"  - Dark fraction: {dark_metrics.mass_fraction:.2%} (target: 27%)")
        print(f"  - Fractal dimension: {dark_metrics.fractal_dimension:.2f} (cosmic web: 2.6)")
        print(f"  - Rotation curves: {'Flat' if dark_metrics.rotation_curve_match > 0.7 else 'Keplerian'}")
        print(f"  - Entropy conserved: {pruning_result.entropy_conservation:.3f}")
        print()
        print("Implications:")
        print("  - Dark matter may emerge from holographic pruning")
        print("  - Multiverse branches leave gravitational residue")
        print("  - No exotic particles needed")
        print()
        print("Next steps:")
        print("  - Test at larger scales (cosmological simulations)")
        print("  - Compare with particle dark matter models")
        print("  - Design observational tests")

    elif cosmological_tests.overall_validity_score >= 0.5:
        print("⚠ THEORY PARTIALLY SUPPORTED")
        print()
        print("Some predictions match observations, others fail.")
        print("Key findings:")
        print(f"  - Dark fraction: {dark_metrics.mass_fraction:.2%} (target: 27%)")
        print(f"  - Tests passed: {cosmological_tests.tests_passed}/6")
        print()
        print("Interpretation:")
        print("  - Pruning mechanism may be partially correct")
        print("  - Needs refinement (threshold, scale, or algorithm)")
        print("  - Hybrid model possible (residue + particles)")
        print()
        print("Next steps:")
        print("  - Investigate failed tests")
        print("  - Adjust pruning algorithm")
        print("  - Test at multiple scales")

    else:
        print("✗✗✗ THEORY FALSIFIED")
        print()
        print("Predictions do not match cosmological observations.")
        print("Key findings:")
        print(f"  - Dark fraction: {dark_metrics.mass_fraction:.2%} (target: 27%)")
        print(f"  - Tests passed: {cosmological_tests.tests_passed}/6")
        print(f"  - Overall score: {cosmological_tests.overall_validity_score:.3f}")
        print()
        print("Conclusion:")
        print("  - Pruning residue does not explain dark matter")
        print("  - Dark matter likely requires new particles")
        print()
        print("Lessons learned:")
        print("  - Holographic pruning creates residue, but wrong properties")
        print("  - Information residue ≠ dark matter")

    print()
    print("="*80)
    print("OUTPUT FILES")
    print("="*80)
    print()
    print(f"Results directory: {run_dir}")
    print()
    print("Generated files:")
    print(f"  1. config.json - Test configuration")
    print(f"  2. multiverse_ensemble.png - Multiverse visualization")
    print(f"  3. multiverse_branches.pt - Branch data export")
    print(f"  4. pruning_analysis.png - Pruning results")
    print(f"  5. dark_matter_signatures.png - DM signatures")
    print(f"  6. cosmological_tests.png - Validation tests")
    print(f"  7. cosmological_validation_report.txt - Full report")
    print()

    # Save summary JSON
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'multiverse': {
            'num_branches': len(branches),
            'perturbation_scale': args.perturbation_scale,
            'perturbation_type': args.perturbation_type
        },
        'pruning': {
            'threshold': pruning_result.threshold_used,
            'kept_branches': len(pruning_result.kept_branches),
            'pruned_branches': len(pruning_result.pruned_branches),
            'dark_fraction': pruning_result.dark_fraction,
            'hologram_quality': pruning_result.hologram_quality,
            'entropy_conservation': pruning_result.entropy_conservation
        },
        'dark_matter_metrics': {
            'mass_fraction': dark_metrics.mass_fraction,
            'entropy_ratio': dark_metrics.entropy_ratio,
            'fractal_dimension': dark_metrics.fractal_dimension,
            'hopkins_clustering': dark_metrics.spatial_clustering,
            'rotation_curve_match': dark_metrics.rotation_curve_match,
            'curvature_residue': dark_metrics.curvature_residue,
            'field_coherence': dark_metrics.field_coherence_residue
        },
        'cosmological_tests': {
            'lambda_cdm_match': cosmological_tests.lambda_cdm_match,
            'cmb_match': cosmological_tests.cmb_power_spectrum_match,
            'lss_match': cosmological_tests.large_scale_structure_match,
            'lensing_match': cosmological_tests.lensing_signature_match,
            'rotation_curve': cosmological_tests.rotation_curve_consistency,
            'entropy_score': cosmological_tests.entropy_conservation_score,
            'overall_score': cosmological_tests.overall_validity_score,
            'tests_passed': cosmological_tests.tests_passed,
            'tests_failed': cosmological_tests.tests_failed
        },
        'verdict': (
            'VALIDATED' if cosmological_tests.overall_validity_score >= 0.7 and cosmological_tests.tests_passed >= 4
            else 'PARTIAL' if cosmological_tests.overall_validity_score >= 0.5
            else 'FALSIFIED'
        )
    }

    summary_path = run_dir / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  8. summary.json - Machine-readable summary")
    print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
