#!/usr/bin/env python3
"""
Master Investigation Script: Hash Quine Discovery Reverse-Mapping
==================================================================

Runs all 4 investigation scripts to systematically explore hash quine mechanism
and test alternative applications.

Investigations:
1. Topology Comparison (Mobius vs Torus vs Sphere)
2. TSP Optimization (continuous landscape test)
3. Entropy Analysis (information-theoretic properties)
4. Holographic Interpretation (bulk-boundary correspondence)

This addresses all CRITICAL TODO items from CLAUDE.md:
- Analyze mechanism (why does recursion create self-similarity?)
- Test on other problems (TSP where smooth gradients exist)
- Mathematical formalization (entropy, information theory)
- Holographic implications (bulk-boundary duality)

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import subprocess
import argparse
import time
import json
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_investigation(script_name: str, args_dict: dict):
    """Run a single investigation script."""

    script_path = Path(__file__).parent / script_name

    # Build command
    cmd = [sys.executable, str(script_path)]

    for key, value in args_dict.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))

    print(f"\nRunning: {script_name}")
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per investigation
        )

        duration = time.time() - start_time

        # Print output
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        return {
            'script': script_name,
            'exit_code': result.exit_code,
            'duration_sec': duration,
            'success': result.exit_code == 0,
            'stdout': result.stdout,
            'stderr': result.stderr
        }

    except subprocess.TimeoutExpired:
        print(f"ERROR: {script_name} timed out after 600s")
        return {
            'script': script_name,
            'exit_code': -1,
            'duration_sec': 600,
            'success': False,
            'error': 'timeout'
        }

    except Exception as e:
        print(f"ERROR running {script_name}: {e}")
        return {
            'script': script_name,
            'exit_code': -1,
            'duration_sec': time.time() - start_time,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='Run All Hash Quine Investigations')
    parser.add_argument('--nodes', type=int, default=5000,
                       help='Number of lattice nodes (for topology/entropy/holography)')
    parser.add_argument('--max-depth', type=int, default=2,
                       help='Max recursion depth (for all tests)')
    parser.add_argument('--num-cities', type=int, default=50,
                       help='Number of cities for TSP test')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output-dir', type=str, default='HASH-QUINE/investigations/results',
                       help='Output directory for results')
    parser.add_argument('--skip', type=str, nargs='*', default=[],
                       help='Skip specific investigations (1, 2, 3, or 4)')

    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("HASH QUINE DISCOVERY: REVERSE-MAPPING INVESTIGATION SUITE")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Nodes: {args.nodes}")
    print(f"  Max depth: {args.max_depth}")
    print(f"  TSP cities: {args.num_cities}")
    print(f"  Device: {args.device}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output dir: {output_dir}")
    print()

    # Define investigations
    investigations = [
        {
            'id': '1',
            'name': 'Topology Comparison',
            'script': '1_topology_comparison.py',
            'args': {
                'nodes': args.nodes,
                'max-depth': args.max_depth,
                'device': args.device,
                'seed': args.seed,
                'output-dir': args.output_dir
            },
            'question': 'Is Mobius twist critical for hash quine emergence?'
        },
        {
            'id': '2',
            'name': 'TSP Optimization',
            'script': '2_tsp_optimization.py',
            'args': {
                'num-cities': args.num_cities,
                'lattice-nodes': args.nodes,
                'max-depth': args.max_depth,
                'device': args.device,
                'seed': args.seed,
                'output-dir': args.output_dir
            },
            'question': 'Does recursive topology help continuous optimization (unlike mining)?'
        },
        {
            'id': '3',
            'name': 'Entropy Analysis',
            'script': '3_entropy_analysis.py',
            'args': {
                'nodes': args.nodes,
                'max-depth': args.max_depth,
                'device': args.device,
                'seed': args.seed,
                'output-dir': args.output_dir
            },
            'question': 'Do hash quines have true mathematical structure (entropy/complexity)?'
        },
        {
            'id': '4',
            'name': 'Holographic Interpretation',
            'script': '4_holographic_interpretation.py',
            'args': {
                'nodes': args.nodes,
                'max-depth': args.max_depth,
                'device': args.device,
                'seed': args.seed,
                'output-dir': args.output_dir
            },
            'question': 'Do recursive layers exhibit bulk-boundary holographic duality?'
        }
    ]

    # Run investigations
    results = []
    total_start = time.time()

    for inv in investigations:
        if inv['id'] in args.skip:
            print(f"\nSkipping Investigation {inv['id']}: {inv['name']}")
            continue

        print()
        print("="*80)
        print(f"INVESTIGATION {inv['id']}: {inv['name']}")
        print("="*80)
        print(f"Question: {inv['question']}")
        print()

        result = run_investigation(inv['script'], inv['args'])
        result['investigation_name'] = inv['name']
        result['question'] = inv['question']
        results.append(result)

    total_duration = time.time() - total_start

    # Summary
    print()
    print("="*80)
    print("INVESTIGATION SUITE SUMMARY")
    print("="*80)
    print()

    print(f"Total duration: {total_duration:.2f}s")
    print()

    print(f"{'ID':<5} {'Investigation':<30} {'Status':<10} {'Duration':<12}")
    print("-" * 80)

    for i, result in enumerate(results, 1):
        status = "SUCCESS" if result['success'] else "FAILED"
        print(f"{i:<5} {result['investigation_name']:<30} {status:<10} {result['duration_sec']:<12.2f}s")

    print()

    num_success = sum(1 for r in results if r['success'])
    num_total = len(results)

    print(f"Completed: {num_success}/{num_total} investigations successful")
    print()

    # Key findings summary
    print("="*80)
    print("KEY FINDINGS SUMMARY")
    print("="*80)
    print()

    for inv_id, inv in enumerate(investigations, 1):
        if inv['id'] in args.skip:
            continue

        print(f"{inv_id}. {inv['name']}")
        print(f"   Question: {inv['question']}")

        # Try to extract key result from JSON
        result_files = list(output_dir.glob(f"{inv['script'].replace('.py', '')}_*.json"))
        if result_files:
            latest_result = sorted(result_files)[-1]
            with open(latest_result) as f:
                data = json.load(f)

            # Extract key insights based on investigation
            if '1_topology' in inv['script']:
                interp = data.get('interpretation', 'Unknown')
                print(f"   Result: {interp}")

            elif '2_tsp' in inv['script']:
                success = data.get('success', False)
                if success:
                    print(f"   Result: Recursive topology helps TSP optimization")
                else:
                    print(f"   Result: Recursive topology does not help TSP")

            elif '3_entropy' in inv['script']:
                is_structured = data.get('interpretation', {}).get('is_structured', False)
                if is_structured:
                    print(f"   Result: Hash quines have TRUE mathematical structure (compressible)")
                else:
                    print(f"   Result: Hash quines may be statistical fluctuation")

            elif '4_holographic' in inv['script']:
                evidence = data.get('summary', {}).get('holographic_evidence', 'unknown')
                print(f"   Result: Holographic evidence is {evidence}")

        print()

    # Implications
    print("="*80)
    print("IMPLICATIONS FOR HHmL FRAMEWORK")
    print("="*80)
    print()

    print("Based on these investigations, we can now understand:")
    print()
    print("1. MECHANISM: Why recursive topology creates hash quines")
    print("   -> Answer from topology comparison + entropy analysis")
    print()
    print("2. APPLICATIONS: Where recursive topology helps vs. fails")
    print("   -> Mining: FAILS (chaotic landscape)")
    print("   -> TSP: ? (smooth landscape) - check results above")
    print()
    print("3. MATHEMATICS: Formal properties of hash quines")
    print("   -> Entropy, compression, complexity - check results above")
    print()
    print("4. PHYSICS: Holographic interpretation")
    print("   -> Bulk-boundary duality - check results above")
    print()

    print("NEXT STEPS:")
    print("- If TSP succeeded: Test on protein folding, SAT solving")
    print("- If holographic evidence strong: Formalize bulk-boundary equations")
    print("- If Mobius is critical: Study what makes twist special")
    print("- Scale up on H200 (100K-1M nodes) for definitive tests")

    print()

    # Save master summary
    master_summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'total_duration_sec': total_duration,
        'investigations_run': num_total,
        'investigations_successful': num_success,
        'results': results
    }

    summary_path = output_dir / f'master_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(master_summary, f, indent=2)

    print(f"Master summary saved: {summary_path}")
    print()

    return 0 if num_success == num_total else 1


if __name__ == '__main__':
    sys.exit(main())
