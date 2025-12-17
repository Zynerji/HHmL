#!/usr/bin/env python3
"""
Multi-Strip Flux Tube Benchmark Suite
======================================
Comprehensive benchmarking for sparse/dense tokamak multi-strip simulations.

Auto-scales from local CPU to H200 GPU.

Benchmarks:
1. Geometry generation time (vs N strips, nodes/strip)
2. Graph construction time (sparse vs dense)
3. Wave propagation throughput (nodes/sec)
4. Memory usage scaling
5. Sparse vs Dense accuracy comparison
6. Optimal configurations for each hardware tier

Usage:
    # Quick benchmark (1 minute)
    python benchmark_multi_strip.py --mode quick

    # Full benchmark suite (10 minutes)
    python benchmark_multi_strip.py --mode full

    # H200 production scale (30+ minutes)
    python benchmark_multi_strip.py --mode production

Author: HHmL Framework
Date: 2025-12-16
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import time
import json
from datetime import datetime
import argparse

from hhml.utils.hardware_config import HardwareConfig, SimulationParams
from hhml.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips


class MultiStripBenchmark:
    """Benchmark suite for multi-strip flux tube simulations"""

    def __init__(self, output_dir: str = "results/benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Hardware detection
        self.hw_config = HardwareConfig()
        self.device = self.hw_config.device_type

        # Results storage
        self.results = {
            'hardware': {
                'tier': self.hw_config.get_hardware_tier(),
                'device': self.device,
                'gpu_name': self.hw_config.gpu_name,
                'vram_gb': self.hw_config.vram_gb,
                'ram_gb': self.hw_config.ram_gb,
                'cpu_cores': self.hw_config.cpu_cores,
                'is_h200': self.hw_config.is_h200
            },
            'benchmarks': []
        }

    def benchmark_geometry_generation(self, num_strips_list, nodes_per_strip):
        """Benchmark 1: Geometry generation time vs N strips"""
        print("\n" + "=" * 80)
        print("BENCHMARK 1: Geometry Generation Time")
        print("=" * 80)

        results = []

        for num_strips in num_strips_list:
            print(f"\nTesting N={num_strips} strips, {nodes_per_strip:,} nodes/strip...")

            start = time.time()
            strips = SparseTokamakMobiusStrips(
                num_strips=num_strips,
                nodes_per_strip=nodes_per_strip,
                device=self.device,
                sparse_threshold=0.3,
                max_neighbors=100
            )
            elapsed = time.time() - start

            total_nodes = num_strips * nodes_per_strip

            result = {
                'num_strips': num_strips,
                'nodes_per_strip': nodes_per_strip,
                'total_nodes': total_nodes,
                'time_seconds': elapsed,
                'nodes_per_second': total_nodes / elapsed,
                'mode': 'sparse' if strips.use_sparse else 'dense',
                'memory_mb': strips._estimate_memory() / 1e6
            }

            results.append(result)

            print(f"  Time: {elapsed:.2f}s")
            print(f"  Throughput: {result['nodes_per_second']:,.0f} nodes/sec")
            print(f"  Memory: {result['memory_mb']:.1f} MB")
            print(f"  Mode: {result['mode'].upper()}")

            del strips
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.results['benchmarks'].append({
            'name': 'geometry_generation',
            'results': results
        })

        print("\n[OK] Geometry generation benchmark complete")

    def benchmark_wave_propagation(self, num_strips, nodes_per_strip, num_iterations=10):
        """Benchmark 2: Wave propagation throughput"""
        print("\n" + "=" * 80)
        print("BENCHMARK 2: Wave Propagation Throughput")
        print("=" * 80)

        print(f"\nConfiguration: N={num_strips} strips, {nodes_per_strip:,} nodes/strip")
        print(f"Iterations: {num_iterations}")

        strips = SparseTokamakMobiusStrips(
            num_strips=num_strips,
            nodes_per_strip=nodes_per_strip,
            device=self.device,
            sparse_threshold=0.3,
            max_neighbors=100
        )

        mode = 'sparse' if strips.use_sparse else 'dense'

        # Warmup
        print("\nWarming up...")
        for _ in range(3):
            strips.evolve_field(t=0.0, sample_ratio=0.1)

        # Benchmark
        print("Running benchmark...")
        times = []
        nodes_updated_list = []

        for i in range(num_iterations):
            start = time.time()
            field_updates, sample_indices = strips.evolve_field(t=float(i), sample_ratio=0.1)
            elapsed = time.time() - start

            times.append(elapsed)
            nodes_updated_list.append(len(sample_indices))

        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_nodes = np.mean(nodes_updated_list)
        throughput = avg_nodes / avg_time

        result = {
            'num_strips': num_strips,
            'nodes_per_strip': nodes_per_strip,
            'total_nodes': num_strips * nodes_per_strip,
            'mode': mode,
            'iterations': num_iterations,
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'avg_nodes_updated': avg_nodes,
            'throughput_nodes_per_sec': throughput,
            'throughput_Mnodes_per_sec': throughput / 1e6
        }

        self.results['benchmarks'].append({
            'name': 'wave_propagation',
            'results': [result]
        })

        print(f"\n  Mode: {mode.upper()}")
        print(f"  Avg time per iteration: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
        print(f"  Avg nodes updated: {avg_nodes:.0f}")
        print(f"  Throughput: {throughput:,.0f} nodes/sec ({throughput/1e6:.3f} Mnodes/sec)")

        del strips
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("\n[OK] Wave propagation benchmark complete")

    def benchmark_scaling(self, base_strips=2, nodes_list=[1000, 2000, 5000, 10000]):
        """Benchmark 3: Scaling analysis (nodes per strip)"""
        print("\n" + "=" * 80)
        print("BENCHMARK 3: Scaling Analysis (Nodes per Strip)")
        print("=" * 80)

        results = []

        for nodes in nodes_list:
            print(f"\nTesting {nodes:,} nodes/strip...")

            start = time.time()
            strips = SparseTokamakMobiusStrips(
                num_strips=base_strips,
                nodes_per_strip=nodes,
                device=self.device
            )

            # Run one propagation cycle
            prop_start = time.time()
            field_updates, sample_indices = strips.evolve_field(t=0.0, sample_ratio=0.1)
            prop_time = time.time() - prop_start

            total_time = time.time() - start

            result = {
                'num_strips': base_strips,
                'nodes_per_strip': nodes,
                'total_nodes': base_strips * nodes,
                'mode': 'sparse' if strips.use_sparse else 'dense',
                'total_time': total_time,
                'propagation_time_ms': prop_time * 1000,
                'memory_mb': strips._estimate_memory() / 1e6,
                'num_edges': strips.num_edges,
                'sparsity_%': 100 * (1 - strips.num_edges / (base_strips * nodes * (base_strips * nodes - 1)))
            }

            results.append(result)

            print(f"  Mode: {result['mode'].upper()}")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Propagation: {prop_time*1000:.2f}ms")
            print(f"  Memory: {result['memory_mb']:.1f} MB")
            print(f"  Edges: {result['num_edges']:,} ({result['sparsity_%']:.1f}% sparse)")

            del strips
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.results['benchmarks'].append({
            'name': 'scaling_nodes',
            'results': results
        })

        print("\n[OK] Scaling benchmark complete")

    def benchmark_sparse_vs_dense(self, num_strips=2, nodes_per_strip=5000):
        """Benchmark 4: Sparse vs Dense mode comparison (if both supported)"""
        print("\n" + "=" * 80)
        print("BENCHMARK 4: Sparse vs Dense Comparison")
        print("=" * 80)

        # Test if dense mode is feasible
        total_nodes = num_strips * nodes_per_strip
        dense_memory_gb = (total_nodes ** 2) * 4 / 1e9  # Distance matrix

        if dense_memory_gb > self.hw_config.vram_gb * 0.8 and self.device == 'cuda':
            print(f"\nSkipping: Dense mode would require {dense_memory_gb:.1f} GB")
            print(f"Available VRAM: {self.hw_config.vram_gb:.1f} GB")
            return

        if self.device == 'cpu' and dense_memory_gb > self.hw_config.ram_gb * 0.5:
            print(f"\nSkipping: Dense mode would require {dense_memory_gb:.1f} GB")
            print(f"Available RAM: {self.hw_config.ram_gb:.1f} GB")
            return

        results = []

        for force_sparse, mode_name in [(True, 'sparse'), (False, 'dense')]:
            print(f"\nTesting {mode_name.upper()} mode...")

            start = time.time()
            strips = SparseTokamakMobiusStrips(
                num_strips=num_strips,
                nodes_per_strip=nodes_per_strip,
                device=self.device,
                force_sparse=force_sparse,
                force_dense=not force_sparse
            )

            # Propagation benchmark
            prop_times = []
            for _ in range(5):
                prop_start = time.time()
                field_updates, sample_indices = strips.evolve_field(t=0.0, sample_ratio=0.1)
                prop_times.append(time.time() - prop_start)

            avg_prop_time = np.mean(prop_times)

            result = {
                'mode': mode_name,
                'num_strips': num_strips,
                'nodes_per_strip': nodes_per_strip,
                'total_nodes': total_nodes,
                'init_time': time.time() - start - np.sum(prop_times),
                'avg_propagation_time_ms': avg_prop_time * 1000,
                'memory_mb': strips._estimate_memory() / 1e6,
                'num_edges': strips.num_edges
            }

            results.append(result)

            print(f"  Init time: {result['init_time']:.2f}s")
            print(f"  Propagation: {avg_prop_time*1000:.2f}ms")
            print(f"  Memory: {result['memory_mb']:.1f} MB")

            del strips
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Compare
        if len(results) == 2:
            speedup = results[0]['avg_propagation_time_ms'] / results[1]['avg_propagation_time_ms']
            print(f"\n  Dense speedup: {speedup:.2f}x faster than sparse")

        self.results['benchmarks'].append({
            'name': 'sparse_vs_dense',
            'results': results
        })

        print("\n[OK] Sparse vs Dense comparison complete")

    def run_quick_benchmark(self):
        """Quick benchmark suite (1-2 minutes)"""
        print("\n" + "=" * 80)
        print("QUICK BENCHMARK SUITE")
        print("=" * 80)

        self.hw_config.print_info()

        params = self.hw_config.get_optimal_params('benchmark')

        # Benchmark 1: Geometry generation (small scale)
        self.benchmark_geometry_generation(
            num_strips_list=[1, 2, 3],
            nodes_per_strip=min(params.nodes_per_strip, 2000)
        )

        # Benchmark 2: Wave propagation
        self.benchmark_wave_propagation(
            num_strips=2,
            nodes_per_strip=min(params.nodes_per_strip, 2000),
            num_iterations=10
        )

        # Benchmark 3: Scaling
        max_nodes = min(params.nodes_per_strip, 5000)
        self.benchmark_scaling(
            base_strips=2,
            nodes_list=[1000, 2000, max_nodes]
        )

    def run_full_benchmark(self):
        """Full benchmark suite (10-15 minutes)"""
        print("\n" + "=" * 80)
        print("FULL BENCHMARK SUITE")
        print("=" * 80)

        self.hw_config.print_info()

        params = self.hw_config.get_optimal_params('training')

        # Benchmark 1: Geometry generation
        self.benchmark_geometry_generation(
            num_strips_list=[1, 2, 4, 6, 8],
            nodes_per_strip=min(params.nodes_per_strip, 10000)
        )

        # Benchmark 2: Wave propagation
        self.benchmark_wave_propagation(
            num_strips=params.num_strips,
            nodes_per_strip=params.nodes_per_strip,
            num_iterations=20
        )

        # Benchmark 3: Scaling
        self.benchmark_scaling(
            base_strips=2,
            nodes_list=[1000, 5000, 10000, 20000, 50000][: 4 if params.nodes_per_strip < 50000 else 5]
        )

        # Benchmark 4: Sparse vs Dense (if feasible)
        self.benchmark_sparse_vs_dense(
            num_strips=2,
            nodes_per_strip=min(10000, params.nodes_per_strip)
        )

    def run_production_benchmark(self):
        """Production benchmark for H200 (30+ minutes)"""
        print("\n" + "=" * 80)
        print("PRODUCTION BENCHMARK SUITE (H200)")
        print("=" * 80)

        if not self.hw_config.is_h200:
            print("\n[WARNING] Not running on H200 - this will take a very long time!")
            print("Consider using --mode full instead")
            user_input = input("Continue anyway? (yes/no): ")
            if user_input.lower() != 'yes':
                return

        self.hw_config.print_info()

        params = self.hw_config.get_optimal_params('production')

        # Benchmark 1: Large-scale geometry
        self.benchmark_geometry_generation(
            num_strips_list=[2, 4, 8, 12, 16, 20],
            nodes_per_strip=min(params.nodes_per_strip, 100000)
        )

        # Benchmark 2: Production wave propagation
        self.benchmark_wave_propagation(
            num_strips=params.num_strips,
            nodes_per_strip=params.nodes_per_strip,
            num_iterations=50
        )

        # Benchmark 3: Scaling to limits
        self.benchmark_scaling(
            base_strips=10,
            nodes_list=[10000, 50000, 100000, 250000, 500000]
        )

        # Benchmark 4: Sparse vs Dense at scale
        self.benchmark_sparse_vs_dense(
            num_strips=10,
            nodes_per_strip=50000
        )

    def save_results(self):
        """Save benchmark results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tier = self.hw_config.get_hardware_tier()
        filename = f"benchmark_{tier}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n[OK] Results saved to: {filepath}")
        return filepath

    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        print(f"\nHardware: {self.results['hardware']['tier'].upper()}")
        print(f"Device: {self.results['hardware']['device'].upper()}")
        if self.results['hardware']['gpu_name']:
            print(f"GPU: {self.results['hardware']['gpu_name']}")
        print(f"Benchmarks run: {len(self.results['benchmarks'])}")

        print("\nKey Results:")
        for benchmark in self.results['benchmarks']:
            name = benchmark['name']
            results = benchmark['results']

            if name == 'geometry_generation':
                max_result = max(results, key=lambda x: x['total_nodes'])
                print(f"  Max geometry: {max_result['total_nodes']:,} nodes in {max_result['time_seconds']:.2f}s")

            elif name == 'wave_propagation':
                r = results[0]
                print(f"  Wave propagation: {r['throughput_nodes_per_sec']:,.0f} nodes/sec ({r['throughput_Mnodes_per_sec']:.3f} Mnodes/sec)")

        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Multi-Strip Flux Tube Benchmark Suite')
    parser.add_argument('--mode', choices=['quick', 'full', 'production'], default='quick',
                        help='Benchmark mode: quick (1-2min), full (10-15min), production (30+ min)')

    args = parser.parse_args()

    benchmark = MultiStripBenchmark()

    if args.mode == 'quick':
        benchmark.run_quick_benchmark()
    elif args.mode == 'full':
        benchmark.run_full_benchmark()
    elif args.mode == 'production':
        benchmark.run_production_benchmark()

    benchmark.print_summary()
    filepath = benchmark.save_results()

    print(f"\nTo view results:")
    print(f"  cat {filepath}")


if __name__ == "__main__":
    main()
