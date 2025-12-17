#!/usr/bin/env python3
"""
Hardware Detection and Auto-Scaling Configuration
==================================================
Automatically detects available hardware (CPU, CUDA, H200) and configures
optimal parameters for multi-scale flux tube simulations.

Usage:
    from hhml.utils.hardware_config import HardwareConfig

    config = HardwareConfig()
    config.print_info()

    # Get optimal parameters for current hardware
    params = config.get_optimal_params()
"""

import torch
import platform
import psutil
import os
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass
class SimulationParams:
    """Optimal parameters for multi-strip flux tube simulation"""
    num_strips: int
    nodes_per_strip: int
    hidden_dim: int
    batch_size: int
    precision: str  # 'float32' or 'float16'
    sparse_threshold: float  # Distance cutoff for sparse interactions
    max_neighbors: int  # Max neighbors per node in sparse graph
    use_compile: bool  # Whether to use torch.compile
    memory_budget_gb: float


class HardwareConfig:
    """Auto-detect hardware and provide optimal simulation parameters"""

    def __init__(self):
        self.device_type = self._detect_device()
        self.device = torch.device(self.device_type)
        self.cuda_available = torch.cuda.is_available()
        self.gpu_name = self._get_gpu_name() if self.cuda_available else None
        self.vram_gb = self._get_vram_gb() if self.cuda_available else 0
        self.ram_gb = self._get_ram_gb()
        self.cpu_cores = os.cpu_count()
        self.is_h200 = self._detect_h200()

    def _detect_device(self) -> str:
        """Detect best available device"""
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'

    def _get_gpu_name(self) -> str:
        """Get GPU name"""
        try:
            return torch.cuda.get_device_name(0)
        except:
            return "Unknown GPU"

    def _get_vram_gb(self) -> float:
        """Get VRAM in GB"""
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / 1e9
        except:
            return 0.0

    def _get_ram_gb(self) -> float:
        """Get system RAM in GB"""
        return psutil.virtual_memory().total / 1e9

    def _detect_h200(self) -> bool:
        """Detect if running on H200 GPU"""
        if not self.cuda_available:
            return False

        # H200 has ~140GB VRAM
        if self.vram_gb > 100:
            return True

        # Check GPU name for H200
        if self.gpu_name and 'H200' in self.gpu_name:
            return True

        return False

    def get_hardware_tier(self) -> str:
        """
        Categorize hardware into tiers for auto-scaling

        Returns:
            'h200': NVIDIA H200 GPU (140GB VRAM)
            'high_gpu': High-end GPU (24-80GB VRAM)
            'mid_gpu': Mid-range GPU (8-24GB VRAM)
            'low_gpu': Low-end GPU (4-8GB VRAM)
            'high_cpu': High-RAM CPU (32GB+)
            'low_cpu': Low-RAM CPU (<32GB)
        """
        if self.is_h200:
            return 'h200'

        if self.cuda_available:
            if self.vram_gb >= 24:
                return 'high_gpu'
            elif self.vram_gb >= 8:
                return 'mid_gpu'
            else:
                return 'low_gpu'
        else:
            if self.ram_gb >= 32:
                return 'high_cpu'
            else:
                return 'low_cpu'

    def get_optimal_params(self, mode: str = 'benchmark') -> SimulationParams:
        """
        Get optimal parameters for current hardware

        Args:
            mode: 'benchmark' (quick test), 'training' (full run), or 'production' (max scale)

        Returns:
            SimulationParams with optimal configuration
        """
        tier = self.get_hardware_tier()

        # Base configurations for each tier and mode
        configs = {
            'h200': {
                'benchmark': SimulationParams(
                    num_strips=4,
                    nodes_per_strip=50000,
                    hidden_dim=2048,
                    batch_size=2000,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=100,
                    use_compile=True,
                    memory_budget_gb=120.0
                ),
                'training': SimulationParams(
                    num_strips=10,
                    nodes_per_strip=100000,
                    hidden_dim=4096,
                    batch_size=5000,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=150,
                    use_compile=True,
                    memory_budget_gb=130.0
                ),
                'production': SimulationParams(
                    num_strips=20,
                    nodes_per_strip=500000,
                    hidden_dim=4096,
                    batch_size=10000,
                    precision='float16',  # Use mixed precision for max scale
                    sparse_threshold=0.25,
                    max_neighbors=200,
                    use_compile=True,
                    memory_budget_gb=135.0
                ),
            },
            'high_gpu': {
                'benchmark': SimulationParams(
                    num_strips=3,
                    nodes_per_strip=20000,
                    hidden_dim=1024,
                    batch_size=1000,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=80,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.8
                ),
                'training': SimulationParams(
                    num_strips=6,
                    nodes_per_strip=50000,
                    hidden_dim=2048,
                    batch_size=2000,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=100,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.85
                ),
                'production': SimulationParams(
                    num_strips=8,
                    nodes_per_strip=100000,
                    hidden_dim=2048,
                    batch_size=3000,
                    precision='float16',
                    sparse_threshold=0.25,
                    max_neighbors=120,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.9
                ),
            },
            'mid_gpu': {
                'benchmark': SimulationParams(
                    num_strips=2,
                    nodes_per_strip=8000,
                    hidden_dim=512,
                    batch_size=500,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=50,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.8
                ),
                'training': SimulationParams(
                    num_strips=4,
                    nodes_per_strip=20000,
                    hidden_dim=1024,
                    batch_size=1000,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=80,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.85
                ),
                'production': SimulationParams(
                    num_strips=6,
                    nodes_per_strip=40000,
                    hidden_dim=1024,
                    batch_size=1500,
                    precision='float16',
                    sparse_threshold=0.25,
                    max_neighbors=100,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.9
                ),
            },
            'low_gpu': {
                'benchmark': SimulationParams(
                    num_strips=2,
                    nodes_per_strip=4000,
                    hidden_dim=256,
                    batch_size=200,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=40,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.8
                ),
                'training': SimulationParams(
                    num_strips=3,
                    nodes_per_strip=8000,
                    hidden_dim=512,
                    batch_size=400,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=50,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.85
                ),
                'production': SimulationParams(
                    num_strips=4,
                    nodes_per_strip=16000,
                    hidden_dim=512,
                    batch_size=600,
                    precision='float16',
                    sparse_threshold=0.25,
                    max_neighbors=60,
                    use_compile=True,
                    memory_budget_gb=self.vram_gb * 0.9
                ),
            },
            'high_cpu': {
                'benchmark': SimulationParams(
                    num_strips=2,
                    nodes_per_strip=2000,
                    hidden_dim=256,
                    batch_size=100,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=30,
                    use_compile=False,  # torch.compile may not work on all CPUs
                    memory_budget_gb=self.ram_gb * 0.5
                ),
                'training': SimulationParams(
                    num_strips=3,
                    nodes_per_strip=4000,
                    hidden_dim=512,
                    batch_size=200,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=40,
                    use_compile=False,
                    memory_budget_gb=self.ram_gb * 0.6
                ),
                'production': SimulationParams(
                    num_strips=4,
                    nodes_per_strip=8000,
                    hidden_dim=512,
                    batch_size=300,
                    precision='float32',
                    sparse_threshold=0.25,
                    max_neighbors=50,
                    use_compile=False,
                    memory_budget_gb=self.ram_gb * 0.7
                ),
            },
            'low_cpu': {
                'benchmark': SimulationParams(
                    num_strips=2,
                    nodes_per_strip=1000,
                    hidden_dim=128,
                    batch_size=50,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=20,
                    use_compile=False,
                    memory_budget_gb=self.ram_gb * 0.5
                ),
                'training': SimulationParams(
                    num_strips=2,
                    nodes_per_strip=2000,
                    hidden_dim=256,
                    batch_size=100,
                    precision='float32',
                    sparse_threshold=0.3,
                    max_neighbors=30,
                    use_compile=False,
                    memory_budget_gb=self.ram_gb * 0.6
                ),
                'production': SimulationParams(
                    num_strips=3,
                    nodes_per_strip=4000,
                    hidden_dim=256,
                    batch_size=150,
                    precision='float32',
                    sparse_threshold=0.25,
                    max_neighbors=40,
                    use_compile=False,
                    memory_budget_gb=self.ram_gb * 0.7
                ),
            },
        }

        return configs[tier][mode]

    def estimate_memory_usage(self, params: SimulationParams) -> Dict[str, float]:
        """
        Estimate memory usage for given parameters

        Returns dict with breakdown in GB
        """
        total_nodes = params.num_strips * params.nodes_per_strip

        # Node positions (x, y, z) - 3 floats per node
        bytes_per_float = 4 if params.precision == 'float32' else 2
        positions_gb = (total_nodes * 3 * bytes_per_float) / 1e9

        # Field values (complex) - 2 floats per node
        field_gb = (total_nodes * 2 * bytes_per_float) / 1e9

        # Wave properties (amplitude, phase, frequency) - 3 floats per node
        properties_gb = (total_nodes * 3 * bytes_per_float) / 1e9

        # Sparse interaction graph (max_neighbors edges per node)
        # Each edge: 2 int indices + 1 float distance
        sparse_graph_gb = (total_nodes * params.max_neighbors * (2*4 + bytes_per_float)) / 1e9

        # RNN hidden states
        rnn_gb = (params.hidden_dim * 4 * bytes_per_float * 4) / 1e9  # 4 layers, 4 = LSTM gates

        # Batch processing buffers
        batch_gb = (params.batch_size * total_nodes * bytes_per_float * 3) / 1e9  # 3 = working memory

        total_gb = positions_gb + field_gb + properties_gb + sparse_graph_gb + rnn_gb + batch_gb

        return {
            'positions': positions_gb,
            'field': field_gb,
            'properties': properties_gb,
            'sparse_graph': sparse_graph_gb,
            'rnn': rnn_gb,
            'batch_buffers': batch_gb,
            'total': total_gb,
            'budget': params.memory_budget_gb,
            'utilization': (total_gb / params.memory_budget_gb) * 100 if params.memory_budget_gb > 0 else 0
        }

    def print_info(self):
        """Print hardware configuration info"""
        print("=" * 80)
        print("HARDWARE CONFIGURATION")
        print("=" * 80)
        print(f"Device: {self.device_type.upper()}")
        print(f"Platform: {platform.system()} {platform.release()}")
        print(f"Python: {platform.python_version()}")
        print(f"PyTorch: {torch.__version__}")
        print()

        if self.cuda_available:
            print(f"GPU: {self.gpu_name}")
            print(f"VRAM: {self.vram_gb:.1f} GB")
            print(f"CUDA Compute Capability: {torch.cuda.get_device_capability(0)}")
            print(f"H200 Detected: {'YES' if self.is_h200 else 'NO'}")
        else:
            print("GPU: Not available")

        print()
        print(f"CPU Cores: {self.cpu_cores}")
        print(f"System RAM: {self.ram_gb:.1f} GB")
        print(f"Hardware Tier: {self.get_hardware_tier().upper()}")
        print("=" * 80)

    def print_optimal_params(self, mode: str = 'benchmark'):
        """Print optimal parameters for given mode"""
        params = self.get_optimal_params(mode)
        memory = self.estimate_memory_usage(params)

        print()
        print("=" * 80)
        print(f"OPTIMAL PARAMETERS ({mode.upper()} MODE)")
        print("=" * 80)
        print(f"Number of Strips: {params.num_strips}")
        print(f"Nodes per Strip: {params.nodes_per_strip:,}")
        print(f"Total Nodes: {params.num_strips * params.nodes_per_strip:,}")
        print(f"Hidden Dim: {params.hidden_dim}")
        print(f"Batch Size: {params.batch_size}")
        print(f"Precision: {params.precision}")
        print(f"Sparse Threshold: {params.sparse_threshold}")
        print(f"Max Neighbors: {params.max_neighbors}")
        print(f"Use torch.compile: {params.use_compile}")
        print()
        print("ESTIMATED MEMORY USAGE:")
        print(f"  Positions:       {memory['positions']:.2f} GB")
        print(f"  Field:           {memory['field']:.2f} GB")
        print(f"  Properties:      {memory['properties']:.2f} GB")
        print(f"  Sparse Graph:    {memory['sparse_graph']:.2f} GB")
        print(f"  RNN:             {memory['rnn']:.2f} GB")
        print(f"  Batch Buffers:   {memory['batch_buffers']:.2f} GB")
        print(f"  -------------------------")
        print(f"  Total:           {memory['total']:.2f} GB")
        print(f"  Budget:          {memory['budget']:.2f} GB")
        print(f"  Utilization:     {memory['utilization']:.1f}%")
        print("=" * 80)


if __name__ == "__main__":
    # Test hardware detection
    config = HardwareConfig()
    config.print_info()

    print()
    config.print_optimal_params('benchmark')

    print()
    config.print_optimal_params('training')

    print()
    config.print_optimal_params('production')
