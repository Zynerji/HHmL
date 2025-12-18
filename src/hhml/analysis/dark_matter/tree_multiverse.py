#!/usr/bin/env python3
"""
Tree-Structured Multiverse Generator
======================================

Generates multiverse branches as a tree where branches split sequentially
from common ancestors, rather than all being independent.

This creates natural clustering and partial coherence, representing a more
realistic quantum decoherence scenario where timelines diverge gradually.

Author: HHmL Project
Date: 2025-12-17
"""

import torch
import numpy as np
import copy
from typing import List
from .multiverse_generator import MultiverseBranch, MultiverseConfig


def generate_tree_multiverse(base_strips_geometry,
                              config: MultiverseConfig,
                              branching_factor: int = 2,
                              device: str = 'cuda') -> List[MultiverseBranch]:
    """
    Generate multiverse as binary tree where branches split sequentially.

    Algorithm:
    1. Start with base branch (root)
    2. For each generation:
       - Each existing branch splits into `branching_factor` children
       - Children inherit parent's field + small perturbation
       - Creates (branching_factor)^depth total branches

    Args:
        base_strips_geometry: Reference Möbius configuration
        config: Multiverse parameters
        branching_factor: How many children per branch (default: 2 = binary tree)
        device: 'cuda' or 'cpu'

    Returns:
        List of MultiverseBranch with tree structure

    Example:
        With branching_factor=2, depth=4:
        Generation 0: 1 branch (root)
        Generation 1: 2 branches
        Generation 2: 4 branches
        Generation 3: 8 branches
        Generation 4: 16 branches
        Total: 31 branches (sum of geometric series)

        But we only keep LEAVES (generation 4): 16 branches
    """
    torch.manual_seed(config.coherence_seed)
    np.random.seed(config.coherence_seed)

    # Compute required depth for target branch count
    depth = int(np.ceil(np.log(config.num_branches) / np.log(branching_factor)))

    print(f"  Tree structure: branching_factor={branching_factor}, depth={depth}")
    print(f"  Target branches: {config.num_branches}, will generate: {branching_factor**depth}")

    # Generation 0: Root branch (base geometry)
    root_geometry = copy.deepcopy(base_strips_geometry)
    root = MultiverseBranch(root_geometry, branch_id=0, divergence_time=0.0, parent_branch=None)

    current_generation = [root]
    branch_id_counter = 1

    # Grow tree for `depth` generations
    for gen in range(1, depth + 1):
        next_generation = []

        for parent in current_generation:
            # Each parent spawns branching_factor children
            for child_idx in range(branching_factor):
                # Clone parent
                child_geometry = copy.deepcopy(parent.geometry)

                # Apply perturbation (diminishing with generation to allow convergence)
                scale = config.perturbation_scale * (0.8 ** gen)  # Decay with generation

                # Perturb from parent
                _apply_branching_perturbation(child_geometry, scale, config.quantum_decoherence)

                # Create child branch
                child = MultiverseBranch(
                    child_geometry,
                    branch_id=branch_id_counter,
                    divergence_time=float(gen),
                    parent_branch=parent
                )

                child.compute_entropy()
                next_generation.append(child)
                branch_id_counter += 1

        current_generation = next_generation

        print(f"  Generation {gen}: {len(current_generation)} branches")

    # Return only leaf nodes (final generation)
    # Trim to exact target count
    leaves = current_generation[:config.num_branches]

    print(f"  Returning {len(leaves)} leaf branches")

    return leaves


def _apply_branching_perturbation(geometry, scale: float, decoherence: float):
    """
    Apply perturbation appropriate for tree branching.

    Smaller than full quantum_noise to maintain parent-child similarity.
    """
    # Moderate phase perturbation
    phase_kicks = torch.randn_like(geometry.phases) * scale * 0.5 * 2 * np.pi
    geometry.phases = (geometry.phases + phase_kicks) % (2 * np.pi)

    # Slight amplitude variation
    amp_noise = torch.randn_like(geometry.amplitudes) * scale * 0.3
    geometry.amplitudes = geometry.amplitudes * (1.0 + amp_noise)
    geometry.amplitudes = torch.clamp(geometry.amplitudes, min=0.1, max=10.0)

    # Field perturbation
    field_noise_real = torch.randn_like(geometry.field.real) * scale * 0.5
    field_noise_imag = torch.randn_like(geometry.field.imag) * scale * 0.5
    field_noise = torch.complex(field_noise_real, field_noise_imag)

    geometry.field = geometry.field + field_noise * torch.abs(geometry.field)
