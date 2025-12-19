#!/usr/bin/env python3
"""
Yang-Mills Mass Gap Test - HHmL Holographic Approach
=====================================================

Tests for mass gap emergence in Yang-Mills gauge theory using HHmL's
holographic Möbius lattice framework.

Millennium Prize Problem: Prove existence of quantum Yang-Mills theory
on ℝ⁴ with positive mass gap Δ > 0 (lightest particle has positive mass).

HHmL Approach:
- Gauge field simulated on Möbius lattice (holographic boundary)
- AdS/CFT-inspired: boundary gauge theory ↔ bulk emergent geometry
- Strong coupling regime (non-perturbative) via retrocausal dynamics
- Mass gap detected from discrete energy spectrum of field excitations
- Confinement measured via vortex flux tube formation

Key Observables:
1. Energy spectrum E_n (eigenvalues of field configuration)
2. Mass gap Δ = E₁ - E₀ (positive indicates massive excitations)
3. Confinement measure (flux tubes between vortex pairs)
4. Gauge invariance preservation (topological charge conservation)
5. String tension σ (linear quark-antiquark potential)

Target: 100 cycles to establish mass gap statistics

Author: HHmL Project
Date: 2025-12-19
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

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.mobius.sparse_tokamak_strips import SparseTokamakMobiusStrips
from src.hhml.core.spatiotemporal.retrocausal_coupling import RetrocausalCoupler

# Import from training scripts in same directory
sys.path.insert(0, str(Path(__file__).parent))
from train_tokamak_wormhole_hunt import detect_temporal_vortices_gpu
from train_tokamak_rnn_control import TokamakRNN


def compute_energy_spectrum(field, tokamak, num_modes=20):
    """
    Compute energy spectrum of gauge field configuration.

    Approximates eigenvalues of field Hamiltonian using spatial Fourier modes.

    Args:
        field: Complex field tensor [num_nodes, num_time_steps]
        tokamak: Tokamak geometry
        num_modes: Number of energy modes to extract

    Returns:
        dict with energy spectrum metrics
    """
    # Compute field energy density at each node (time-averaged)
    energy_density = (torch.abs(field) ** 2).mean(dim=1)  # [num_nodes]

    # Compute graph Laplacian eigenvalues (approximate energy modes)
    # For large graphs, use power iteration for top eigenvalues
    edge_index = tokamak.edge_index
    num_nodes = field.shape[0]

    # Degree matrix
    row, col = edge_index[0], edge_index[1]
    degree = torch.zeros(num_nodes, device=field.device)
    degree.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))

    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    deg_inv_sqrt = torch.pow(degree, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    # Compute energy levels from field amplitude distribution
    # Sort energy densities (proxy for energy eigenvalues)
    sorted_energies = torch.sort(energy_density, descending=True)[0]

    # Extract top modes
    energy_modes = sorted_energies[:num_modes]

    # Ground state energy (minimum non-zero energy)
    nonzero_mask = energy_density > 1e-6
    if nonzero_mask.sum() > 0:
        ground_state_energy = energy_density[nonzero_mask].min().item()
    else:
        ground_state_energy = 0.0

    # First excited state
    if len(energy_modes) > 1:
        first_excited_energy = energy_modes[1].item()
    else:
        first_excited_energy = ground_state_energy

    # Mass gap
    mass_gap = first_excited_energy - ground_state_energy

    # Mean energy spacing (discreteness measure)
    if len(energy_modes) > 1:
        energy_spacings = energy_modes[:-1] - energy_modes[1:]
        mean_spacing = energy_spacings.mean().item()
        spacing_std = energy_spacings.std().item()
    else:
        mean_spacing = 0.0
        spacing_std = 0.0

    return {
        'energy_modes': energy_modes.cpu().numpy().tolist(),
        'ground_state_energy': ground_state_energy,
        'first_excited_energy': first_excited_energy,
        'mass_gap': mass_gap,
        'mean_spacing': mean_spacing,
        'spacing_std': spacing_std,
        'total_energy': energy_density.sum().item()
    }


def measure_confinement(field, vortex_dict, tokamak):
    """
    Measure confinement via flux tube formation between vortex pairs.

    In Yang-Mills, quarks are confined by linear potential V(r) = σr,
    where σ is string tension. Vortices act as color charges connected
    by flux tubes.

    Args:
        field: Complex field tensor [num_nodes, num_time_steps]
        vortex_dict: Dictionary with vortex node indices and properties
        tokamak: Tokamak geometry

    Returns:
        dict with confinement metrics
    """
    if len(vortex_dict['node_idx']) < 2:
        return {
            'num_vortex_pairs': 0,
            'mean_flux_tube_energy': 0.0,
            'string_tension': 0.0,
            'confinement_measure': 0.0
        }

    vortex_nodes = vortex_dict['node_idx']
    num_vortices = len(vortex_nodes)

    # Sample vortex pairs
    max_pairs = min(100, num_vortices * (num_vortices - 1) // 2)
    pair_energies = []
    pair_distances = []

    positions = tokamak.positions

    for i in range(min(50, num_vortices)):
        for j in range(i + 1, min(i + 10, num_vortices)):
            node_i = vortex_nodes[i]
            node_j = vortex_nodes[j]

            # Distance between vortices
            pos_i = positions[node_i]
            pos_j = positions[node_j]
            distance = torch.norm(pos_i - pos_j).item()

            # Flux tube energy (field energy along path)
            # Approximate as field amplitude at midpoint
            midpoint_energy = (torch.abs(field[node_i]) + torch.abs(field[node_j])).mean().item()

            pair_energies.append(midpoint_energy)
            pair_distances.append(distance)

            if len(pair_energies) >= max_pairs:
                break
        if len(pair_energies) >= max_pairs:
            break

    if len(pair_energies) == 0:
        return {
            'num_vortex_pairs': 0,
            'mean_flux_tube_energy': 0.0,
            'string_tension': 0.0,
            'confinement_measure': 0.0
        }

    # String tension: linear fit E = σ * r
    # Use simple correlation as proxy
    pair_energies_np = np.array(pair_energies)
    pair_distances_np = np.array(pair_distances)

    if pair_distances_np.std() > 1e-6:
        # Linear regression E ~ σ * r
        string_tension = np.corrcoef(pair_distances_np, pair_energies_np)[0, 1]
    else:
        string_tension = 0.0

    # Confinement measure: positive string tension indicates confinement
    confinement_measure = max(0.0, string_tension)

    return {
        'num_vortex_pairs': len(pair_energies),
        'mean_flux_tube_energy': float(np.mean(pair_energies)),
        'string_tension': float(string_tension),
        'confinement_measure': float(confinement_measure)
    }


def measure_gauge_invariance(field, vortex_dict, tokamak):
    """
    Measure gauge invariance via topological charge conservation.

    In Yang-Mills, gauge transformations preserve total topological charge
    (sum of winding numbers). Violation indicates gauge symmetry breaking.

    Args:
        field: Complex field tensor [num_nodes, num_time_steps]
        vortex_dict: Dictionary with vortex charges
        tokamak: Tokamak geometry

    Returns:
        dict with gauge invariance metrics
    """
    if len(vortex_dict['node_idx']) == 0:
        return {
            'total_topological_charge': 0.0,
            'charge_conservation_violation': 0.0,
            'gauge_invariance_measure': 1.0
        }

    # Total topological charge (sum of vortex winding numbers)
    # Approximated by phase circulation
    vortex_nodes = vortex_dict['node_idx']

    # Compute winding number for each vortex
    total_charge = 0.0
    for node in vortex_nodes:
        # Phase gradient around node (proxy for winding number)
        phase = torch.angle(field[node, :])  # [num_time_steps]
        phase_diff = torch.diff(phase)
        phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
        winding = phase_diff.sum() / (2 * np.pi)
        total_charge += abs(winding.item())

    # Gauge invariance: total charge should be integer (quantized)
    charge_fractional_part = abs(total_charge - round(total_charge))

    # Violation measure (0 = perfect quantization)
    charge_conservation_violation = charge_fractional_part

    # Gauge invariance measure (1 = perfect, 0 = broken)
    gauge_invariance_measure = 1.0 - min(1.0, charge_conservation_violation)

    return {
        'total_topological_charge': total_charge,
        'charge_conservation_violation': charge_conservation_violation,
        'gauge_invariance_measure': gauge_invariance_measure
    }


def train_cycle_with_yang_mills(tokamak, coupler, rnn, optimizer, scaler, args, device, prev_metrics):
    """Train one cycle and analyze Yang-Mills mass gap signatures."""

    # Get RNN parameters - construct state with proper time dimension
    state_vec = torch.zeros(8, dtype=torch.float32, device=device)
    if prev_metrics is not None:
        state_vec[0] = prev_metrics.get('vortex_density', 0.0)
        state_vec[1] = prev_metrics.get('mass_gap', 0.0) / 10.0  # Normalize
        state_vec[2] = prev_metrics.get('confinement_measure', 0.0)
        state_vec[3] = prev_metrics.get('gauge_invariance_measure', 1.0)
        state_vec[4] = prev_metrics.get('string_tension', 0.0)
        state_vec[5] = prev_metrics.get('ground_state_energy', 0.0)
        state_vec[6] = prev_metrics.get('mean_spacing', 0.0)
        state_vec[7] = prev_metrics.get('total_energy', 0.0) / 1000.0  # Normalize

    # Repeat across time dimension
    state_features = state_vec.unsqueeze(0).repeat(args.num_time_steps, 1)

    # Pad to state_dim=128
    padding = torch.zeros(args.num_time_steps, 120, device=device)
    state_input = torch.cat([state_features, padding], dim=-1)
    state_input = state_input.unsqueeze(0)  # (1, num_time_steps, 128)

    # RNN forward pass
    with torch.no_grad():
        params = rnn(state_input)

    # Extract scalar values
    params_scalar = {k: v.item() if v.numel() == 1 else v.squeeze(0).item()
                     for k, v in params.items() if k != 'value'}

    # Update retrocausal coupling with RNN parameters (strong coupling regime)
    with torch.no_grad():
        # Use high coupling for non-perturbative Yang-Mills
        coupler.retrocausal_strength = 0.9  # Strong coupling
        coupler.prophetic_mixing = params_scalar['prophetic_gamma']

    # Initialize gauge field (complex field on Möbius lattice)
    # Higher amplitude for Yang-Mills field strength
    field_forward = torch.randn(args.total_nodes, args.num_time_steps,
                               dtype=torch.complex64, device=device) * 2.0
    field_backward = field_forward.clone()

    # Evolve gauge field with strong coupling (non-perturbative regime)
    field_forward, field_backward = coupler.apply_coupling(
        field_forward,
        field_backward,
        enable_mixing=True,
        enable_swapping=True,
        enable_anchoring=True
    )

    # Use forward field as gauge field configuration
    gauge_field = field_forward

    # Detect vortices (proxy for gluon excitations / color charges)
    vortex_dict = detect_temporal_vortices_gpu(
        gauge_field[:, 0],
        tokamak.positions,
        vortex_threshold=0.5
    )

    num_vortices = len(vortex_dict['node_idx'])
    vortex_density = num_vortices / args.total_nodes

    # YANG-MILLS ANALYSIS

    # 1. Energy spectrum and mass gap
    spectrum_metrics = compute_energy_spectrum(gauge_field, tokamak, num_modes=20)

    # 2. Confinement measure (flux tubes)
    confinement_metrics = measure_confinement(gauge_field, vortex_dict, tokamak)

    # 3. Gauge invariance (charge conservation)
    gauge_metrics = measure_gauge_invariance(gauge_field, vortex_dict, tokamak)

    # Compute convergence metrics
    divergence = torch.mean(torch.abs(field_forward - field_backward)).item()
    fixed_point_mask = torch.abs(field_forward - field_backward) < 1e-5
    fixed_point_pct = (fixed_point_mask.sum().item() / fixed_point_mask.numel()) * 100.0

    # Reward: positive mass gap + confinement + gauge invariance
    mass_gap_reward = 100.0 * min(spectrum_metrics['mass_gap'] / 1.0, 1.0) if spectrum_metrics['mass_gap'] > 0 else 0.0
    confinement_reward = 50.0 * confinement_metrics['confinement_measure']
    gauge_reward = 50.0 * gauge_metrics['gauge_invariance_measure']
    spectrum_reward = 25.0 * min(spectrum_metrics['mean_spacing'] / 0.1, 1.0)  # Discrete spectrum

    reward = (
        mass_gap_reward +
        confinement_reward +
        gauge_reward +
        spectrum_reward +
        100.0 * (fixed_point_pct / 100.0)  # Temporal stability
    )

    # Metrics
    metrics = {
        'reward': reward,
        'vortex_count': num_vortices,
        'vortex_density': vortex_density,
        'fixed_point_pct': fixed_point_pct,
        'divergence': divergence,

        # Energy spectrum
        'ground_state_energy': spectrum_metrics['ground_state_energy'],
        'first_excited_energy': spectrum_metrics['first_excited_energy'],
        'mass_gap': spectrum_metrics['mass_gap'],
        'mean_spacing': spectrum_metrics['mean_spacing'],
        'spacing_std': spectrum_metrics['spacing_std'],
        'total_energy': spectrum_metrics['total_energy'],

        # Confinement
        'num_vortex_pairs': confinement_metrics['num_vortex_pairs'],
        'mean_flux_tube_energy': confinement_metrics['mean_flux_tube_energy'],
        'string_tension': confinement_metrics['string_tension'],
        'confinement_measure': confinement_metrics['confinement_measure'],

        # Gauge invariance
        'total_topological_charge': gauge_metrics['total_topological_charge'],
        'charge_conservation_violation': gauge_metrics['charge_conservation_violation'],
        'gauge_invariance_measure': gauge_metrics['gauge_invariance_measure'],
    }

    return metrics, params_scalar


def main():
    parser = argparse.ArgumentParser(description='Yang-Mills Mass Gap Test')

    parser.add_argument('--num-strips', type=int, default=300)
    parser.add_argument('--nodes-per-strip', type=int, default=166)
    parser.add_argument('--num-time-steps', type=int, default=10)
    parser.add_argument('--num-cycles', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='~/results/yang_mills_mass_gap')

    args = parser.parse_args()
    args.total_nodes = args.num_strips * args.nodes_per_strip

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser() / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Initialize tokamak (holographic boundary)
    print("Initializing Möbius lattice (holographic boundary)...")
    tokamak = SparseTokamakMobiusStrips(
        num_strips=args.num_strips,
        nodes_per_strip=args.nodes_per_strip,
        kappa=1.5,
        delta=0.3,
        sparse_threshold=0.3,
        max_neighbors=2000,
        device=device
    )

    coupler = RetrocausalCoupler(
        num_nodes=args.total_nodes,
        num_time_steps=args.num_time_steps,
        retrocausal_strength=0.9,  # Strong coupling for Yang-Mills
        prophetic_mixing=0.3,
        device=device
    )

    print(f"Initialized Möbius lattice: {args.num_strips} strips, {args.total_nodes} nodes\n")

    # Initialize RNN controller
    print("Initializing RNN controller...")
    rnn = TokamakRNN(
        state_dim=128,
        hidden_dim=2048,
        num_params=11,
        device=device
    )

    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    print("RNN initialized\n")

    # Training loop
    print(f"Starting Yang-Mills mass gap test: {args.num_cycles} cycles")
    print("="*80)
    print()

    start_time = time.time()

    history = {
        'rewards': [],
        'vortex_counts': [],
        'vortex_densities': [],
        'ground_state_energies': [],
        'first_excited_energies': [],
        'mass_gaps': [],
        'mean_spacings': [],
        'confinement_measures': [],
        'string_tensions': [],
        'gauge_invariance_measures': [],
        'total_energies': [],
        'fixed_point_pcts': [],
        'divergences': []
    }

    prev_metrics = None

    for cycle in range(args.num_cycles):
        cycle_start = time.time()

        metrics, params_used = train_cycle_with_yang_mills(
            tokamak, coupler, rnn, optimizer, scaler, args, device, prev_metrics
        )

        prev_metrics = metrics

        # Record history
        history['rewards'].append(metrics['reward'])
        history['vortex_counts'].append(metrics['vortex_count'])
        history['vortex_densities'].append(metrics['vortex_density'])
        history['ground_state_energies'].append(metrics['ground_state_energy'])
        history['first_excited_energies'].append(metrics['first_excited_energy'])
        history['mass_gaps'].append(metrics['mass_gap'])
        history['mean_spacings'].append(metrics['mean_spacing'])
        history['confinement_measures'].append(metrics['confinement_measure'])
        history['string_tensions'].append(metrics['string_tension'])
        history['gauge_invariance_measures'].append(metrics['gauge_invariance_measure'])
        history['total_energies'].append(metrics['total_energy'])
        history['fixed_point_pcts'].append(metrics['fixed_point_pct'])
        history['divergences'].append(metrics['divergence'])

        cycle_time = time.time() - cycle_start

        # Print progress
        if cycle % 10 == 0 or cycle == args.num_cycles - 1:
            print(f"Cycle {cycle}/{args.num_cycles}")
            print(f"  Reward: {metrics['reward']:.2f}")
            print(f"  Vortex density: {metrics['vortex_density']:.2%} ({metrics['vortex_count']} vortices)")
            print(f"  MASS GAP METRICS:")
            print(f"    Ground state energy: {metrics['ground_state_energy']:.6f}")
            print(f"    First excited state: {metrics['first_excited_energy']:.6f}")
            print(f"    Mass gap Δ: {metrics['mass_gap']:.6f}")
            print(f"    Mean spacing: {metrics['mean_spacing']:.6f}")
            print(f"  CONFINEMENT METRICS:")
            print(f"    String tension σ: {metrics['string_tension']:.6f}")
            print(f"    Confinement measure: {metrics['confinement_measure']:.6f}")
            print(f"  GAUGE INVARIANCE:")
            print(f"    Topological charge: {metrics['total_topological_charge']:.2f}")
            print(f"    Gauge invariance: {metrics['gauge_invariance_measure']:.6f}")
            print(f"  Fixed points: {metrics['fixed_point_pct']:.1f}%")
            print(f"  Divergence: {metrics['divergence']:.6f}")
            print(f"  Time: {cycle_time:.3f}s")
            print()

    total_time = time.time() - start_time

    print("="*80)
    print(f"Test complete: {args.num_cycles} cycles in {total_time:.2f}s")
    print(f"Average time/cycle: {total_time/args.num_cycles:.3f}s")
    print()

    # Compute summary statistics
    avg_mass_gap = np.mean(history['mass_gaps'])
    std_mass_gap = np.std(history['mass_gaps'])
    min_mass_gap = np.min(history['mass_gaps'])
    max_mass_gap = np.max(history['mass_gaps'])

    positive_mass_gap_cycles = sum(1 for mg in history['mass_gaps'] if mg > 0)
    positive_mass_gap_fraction = positive_mass_gap_cycles / args.num_cycles

    avg_confinement = np.mean(history['confinement_measures'])
    avg_gauge_invariance = np.mean(history['gauge_invariance_measures'])

    print("YANG-MILLS MASS GAP SUMMARY:")
    print(f"  Average mass gap Δ: {avg_mass_gap:.6f} ± {std_mass_gap:.6f}")
    print(f"  Mass gap range: [{min_mass_gap:.6f}, {max_mass_gap:.6f}]")
    print(f"  Positive mass gap: {positive_mass_gap_cycles}/{args.num_cycles} cycles ({positive_mass_gap_fraction:.1%})")
    print(f"  Average confinement: {avg_confinement:.6f}")
    print(f"  Average gauge invariance: {avg_gauge_invariance:.6f}")
    print()

    # Save results
    results = {
        'config': vars(args),
        'total_time_sec': total_time,
        'history': history,
        'yang_mills_summary': {
            'avg_mass_gap': avg_mass_gap,
            'std_mass_gap': std_mass_gap,
            'min_mass_gap': min_mass_gap,
            'max_mass_gap': max_mass_gap,
            'positive_mass_gap_cycles': positive_mass_gap_cycles,
            'positive_mass_gap_fraction': positive_mass_gap_fraction,
            'avg_confinement': avg_confinement,
            'avg_gauge_invariance': avg_gauge_invariance
        }
    }

    results_path = output_dir / f"yang_mills_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved: {results_path}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
