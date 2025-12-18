"""
HHmL Parameter Mapping - Test Concept to Implementation Guide
==============================================================

Maps test concepts and physical phenomena to appropriate HHmL parameters,
topologies, and field dynamics. Use this when creating new test scripts to
ensure proper parameter selection.

Usage:
    from hhml.utils.hhml_parameter_mapping import (
        get_topology_for_concept,
        get_field_dynamics_for_concept,
        get_observables_for_concept,
        get_verification_type_for_concept
    )

    # Map user concept to implementation
    concept = "topological phase transition"
    topology = get_topology_for_concept(concept)
    dynamics = get_field_dynamics_for_concept(concept)
    observables = get_observables_for_concept(concept)
    verification = get_verification_type_for_concept(concept)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ConceptMapping:
    """Mapping from concept to HHmL implementation."""
    concept: str
    topology: str  # 'mobius', 'torus', 'sphere', 'klein_bottle'
    field_dynamics: str  # 'holographic_resonance', 'gft_condensate', 'wave_equation'
    key_rnn_parameters: List[str]  # Which RNN parameters are most important
    observables: List[str]  # What to measure
    verification_type: str  # 'oscillatory', 'spatial', 'energetic', 'auto'
    suggested_phases: List[str]  # Suggested test phases beyond mandatory
    description: str


# Concept mapping database
CONCEPT_MAPPINGS = [
    # =========================================================================
    # Topological Concepts
    # =========================================================================
    ConceptMapping(
        concept="möbius topology",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['winding_density', 'twist_rate', 'cross_coupling'],
        observables=['vortex_density', 'topological_charge', 'winding_number'],
        verification_type='spatial',
        suggested_phases=['Möbius geometry generation', 'Twist parameter sweep', 'Topological invariant computation'],
        description="Basic Möbius strip topology with single twist"
    ),

    ConceptMapping(
        concept="klein bottle",
        topology="klein_bottle",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['winding_density', 'twist_rate', 'cross_coupling'],
        observables=['vortex_density', 'topological_charge', 'non_orientability'],
        verification_type='spatial',
        suggested_phases=['Klein bottle geometry generation', 'Double-twist dynamics', 'Non-orientability measurement'],
        description="Klein bottle (double-twisted Möbius) topology"
    ),

    ConceptMapping(
        concept="topological phase transition",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['twist_rate', 'winding_density', 'vortex_target'],
        observables=['topological_order_parameter', 'vortex_density', 'phase_transition_point'],
        verification_type='spatial',
        suggested_phases=['Phase space sweep', 'Order parameter tracking', 'Critical point detection'],
        description="Transition between topologically distinct phases"
    ),

    ConceptMapping(
        concept="topological charge conservation",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['vortex_seed_strength', 'antivortex_strength', 'annihilation_radius'],
        observables=['total_winding_number', 'charge_flux', 'conservation_violation'],
        verification_type='spatial',
        suggested_phases=['Charge injection', 'Vortex-antivortex dynamics', 'Conservation measurement'],
        description="Test if topological charge is conserved during evolution"
    ),

    # =========================================================================
    # Vortex Dynamics
    # =========================================================================
    ConceptMapping(
        concept="vortex annihilation",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['antivortex_strength', 'annihilation_radius', 'pruning_threshold', 'preserve_ratio'],
        observables=['annihilation_rate', 'vortex_quality', 'residual_density'],
        verification_type='spatial',
        suggested_phases=['Vortex generation', 'Antivortex injection', 'Annihilation dynamics', 'Quality tracking'],
        description="Selective vortex annihilation via antivortex injection"
    ),

    ConceptMapping(
        concept="vortex stability",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['damping', 'nonlinearity', 'vortex_target'],
        observables=['vortex_lifetime', 'stability_index', 'perturbation_response'],
        verification_type='spatial',
        suggested_phases=['Vortex initialization', 'Perturbation injection', 'Stability measurement'],
        description="Test vortex stability under perturbations"
    ),

    # =========================================================================
    # Holographic / AdS-CFT
    # =========================================================================
    ConceptMapping(
        concept="ads/cft correspondence",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['omega', 'spectral_weight', 'num_qec_layers'],
        observables=['bulk_entropy', 'boundary_entanglement', 'rt_formula_check'],
        verification_type='spatial',
        suggested_phases=['Bulk field evolution', 'Boundary encoding', 'Holographic matching'],
        description="Test bulk-boundary correspondence (AdS/CFT)"
    ),

    ConceptMapping(
        concept="holographic encoding",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['num_qec_layers', 'omega', 'spectral_weight'],
        observables=['encoding_efficiency', 'information_density', 'quantum_error_rate'],
        verification_type='spatial',
        suggested_phases=['Boundary data preparation', 'Bulk encoding', 'Decoding verification'],
        description="Test holographic encoding efficiency"
    ),

    # =========================================================================
    # Quantum / Coherence
    # =========================================================================
    ConceptMapping(
        concept="quantum coherence",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['damping', 'num_qec_layers', 'diffusion_dt'],
        observables=['coherence_time', 'decoherence_rate', 'purity'],
        verification_type='spatial',
        suggested_phases=['Coherent state preparation', 'Evolution', 'Coherence measurement'],
        description="Test quantum coherence preservation"
    ),

    ConceptMapping(
        concept="quantum entanglement",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['cross_coupling', 'num_qec_layers', 'spectral_weight'],
        observables=['entanglement_entropy', 'mutual_information', 'concurrence'],
        verification_type='spatial',
        suggested_phases=['Entangled state preparation', 'Subsystem tracing', 'Entanglement measurement'],
        description="Test quantum entanglement in Möbius topology"
    ),

    # =========================================================================
    # Wave / Oscillation
    # =========================================================================
    ConceptMapping(
        concept="gravitational waves",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['omega', 'damping', 'nonlinearity'],
        observables=['strain_amplitude', 'frequency_chirp', 'waveform_match'],
        verification_type='oscillatory',
        suggested_phases=['Wave generation', 'Propagation', 'LIGO comparison'],
        description="Test gravitational wave-like oscillations"
    ),

    ConceptMapping(
        concept="wave propagation",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['omega', 'damping', 'spectral_weight'],
        observables=['wave_speed', 'dispersion_relation', 'attenuation'],
        verification_type='oscillatory',
        suggested_phases=['Wave packet initialization', 'Propagation tracking', 'Dispersion measurement'],
        description="Test wave propagation in Möbius topology"
    ),

    # =========================================================================
    # Dark Matter / Cosmology
    # =========================================================================
    ConceptMapping(
        concept="dark matter",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['pruning_threshold', 'preserve_ratio', 'damping'],
        observables=['dark_fraction', 'clustering', 'rotation_curves'],
        verification_type='spatial',
        suggested_phases=['Multiverse generation', 'Pruning', 'Residue measurement', 'Cosmological validation'],
        description="Dark matter as multiverse pruning residue"
    ),

    ConceptMapping(
        concept="cmb fluctuations",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['amp_variance', 'omega', 'spectral_weight'],
        observables=['power_spectrum', 'acoustic_peaks', 'angular_correlation'],
        verification_type='spatial',
        suggested_phases=['Field fluctuation generation', 'Power spectrum computation', 'CMB comparison'],
        description="Test CMB-like spatial fluctuations"
    ),

    # =========================================================================
    # Particle Physics
    # =========================================================================
    ConceptMapping(
        concept="particle masses",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['vortex_seed_strength', 'nonlinearity', 'damping'],
        observables=['energy_spectrum', 'mass_peaks', 'resonance_widths'],
        verification_type='energetic',
        suggested_phases=['Vortex excitation generation', 'Energy spectrum measurement', 'PDG comparison'],
        description="Test if vortex energies match particle masses"
    ),

    # =========================================================================
    # Numerical / Computational
    # =========================================================================
    ConceptMapping(
        concept="numerical stability",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['diffusion_dt', 'sample_ratio', 'damping'],
        observables=['max_field_value', 'energy_conservation', 'error_growth'],
        verification_type='auto',
        suggested_phases=['Timestep sweep', 'Long-term evolution', 'Error accumulation measurement'],
        description="Test numerical stability of evolution"
    ),

    ConceptMapping(
        concept="hardware scalability",
        topology="mobius",
        field_dynamics="holographic_resonance",
        key_rnn_parameters=['sample_ratio', 'max_neighbors_factor', 'sparse_density'],
        observables=['execution_time', 'memory_usage', 'scaling_exponent'],
        verification_type='auto',
        suggested_phases=['Node count sweep', 'Performance measurement', 'Scaling law fitting'],
        description="Test how simulation scales with hardware"
    ),
]


# Build lookup dictionary
_CONCEPT_LOOKUP = {mapping.concept.lower(): mapping for mapping in CONCEPT_MAPPINGS}


def get_mapping_for_concept(concept: str) -> Optional[ConceptMapping]:
    """
    Get complete mapping for a concept.

    Args:
        concept: Concept name (case-insensitive, partial match supported)

    Returns:
        ConceptMapping object or None if not found

    Example:
        >>> mapping = get_mapping_for_concept("vortex annihilation")
        >>> print(mapping.topology)
        mobius
        >>> print(mapping.key_rnn_parameters)
        ['antivortex_strength', 'annihilation_radius', 'pruning_threshold', 'preserve_ratio']
    """
    concept_lower = concept.lower()

    # Exact match
    if concept_lower in _CONCEPT_LOOKUP:
        return _CONCEPT_LOOKUP[concept_lower]

    # Partial match
    for key, mapping in _CONCEPT_LOOKUP.items():
        if concept_lower in key or key in concept_lower:
            return mapping

    return None


def get_topology_for_concept(concept: str) -> str:
    """Get appropriate topology for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.topology if mapping else "mobius"  # Default


def get_field_dynamics_for_concept(concept: str) -> str:
    """Get appropriate field dynamics for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.field_dynamics if mapping else "holographic_resonance"  # Default


def get_key_parameters_for_concept(concept: str) -> List[str]:
    """Get key RNN parameters for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.key_rnn_parameters if mapping else []


def get_observables_for_concept(concept: str) -> List[str]:
    """Get observables to measure for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.observables if mapping else []


def get_verification_type_for_concept(concept: str) -> str:
    """Get verification type for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.verification_type if mapping else "auto"


def get_suggested_phases_for_concept(concept: str) -> List[str]:
    """Get suggested test phases for a concept."""
    mapping = get_mapping_for_concept(concept)
    return mapping.suggested_phases if mapping else []


def list_all_concepts() -> List[str]:
    """
    List all mapped concepts.

    Returns:
        List of concept names
    """
    return [mapping.concept for mapping in CONCEPT_MAPPINGS]


def print_concept_mapping(concept: str):
    """
    Print detailed mapping for a concept.

    Args:
        concept: Concept name
    """
    mapping = get_mapping_for_concept(concept)

    if mapping is None:
        print(f"No mapping found for concept: {concept}")
        print(f"Available concepts: {list_all_concepts()}")
        return

    print("=" * 80)
    print(f"CONCEPT MAPPING: {mapping.concept.upper()}")
    print("=" * 80)
    print()
    print(f"Description: {mapping.description}")
    print()
    print(f"Topology: {mapping.topology}")
    print(f"Field Dynamics: {mapping.field_dynamics}")
    print(f"Verification Type: {mapping.verification_type}")
    print()
    print(f"Key RNN Parameters:")
    for param in mapping.key_rnn_parameters:
        print(f"  - {param}")
    print()
    print(f"Observables to Measure:")
    for obs in mapping.observables:
        print(f"  - {obs}")
    print()
    print(f"Suggested Test Phases:")
    for i, phase in enumerate(mapping.suggested_phases, 1):
        print(f"  {i}. {phase}")
    print()


def generate_test_template_for_concept(concept: str) -> str:
    """
    Generate test script template for a concept.

    Args:
        concept: Concept name

    Returns:
        Python code template as string
    """
    mapping = get_mapping_for_concept(concept)

    if mapping is None:
        return f"# Error: No mapping found for concept '{concept}'"

    template = f'''#!/usr/bin/env python3
"""
{mapping.concept.title()} Test - Hardware Scalable
{"=" * (len(mapping.concept) + 28)}

{mapping.description}

Phases:
- PHASE 0: Hardware detection and auto-scaling
'''

    for i, phase in enumerate(mapping.suggested_phases, 1):
        template += f"- PHASE {i}: {phase}\n"

    template += f"- PHASE {len(mapping.suggested_phases) + 1}: Emergent verification and whitepaper\n\n"

    template += f'''Key Parameters: {", ".join(mapping.key_rnn_parameters)}
Observables: {", ".join(mapping.observables)}
Verification: {mapping.verification_type}

Target Hardware: Auto-scaled (CPU → H200)

Author: HHmL Project
Date: YYYY-MM-DD
"""

import sys
from pathlib import Path
import argparse
import torch

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hhml.utils import HardwareConfig, EmergentVerifier, EmergentWhitepaperGenerator
from hhml.utils.rnn_parameter_mapping import get_parameter_info

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='{mapping.concept.title()} Test')

    # Hardware auto-scaling (MANDATORY)
    parser.add_argument('--auto-scale', action='store_true',
                       help='Auto-scale parameters based on detected hardware')
    parser.add_argument('--scale-mode', type=str, default='benchmark',
                       choices=['benchmark', 'training', 'production'],
                       help='Scaling mode')

    # Test-specific parameters
    # TODO: Add test-specific arguments here

    return parser.parse_args()

def main():
    """Run {mapping.concept} test."""
    args = parse_args()

    # PHASE 0: Hardware Detection and Auto-Scaling
    print("=" * 80)
    print("PHASE 0: HARDWARE DETECTION AND AUTO-SCALING")
    print("=" * 80)

    hw_config = HardwareConfig()
    hw_config.print_info()

    if args.auto_scale:
        optimal_params = hw_config.get_optimal_params(mode=args.scale_mode)
        # Override args with optimal values
        # TODO: Map optimal_params to your test parameters

    # TODO: Implement test phases
    # PHASE 1: {mapping.suggested_phases[0] if mapping.suggested_phases else 'Setup'}
    # PHASE 2: {mapping.suggested_phases[1] if len(mapping.suggested_phases) > 1 else 'Execution'}
    # ...

    # PHASE N+1: Emergent Verification
    print("=" * 80)
    print("PHASE N+1: EMERGENT VERIFICATION")
    print("=" * 80)

    # TODO: Extract final_field from test
    # final_field = ...

    # if test_score >= 0.5:
    #     verifier = EmergentVerifier(data_dir="data")
    #     verification_results = verifier.verify_phenomenon(
    #         field_tensor=final_field,
    #         phenomenon_type='{mapping.verification_type}',
    #         save_results=True
    #     )
    #
    #     generator = EmergentWhitepaperGenerator()
    #     whitepaper = generator.generate(
    #         phenomenon_name="{mapping.concept.title()}",
    #         discovery_data={{...}},
    #         verification_results=verification_results
    #     )

    return 0

if __name__ == '__main__':
    sys.exit(main())
'''

    return template


if __name__ == "__main__":
    # Print all available concepts
    print("=" * 80)
    print("AVAILABLE CONCEPT MAPPINGS")
    print("=" * 80)
    print()

    for concept in list_all_concepts():
        print(f"- {concept}")

    print()
    print("=" * 80)
    print("EXAMPLE: Vortex Annihilation Mapping")
    print("=" * 80)
    print()

    print_concept_mapping("vortex annihilation")
