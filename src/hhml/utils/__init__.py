"""HHmL Utilities - Core utility modules."""

from .hardware_config import HardwareConfig
from .emergent_verifier import EmergentVerifier
from .emergent_whitepaper import EmergentWhitepaperGenerator

# RNN parameter mapping
from .rnn_parameter_mapping import (
    RNN_PARAMETERS,
    get_parameter_info,
    get_parameters_by_category,
    get_all_parameter_names,
    get_parameter_count,
    get_parameter_ranges,
    create_parameter_dict_from_tensor,
    scale_parameters,
)

# HHmL concept mapping
from .hhml_parameter_mapping import (
    get_mapping_for_concept,
    get_topology_for_concept,
    get_field_dynamics_for_concept,
    get_key_parameters_for_concept,
    get_observables_for_concept,
    get_verification_type_for_concept,
    get_suggested_phases_for_concept,
    list_all_concepts,
    generate_test_template_for_concept,
)

__all__ = [
    # Hardware
    'HardwareConfig',

    # Emergent detection
    'EmergentVerifier',
    'EmergentWhitepaperGenerator',

    # RNN parameter mapping
    'RNN_PARAMETERS',
    'get_parameter_info',
    'get_parameters_by_category',
    'get_all_parameter_names',
    'get_parameter_count',
    'get_parameter_ranges',
    'create_parameter_dict_from_tensor',
    'scale_parameters',

    # HHmL concept mapping
    'get_mapping_for_concept',
    'get_topology_for_concept',
    'get_field_dynamics_for_concept',
    'get_key_parameters_for_concept',
    'get_observables_for_concept',
    'get_verification_type_for_concept',
    'get_suggested_phases_for_concept',
    'list_all_concepts',
    'generate_test_template_for_concept',
]
