"""
RNN Parameter Mapping - Programmatic Definition
=================================================

Defines the complete mapping of 23 RNN-controlled parameters for HHmL.
This module provides the canonical source for parameter definitions that
should be used by all test scripts and training workflows.

Based on: docs/guides/RNN_PARAMETER_MAPPING.md

Usage:
    from hhml.utils.rnn_parameter_mapping import RNN_PARAMETERS, get_parameter_info

    # Get all parameter names
    param_names = [p['name'] for p in RNN_PARAMETERS]

    # Get parameter info
    info = get_parameter_info('kappa')
    print(f"{info['name']}: {info['purpose']}")
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ParameterDef:
    """Definition of a single RNN parameter."""
    name: str
    index: int
    range_min: float
    range_max: float
    purpose: str
    category: str


# Complete RNN parameter definitions (23 parameters)
RNN_PARAMETERS = [
    # =========================================================================
    # Category 1: Geometry (4 parameters)
    # =========================================================================
    ParameterDef(
        name='kappa',
        index=0,
        range_min=1.0,
        range_max=2.0,
        purpose='Tokamak elongation (D-shape vertical stretch)',
        category='geometry'
    ),
    ParameterDef(
        name='delta',
        index=1,
        range_min=0.0,
        range_max=0.5,
        purpose='Tokamak triangularity (D-shape tip sharpness)',
        category='geometry'
    ),
    ParameterDef(
        name='vortex_target',
        index=2,
        range_min=0.5,
        range_max=0.8,
        purpose='Target vortex density (spectral reset goal)',
        category='geometry'
    ),
    ParameterDef(
        name='num_qec_layers',
        index=3,
        range_min=1.0,
        range_max=10.0,
        purpose='Quantum error correction depth',
        category='geometry'
    ),

    # =========================================================================
    # Category 2: Physics (4 parameters)
    # =========================================================================
    ParameterDef(
        name='damping',
        index=4,
        range_min=0.01,
        range_max=0.2,
        purpose='Wave energy dissipation rate',
        category='physics'
    ),
    ParameterDef(
        name='nonlinearity',
        index=5,
        range_min=-2.0,
        range_max=2.0,
        purpose='Self-interaction strength (can enhance or suppress)',
        category='physics'
    ),
    ParameterDef(
        name='amp_variance',
        index=6,
        range_min=0.1,
        range_max=3.0,
        purpose='Amplitude spread control (diversity vs uniformity)',
        category='physics'
    ),
    ParameterDef(
        name='vortex_seed_strength',
        index=7,
        range_min=0.0,
        range_max=1.0,
        purpose='Probability of injecting vortex cores',
        category='physics'
    ),

    # =========================================================================
    # Category 3: Spectral (3 parameters)
    # =========================================================================
    ParameterDef(
        name='omega',
        index=8,
        range_min=0.1,
        range_max=1.0,
        purpose='Helical phase frequency (spectral weighting)',
        category='spectral'
    ),
    ParameterDef(
        name='diffusion_dt',
        index=9,
        range_min=0.01,
        range_max=0.5,
        purpose='Laplacian diffusion timestep',
        category='spectral'
    ),
    ParameterDef(
        name='reset_strength',
        index=10,
        range_min=0.0,
        range_max=1.0,
        purpose='Spectral vortex reset blend factor',
        category='spectral'
    ),

    # =========================================================================
    # Category 4: Sampling (3 parameters)
    # =========================================================================
    ParameterDef(
        name='sample_ratio',
        index=11,
        range_min=0.01,
        range_max=0.5,
        purpose='Fraction of nodes to update per cycle',
        category='sampling'
    ),
    ParameterDef(
        name='max_neighbors_factor',
        index=12,
        range_min=0.1,
        range_max=2.0,
        purpose='Multiplier for sparse graph connectivity',
        category='sampling'
    ),
    ParameterDef(
        name='sparsity_threshold',
        index=13,
        range_min=0.1,
        range_max=0.5,
        purpose='Field magnitude cutoff for vortex detection',
        category='sampling'
    ),

    # =========================================================================
    # Category 5: Mode Selection (2 parameters)
    # =========================================================================
    ParameterDef(
        name='sparse_density',
        index=14,
        range_min=0.0,
        range_max=1.0,
        purpose='Graph density (0=dense, 1=sparse)',
        category='mode_selection'
    ),
    ParameterDef(
        name='spectral_weight',
        index=15,
        range_min=0.0,
        range_max=1.0,
        purpose='Propagation method (0=spatial, 1=spectral)',
        category='mode_selection'
    ),

    # =========================================================================
    # Category 6: Geometry Extended (3 parameters)
    # =========================================================================
    ParameterDef(
        name='winding_density',
        index=16,
        range_min=0.5,
        range_max=2.5,
        purpose='Möbius winding frequency',
        category='geometry_extended'
    ),
    ParameterDef(
        name='twist_rate',
        index=17,
        range_min=0.5,
        range_max=2.0,
        purpose='Topological twist rate',
        category='geometry_extended'
    ),
    ParameterDef(
        name='cross_coupling',
        index=18,
        range_min=0.0,
        range_max=1.0,
        purpose='Inter-strip coupling strength',
        category='geometry_extended'
    ),

    # =========================================================================
    # Category 7: Vortex Annihilation (4 parameters) - NEW
    # =========================================================================
    ParameterDef(
        name='antivortex_strength',
        index=19,
        range_min=0.0,
        range_max=1.0,
        purpose='Strength of phase-inverted antivortex injection',
        category='vortex_annihilation'
    ),
    ParameterDef(
        name='annihilation_radius',
        index=20,
        range_min=0.1,
        range_max=1.0,
        purpose='Spatial extent of annihilation zone',
        category='vortex_annihilation'
    ),
    ParameterDef(
        name='pruning_threshold',
        index=21,
        range_min=0.0,
        range_max=1.0,
        purpose='Quality score below which vortices are targeted',
        category='vortex_annihilation'
    ),
    ParameterDef(
        name='preserve_ratio',
        index=22,
        range_min=0.3,
        range_max=0.9,
        purpose='Minimum vortex density to maintain (safety limit)',
        category='vortex_annihilation'
    ),
]


# Build lookup dictionaries for fast access
_PARAM_BY_NAME = {p.name: p for p in RNN_PARAMETERS}
_PARAM_BY_INDEX = {p.index: p for p in RNN_PARAMETERS}
_PARAMS_BY_CATEGORY = {}
for param in RNN_PARAMETERS:
    if param.category not in _PARAMS_BY_CATEGORY:
        _PARAMS_BY_CATEGORY[param.category] = []
    _PARAMS_BY_CATEGORY[param.category].append(param)


def get_parameter_info(name_or_index) -> Optional[ParameterDef]:
    """
    Get parameter information by name or index.

    Args:
        name_or_index: Parameter name (str) or index (int)

    Returns:
        ParameterDef object or None if not found

    Example:
        >>> info = get_parameter_info('kappa')
        >>> print(f"{info.name}: {info.purpose}")
        kappa: Tokamak elongation (D-shape vertical stretch)

        >>> info = get_parameter_info(0)
        >>> print(info.name)
        kappa
    """
    if isinstance(name_or_index, str):
        return _PARAM_BY_NAME.get(name_or_index)
    elif isinstance(name_or_index, int):
        return _PARAM_BY_INDEX.get(name_or_index)
    else:
        return None


def get_parameters_by_category(category: str) -> List[ParameterDef]:
    """
    Get all parameters in a category.

    Args:
        category: Category name (e.g., 'geometry', 'physics', 'vortex_annihilation')

    Returns:
        List of ParameterDef objects in that category

    Example:
        >>> physics_params = get_parameters_by_category('physics')
        >>> print([p.name for p in physics_params])
        ['damping', 'nonlinearity', 'amp_variance', 'vortex_seed_strength']
    """
    return _PARAMS_BY_CATEGORY.get(category, [])


def get_all_parameter_names() -> List[str]:
    """
    Get list of all parameter names in index order.

    Returns:
        List of parameter names

    Example:
        >>> names = get_all_parameter_names()
        >>> print(len(names))
        23
    """
    return [p.name for p in RNN_PARAMETERS]


def get_parameter_count() -> int:
    """
    Get total number of RNN parameters.

    Returns:
        Number of parameters (should be 23)
    """
    return len(RNN_PARAMETERS)


def get_parameter_ranges() -> Dict[str, Tuple[float, float]]:
    """
    Get parameter ranges as dictionary.

    Returns:
        Dictionary mapping parameter names to (min, max) tuples

    Example:
        >>> ranges = get_parameter_ranges()
        >>> print(ranges['kappa'])
        (1.0, 2.0)
    """
    return {p.name: (p.range_min, p.range_max) for p in RNN_PARAMETERS}


def create_parameter_dict_from_tensor(param_tensor) -> Dict[str, float]:
    """
    Convert RNN output tensor to named parameter dictionary.

    Args:
        param_tensor: Tensor of shape (23,) with RNN outputs

    Returns:
        Dictionary mapping parameter names to values

    Example:
        >>> import torch
        >>> rnn_output = torch.randn(23)
        >>> params = create_parameter_dict_from_tensor(rnn_output)
        >>> print(params['kappa'])
        1.234
    """
    if len(param_tensor) != 23:
        raise ValueError(f"Expected 23 parameters, got {len(param_tensor)}")

    return {
        param.name: param_tensor[param.index].item()
        for param in RNN_PARAMETERS
    }


def scale_parameters(raw_params: Dict[str, float]) -> Dict[str, float]:
    """
    Scale raw RNN outputs to parameter ranges.

    Assumes raw_params are in [0, 1] range (e.g., from sigmoid/tanh activation).

    Args:
        raw_params: Dictionary of raw parameter values in [0, 1]

    Returns:
        Dictionary of scaled parameter values in proper ranges

    Example:
        >>> raw = {'kappa': 0.5, 'delta': 0.3}
        >>> scaled = scale_parameters(raw)
        >>> print(scaled['kappa'])  # Should be in [1.0, 2.0]
        1.5
    """
    scaled = {}
    for name, raw_value in raw_params.items():
        param = get_parameter_info(name)
        if param is None:
            raise ValueError(f"Unknown parameter: {name}")

        # Linear scaling: raw [0,1] -> [range_min, range_max]
        scaled[name] = param.range_min + raw_value * (param.range_max - param.range_min)

    return scaled


def get_category_names() -> List[str]:
    """
    Get list of all parameter categories.

    Returns:
        List of category names

    Example:
        >>> categories = get_category_names()
        >>> print(categories)
        ['geometry', 'physics', 'spectral', 'sampling', 'mode_selection',
         'geometry_extended', 'vortex_annihilation']
    """
    return list(_PARAMS_BY_CATEGORY.keys())


def print_parameter_summary():
    """
    Print a formatted summary of all parameters.

    Useful for debugging and documentation.
    """
    print("=" * 80)
    print("RNN PARAMETER MAPPING SUMMARY")
    print("=" * 80)
    print(f"Total Parameters: {get_parameter_count()}")
    print()

    for category in get_category_names():
        params = get_parameters_by_category(category)
        print(f"Category: {category.upper()} ({len(params)} parameters)")
        print("-" * 80)

        for param in params:
            print(f"  [{param.index:2d}] {param.name:25s} [{param.range_min:6.2f}, {param.range_max:6.2f}]")
            print(f"       {param.purpose}")
        print()


if __name__ == "__main__":
    # Test the module
    print_parameter_summary()

    # Example usage
    print("=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    print()

    # Get parameter info
    kappa = get_parameter_info('kappa')
    print(f"Parameter: {kappa.name}")
    print(f"Index: {kappa.index}")
    print(f"Range: [{kappa.range_min}, {kappa.range_max}]")
    print(f"Purpose: {kappa.purpose}")
    print()

    # Get parameters by category
    physics_params = get_parameters_by_category('physics')
    print(f"Physics parameters: {[p.name for p in physics_params]}")
    print()

    # Get all parameter names
    all_names = get_all_parameter_names()
    print(f"All parameter names ({len(all_names)}): {all_names[:5]}...")
