"""
HHmL Simulation Mapper

Maps environment configurations to actual HHmL simulation objects.
Handles translation between generic parameters and HHmL-specific implementations.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

from .environment_manager import Environment
from .hardware_config import detect_device


class SimulationMapper:
    """
    Maps environment configurations to HHmL simulation objects.

    Handles the translation layer between generic simulation parameters
    and HHmL-specific topology, field dynamics, and RNN control.
    """

    def __init__(self, environment: Environment):
        """
        Initialize mapper with environment.

        Args:
            environment: Environment configuration to map
        """
        self.env = environment
        self.device = self._resolve_device()

        # Set random seed for reproducibility
        if self.env.reproducibility.random_seed is not None:
            self._set_random_seed(self.env.reproducibility.random_seed)

    def _resolve_device(self) -> str:
        """
        Resolve device from environment configuration.

        Returns:
            Device string (e.g., 'cuda', 'cpu')
        """
        if self.env.hardware.device == 'auto':
            return detect_device()
        return self.env.hardware.device

    def _set_random_seed(self, seed: int):
        """Set all random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.env.reproducibility.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def create_topology(self):
        """
        Create topology object based on environment configuration.

        Returns:
            Topology object (MobiusStrip, Torus, etc.)

        Raises:
            ValueError: If topology type not supported
        """
        topo_type = self.env.topology.type

        if topo_type == 'mobius':
            return self._create_mobius_topology()
        elif topo_type == 'torus':
            return self._create_torus_topology()
        elif topo_type == 'sphere':
            return self._create_sphere_topology()
        else:
            raise ValueError(f"Unsupported topology type: {topo_type}")

    def _create_mobius_topology(self):
        """Create Möbius strip topology."""
        from hhml.core.mobius.optimized_sphere import OptimizedMobiusSphere

        params = self.env.topology.mobius

        # Map environment parameters to Möbius constructor
        mobius = OptimizedMobiusSphere(
            num_sites=self.env.simulation.nodes,
            radius=params.get('radius', 1.0),
            width=params.get('width', 0.5),
            twist=params.get('twist', np.pi),
            windings=params.get('windings', 109),
            device=self.device
        )

        return mobius

    def _create_torus_topology(self):
        """Create torus topology."""
        # TODO: Implement torus topology when available
        raise NotImplementedError("Torus topology not yet implemented")

    def _create_sphere_topology(self):
        """Create sphere topology."""
        # TODO: Implement sphere topology when available
        raise NotImplementedError("Sphere topology not yet implemented")

    def create_field_dynamics(self, topology):
        """
        Create field dynamics object.

        Args:
            topology: Topology object to attach dynamics to

        Returns:
            Configured topology with field dynamics
        """
        equation_type = self.env.field_dynamics.equation

        if equation_type == 'holographic_resonance':
            return self._configure_holographic_resonance(topology)
        elif equation_type == 'gft_condensate':
            return self._configure_gft_condensate(topology)
        else:
            raise ValueError(f"Unsupported field dynamics: {equation_type}")

    def _configure_holographic_resonance(self, topology):
        """Configure holographic resonance on topology."""
        params = self.env.field_dynamics.holographic

        # Set field parameters on topology
        if hasattr(topology, 'set_field_params'):
            topology.set_field_params(
                damping=params.get('damping', 0.1),
                nonlinearity=params.get('nonlinearity', 0.01),
                wavenumber=params.get('wavenumber', 10.0),
                amplitude_variance=params.get('amplitude_variance', 0.5)
            )

        return topology

    def _configure_gft_condensate(self, topology):
        """Configure GFT condensate dynamics."""
        # TODO: Implement GFT configuration when available
        raise NotImplementedError("GFT condensate not yet implemented")

    def create_rnn_controller(self, topology):
        """
        Create RNN controller for topology.

        Args:
            topology: Topology to control

        Returns:
            RNN controller object
        """
        if not self.env.rnn_control.enabled:
            return None

        # Count total parameters
        param_count = self.env.get_rnn_param_count()

        # Create RNN based on architecture
        arch = self.env.rnn_control.architecture

        if arch == 'lstm':
            return self._create_lstm_controller(topology, param_count)
        elif arch == 'gru':
            return self._create_gru_controller(topology, param_count)
        elif arch == 'transformer':
            return self._create_transformer_controller(topology, param_count)
        else:
            raise ValueError(f"Unsupported RNN architecture: {arch}")

    def _create_lstm_controller(self, topology, param_count: int):
        """Create LSTM-based RNN controller."""
        import torch.nn as nn

        # Create LSTM
        lstm = nn.LSTM(
            input_size=10,  # State features
            hidden_size=self.env.rnn_control.hidden_dim,
            num_layers=self.env.rnn_control.num_layers,
            batch_first=True
        ).to(self.device)

        # Create parameter output head
        control_head = nn.Linear(
            self.env.rnn_control.hidden_dim,
            param_count
        ).to(self.device)

        return {
            'lstm': lstm,
            'control_head': control_head,
            'param_count': param_count
        }

    def _create_gru_controller(self, topology, param_count: int):
        """Create GRU-based RNN controller."""
        # TODO: Implement GRU controller
        raise NotImplementedError("GRU controller not yet implemented")

    def _create_transformer_controller(self, topology, param_count: int):
        """Create Transformer-based controller."""
        # TODO: Implement Transformer controller
        raise NotImplementedError("Transformer controller not yet implemented")

    def create_training_config(self) -> Dict[str, Any]:
        """
        Create training configuration dictionary.

        Returns:
            Dictionary of training parameters
        """
        rl_params = self.env.rnn_control.reinforcement_learning

        return {
            'cycles': self.env.simulation.cycles,
            'timestep': self.env.simulation.timestep,
            'learning_rate': rl_params.get('learning_rate', 0.0001),
            'discount_factor': rl_params.get('discount_factor', 0.99),
            'buffer_size': rl_params.get('buffer_size', 100000),
            'batch_size': rl_params.get('batch_size', 256),
            'device': self.device,
        }

    def create_monitoring_config(self) -> Dict[str, Any]:
        """
        Create monitoring configuration.

        Returns:
            Dictionary of monitoring parameters
        """
        return {
            'dashboard_enabled': self.env.monitoring.dashboard.get('enabled', True),
            'dashboard_port': self.env.monitoring.dashboard.get('port', 8000),
            'update_frequency': self.env.monitoring.dashboard.get('update_frequency', 1),
            'save_frequency': self.env.monitoring.metrics.get('save_frequency', 10),
            'checkpoint_frequency': self.env.monitoring.metrics.get('checkpoint_frequency', 100),
            'logging_level': self.env.monitoring.logging.get('level', 'INFO'),
        }

    def create_output_config(self) -> Dict[str, Any]:
        """
        Create output configuration.

        Returns:
            Dictionary of output parameters
        """
        # Replace {experiment} placeholder with environment name
        output_dir = self.env.output.directory.replace(
            '{experiment}',
            self.env.name
        )

        return {
            'directory': output_dir,
            'save_final_state': self.env.output.save_final_state,
            'save_trajectories': self.env.output.save_trajectories,
            'generate_whitepaper': self.env.output.generate_whitepaper,
            'artifacts': self.env.output.artifacts,
        }

    def get_validation_targets(self) -> Dict[str, Dict[str, float]]:
        """
        Get validation targets.

        Returns:
            Dictionary of target metrics
        """
        return self.env.validation.targets

    def create_complete_simulation(self) -> Dict[str, Any]:
        """
        Create complete simulation setup.

        Returns:
            Dictionary containing all simulation components:
                - 'topology': Configured topology
                - 'rnn_controller': RNN controller (or None)
                - 'training_config': Training parameters
                - 'monitoring_config': Monitoring parameters
                - 'output_config': Output parameters
                - 'validation_targets': Validation targets
                - 'environment': Original environment object
        """
        # Create topology
        topology = self.create_topology()

        # Configure field dynamics
        topology = self.create_field_dynamics(topology)

        # Create RNN controller
        rnn_controller = self.create_rnn_controller(topology)

        # Create configurations
        training_config = self.create_training_config()
        monitoring_config = self.create_monitoring_config()
        output_config = self.create_output_config()
        validation_targets = self.get_validation_targets()

        return {
            'topology': topology,
            'rnn_controller': rnn_controller,
            'training_config': training_config,
            'monitoring_config': monitoring_config,
            'output_config': output_config,
            'validation_targets': validation_targets,
            'environment': self.env,
            'device': self.device,
        }

    def validate_simulation(self, simulation: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Validate simulation configuration.

        Args:
            simulation: Simulation dictionary from create_complete_simulation()

        Returns:
            (is_valid, errors) tuple
        """
        errors = []

        # Validate hardware
        hw_valid, hw_msg = self.env.validate_hardware()
        if not hw_valid:
            errors.append(f"Hardware validation failed: {hw_msg}")

        # Validate topology
        if simulation['topology'] is None:
            errors.append("Topology creation failed")

        # Validate RNN controller
        if self.env.rnn_control.enabled and simulation['rnn_controller'] is None:
            errors.append("RNN controller required but not created")

        # Validate parameter count
        if simulation['rnn_controller']:
            expected_params = self.env.get_rnn_param_count()
            actual_params = simulation['rnn_controller']['param_count']
            if expected_params != actual_params:
                errors.append(
                    f"Parameter mismatch: expected {expected_params}, got {actual_params}"
                )

        return (len(errors) == 0, errors)


def create_simulation_from_environment(
    env_name: str,
    env_dir: str = "configs/environments"
) -> Dict[str, Any]:
    """
    Convenience function to create simulation from environment name.

    Args:
        env_name: Name of environment to load
        env_dir: Directory containing environment files

    Returns:
        Complete simulation dictionary

    Example:
        >>> sim = create_simulation_from_environment('benchmark_mobius')
        >>> topology = sim['topology']
        >>> rnn = sim['rnn_controller']
        >>> # Run training...
    """
    from .environment_manager import EnvironmentManager

    # Load environment
    manager = EnvironmentManager(env_dir)
    env = manager.get(env_name)

    if env is None:
        available = manager.list()
        raise ValueError(
            f"Environment '{env_name}' not found. "
            f"Available: {available}"
        )

    # Create mapper and simulation
    mapper = SimulationMapper(env)
    simulation = mapper.create_complete_simulation()

    # Validate
    valid, errors = mapper.validate_simulation(simulation)
    if not valid:
        raise RuntimeError(f"Simulation validation failed: {errors}")

    return simulation
