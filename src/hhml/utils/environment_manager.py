"""
HHmL Environment Manager

Loads and manages simulation environments from YAML configuration files.
Provides mapping from generic simulation parameters to HHmL-specific parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class TopologyConfig:
    """Topology configuration."""
    type: str
    dimensions: int = 11
    mobius: Dict[str, Any] = field(default_factory=dict)
    torus: Dict[str, Any] = field(default_factory=dict)
    sphere: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FieldDynamicsConfig:
    """Field dynamics configuration."""
    equation: str
    holographic: Dict[str, Any] = field(default_factory=dict)
    gft: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RNNControlConfig:
    """RNN control configuration."""
    enabled: bool
    architecture: str
    hidden_dim: int
    num_layers: int
    parameters: Dict[str, List[str]]
    reinforcement_learning: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Simulation parameters."""
    nodes: int
    cycles: int
    timestep: float
    scaling: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HardwareConfig:
    """Hardware requirements."""
    device: str
    min_memory_gb: int
    recommended_memory_gb: int
    precision: str
    multi_gpu: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Validation criteria."""
    targets: Dict[str, Dict[str, float]]
    tests: List[str]


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    dashboard: Dict[str, Any]
    metrics: Dict[str, Any]
    logging: Dict[str, str]


@dataclass
class OutputConfig:
    """Output configuration."""
    directory: str
    save_final_state: bool
    save_trajectories: bool
    generate_whitepaper: bool
    artifacts: List[str]


@dataclass
class ReproducibilityConfig:
    """Reproducibility settings."""
    random_seed: int
    deterministic: bool
    track_provenance: bool


class Environment:
    """
    Represents a complete simulation environment.

    Loads from YAML file and provides structured access to all configuration.
    """

    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize environment from file or dictionary.

        Args:
            config_path: Path to YAML environment file
            config_dict: Dictionary containing environment config

        Raises:
            ValueError: If neither config_path nor config_dict provided
            FileNotFoundError: If config_path doesn't exist
        """
        if config_path is None and config_dict is None:
            raise ValueError("Must provide either config_path or config_dict")

        if config_path:
            self.config_path = Path(config_path)
            if not self.config_path.exists():
                raise FileNotFoundError(f"Environment file not found: {config_path}")

            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config_dict
            self.config_path = None

        # Parse configuration into structured objects
        self._parse_config()

    def _parse_config(self):
        """Parse YAML config into structured dataclasses."""
        # Metadata
        self.metadata = self.config.get('metadata', {})
        self.name = self.metadata.get('name', 'unnamed')
        self.version = self.metadata.get('version', '1.0.0')
        self.description = self.metadata.get('description', '')

        # Topology
        topo_dict = self.config.get('topology', {})
        self.topology = TopologyConfig(
            type=topo_dict.get('type', 'mobius'),
            dimensions=topo_dict.get('dimensions', 11),
            mobius=topo_dict.get('mobius', {}),
            torus=topo_dict.get('torus', {}),
            sphere=topo_dict.get('sphere', {})
        )

        # Field dynamics
        field_dict = self.config.get('field_dynamics', {})
        self.field_dynamics = FieldDynamicsConfig(
            equation=field_dict.get('equation', 'holographic_resonance'),
            holographic=field_dict.get('holographic', {}),
            gft=field_dict.get('gft', {})
        )

        # RNN control
        rnn_dict = self.config.get('rnn_control', {})
        self.rnn_control = RNNControlConfig(
            enabled=rnn_dict.get('enabled', True),
            architecture=rnn_dict.get('architecture', 'lstm'),
            hidden_dim=rnn_dict.get('hidden_dim', 4096),
            num_layers=rnn_dict.get('num_layers', 4),
            parameters=rnn_dict.get('parameters', {}),
            reinforcement_learning=rnn_dict.get('reinforcement_learning', {})
        )

        # Simulation
        sim_dict = self.config.get('simulation', {})
        self.simulation = SimulationConfig(
            nodes=sim_dict.get('nodes', 4000),
            cycles=sim_dict.get('cycles', 1000),
            timestep=sim_dict.get('timestep', 0.01),
            scaling=sim_dict.get('scaling', {})
        )

        # Hardware
        hw_dict = self.config.get('hardware', {})
        self.hardware = HardwareConfig(
            device=hw_dict.get('device', 'auto'),
            min_memory_gb=hw_dict.get('min_memory_gb', 8),
            recommended_memory_gb=hw_dict.get('recommended_memory_gb', 16),
            precision=hw_dict.get('precision', 'float32'),
            multi_gpu=hw_dict.get('multi_gpu', {})
        )

        # Validation
        val_dict = self.config.get('validation', {})
        self.validation = ValidationConfig(
            targets=val_dict.get('targets', {}),
            tests=val_dict.get('tests', [])
        )

        # Monitoring
        mon_dict = self.config.get('monitoring', {})
        self.monitoring = MonitoringConfig(
            dashboard=mon_dict.get('dashboard', {}),
            metrics=mon_dict.get('metrics', {}),
            logging=mon_dict.get('logging', {})
        )

        # Output
        out_dict = self.config.get('output', {})
        self.output = OutputConfig(
            directory=out_dict.get('directory', 'data/results/experiment'),
            save_final_state=out_dict.get('save_final_state', True),
            save_trajectories=out_dict.get('save_trajectories', True),
            generate_whitepaper=out_dict.get('generate_whitepaper', True),
            artifacts=out_dict.get('artifacts', [])
        )

        # Reproducibility
        repro_dict = self.config.get('reproducibility', {})
        self.reproducibility = ReproducibilityConfig(
            random_seed=repro_dict.get('random_seed', 42),
            deterministic=repro_dict.get('deterministic', True),
            track_provenance=repro_dict.get('track_provenance', True)
        )

    def get_topology_params(self) -> Dict[str, Any]:
        """
        Get topology-specific parameters.

        Returns:
            Dictionary of parameters for the selected topology type
        """
        if self.topology.type == 'mobius':
            return self.topology.mobius
        elif self.topology.type == 'torus':
            return self.topology.torus
        elif self.topology.type == 'sphere':
            return self.topology.sphere
        else:
            return {}

    def get_rnn_param_count(self) -> int:
        """
        Count total RNN-controlled parameters.

        Returns:
            Total number of parameters controlled by RNN
        """
        total = 0
        for category, params in self.rnn_control.parameters.items():
            total += len(params)
        return total

    def validate_hardware(self) -> tuple[bool, str]:
        """
        Validate that current hardware meets requirements.

        Returns:
            (is_valid, message) tuple
        """
        import torch

        # Check if CUDA required
        if self.hardware.device.startswith('cuda'):
            if not torch.cuda.is_available():
                return False, "CUDA required but not available"

            # Check VRAM
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb < self.hardware.min_memory_gb:
                    return False, f"Insufficient VRAM: {vram_gb:.1f}GB < {self.hardware.min_memory_gb}GB"

        return True, "Hardware requirements met"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert environment to dictionary.

        Returns:
            Complete environment as dictionary
        """
        return self.config

    def to_json(self, path: Optional[str] = None) -> str:
        """
        Export environment to JSON.

        Args:
            path: Optional file path to save JSON

        Returns:
            JSON string
        """
        json_str = json.dumps(self.config, indent=2)

        if path:
            with open(path, 'w') as f:
                f.write(json_str)

        return json_str

    def __repr__(self) -> str:
        return (f"Environment(name='{self.name}', "
                f"topology='{self.topology.type}', "
                f"nodes={self.simulation.nodes}, "
                f"cycles={self.simulation.cycles})")


class EnvironmentManager:
    """
    Manages multiple environments and provides discovery/loading.
    """

    def __init__(self, env_dir: str = "configs/environments"):
        """
        Initialize environment manager.

        Args:
            env_dir: Directory containing environment YAML files
        """
        self.env_dir = Path(env_dir)
        self._environments = {}
        self._load_environments()

    def _load_environments(self):
        """Load all environments from directory."""
        if not self.env_dir.exists():
            print(f"Warning: Environment directory not found: {self.env_dir}")
            return

        for yaml_file in self.env_dir.glob("*.yaml"):
            if yaml_file.name == 'schema.yaml':
                continue  # Skip schema file

            try:
                env = Environment(config_path=str(yaml_file))
                self._environments[env.name] = env
            except Exception as e:
                print(f"Error loading {yaml_file}: {e}")

    def get(self, name: str) -> Optional[Environment]:
        """
        Get environment by name.

        Args:
            name: Environment name

        Returns:
            Environment object or None if not found
        """
        return self._environments.get(name)

    def list(self) -> List[str]:
        """
        List all available environment names.

        Returns:
            List of environment names
        """
        return list(self._environments.keys())

    def list_detailed(self) -> List[Dict[str, str]]:
        """
        List environments with details.

        Returns:
            List of dictionaries with environment info
        """
        return [
            {
                'name': env.name,
                'version': env.version,
                'description': env.description,
                'topology': env.topology.type,
                'nodes': env.simulation.nodes,
                'cycles': env.simulation.cycles
            }
            for env in self._environments.values()
        ]

    def filter(self, **criteria) -> List[Environment]:
        """
        Filter environments by criteria.

        Args:
            **criteria: Key-value pairs to filter by
                       (e.g., topology='mobius', nodes=4000)

        Returns:
            List of matching environments
        """
        results = []

        for env in self._environments.values():
            match = True

            for key, value in criteria.items():
                if '.' in key:
                    # Handle nested attributes (e.g., 'topology.type')
                    parts = key.split('.')
                    obj = env
                    for part in parts:
                        obj = getattr(obj, part, None)
                        if obj is None:
                            match = False
                            break
                    if obj != value:
                        match = False
                else:
                    # Handle top-level attributes
                    if not hasattr(env, key) or getattr(env, key) != value:
                        match = False

            if match:
                results.append(env)

        return results

    def create_environment(
        self,
        name: str,
        template: str = "default",
        **overrides
    ) -> Environment:
        """
        Create new environment from template with overrides.

        Args:
            name: Name for new environment
            template: Template environment to base on
            **overrides: Parameter overrides

        Returns:
            New Environment object
        """
        # Get template
        template_env = self.get(template)
        if template_env is None:
            raise ValueError(f"Template environment '{template}' not found")

        # Copy config
        new_config = template_env.to_dict().copy()

        # Update metadata
        new_config['metadata']['name'] = name
        new_config['metadata']['created'] = datetime.now().isoformat()

        # Apply overrides
        for key, value in overrides.items():
            if '.' in key:
                # Handle nested keys (e.g., 'simulation.nodes')
                parts = key.split('.')
                current = new_config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                new_config[key] = value

        # Create environment
        env = Environment(config_dict=new_config)

        return env

    def save_environment(self, env: Environment, path: Optional[str] = None):
        """
        Save environment to YAML file.

        Args:
            env: Environment to save
            path: Optional custom path (defaults to env_dir/{name}.yaml)
        """
        if path is None:
            path = self.env_dir / f"{env.name}.yaml"

        with open(path, 'w') as f:
            yaml.dump(env.to_dict(), f, default_flow_style=False, sort_keys=False)

        # Reload environments
        self._load_environments()

    def __repr__(self) -> str:
        return f"EnvironmentManager(environments={len(self._environments)})"
