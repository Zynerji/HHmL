"""
Unit tests for environment system.

Tests environment loading, mapping, and simulation creation.
"""

import pytest
import torch
from pathlib import Path


class TestEnvironmentManager:
    """Test EnvironmentManager functionality."""

    def test_load_environments(self, env_manager):
        """Test that environments are loaded correctly."""
        envs = env_manager.list()
        assert len(envs) > 0
        assert 'test_small' in envs
        assert 'benchmark_mobius' in envs

    def test_get_environment(self, env_manager):
        """Test getting environment by name."""
        env = env_manager.get('test_small')
        assert env is not None
        assert env.name == 'test_small'
        assert env.topology.type == 'mobius'

    def test_get_nonexistent_environment(self, env_manager):
        """Test getting non-existent environment returns None."""
        env = env_manager.get('nonexistent')
        assert env is None

    def test_list_detailed(self, env_manager):
        """Test detailed environment listing."""
        details = env_manager.list_detailed()
        assert len(details) > 0
        assert all('name' in d for d in details)
        assert all('topology' in d for d in details)
        assert all('nodes' in d for d in details)

    def test_filter_environments(self, env_manager):
        """Test filtering environments by criteria."""
        # Filter by topology type
        mobius_envs = env_manager.filter(**{'topology.type': 'mobius'})
        assert len(mobius_envs) > 0
        assert all(e.topology.type == 'mobius' for e in mobius_envs)

    def test_create_environment_from_template(self, env_manager):
        """Test creating new environment from template."""
        new_env = env_manager.create_environment(
            name="custom_test",
            template="test_small",
            **{"simulation.nodes": 2000}
        )

        assert new_env.name == "custom_test"
        assert new_env.simulation.nodes == 2000


class TestEnvironment:
    """Test Environment class functionality."""

    def test_environment_attributes(self, test_env):
        """Test environment has correct attributes."""
        assert test_env.name == 'test_small'
        assert test_env.topology.type == 'mobius'
        assert test_env.simulation.nodes == 1000
        assert test_env.simulation.cycles == 10

    def test_get_topology_params(self, test_env):
        """Test getting topology-specific parameters."""
        params = test_env.get_topology_params()
        assert 'radius' in params
        assert 'width' in params
        assert 'windings' in params

    def test_get_rnn_param_count(self, test_env):
        """Test counting RNN parameters."""
        count = test_env.get_rnn_param_count()
        assert count == 10  # test_small has 10 parameters

    def test_get_rnn_param_count_benchmark(self, benchmark_env):
        """Test benchmark has 23 parameters."""
        count = benchmark_env.get_rnn_param_count()
        assert count == 23  # Full HHmL parameter set

    def test_to_dict(self, test_env):
        """Test converting environment to dictionary."""
        config = test_env.to_dict()
        assert isinstance(config, dict)
        assert 'metadata' in config
        assert 'topology' in config
        assert 'simulation' in config

    def test_validate_hardware_cpu(self, test_env):
        """Test hardware validation for CPU."""
        valid, msg = test_env.validate_hardware()
        # Should always pass on CPU
        assert valid

    @pytest.mark.gpu
    def test_validate_hardware_cuda(self, benchmark_env):
        """Test hardware validation for CUDA."""
        # Modify to require CUDA
        benchmark_env.hardware.device = "cuda"

        valid, msg = benchmark_env.validate_hardware()

        if torch.cuda.is_available():
            # Should validate on GPU system
            assert valid or "VRAM" in msg  # May fail if insufficient VRAM
        else:
            # Should fail on CPU-only system
            assert not valid
            assert "CUDA" in msg


class TestSimulationMapper:
    """Test SimulationMapper functionality."""

    def test_create_topology(self, test_simulation):
        """Test topology creation."""
        topology = test_simulation['topology']
        assert topology is not None
        # Should have basic attributes
        assert hasattr(topology, 'device')

    def test_create_rnn_controller(self, test_simulation):
        """Test RNN controller creation."""
        rnn = test_simulation['rnn_controller']
        assert rnn is not None
        assert 'lstm' in rnn
        assert 'control_head' in rnn
        assert 'param_count' in rnn

    def test_rnn_parameter_count(self, test_simulation):
        """Test RNN outputs correct number of parameters."""
        rnn = test_simulation['rnn_controller']
        assert rnn['param_count'] == 10  # test_small has 10 params

    def test_training_config(self, test_simulation):
        """Test training configuration creation."""
        config = test_simulation['training_config']
        assert 'cycles' in config
        assert 'learning_rate' in config
        assert 'device' in config
        assert config['cycles'] == 10  # test_small cycles

    def test_monitoring_config(self, test_simulation):
        """Test monitoring configuration."""
        config = test_simulation['monitoring_config']
        assert 'dashboard_enabled' in config
        assert 'save_frequency' in config

    def test_output_config(self, test_simulation):
        """Test output configuration."""
        config = test_simulation['output_config']
        assert 'directory' in config
        assert 'test_small' in config['directory']  # Should have env name

    def test_validation_targets(self, test_simulation):
        """Test validation targets."""
        targets = test_simulation['validation_targets']
        assert 'vortex_density' in targets

    def test_device_resolution_auto(self, test_env):
        """Test device resolution with 'auto'."""
        from hhml.utils.simulation_mapper import SimulationMapper

        test_env.hardware.device = 'auto'
        mapper = SimulationMapper(test_env)

        # Should resolve to cuda or cpu
        assert mapper.device in ['cuda', 'cpu']

    def test_random_seed_setting(self, test_env):
        """Test random seed is set correctly."""
        from hhml.utils.simulation_mapper import SimulationMapper

        seed = test_env.reproducibility.random_seed
        mapper = SimulationMapper(test_env)

        # Create some random numbers
        import torch
        import numpy as np

        t1 = torch.rand(10)
        n1 = np.random.rand(10)

        # Reset mapper (should reset seed)
        mapper = SimulationMapper(test_env)

        t2 = torch.rand(10)
        n2 = np.random.rand(10)

        # Should get same random numbers
        assert torch.allclose(t1, t2)
        assert np.allclose(n1, n2)


class TestSimulationCreation:
    """Test complete simulation creation."""

    def test_create_from_environment_name(self):
        """Test convenience function."""
        from hhml.utils.simulation_mapper import create_simulation_from_environment

        sim = create_simulation_from_environment('test_small')

        assert sim is not None
        assert 'topology' in sim
        assert 'rnn_controller' in sim
        assert 'environment' in sim

    def test_create_from_nonexistent_environment(self):
        """Test error on non-existent environment."""
        from hhml.utils.simulation_mapper import create_simulation_from_environment

        with pytest.raises(ValueError, match="not found"):
            create_simulation_from_environment('nonexistent')

    def test_simulation_validation(self, test_env):
        """Test simulation validation."""
        from hhml.utils.simulation_mapper import SimulationMapper

        mapper = SimulationMapper(test_env)
        sim = mapper.create_complete_simulation()

        valid, errors = mapper.validate_simulation(sim)

        assert valid
        assert len(errors) == 0


@pytest.mark.all_environments
class TestAllEnvironments:
    """Tests that run on all available environments."""

    def test_environment_loads(self, environment):
        """Test each environment loads without error."""
        assert environment is not None
        assert environment.name is not None

    def test_topology_params_exist(self, environment):
        """Test each environment has topology parameters."""
        params = environment.get_topology_params()
        assert isinstance(params, dict)

    def test_simulation_creates(self, environment):
        """Test simulation can be created from each environment."""
        from hhml.utils.simulation_mapper import SimulationMapper

        mapper = SimulationMapper(environment)
        sim = mapper.create_complete_simulation()

        assert sim is not None
        assert 'topology' in sim


class TestCustomEnvironments:
    """Test custom environment creation."""

    def test_create_custom_nodes(self, custom_env):
        """Test creating environment with custom node count."""
        env = custom_env(
            name="custom_nodes",
            template="test_small",
            **{"simulation.nodes": 5000}
        )

        assert env.simulation.nodes == 5000

    def test_create_custom_multiple_params(self, custom_env):
        """Test creating environment with multiple overrides."""
        env = custom_env(
            name="custom_multi",
            template="test_small",
            **{
                "simulation.nodes": 3000,
                "simulation.cycles": 50,
                "topology.mobius.windings": 75
            }
        )

        assert env.simulation.nodes == 3000
        assert env.simulation.cycles == 50
        assert env.topology.mobius['windings'] == 75

    def test_custom_simulation(self, custom_simulation):
        """Test creating custom simulation."""
        sim = custom_simulation(
            name="custom_sim",
            template="test_small",
            **{"simulation.nodes": 2000}
        )

        assert sim is not None
        assert sim['environment'].simulation.nodes == 2000
