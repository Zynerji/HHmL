"""
Pytest configuration and fixtures for HHmL tests.

Provides common fixtures for environment-based testing.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhml.utils.environment_manager import EnvironmentManager, Environment
from hhml.utils.simulation_mapper import SimulationMapper, create_simulation_from_environment


@pytest.fixture(scope="session")
def env_manager():
    """
    Session-scoped environment manager.

    Returns:
        EnvironmentManager instance with all environments loaded

    Example:
        >>> def test_environments(env_manager):
        ...     envs = env_manager.list()
        ...     assert 'benchmark_mobius' in envs
    """
    return EnvironmentManager("configs/environments")


@pytest.fixture
def test_env(env_manager):
    """
    Small test environment for fast unit tests.

    Returns:
        Environment configured for quick testing (1K nodes, 10 cycles)

    Example:
        >>> def test_small_simulation(test_env):
        ...     assert test_env.simulation.nodes == 1000
        ...     assert test_env.simulation.cycles == 10
    """
    return env_manager.get("test_small")


@pytest.fixture
def benchmark_env(env_manager):
    """
    Standard benchmark environment.

    Returns:
        Environment configured for benchmarking (4K nodes, 1000 cycles)

    Example:
        >>> def test_benchmark(benchmark_env):
        ...     assert benchmark_env.simulation.nodes == 4000
        ...     assert benchmark_env.topology.type == 'mobius'
    """
    return env_manager.get("benchmark_mobius")


@pytest.fixture
def test_simulation(test_env):
    """
    Complete simulation from test environment.

    Returns:
        Dictionary containing:
            - topology: Configured Möbius topology
            - rnn_controller: RNN controller
            - training_config: Training parameters
            - monitoring_config: Monitoring parameters
            - output_config: Output parameters
            - validation_targets: Validation targets
            - environment: Environment object

    Example:
        >>> def test_topology(test_simulation):
        ...     topology = test_simulation['topology']
        ...     assert topology is not None
        ...     assert hasattr(topology, 'evolve')
    """
    mapper = SimulationMapper(test_env)
    return mapper.create_complete_simulation()


@pytest.fixture
def benchmark_simulation(benchmark_env):
    """
    Complete simulation from benchmark environment.

    Example:
        >>> def test_training(benchmark_simulation):
        ...     rnn = benchmark_simulation['rnn_controller']
        ...     assert rnn is not None
    """
    mapper = SimulationMapper(benchmark_env)
    return mapper.create_complete_simulation()


@pytest.fixture
def custom_env(env_manager):
    """
    Factory fixture for creating custom environments.

    Returns:
        Function that creates environment with overrides

    Example:
        >>> def test_custom(custom_env):
        ...     env = custom_env(
        ...         name="my_test",
        ...         template="test_small",
        ...         simulation_nodes=2000,
        ...         simulation_cycles=20
        ...     )
        ...     assert env.simulation.nodes == 2000
    """
    def _create_custom(**overrides):
        return env_manager.create_environment(**overrides)

    return _create_custom


@pytest.fixture
def custom_simulation(custom_env):
    """
    Factory fixture for creating custom simulations.

    Returns:
        Function that creates simulation with environment overrides

    Example:
        >>> def test_scaling(custom_simulation):
        ...     sim = custom_simulation(
        ...         name="scaling_test",
        ...         simulation_nodes=10000
        ...     )
        ...     assert sim['topology'] is not None
    """
    def _create_simulation(**overrides):
        env = custom_env(**overrides)
        mapper = SimulationMapper(env)
        return mapper.create_complete_simulation()

    return _create_simulation


# Parametrize helpers for testing across multiple environments
def pytest_generate_tests(metafunc):
    """
    Automatically parametrize tests with environment marker.

    Usage in tests:
        @pytest.mark.all_environments
        def test_across_envs(environment):
            # Test runs for each environment
            assert environment is not None
    """
    if "environment" in metafunc.fixturenames:
        if "all_environments" in metafunc.keywords:
            manager = EnvironmentManager("configs/environments")
            envs = [manager.get(name) for name in manager.list()]
            metafunc.parametrize("environment", envs, ids=manager.list())


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "all_environments: run test on all available environments"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests requiring GPU (deselect if no GPU)"
    )
