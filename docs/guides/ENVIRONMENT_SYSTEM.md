# HHmL Environment System

**Flexible simulation-to-topology mapping for standardized testing and benchmarking**

---

## Overview

The HHmL Environment System provides a standardized way to define, load, and execute simulations across different configurations. It acts as a **mapping layer** between generic simulation parameters and HHmL's specific topologies, field dynamics, and RNN control.

### Key Benefits

✅ **Standardized Testing** - Consistent test environments across development
✅ **Easy Benchmarking** - Compare performance across configurations
✅ **Reproducible Research** - Fixed seeds and deterministic execution
✅ **Flexible Configuration** - Override any parameter without code changes
✅ **Hardware Abstraction** - Automatic device selection and validation

---

## Quick Start

### Using Pre-defined Environments

```python
from hhml.utils.simulation_mapper import create_simulation_from_environment

# Create simulation from environment name
sim = create_simulation_from_environment('benchmark_mobius')

# Extract components
topology = sim['topology']
rnn_controller = sim['rnn_controller']
training_config = sim['training_config']

# Run training
for cycle in range(training_config['cycles']):
    # Your training loop
    pass
```

### Using in Tests

```python
import pytest

def test_vortex_density(test_simulation):
    """Uses test_small environment automatically."""
    topology = test_simulation['topology']
    targets = test_simulation['validation_targets']

    # Run simulation
    result = run_training(topology)

    # Validate against targets
    assert result['density'] >= targets['vortex_density']['min']
```

---

## Architecture

### Components

```
┌─────────────────────────────────────────────────┐
│          Environment YAML File                  │
│  (configs/environments/benchmark_mobius.yaml)   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Environment Manager                     │
│  - Loads environments from directory            │
│  - Provides discovery and filtering             │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Environment Object                      │
│  - Structured configuration                     │
│  - Validation                                   │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Simulation Mapper                       │
│  - Maps config → HHmL objects                  │
│  - Creates topology, RNN, configs               │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────┐
│         Complete Simulation                     │
│  - Topology ready to use                        │
│  - RNN controller configured                    │
│  - All parameters set                           │
└─────────────────────────────────────────────────┘
```

---

## Environment Files

### Location

All environment files are in:
```
configs/environments/
├── schema.yaml              # Schema definition (reference)
├── benchmark_mobius.yaml    # Standard benchmark
├── test_small.yaml          # Fast test environment
└── your_custom.yaml         # Your environments
```

### Structure

```yaml
metadata:
  name: "environment_name"
  version: "1.0.0"
  description: "What this environment is for"
  author: "@Conceptual1"
  tags: ["benchmark", "test"]

topology:
  type: "mobius"
  mobius:
    strips: 1
    radius: 1.0
    windings: 109

field_dynamics:
  equation: "holographic_resonance"
  holographic:
    damping: 0.1
    nonlinearity: 0.01

rnn_control:
  enabled: true
  architecture: "lstm"
  hidden_dim: 4096
  parameters:
    geometry: [kappa, delta, qec_layers, num_sites]
    physics: [damping, nonlinearity, amplitude_variance, diffusion]
    # ... (see schema.yaml for complete list)

simulation:
  nodes: 4000
  cycles: 1000
  timestep: 0.01

hardware:
  device: "auto"
  min_memory_gb: 8

validation:
  targets:
    vortex_density:
      min: 0.70
      target: 0.82

monitoring:
  dashboard:
    enabled: true
    port: 8000

output:
  directory: "data/results/{experiment}"
  generate_whitepaper: true

reproducibility:
  random_seed: 42
  deterministic: true
```

### Complete Schema

See `configs/environments/schema.yaml` for the complete schema with all available options and documentation.

---

## Python API

### EnvironmentManager

```python
from hhml.utils.environment_manager import EnvironmentManager

# Initialize manager
manager = EnvironmentManager("configs/environments")

# List available environments
envs = manager.list()
# ['benchmark_mobius', 'test_small', ...]

# Get environment
env = manager.get('benchmark_mobius')

# List with details
details = manager.list_detailed()
# [{'name': '...', 'topology': 'mobius', 'nodes': 4000, ...}, ...]

# Filter environments
mobius_envs = manager.filter(**{'topology.type': 'mobius'})
small_envs = manager.filter(**{'simulation.nodes': 1000})

# Create new environment from template
new_env = manager.create_environment(
    name="my_experiment",
    template="benchmark_mobius",
    **{"simulation.nodes": 10000}
)

# Save environment
manager.save_environment(new_env, "configs/environments/my_experiment.yaml")
```

### Environment

```python
from hhml.utils.environment_manager import Environment

# Load from file
env = Environment("configs/environments/benchmark_mobius.yaml")

# Load from dictionary
config = {...}
env = Environment(config_dict=config)

# Access structured configuration
print(env.name)                    # "benchmark_mobius"
print(env.topology.type)           # "mobius"
print(env.simulation.nodes)        # 4000
print(env.hardware.device)         # "auto"

# Get topology-specific parameters
params = env.get_topology_params()
# {'radius': 1.0, 'width': 0.5, 'windings': 109, ...}

# Count RNN parameters
param_count = env.get_rnn_param_count()  # 23

# Validate hardware
valid, message = env.validate_hardware()
if not valid:
    print(f"Hardware issue: {message}")

# Export to JSON
env.to_json("configs/my_env.json")
```

### SimulationMapper

```python
from hhml.utils.simulation_mapper import SimulationMapper

# Create mapper
mapper = SimulationMapper(env)

# Create individual components
topology = mapper.create_topology()
rnn_controller = mapper.create_rnn_controller(topology)
training_config = mapper.create_training_config()

# Or create everything at once
sim = mapper.create_complete_simulation()
# Returns dictionary with:
#   - topology
#   - rnn_controller
#   - training_config
#   - monitoring_config
#   - output_config
#   - validation_targets
#   - environment
#   - device

# Validate simulation
valid, errors = mapper.validate_simulation(sim)
if not valid:
    print(f"Validation errors: {errors}")
```

### Convenience Function

```python
from hhml.utils.simulation_mapper import create_simulation_from_environment

# One-liner to create complete simulation
sim = create_simulation_from_environment('benchmark_mobius')

# Use components
topology = sim['topology']
rnn = sim['rnn_controller']
config = sim['training_config']

# Run training
for cycle in range(config['cycles']):
    # Training loop
    pass
```

---

## Test Integration

### Pytest Fixtures

The environment system integrates seamlessly with pytest via fixtures in `tests/conftest.py`:

#### Available Fixtures

**`env_manager`** - Session-scoped EnvironmentManager
```python
def test_environments(env_manager):
    envs = env_manager.list()
    assert 'benchmark_mobius' in envs
```

**`test_env`** - Small test environment (1K nodes, 10 cycles)
```python
def test_quick(test_env):
    assert test_env.simulation.nodes == 1000
```

**`benchmark_env`** - Standard benchmark (4K nodes, 1000 cycles)
```python
def test_benchmark(benchmark_env):
    assert benchmark_env.simulation.nodes == 4000
```

**`test_simulation`** - Complete simulation from test_env
```python
def test_topology(test_simulation):
    topology = test_simulation['topology']
    assert topology is not None
```

**`benchmark_simulation`** - Complete simulation from benchmark_env
```python
def test_training(benchmark_simulation):
    rnn = benchmark_simulation['rnn_controller']
    config = benchmark_simulation['training_config']
    # Run training...
```

**`custom_env`** - Factory for custom environments
```python
def test_custom(custom_env):
    env = custom_env(
        name="my_test",
        template="test_small",
        simulation_nodes=2000
    )
    assert env.simulation.nodes == 2000
```

**`custom_simulation`** - Factory for custom simulations
```python
def test_custom_sim(custom_simulation):
    sim = custom_simulation(
        name="scaling_test",
        simulation_nodes=10000
    )
    assert sim['topology'] is not None
```

### Test Markers

**`@pytest.mark.all_environments`** - Run test on all environments
```python
@pytest.mark.all_environments
def test_loading(environment):
    """Runs once for each environment."""
    assert environment is not None
```

**`@pytest.mark.slow`** - Mark slow tests
```python
@pytest.mark.slow
def test_long_training(benchmark_simulation):
    # Long-running test
    pass

# Skip with: pytest -m "not slow"
```

**`@pytest.mark.gpu`** - Mark GPU-required tests
```python
@pytest.mark.gpu
def test_cuda(benchmark_simulation):
    assert benchmark_simulation['device'] == 'cuda'

# Skip without GPU: pytest -m "not gpu"
```

### Example Test

```python
import pytest

class TestVortexFormation:
    """Test vortex formation across environments."""

    def test_basic_vortex_creation(self, test_simulation):
        """Test vortex creation in small environment."""
        topology = test_simulation['topology']
        targets = test_simulation['validation_targets']

        # Run simulation
        topology.evolve(cycles=10)

        # Check vortex density
        density = topology.get_vortex_density()
        assert density >= targets['vortex_density']['min']

    def test_convergence(self, benchmark_simulation):
        """Test convergence in benchmark environment."""
        topology = benchmark_simulation['topology']
        rnn = benchmark_simulation['rnn_controller']
        config = benchmark_simulation['training_config']
        targets = benchmark_simulation['validation_targets']

        # Run training
        for cycle in range(config['cycles']):
            # Training step
            pass

        # Check convergence
        max_cycles = targets['convergence_cycles']['max']
        assert cycle < max_cycles

    @pytest.mark.all_environments
    def test_numerical_stability(self, environment):
        """Test numerical stability across all environments."""
        from hhml.utils.simulation_mapper import SimulationMapper

        mapper = SimulationMapper(environment)
        sim = mapper.create_complete_simulation()

        # Run short simulation
        topology = sim['topology']
        topology.evolve(cycles=5)

        # Check for NaN/Inf
        import torch
        assert torch.isfinite(topology.field).all()

    def test_custom_config(self, custom_simulation):
        """Test custom configuration."""
        sim = custom_simulation(
            name="custom_test",
            template="test_small",
            simulation_nodes=2500,
            simulation_cycles=25
        )

        assert sim['environment'].simulation.nodes == 2500
        assert sim['environment'].simulation.cycles == 25
```

---

## Creating Custom Environments

### Method 1: YAML File

Create `configs/environments/my_experiment.yaml`:

```yaml
metadata:
  name: "my_experiment"
  version: "1.0.0"
  description: "My custom experiment"
  author: "@YourHandle"
  tags: ["experiment", "custom"]

topology:
  type: "mobius"
  mobius:
    windings: 120  # Custom winding number

simulation:
  nodes: 8000     # Custom node count
  cycles: 500

# ... rest of configuration
```

Load and use:

```python
from hhml.utils.simulation_mapper import create_simulation_from_environment

sim = create_simulation_from_environment('my_experiment')
```

### Method 2: Programmatic Creation

```python
from hhml.utils.environment_manager import EnvironmentManager

manager = EnvironmentManager()

# Create from template with overrides
env = manager.create_environment(
    name="scaling_study_10K",
    template="benchmark_mobius",
    **{
        "simulation.nodes": 10000,
        "simulation.cycles": 2000,
        "topology.mobius.windings": 150,
        "hardware.min_memory_gb": 32,
    }
)

# Save for future use
manager.save_environment(env)

# Use immediately
from hhml.utils.simulation_mapper import SimulationMapper
mapper = SimulationMapper(env)
sim = mapper.create_complete_simulation()
```

### Method 3: Dictionary-based

```python
from hhml.utils.environment_manager import Environment

config = {
    'metadata': {
        'name': 'inline_experiment',
        'version': '1.0.0',
    },
    'topology': {
        'type': 'mobius',
        'mobius': {'windings': 95}
    },
    'simulation': {
        'nodes': 3000,
        'cycles': 100
    },
    # ... rest of config
}

env = Environment(config_dict=config)
```

---

## Best Practices

### 1. Use Appropriate Environment for Context

```python
# Unit tests - use test_small
def test_function(test_simulation):
    # Fast, minimal environment
    pass

# Integration tests - use benchmark_mobius
def test_workflow(benchmark_simulation):
    # Full benchmark configuration
    pass

# Production runs - create custom environment
sim = create_simulation_from_environment('production_h200')
```

### 2. Version Your Environments

```yaml
metadata:
  name: "experiment_v2"
  version: "2.0.0"  # Increment when changing
```

### 3. Tag Appropriately

```yaml
metadata:
  tags: ["benchmark", "production", "gpu-required", "large-scale"]
```

### 4. Document Custom Parameters

```yaml
topology:
  mobius:
    windings: 125  # Optimized for 10K nodes based on experiment XYZ
```

### 5. Use Templates

Don't copy entire environments - use templates with overrides:

```python
# Good
env = manager.create_environment(
    name="my_exp",
    template="benchmark_mobius",
    simulation_nodes=20000
)

# Bad - duplicates entire config
# (Just create new YAML file)
```

---

## Advanced Usage

### Filtering Environments

```python
manager = EnvironmentManager()

# Find all GPU environments
gpu_envs = [
    env for env in manager.list()
    if manager.get(env).hardware.device.startswith('cuda')
]

# Find all benchmarks
benchmarks = manager.filter(**{'metadata.tags': 'benchmark'})

# Find small test environments
small = [
    env for env in manager.list()
    if manager.get(env).simulation.nodes < 2000
]
```

### Batch Creation

```python
# Create environments for scaling study
manager = EnvironmentManager()

node_counts = [1000, 2000, 5000, 10000, 20000]

for nodes in node_counts:
    env = manager.create_environment(
        name=f"scaling_{nodes}",
        template="benchmark_mobius",
        **{"simulation.nodes": nodes}
    )
    manager.save_environment(env)
```

### Parametric Studies

```python
# Study effect of winding number
import numpy as np

windings = np.linspace(50, 150, 11)
results = []

for w in windings:
    env = manager.create_environment(
        name=f"winding_study_{int(w)}",
        template="benchmark_mobius",
        **{"topology.mobius.windings": int(w)}
    )

    sim = SimulationMapper(env).create_complete_simulation()

    # Run simulation and collect results
    result = run_training(sim)
    results.append({
        'windings': w,
        'final_density': result['density']
    })
```

---

## Environment Schema Reference

### Complete Parameter List

See `configs/environments/schema.yaml` for the authoritative schema.

**Main Sections:**
- `metadata` - Environment metadata (name, version, description, author, tags)
- `topology` - Topology configuration (type, mobius, torus, sphere parameters)
- `field_dynamics` - Field equation and parameters (holographic, GFT)
- `rnn_control` - RNN architecture and controlled parameters (23 in HHmL)
- `simulation` - Simulation parameters (nodes, cycles, timestep, scaling)
- `hardware` - Hardware requirements (device, memory, precision, multi-GPU)
- `validation` - Validation criteria (targets, tests)
- `monitoring` - Monitoring configuration (dashboard, metrics, logging)
- `output` - Output configuration (directory, artifacts, whitepaper)
- `reproducibility` - Reproducibility settings (seed, deterministic, provenance)

---

## Troubleshooting

### Environment Not Found

```python
# Error: ValueError: Environment 'my_env' not found

# Solution 1: List available environments
manager = EnvironmentManager()
print(manager.list())

# Solution 2: Check file exists
import os
path = "configs/environments/my_env.yaml"
print(os.path.exists(path))

# Solution 3: Reload manager
manager = EnvironmentManager("configs/environments")
```

### Hardware Validation Fails

```python
# Error: Hardware validation failed: CUDA required but not available

# Solution: Use CPU or auto
env.hardware.device = 'cpu'
# or
env.hardware.device = 'auto'
```

### Import Errors

```python
# Error: ModuleNotFoundError: No module named 'hhml.utils.environment_manager'

# Solution: Install package
pip install -e .

# Or add to path
import sys
sys.path.insert(0, 'src')
```

### YAML Parse Errors

```
# Error: yaml.scanner.ScannerError: mapping values are not allowed here

# Solution: Check YAML indentation
# All nested keys must be indented with spaces (not tabs)
# Use 2 or 4 spaces consistently
```

---

## Examples

### Complete Training Script

```python
#!/usr/bin/env python3
"""
Training script using environment system.
"""

from hhml.utils.simulation_mapper import create_simulation_from_environment
from hhml.monitoring.live_dashboard import TrainingDashboard
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='benchmark_mobius',
                       help='Environment name')
    args = parser.parse_args()

    # Load environment and create simulation
    print(f"Loading environment: {args.env}")
    sim = create_simulation_from_environment(args.env)

    # Extract components
    topology = sim['topology']
    rnn = sim['rnn_controller']
    training_cfg = sim['training_config']
    monitor_cfg = sim['monitoring_config']
    output_cfg = sim['output_config']
    targets = sim['validation_targets']

    # Start monitoring
    if monitor_cfg['dashboard_enabled']:
        dashboard = TrainingDashboard(port=monitor_cfg['dashboard_port'])
        dashboard.start()

    # Training loop
    print(f"Training for {training_cfg['cycles']} cycles...")

    for cycle in range(training_cfg['cycles']):
        # Evolve topology
        topology.evolve(timestep=training_cfg['timestep'])

        # Get RNN control
        state = topology.get_state()
        controls = rnn['lstm'](state)
        params = rnn['control_head'](controls)

        # Apply parameters
        topology.set_parameters(params)

        # Monitor
        if monitor_cfg['dashboard_enabled'] and cycle % monitor_cfg['update_frequency'] == 0:
            metrics = topology.get_metrics()
            dashboard.update({
                'cycle': cycle,
                'density': metrics['vortex_density'],
                'quality': metrics['vortex_quality'],
                'reward': metrics['reward']
            })

        # Save checkpoint
        if cycle % monitor_cfg['checkpoint_frequency'] == 0:
            torch.save(rnn, f"{output_cfg['directory']}/checkpoint_{cycle}.pt")

    # Validate results
    final_metrics = topology.get_metrics()

    print("\nValidation Results:")
    print(f"  Vortex Density: {final_metrics['vortex_density']:.2%}")
    print(f"    Target: {targets['vortex_density']['target']:.2%}")
    print(f"    Status: {'PASS' if final_metrics['vortex_density'] >= targets['vortex_density']['min'] else 'FAIL'}")

    if monitor_cfg['dashboard_enabled']:
        dashboard.stop()

    print(f"\nResults saved to: {output_cfg['directory']}")


if __name__ == '__main__':
    main()
```

Run with:
```bash
# Use default benchmark environment
python train.py

# Use custom environment
python train.py --env my_experiment

# Use test environment
python train.py --env test_small
```

---

## Contributing

When adding new environments:

1. **Follow schema** - Use `schema.yaml` as reference
2. **Add metadata** - Include name, version, description, author, tags
3. **Set validation targets** - Define expected outcomes
4. **Test environment** - Ensure it loads and validates
5. **Document purpose** - Add comments explaining custom parameters
6. **Commit to repo** - Save in `configs/environments/`

---

## See Also

- [Schema Reference](../../configs/environments/schema.yaml) - Complete schema
- [Example Environments](../../configs/environments/) - Pre-defined environments
- [Test Fixtures](../../tests/conftest.py) - Pytest integration
- [Hardware Configuration](hardware_config.md) - Device detection
- [Checkpoint Manager](checkpoint_manager.md) - Saving/loading

---

**Contact:** [@Conceptual1](https://twitter.com/Conceptual1)
**Repository:** https://github.com/Zynerji/HHmL
