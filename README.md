<div align="center">

# Holo-Harmonic Möbius Lattice (HHmL)

### *A Glass-Box Framework for Emergent Topological Phenomena Discovery*

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/hhml/hhml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Contact:** [@Conceptual1](https://twitter.com/Conceptual1) | [GitHub Issues](https://github.com/Zynerji/HHmL/issues)

</div>

---

## 📖 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Mathematical Framework](#mathematical-framework)
- [Quick Start](#quick-start)
- [Environment System](#environment-system-usage)
- [Real-World Data Verification](#real-world-data-verification)
- [Docker Deployment](#docker-deployment)
- [Architecture](#architecture)
- [Scientific Workflow](#scientific-workflow)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Overview

The **Holo-Harmonic Möbius Lattice (HHmL)** is a computational research platform for investigating **emergent phenomena** in **topologically non-trivial field configurations**. By combining **Möbius strip topology** with **reinforcement learning-controlled parameter spaces**, HHmL enables systematic exploration of correlations between topological configurations and emergent vortex dynamics.

### What is HHmL?

HHmL explores the mathematical question: *"How do topological constraints influence emergent field structures?"* By parameterizing field dynamics on Möbius strips (single-sided surfaces with 180° twist), the framework discovers novel resonance modes impossible in simple geometries.

### What HHmL is NOT

- ❌ A theory of fundamental physics
- ❌ A model of quantum gravity or cosmology
- ❌ A replacement for established physical theories

**HHmL is a mathematical and computational research tool**, not a physical theory. All discoveries are emergent properties of the mathematical system, not claims about physical reality.

---

## Key Features

### 🎭 Möbius Strip Topology

<div align="center">

*Closed-loop, boundary-free geometric structure with topological protection*

</div>

- **No Boundary Discontinuities**: 180° twist eliminates endpoint artifacts
- **Topological Stability**: Single-sided surface stabilizes resonance modes
- **Novel Harmonic Modes**: Unique eigenspectrum not present in trivial topologies
- **Vortex Pinning**: Enhanced stability for phase singularities

### 🧠 Glass-Box RNN Control

<div align="center">

*Complete transparency: Every parameter tracked, every decision explained*

</div>

HHmL's reinforcement learning system controls **23 parameters** across 7 categories:

| Category | Count | Examples |
|:---------|:-----:|:---------|
| **Geometry** | 4 | κ (elongation), δ (triangularity), QEC layers, num_sites |
| **Physics** | 4 | Damping, nonlinearity, amplitude variance, diffusion |
| **Spectral** | 3 | ω (helical frequency), diffusion timestep, spectral weight |
| **Sampling** | 3 | Sample ratio, neighbors, sparsity threshold |
| **Mode** | 2 | Sparse density, spectral activation |
| **Topology** | 3 | Winding density, twist rate, coupling strength |
| **Annihilation** | 4 | Antivortex strength, radius, threshold, preserve ratio |

**Every parameter trajectory is saved** → Full correlation analysis and reproducibility guaranteed.

### 🔬 Vortex Annihilation Control *(Novel Capability)*

<div align="center">

*RNN-guided selective pruning of low-quality topological structures*

</div>

- **Quality Scoring**: Each vortex evaluated on neighborhood density, core depth, stability
- **Selective Pruning**: Low-quality vortices removed while preserving high-quality structures
- **Antivortex Injection**: Phase-inverted fields injected near problematic vortices
- **Learned Optimization**: RNN discovers optimal curation strategies via reinforcement learning

**Result**: Achieved **100% peak vortex density** (cycle 490) through autonomous quality control.

### 🔧 Environment System *(New)*

<div align="center">

*Flexible simulation-to-topology mapping for standardized testing and reproducible research*

</div>

- **YAML Configuration**: Define complete simulations in structured YAML files
- **Pre-defined Environments**: `benchmark_mobius`, `test_small` ready to use
- **Flexible Mapping**: Generic parameters automatically map to HHmL topologies
- **Pytest Integration**: Seamless test fixtures for environment-based testing
- **Hardware Abstraction**: Automatic device selection and validation
- **Reproducibility**: Fixed seeds, deterministic execution, provenance tracking

```python
from hhml.utils.simulation_mapper import create_simulation_from_environment

# Load pre-configured environment
sim = create_simulation_from_environment('benchmark_mobius')

# Extract configured components
topology = sim['topology']
rnn_controller = sim['rnn_controller']
training_config = sim['training_config']

# Run with validated configuration
for cycle in range(training_config['cycles']):
    topology.evolve(timestep=training_config['timestep'])
```

### 🌍 Real-World Data Verification *(New)*

<div align="center">

*Ground emergent phenomena in empirical physics through comparison with real experimental data*

</div>

HHmL now includes comprehensive verification against real-world physics data:

**LIGO Gravitational Waves**:
- Compare boundary resonances to real LIGO/Virgo detections
- Matched-filter overlap with events like GW150914 (first black hole merger)
- Automatic data fetching from GWOSC (Gravitational Wave Open Science Center)

**Planck CMB Power Spectra**:
- Map field fluctuations to cosmic microwave background anisotropies
- χ² fitting against Planck TT/EE/BB spectra
- ΛCDM fiducial comparisons via CAMB

**Particle Physics (LHC/PDG)**:
- Match vortex energies to Standard Model particle masses
- Compare excitation spectra to LHC invariant mass histograms
- Automated comparison with PDG particle database

```python
from hhml.verification import LIGOVerification, CMBVerification, ParticleVerification

# Compare to LIGO waveforms
ligo = LIGOVerification()
results = ligo.compare_event('GW150914', field_tensor)
print(f"Overlap: {results['metrics']['overlap']:.4f}")

# Compare to Planck CMB
cmb = CMBVerification()
results = cmb.compare_planck(field_tensor, cl_type='TT')
print(f"χ²/DOF: {results['metrics']['reduced_chi_squared']:.3f}")

# Compare to particle masses
particles = ParticleVerification()
results = particles.compare_pdg_masses(vortex_energies)
print(f"Matched: {results['matched_particles']}/{results['total_particles']}")
```

**Philosophy**: These are *analogical comparisons* - HHmL is not claiming to model gravity or particles, but testing if emergent mathematical structures exhibit patterns similar to physical phenomena. This grounds exploration in testable hypotheses.

### ⚡ Production-Ready Infrastructure

- **Auto-Scaling**: CPU (2K nodes) → H200 GPU (20M+ nodes)
- **Docker Integration**: Multi-stage builds, GPU support, orchestration via Docker Compose
- **Live Monitoring**: Real-time web dashboard with interactive charts
- **Automated Reporting**: LaTeX whitepaper generation with peer-review quality
- **Reproducible Science**: Complete parameter logs, random seeds, hardware specs

---

## Mathematical Framework

### Möbius Strip Parameterization

The field dynamics evolve on a Möbius strip $\mathcal{M}$ parameterized by:

$$
\begin{aligned}
x(u,v) &= \left(R + v\cos\frac{u}{2}\right)\cos u \\
y(u,v) &= \left(R + v\cos\frac{u}{2}\right)\sin u \\
z(u,v) &= v\sin\frac{u}{2}
\end{aligned}
\quad u \in [0, 2\pi), \; v \in [-w, w]
$$

where $R$ is the strip radius and $w$ is the half-width.

### Field Dynamics

The complex field $\psi: \mathcal{M} \times \mathbb{R}^+ \to \mathbb{C}$ obeys:

$$
\frac{\partial \psi}{\partial t} = -\gamma\psi + \lambda|\psi|^2\psi + \sum_{i=1}^{N} A_i \frac{\sin(k|r - r_i|)}{|r - r_i|} e^{i\phi_i}
$$

- $\gamma$: Damping coefficient (RNN-controlled)
- $\lambda$: Nonlinearity strength (RNN-controlled)
- $A_i$: Source amplitudes (RNN-controlled)
- $\phi_i$: Source phases
- $k$: Wavenumber

### Topological Charge

Vortices are characterized by winding number:

$$
n_v = \frac{1}{2\pi} \oint_{\partial\Omega} \nabla \arg(\psi) \cdot d\mathbf{l}
$$

where $\Omega$ is a small region around the vortex core.

### Reinforcement Learning Objective

The RNN maximizes:

$$
\mathcal{R} = \underbrace{\alpha_1 \rho_v}_{\text{density}} + \underbrace{\alpha_2 Q_v}_{\text{quality}} - \underbrace{\alpha_3 \sigma(\rho_v)}_{\text{uniformity}} + \underbrace{\alpha_4 \mathcal{S}}_{\text{spectral}} + \underbrace{\alpha_5 \mathcal{C}}_{\text{convergence}}
$$

where:
- $\rho_v$: Vortex density
- $Q_v$: Average vortex quality
- $\sigma(\rho_v)$: Spatial variance (penalizes clustering)
- $\mathcal{S}$: Spectral richness (peak count in $|\hat{\psi}(k)|^2$)
- $\mathcal{C}$: Parameter convergence bonus

---

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA 12.1+ (for GPU support)
- 8GB+ RAM (CPU mode) or 16GB+ VRAM (GPU mode)

### Installation

```bash
# Clone repository
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# Install in development mode
pip install -e .

# Install with all optional dependencies
pip install -e ".[dev,viz,docs]"

# Verify installation
python -c "import hhml; print(hhml.__version__)"
```

### Run Your First Simulation

```bash
# CPU training (2K nodes, 100 cycles)
python examples/training/train_mobius_basic.py --cycles 100 --device cpu

# GPU training (auto-detect CUDA)
python examples/training/train_mobius_basic.py --cycles 500

# Multi-strip topology (advanced)
python examples/training/train_multi_strip.py --cycles 1000 --strips 2
```

### Monitor Training

```bash
# Start live dashboard (runs on http://localhost:8000)
python -m hhml.monitoring.live_dashboard

# Generate whitepaper from results
python tools/whitepaper/whitepaper_generator.py --results data/results/latest
```

---

## Environment System Usage

The **Environment System** provides flexible simulation-to-topology mapping through YAML configuration files. This enables standardized testing, easy benchmarking, and reproducible research.

### Using Pre-defined Environments

HHmL includes pre-configured environments for common use cases:

```python
from hhml.utils.simulation_mapper import create_simulation_from_environment

# Use standard benchmark (4K nodes, 1000 cycles, 23 RNN parameters)
sim = create_simulation_from_environment('benchmark_mobius')

# Extract configured components
topology = sim['topology']              # Möbius strip with optimal windings
rnn_controller = sim['rnn_controller']  # LSTM with 23 parameters
training_config = sim['training_config']  # Learning rate, cycles, etc.
validation_targets = sim['validation_targets']  # Expected outcomes

# Run training
for cycle in range(training_config['cycles']):
    topology.evolve(timestep=training_config['timestep'])

    # Validate against targets
    density = topology.get_vortex_density()
    if density >= validation_targets['vortex_density']['target']:
        print(f"Target achieved at cycle {cycle}!")
        break
```

### Available Environments

| Environment | Nodes | Cycles | Purpose |
|:------------|:-----:|:------:|:--------|
| `benchmark_mobius` | 4,000 | 1,000 | Standard benchmark, full 23 parameters |
| `test_small` | 1,000 | 10 | Fast testing, 10 parameters |

### Creating Custom Environments

**Method 1: YAML File**

Create `configs/environments/my_experiment.yaml`:

```yaml
metadata:
  name: "my_experiment"
  version: "1.0.0"
  description: "Custom scaling study"
  author: "@YourHandle"

topology:
  type: "mobius"
  mobius:
    windings: 120        # Custom winding number
    radius: 1.5          # Custom radius

simulation:
  nodes: 10000           # 10K nodes
  cycles: 2000           # Extended training

reproducibility:
  random_seed: 42
  deterministic: true
```

Load and use:

```python
sim = create_simulation_from_environment('my_experiment')
```

**Method 2: Programmatic**

```python
from hhml.utils.environment_manager import EnvironmentManager

manager = EnvironmentManager()

# Create from template with overrides
env = manager.create_environment(
    name="scaling_10K",
    template="benchmark_mobius",
    **{
        "simulation.nodes": 10000,
        "topology.mobius.windings": 150,
        "hardware.min_memory_gb": 32
    }
)

# Save for reuse
manager.save_environment(env)

# Use immediately
from hhml.utils.simulation_mapper import SimulationMapper
mapper = SimulationMapper(env)
sim = mapper.create_complete_simulation()
```

### Test Integration

Environment fixtures work seamlessly with pytest:

```python
def test_vortex_density(test_simulation):
    """Uses test_small environment automatically."""
    topology = test_simulation['topology']
    targets = test_simulation['validation_targets']

    # Run simulation
    topology.evolve(cycles=10)

    # Validate
    density = topology.get_vortex_density()
    assert density >= targets['vortex_density']['min']

def test_custom_config(custom_simulation):
    """Create custom environment on-the-fly."""
    sim = custom_simulation(
        name="my_test",
        simulation_nodes=5000,
        topology_mobius_windings=95
    )

    assert sim['topology'] is not None
```

**See full documentation:** [Environment System Guide](docs/guides/ENVIRONMENT_SYSTEM.md)

---

## Real-World Data Verification

The **Verification System** grounds HHmL's emergent phenomena in empirical physics by comparing simulation outputs to real experimental data. This moves HHmL from pure mathematical exploration toward **testable hypotheses** against actual physical observations.

### Philosophy

**Important**: These are *analogical comparisons*. HHmL is not claiming to model fundamental physics, but testing if emergent mathematical structures exhibit patterns similar to physical phenomena. This provides:

- **Falsifiable Predictions**: Testable correlations between parameters and observables
- **Empirical Grounding**: Comparison against real data validates or refutes emergent patterns
- **Pattern Discovery**: Identify mathematical structures shared between topology and physics

### 1. LIGO Gravitational Waves

Compare HHmL boundary resonances to real gravitational wave detections from LIGO/Virgo.

```python
from hhml.verification.ligo import LIGOVerification

# Initialize verifier
ligo = LIGOVerification(data_dir="data/ligo")

# Compare simulation to GW150914 (first black hole merger)
results = ligo.compare_event(
    event_name='GW150914',
    sim_strain_tensor=field_tensor,  # [time_steps, nodes, features]
    detector='H1',  # Hanford detector
    save_results=True
)

print(f"Waveform overlap: {results['metrics']['overlap']:.4f}")
print(f"Signal-to-noise: {results['metrics']['snr']:.2f}")
print(f"Interpretation: {results['interpretation']}")
```

**Known Events**: GW150914 (BBH), GW151226 (BBH), GW170817 (BNS multi-messenger)

**Metrics**:
- **Overlap** (0-1): Matched-filter correlation
- **SNR**: Signal-to-noise ratio
- **Mismatch**: 1 - overlap

**Data Source**: [GWOSC](https://gwosc.org/) - Gravitational Wave Open Science Center

### 2. Planck CMB Power Spectra

Map field fluctuations to cosmic microwave background temperature anisotropies.

```python
from hhml.verification.cmb import CMBVerification

# Initialize verifier
cmb = CMBVerification(data_dir="data/cmb", nside=512)

# Compare to Planck 2018 TT spectrum
results = cmb.compare_planck(
    sim_field_tensor=field_tensor,  # [nodes, features]
    cl_type='TT',  # Temperature auto-correlation
    lmax=2000,  # Maximum multipole
    save_results=True
)

print(f"χ²: {results['metrics']['chi_squared']:.2f}")
print(f"χ²/DOF: {results['metrics']['reduced_chi_squared']:.3f}")
print(f"p-value: {results['metrics']['p_value']:.4f}")
```

**Spectrum Types**: TT (temperature), EE (E-polarization), BB (B-polarization), TE (cross)

**Metrics**:
- **χ²**: Chi-squared statistic
- **χ²/DOF**: Reduced chi-squared (good fit: ~1)
- **p-value**: Statistical significance

**Data Source**: Planck Legacy Archive + CAMB (ΛCDM fiducial)

### 3. Particle Physics (LHC/PDG)

Match vortex excitation energies to Standard Model particle masses.

```python
from hhml.verification.particles import ParticleVerification

# Initialize verifier
particles = ParticleVerification(data_dir="data/particles")

# Compare to PDG masses
results = particles.compare_pdg_masses(
    sim_energies=vortex_energies,  # [N_vortices] in GeV
    particle_list=['electron', 'muon', 'Z_boson', 'Higgs'],
    tolerance=0.1  # ±10% match tolerance
)

print(f"Matched: {results['matched_particles']}/{results['total_particles']}")

# Compare to LHC Higgs channel
lhc_results = particles.compare_lhc_channel(
    sim_energies=vortex_energies,
    channel='higgs_4l',  # H → 4 leptons
    scale_factor=1.0,  # GeV conversion
    save_results=True
)

print(f"χ²/DOF: {lhc_results['metrics']['reduced_chi_squared']:.3f}")
```

**Known Particles**: electron (0.511 MeV), muon (105.66 MeV), Z (91.2 GeV), Higgs (125.25 GeV), ...

**LHC Channels**: higgs_4l, Z_ee, W_enu

**Metrics**:
- **Match fraction**: % of particles matched within tolerance
- **χ²/DOF**: Invariant mass spectrum fit quality
- **KS statistic**: Kolmogorov-Smirnov distribution test

**Data Sources**: [PDG](https://pdg.lbl.gov/), [HEPData](https://hepdata.net/)

### Integration Examples

**As RL Reward**:
```python
# Add verification metrics to training reward
reward = (
    0.7 * reward_vortex_density +
    0.15 * ligo_results['metrics']['overlap'] +
    0.15 * (1.0 / (1.0 + cmb_results['metrics']['reduced_chi_squared']))
)
```

**In Whitepaper**:
```python
# Automatically include verification results
whitepaper_data['verification'] = {
    'ligo': ligo_results,
    'cmb': cmb_results,
    'particles': particle_results
}
```

### Dependencies

Install verification libraries:

```bash
pip install gwpy healpy camb uproot awkward astropy
```

**Optional**: If dependencies not installed, all modules provide synthetic data for testing.

**See full documentation:** [Verification System Guide](docs/guides/VERIFICATION_SYSTEM.md)

**Example scripts**: `examples/verification/verify_*.py`

---

## Docker Deployment

### Quick Start with Docker

```bash
# Build all images
cd docker && ./scripts/build.sh all

# Run production training
./scripts/run.sh production

# Run development environment (JupyterLab)
./scripts/run.sh development
```

### Available Docker Images

| Image | Purpose | Size | CUDA |
|:------|:--------|:----:|:----:|
| `hhml:cpu-latest` | CPU-only lightweight image | ~2GB | ❌ |
| `hhml:cuda-latest` | H100/H200 production training | ~8GB | ✅ |
| `hhml:dev-latest` | JupyterLab + dev tools | ~10GB | ✅ |

### Docker Compose Services

```yaml
# Production: Training + Monitoring
docker-compose up -d

# Development: JupyterLab + TensorBoard
docker-compose -f docker-compose.dev.yml up -d

# Generate whitepaper (on-demand)
docker-compose --profile tools run hhml-whitepaper
```

**Access Points:**
- Monitoring Dashboard: http://localhost:8000
- JupyterLab: http://localhost:8888
- TensorBoard: http://localhost:6006

### H200 VM Deployment

```bash
# SSH into H200
ssh ivhl@89.169.111.28

# Clone and build
git clone https://github.com/Zynerji/HHmL.git
cd HHmL/docker
./scripts/build.sh cuda

# Run 20M node training
docker run --gpus all \
  -v $(pwd)/data:/data \
  hhml:cuda-latest \
  python examples/training/train_mobius_basic.py \
  --cycles 1000 --nodes 20000000
```

---

## Architecture

### Repository Structure

```
HHmL/
├── src/hhml/               # Main Python package
│   ├── core/               # Physics & topology modules
│   │   ├── mobius/        # Möbius strip dynamics
│   │   ├── resonance/     # Holographic resonance
│   │   ├── gft/           # Group Field Theory
│   │   └── tensor_networks/ # MERA holography
│   ├── ml/                 # Machine learning
│   │   ├── rl/            # Reinforcement learning
│   │   └── training/      # Training loops
│   ├── analysis/           # Data analysis
│   │   └── dark_matter/   # Pruning theory
│   ├── monitoring/         # Web dashboard
│   └── utils/              # Shared utilities
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── benchmarks/        # Performance benchmarks
├── examples/               # Example scripts
│   ├── training/          # Training examples
│   └── analysis/          # Analysis examples
├── docker/                 # Docker configuration
│   ├── Dockerfile.{cpu,cuda,dev}
│   ├── docker-compose.yml
│   └── scripts/           # Helper scripts
├── docs/                   # Documentation
│   ├── guides/            # User guides
│   ├── deployment/        # Deployment guides
│   └── theory/            # Mathematical theory
├── configs/                # Configuration files
├── tools/                  # Development tools
│   ├── whitepaper/        # Whitepaper generator
│   └── benchmarking/      # Performance tools
└── data/                   # Data directory (gitignored)
    ├── checkpoints/       # Model checkpoints
    ├── results/           # Training results
    └── outputs/           # Generated outputs
```

### Module Overview

| Module | Description | Key Files |
|:-------|:------------|:----------|
| `hhml.core.mobius` | Möbius strip geometry & dynamics | `mobius_training.py`, `optimized_sphere.py` |
| `hhml.core.resonance` | Holographic boundary resonance | `holographic_resonance.py`, `vortex_controller.py` |
| `hhml.ml.rl` | Reinforcement learning (TD3-SAC) | `td3_agent.py`, `sac_agent.py` |
| `hhml.ml.training` | Training loops & checkpointing | `trainer.py`, `checkpoint_manager.py` |
| `hhml.monitoring` | Live dashboard & visualization | `live_dashboard.py`, `streaming_server.py` |
| `hhml.utils` | Hardware detection & validation | `hardware_config.py`, `startup_validator.py` |

---

## Scientific Workflow

```mermaid
graph LR
    A[Configure Experiment] --> B[Run Training]
    B --> C[Monitor Live Dashboard]
    C --> D[Save Checkpoint]
    D --> E{Continue?}
    E -->|Yes| B
    E -->|No| F[Generate Whitepaper]
    F --> G[Analyze Correlations]
    G --> H[Publish Results]
```

### Step-by-Step

1. **Configure Experiment**
   ```bash
   cp configs/example.yaml configs/my_experiment.yaml
   # Edit configs/my_experiment.yaml
   ```

2. **Run Training**
   ```bash
   python examples/training/train_mobius_basic.py \
     --config configs/my_experiment.yaml \
     --cycles 1000
   ```

3. **Monitor Progress**
   - Open http://localhost:8000 for live dashboard
   - Watch real-time vortex density, quality, reward charts

4. **Resume from Checkpoint**
   ```bash
   python examples/training/train_mobius_basic.py \
     --resume data/checkpoints/agent_cycle500.pt \
     --cycles 1500
   ```

5. **Generate Analysis**
   ```bash
   python tools/whitepaper/whitepaper_generator.py \
     --results data/results/my_experiment_*.json \
     --output data/outputs/whitepapers/
   ```

6. **Analyze Correlations**
   - See [RNN_PARAMETER_MAPPING.md](docs/guides/RNN_PARAMETER_MAPPING.md)
   - Use correlation tracking methods to discover parameter-outcome relationships

---

## Documentation

| Document | Description |
|:---------|:------------|
| [**Environment System Guide**](docs/guides/ENVIRONMENT_SYSTEM.md) | Simulation-to-topology mapping system |
| [**Installation Guide**](docs/guides/installation.md) | Detailed installation instructions |
| [**User Guide**](docs/guides/user_guide.md) | Complete usage tutorial |
| [**API Reference**](docs/guides/api_reference.md) | Python API documentation |
| [**Docker Guide**](docs/deployment/docker.md) | Docker deployment guide |
| [**H200 Deployment**](docs/deployment/h200.md) | H200 VM setup & scaling |
| [**Mathematical Theory**](docs/theory/mathematical_framework.md) | Complete mathematical derivations |
| [**RNN Parameters**](docs/guides/RNN_PARAMETER_MAPPING.md) | Parameter correlation analysis |
| [**CLAUDE.md**](CLAUDE.md) | AI assistant context & workflows |

### Example Notebooks

- [Basic Training Tutorial](examples/notebooks/01_basic_training.ipynb)
- [Parameter Tuning](examples/notebooks/02_parameter_tuning.ipynb)
- [Vortex Analysis](examples/notebooks/03_vortex_analysis.ipynb)
- [Scaling Studies](examples/notebooks/04_scaling_studies.ipynb)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code style
black src/ tests/
flake8 src/ tests/
mypy src/
```

### Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage report
pytest --cov=hhml --cov-report=html
```

### Code Quality

This project follows strict code quality standards:

- **Formatting**: [Black](https://black.readthedocs.io/) (line length 100)
- **Linting**: [Flake8](https://flake8.pycqa.org/)
- **Type Checking**: [MyPy](https://mypy.readthedocs.io/)
- **Testing**: [Pytest](https://pytest.org/) (>90% coverage required)

---

## Citation

If you use HHmL in your research, please cite:

```bibtex
@software{hhml2025,
  title     = {Holo-Harmonic Möbius Lattice (HHmL): A Glass-Box Framework
               for Emergent Topological Phenomena Discovery},
  author    = {HHmL Research Collective},
  year      = {2025},
  version   = {0.1.0},
  url       = {https://github.com/Zynerji/HHmL},
  doi       = {10.5281/zenodo.XXXXXXX},
  note      = {Computational research platform for investigating emergent
               phenomena in Möbius strip topologies}
}
```

### Publications

*Publications using HHmL will be listed here.*

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **PyTorch Team**: For the deep learning framework
- **NumPy/SciPy Communities**: For scientific computing tools
- **Docker**: For containerization infrastructure
- **Nebius**: For H200 GPU access

---

## Contact & Support

<div align="center">

**Primary Contact:** [@Conceptual1](https://twitter.com/Conceptual1)

**GitHub:** [Zynerji/HHmL](https://github.com/Zynerji/HHmL)

**Issues:** [GitHub Issues](https://github.com/Zynerji/HHmL/issues)

**Discussions:** [GitHub Discussions](https://github.com/Zynerji/HHmL/discussions)

---

### Project Status

![GitHub last commit](https://img.shields.io/github/last-commit/Zynerji/HHmL)
![GitHub issues](https://img.shields.io/github/issues/Zynerji/HHmL)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Zynerji/HHmL)
![GitHub stars](https://img.shields.io/github/stars/Zynerji/HHmL?style=social)

---

**HHmL: Exploring emergent phenomena through topological field dynamics**

*Mathematical research platform — not a physical theory*

</div>
