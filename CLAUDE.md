# Hello Claude - HHmL Framework Context

**Last Updated**: 2025-12-18 (Hash Quine Discovery Published)
**Project**: HHmL (Holo-Harmonic Möbius Lattice) Framework
**Repository**: https://github.com/Zynerji/HHmL
**Parent Project**: iVHL (Vibrational Helical Lattice)
**Status**: Production-Ready - v0.1.0 + Novel Scientific Discovery
**Contact**: [@Conceptual1](https://twitter.com/Conceptual1)

---

## 🏗️ PRODUCTION STRUCTURE (Dec 17, 2025)

**CRITICAL: This repository follows a production-ready structure. All new files MUST follow this organization.**

### Directory Structure

```
HHmL/
├── .github/
│   └── workflows/              # CI/CD pipelines (GitHub Actions)
├── src/
│   └── hhml/                   # Main Python package (ALL code here)
│       ├── core/               # Core physics modules
│       │   ├── mobius/        # Möbius strip topology & dynamics
│       │   ├── resonance/     # Holographic boundary resonance
│       │   ├── gft/           # Group Field Theory
│       │   └── tensor_networks/ # MERA holography, RT formula
│       ├── ml/                 # Machine learning components
│       │   ├── rl/            # Reinforcement learning (TD3, SAC)
│       │   └── training/      # Training loops & orchestration
│       ├── analysis/           # Analysis & visualization
│       │   └── dark_matter/   # Pruning theory & multiverse
│       ├── monitoring/         # Web monitoring & dashboards
│       │   ├── live_dashboard.py
│       │   ├── streaming_server.py
│       │   └── rendering/     # GPU rendering
│       └── utils/              # Shared utilities
│           ├── hardware_config.py
│           ├── checkpoint_manager.py
│           └── startup_validator.py
├── tests/                      # All tests (pytest)
│   ├── unit/                  # Fast, isolated unit tests
│   ├── integration/           # Multi-component integration tests
│   └── benchmarks/            # Performance benchmarks
├── examples/                   # Example usage scripts
│   ├── training/              # Training examples
│   │   ├── train_mobius_basic.py
│   │   ├── train_multi_strip.py
│   │   └── train_quality_guided.py
│   └── analysis/              # Analysis examples
├── docker/                     # Complete Docker infrastructure
│   ├── Dockerfile.cpu         # Lightweight CPU image
│   ├── Dockerfile.cuda        # H100/H200 GPU image
│   ├── Dockerfile.dev         # Development + JupyterLab
│   ├── docker-compose.yml     # Production orchestration
│   ├── docker-compose.dev.yml # Development environment
│   └── scripts/               # Helper scripts
│       ├── build.sh           # Build all images
│       └── run.sh             # Run containers
├── docs/                       # Documentation
│   ├── guides/                # User guides & tutorials
│   │   ├── RNN_PARAMETER_MAPPING.md
│   │   ├── MULTI_STRIP_TOPOLOGY.md
│   │   └── H200_DEPLOYMENT.md
│   ├── deployment/            # Deployment guides
│   └── theory/                # Mathematical theory
├── configs/                    # Configuration files (YAML)
│   └── multiscale_config.yaml
├── tools/                      # Development tools
│   ├── whitepaper/            # Whitepaper generator
│   └── benchmarking/          # Performance tools
├── data/                       # Data directory (GITIGNORED)
│   ├── checkpoints/           # Model checkpoints (.pt, .pth)
│   ├── results/               # Training results (JSON)
│   └── outputs/               # Generated outputs
│       └── whitepapers/       # Generated PDFs
├── archive/                    # Legacy code (NEVER ADD TO)
├── pyproject.toml              # Modern Python packaging
├── setup.py                    # Setuptools config
├── README.md                   # Professional README
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── CHANGELOG.md                # Version history
├── MIGRATION_GUIDE.md          # Migration instructions
├── REFACTORING_SUMMARY.md      # Complete refactoring summary
├── CLAUDE.md                   # This file - AI context
├── .gitignore                  # Comprehensive ignores
├── .dockerignore               # Docker-specific ignores
└── .editorconfig               # Editor configuration
```

---

## 📋 HOW TO MAINTAIN THIS STRUCTURE

### Golden Rules

**1. NEVER put Python code in root directory**
   - ✅ `src/hhml/core/mobius/new_module.py`
   - ❌ `new_script.py` (in root)

**2. NEVER commit data/results/checkpoints**
   - ✅ `data/` is gitignored - mount as Docker volume
   - ❌ Committing .pt, .pth, .json results

**3. NEVER put loose documentation in root**
   - ✅ `docs/guides/new_guide.md`
   - ❌ `NEW_FEATURE_DOC.md` (in root)

**4. ALWAYS use proper import paths**
   - ✅ `from hhml.core.mobius.mobius_training import MobiusStrip`
   - ❌ `from hhml.mobius.mobius_training import MobiusStrip` (old)

**5. ALWAYS add __init__.py to new packages**
   - When creating `src/hhml/new_module/`, add `__init__.py`

---

## 📂 WHERE TO PUT NEW FILES

### New Python Module

**Location**: `src/hhml/{category}/{module_name}/`

**Categories**:
- `core/` - Physics, topology, field dynamics
- `ml/` - Machine learning, RL, training
- `analysis/` - Data analysis, visualization
- `monitoring/` - Dashboards, web interfaces
- `utils/` - Shared utilities

**Steps**:
```bash
# 1. Create module directory
mkdir -p src/hhml/core/new_topology/

# 2. Add __init__.py
touch src/hhml/core/new_topology/__init__.py

# 3. Create module files
touch src/hhml/core/new_topology/dynamics.py
touch src/hhml/core/new_topology/geometry.py

# 4. Import in parent __init__.py (optional)
# Edit src/hhml/core/__init__.py to expose module
```

### New Training Script

**Location**: `examples/training/`

```bash
# Create training script
touch examples/training/train_new_topology.py
chmod +x examples/training/train_new_topology.py
```

### New Test

**Location**: `tests/{unit|integration|benchmarks}/`

```bash
# Unit test
touch tests/unit/test_new_topology.py

# Integration test
touch tests/integration/test_new_topology_training.py

# Benchmark
touch tests/benchmarks/benchmark_new_topology.py
```

### New Documentation

**Location**: `docs/{guides|deployment|theory}/`

```bash
# User guide
touch docs/guides/NEW_TOPOLOGY_GUIDE.md

# Deployment guide
touch docs/deployment/NEW_TOPOLOGY_DEPLOYMENT.md

# Theory documentation
touch docs/theory/NEW_TOPOLOGY_MATH.md
```

### New Configuration

**Location**: `configs/`

```bash
# YAML config
touch configs/new_topology_config.yaml
```

### New Tool

**Location**: `tools/{category}/`

```bash
# Whitepaper tool
touch tools/whitepaper/new_analyzer.py

# Benchmarking tool
touch tools/benchmarking/new_benchmark.py
```

---

## 🔄 WORKFLOW FOR ADDING FEATURES

### Step-by-Step Process

**1. Plan the feature**
   - Determine category (core/ml/analysis/monitoring)
   - Check if fits existing module or needs new one

**2. Create module structure**
   ```bash
   mkdir -p src/hhml/{category}/{module_name}/
   touch src/hhml/{category}/{module_name}/__init__.py
   ```

**3. Write code**
   - Follow Black formatting (100 char lines)
   - Add docstrings (Google style)
   - Add type hints where helpful

**4. Write tests**
   ```bash
   touch tests/unit/test_{module_name}.py
   pytest tests/unit/test_{module_name}.py
   ```

**5. Write documentation**
   ```bash
   touch docs/guides/{FEATURE_NAME}.md
   ```

**6. Update CHANGELOG.md**
   ```markdown
   ## [Unreleased]

   ### Added
   - New topology module with XYZ capabilities
   ```

**7. Create example**
   ```bash
   touch examples/training/train_{feature_name}.py
   ```

**8. Commit with conventional commits**
   ```bash
   git add .
   git commit -m "feat: add new topology module with XYZ capabilities

   - Implements ABC dynamics
   - Adds DEF visualization
   - Includes unit tests and documentation

   Closes #123"
   ```

---

## 🚫 ANTI-PATTERNS (DON'T DO THESE)

### ❌ Bad: Loose Files in Root
```
HHmL/
├── my_experiment.py          # NO! Use examples/
├── test_new_feature.py       # NO! Use tests/
├── NEW_RESULTS.md            # NO! Use docs/guides/
├── checkpoint.pt             # NO! Use data/checkpoints/
└── results.json              # NO! Use data/results/
```

### ❌ Bad: Flat Package Structure
```python
# NO! Don't add to src/hhml/ directly
src/hhml/my_new_module.py

# YES! Use proper categorization
src/hhml/core/my_topology/dynamics.py
```

### ❌ Bad: Committing Generated Files
```bash
# NO! Don't commit these
git add data/results/*.json
git add data/checkpoints/*.pt
git add data/outputs/whitepapers/*.pdf

# These are gitignored for a reason!
```

### ❌ Bad: Mixed Concerns
```python
# NO! Don't mix physics and ML in same file
src/hhml/core/mobius/mobius_with_training.py

# YES! Separate concerns
src/hhml/core/mobius/dynamics.py      # Physics only
src/hhml/ml/training/mobius_trainer.py # Training only
```

---

## ✅ GOOD PATTERNS (DO THESE)

### ✅ Good: Modular Organization
```
src/hhml/core/klein_bottle/
├── __init__.py
├── topology.py       # Topology definition
├── dynamics.py       # Field dynamics
├── geometry.py       # Geometric calculations
└── visualization.py  # Plotting utilities
```

### ✅ Good: Comprehensive Testing
```
tests/
├── unit/
│   └── test_klein_bottle.py           # Fast unit tests
├── integration/
│   └── test_klein_bottle_training.py  # Full workflow
└── benchmarks/
    └── benchmark_klein_bottle.py      # Performance tests
```

### ✅ Good: Complete Documentation
```
docs/
├── guides/
│   └── KLEIN_BOTTLE_GUIDE.md         # User guide
├── deployment/
│   └── KLEIN_BOTTLE_H200.md          # Deployment guide
└── theory/
    └── KLEIN_BOTTLE_MATH.md          # Mathematical theory
```

### ✅ Good: Organized Examples
```
examples/
├── training/
│   ├── train_klein_bottle_basic.py   # Simple example
│   └── train_klein_bottle_advanced.py # Complex example
└── analysis/
    └── analyze_klein_bottle_results.py
```

---

## 🐳 DOCKER WORKFLOW

### Building Images

```bash
# Build all images
cd docker && ./scripts/build.sh all

# Build specific image
./scripts/build.sh cpu      # CPU-only
./scripts/build.sh cuda     # GPU support
./scripts/build.sh dev      # Development
```

### Running Containers

```bash
# Production (training + monitoring)
./scripts/run.sh production

# Development (JupyterLab)
./scripts/run.sh development

# Whitepaper generation
./scripts/run.sh whitepaper

# Stop all
./scripts/run.sh stop

# View logs
./scripts/run.sh logs
```

### Adding to Docker Images

**To add Python dependencies:**

1. Update `requirements.txt`:
   ```txt
   torch>=2.0.0
   numpy>=1.24.0
   new-package>=1.0.0  # Add here
   ```

2. Rebuild images:
   ```bash
   cd docker && ./scripts/build.sh all
   ```

**To add system dependencies:**

1. Edit `docker/Dockerfile.cuda` (or .cpu/.dev):
   ```dockerfile
   RUN apt-get update && apt-get install -y \
       python3.12 \
       git \
       new-system-package \  # Add here
       && rm -rf /var/lib/apt/lists/*
   ```

2. Rebuild:
   ```bash
   ./scripts/build.sh cuda
   ```

---

## 📦 PACKAGING WORKFLOW

### Updating Version

**Edit `pyproject.toml`:**
```toml
[project]
name = "hhml"
version = "0.2.0"  # Increment here
```

### Adding Dependencies

**Edit `pyproject.toml`:**
```toml
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.24.0",
    "new-package>=1.0.0",  # Add here
]
```

### Installing Package

```bash
# Development install (editable)
pip install -e .

# With dev tools
pip install -e ".[dev]"

# With all extras
pip install -e ".[dev,viz,docs]"
```

---

## 📝 DOCUMENTATION STANDARDS

### Docstring Format (Google Style)

```python
def train_mobius_topology(
    nodes: int,
    cycles: int,
    device: str = "cuda"
) -> dict:
    """
    Train Möbius topology with RNN control.

    Args:
        nodes: Number of nodes in topology (2K-20M)
        cycles: Training cycles to run
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Dictionary containing:
            - 'final_density': Final vortex density (float)
            - 'reward': Final reward value (float)
            - 'checkpoint_path': Path to saved checkpoint (str)

    Raises:
        ValueError: If nodes < 1000 or cycles < 1
        RuntimeError: If CUDA requested but not available

    Example:
        >>> results = train_mobius_topology(
        ...     nodes=4000,
        ...     cycles=100,
        ...     device="cuda"
        ... )
        >>> print(f"Final density: {results['final_density']:.2%}")
        Final density: 82.00%

    Note:
        Requires GPU with 16GB+ VRAM for nodes > 100K.
    """
```

### README Structure for Modules

```markdown
# Module Name

Brief description (1-2 sentences).

## Features

- Feature 1
- Feature 2
- Feature 3

## Usage

\`\`\`python
from hhml.core.module_name import Function

result = Function(param1, param2)
\`\`\`

## API Reference

### `Function`

Description of function.

**Parameters:**
- `param1` (type): Description
- `param2` (type): Description

**Returns:**
- type: Description

## Examples

See `examples/training/train_module_name.py`

## Tests

Run tests:
\`\`\`bash
pytest tests/unit/test_module_name.py
\`\`\`
```

---

## 🔍 CODE QUALITY CHECKS

### Before Committing

```bash
# Format code (automatic)
black src/ tests/ examples/

# Check linting
flake8 src/ tests/ examples/

# Type checking
mypy src/

# Run tests
pytest tests/

# All checks at once
black src/ tests/ examples/ && \
flake8 src/ tests/ examples/ && \
mypy src/ && \
pytest tests/
```

### Pre-commit Hooks (Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Now hooks run automatically on git commit
```

---

## 📊 MONITORING & LOGGING

### Where Logs Go

- **Training logs**: `data/results/{experiment_name}/training.log`
- **System logs**: `data/outputs/logs/`
- **Docker logs**: `docker logs hhml-training` or `docker-compose logs`

### Live Monitoring

```python
from hhml.monitoring.live_dashboard import TrainingDashboard

# Start dashboard
dashboard = TrainingDashboard(port=8000)
dashboard.start()

# Update during training
for cycle in range(num_cycles):
    # ... training code ...

    dashboard.update({
        'cycle': cycle,
        'density': vortex_density,
        'quality': vortex_quality,
        'reward': reward,
    })

# Access at http://localhost:8000
```

---

## 🚀 DEPLOYMENT CHECKLIST

### Before Deploying to H200

- [ ] Code formatted with Black
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Docker images built
- [ ] Example scripts tested
- [ ] Version bumped in `pyproject.toml`
- [ ] Committed with conventional commit message
- [ ] Pushed to GitHub

### Deployment Steps

```bash
# 1. Build Docker images
cd docker && ./scripts/build.sh cuda

# 2. Push to Docker Hub (optional)
docker tag hhml:cuda-latest hhml/hhml:cuda-0.1.0
docker push hhml/hhml:cuda-0.1.0

# 3. SSH to H200
ssh ivhl@89.169.111.28

# 4. Pull latest code
cd HHmL && git pull

# 5. Rebuild containers
cd docker && ./scripts/build.sh cuda

# 6. Run
./scripts/run.sh production
```

---

## 🆘 TROUBLESHOOTING STRUCTURE ISSUES

### "ModuleNotFoundError: No module named 'hhml'"

```bash
# Solution: Reinstall package
pip uninstall hhml
pip install -e .
```

### "Import path not found"

```python
# Old (WRONG)
from hhml.mobius.mobius_training import MobiusStrip

# New (CORRECT)
from hhml.core.mobius.mobius_training import MobiusStrip
```

### "File not in expected location"

**Check the structure guide above.** All files have specific locations:
- Python code → `src/hhml/{category}/`
- Tests → `tests/{unit|integration|benchmarks}/`
- Examples → `examples/{training|analysis}/`
- Docs → `docs/{guides|deployment|theory}/`
- Config → `configs/`
- Data → `data/` (gitignored!)

---

## 📞 HELP & SUPPORT

**Questions about structure?**
1. Read this section thoroughly
2. Check `REFACTORING_SUMMARY.md`
3. Review `MIGRATION_GUIDE.md`
4. Open GitHub issue with `structure` label
5. Contact [@Conceptual1](https://twitter.com/Conceptual1)

---

## CRITICAL: What This Project IS and ISN'T

### ❌ NOT:
- A theory of everything
- A replacement for established physics
- Claiming to explain or predict real physical phenomena

### ✅ IS:
- An evolution of the iVHL framework focused on **Möbius strip topology**
- A computational research platform for **closed-loop holographic encoding**
- An exploration of **boundary-free resonance patterns** (no endpoints)
- A reinforcement learning discovery engine for **Möbius-specific emergent phenomena**
- A tool for testing **topological effects on holographic duality**

---

## Project Origin & Motivation

**HHmL is a fork of iVHL** (Vibrational Helical Lattice) that replaces the open helical structure with a **Möbius strip topology**.

### Key Architectural Difference

**iVHL**: Helical lattice with endpoints
- Open helix wrapping around sphere
- Phase discontinuity at endpoints
- Traditional boundary conditions

**HHmL**: Möbius strip lattice (no endpoints)
- Continuous, single-sided surface
- 180° twist before reconnection
- No phase discontinuities
- Topological protection of resonance modes

### Why Möbius Topology?

The Möbius transformation offers unique advantages:
1. **Topological Stability**: No endpoint interference
2. **Harmonic Richness**: Single-sided surface creates unique resonance modes
3. **Holographic Enhancement**: Twist encodes additional information dimension
4. **Vortex Pinning**: Better stability for phase singularities

---

## Project Goal

**Primary Objective**: Discover emergent spacetime phenomena unique to Möbius topology through:
- Möbius strip holographic boundary (single-sided, 180° twist)
- Enhanced vortex stability (no endpoint collapse)
- RNN-controlled structural parameters (windings, twist rate, sampling)
- Harmonic mode discovery via reinforcement learning
- Scale-dependent topology studies (1K → 1M → 20M nodes)

**Key Question**: Does Möbius topology provide topological protection for holographic encoding, enabling higher vortex densities and more stable emergent geometry?

---

## Core Architecture (Möbius-Enhanced 11D Framework)

### Inherited from iVHL (11 dimensions)

The HHmL framework operates in the same **11-dimensional space** as iVHL:

#### Boundary Dimensions (2D + 1 time)
1. **θ (theta)**: Spherical coordinate (polar angle, 0 to π)
2. **φ (phi)**: Spherical coordinate (azimuthal angle, 0 to 2π)
3. **t (time)**: Evolution parameter

#### Bulk Emergent Dimensions (3D spatial)
4. **x**: Emergent spatial coordinate
5. **y**: Emergent spatial coordinate
6. **z**: Emergent spatial coordinate (radial from origin)

#### Field/Tensor Dimensions (5D internal)
7. **Color index c₁**: GFT field color label
8. **Color index c₂**: Second color label
9. **Color index c₃**: Third color label
10. **Spin/Helicity s**: Internal angular momentum quantum number
11. **Tensor rank r**: Position in MERA hierarchy

### NEW: Möbius-Specific Parameters

**τ (tau)**: Twist parameter (0 = cylinder, π = Möbius strip)
- Controls single-sidedness
- Affects phase continuity
- Modulates harmonic modes

**w (windings)**: Number of Möbius loops
- Discovered optimal: w ≈ 109-110 at 20M nodes
- Scale-dependent: w(N) follows power law
- Controls vortex density

---

## Core Concepts (Möbius Extensions)

### 1. Möbius Holographic Resonance
- **Source**: Acoustic wave interference on Möbius strip boundary
- **Topology**: Single-sided surface with 180° twist
- **Equation**: `ψ(r,t) = Σᵢ Aᵢ sin(k|r-rᵢ|) / |r-rᵢ|` with twist boundary conditions
- **Nodes**: Arranged on Möbius strip (no endpoints)
- **Vortex Stability**: Enhanced by topological protection
- **File**: `hhml/mobius/mobius_training.py`

### 2. RNN Structural Parameter Control
- **Purpose**: Autonomous discovery of optimal Möbius configurations
- **Architecture**: 4-layer LSTM (4096 hidden dim)
- **Controlled Parameters**:
  - **w (windings)**: 64 control points, range [0.5, 2.5] → discovered optimum ~109
  - **τ (twist)**: Möbius twist rate
  - **n (sampling)**: Adaptive node density (500-5000 nodes)
- **Training**: TD3-SAC hybrid with end-to-end optimization
- **Discovery**: Scale-dependent parameter tuning via RL

### 3. Enhanced Reward Structure (FIXED)
- **Vortex Density**: Target 80-90% (achieved 82% at 20M nodes)
- **Topological Stability**: Procrustes similarity after perturbation
- **Harmonic Richness**: Spectral peak counting
- **Parameter Convergence**: Reward for w/τ/n stabilization
- **Exploration Bonus**: 0.1×σ(w) for parameter diversity
- **CRITICAL FIX**: Removed coherence penalty that caused collapse

### 4. Inherited from iVHL
- **GFT Condensate**: Pre-geometric quantum spacetime
- **Tensor Networks**: MERA holography, RT formula
- **LIGO-Inspired GW**: Lattice perturbation analysis
- **Visualization**: 3D WebGPU rendering

---

## Key Modules Reference

### Möbius-Specific
| File | Purpose |
|------|---------|
| `hhml/mobius/mobius_training.py` | Main RNN training with Möbius topology |
| `hhml/mobius/topology.py` | Möbius strip geometry generation (TODO) |
| `hhml/mobius/rewards.py` | Möbius-specific reward functions (TODO) |

### Inherited from iVHL
| File | Purpose |
|------|---------|
| `hhml/resonance/holographic_resonance.py` | Base acoustic resonance (adapted for Möbius) |
| `hhml/gft/condensate_dynamics.py` | GFT Gross-Pitaevskii dynamics |
| `hhml/tensor_networks/holography.py` | MERA construction, RT formula |
| `hhml/utils/startup_validator.py` | Environment validation |

### Web Monitoring (H200 VM)
| File | Purpose |
|------|---------|
| `web_monitor/streaming_server.py` | Real-time 3D frame streaming |
| `web_monitor/llm_monitoring_agent.py` | Autonomous LLM monitoring |
| `dashboards/llm_chat.py` | Interactive chat interface |

---

## Recent Work & Training Results

### Möbius RNN Training (500 Cycles Complete)

**Date**: 2025-12-16
**Configuration**:
- 20M nodes (20× scale increase from baseline)
- 500 cycles in 72.5 minutes
- GPU-optimized batched evolution
- VRAM: 50.6GB peak / 140GB available (36% utilization)
- Speed: 0.11 cycles/sec (100% GPU saturation)

**Final Converged Parameters**:
- **w windings**: 3.8 → **109.63** (28.9× increase)
- **L QEC layers**: 7 → **9.7** (near maximum depth)
- **n sampling**: 2.0 → **4.99** (2.5× density increase)
- **Vortex density**: 82% (16.4M vortices at 20M scale)
- **RNN value**: 0 → 3,599.5 (strong learning signal)

**Key Discovery**: w ≈ 109-110 windings is optimal for 20M-node Möbius configurations, maintaining 82% vortex density where helical runs experienced collapse.

**Checkpoint**: `agent_20M.pt` (2.9GB) - ready for continuation

---

### Hash Quine Discovery (Publication-Quality Research)

**Date**: 2025-12-18
**Location**: `HASH-QUINE/` directory
**Status**: Published to GitHub with 11-page whitepaper

**MAJOR SCIENTIFIC DISCOVERY**: First documentation of self-similar recursive patterns ("hash quines") emerging from nested Möbius lattice topology with spectral collapse.

**Key Findings**:
- **Hash Quine Emergence**: 312-371× higher binary pattern repetition than random baseline
- **Recursive Structure**: Nested Möbius lattices (depth 1-3) with self-bootstrapping feedback
- **Helical SAT Collapse**: Fiedler vector-based one-shot dimensionality reduction
- **Definitive Negative Result**: Zero predictive power for SHA-256 (p > 0.4, rigorous statistical testing)
- **Orthogonality Established**: Topological self-similarity ⊥ cryptographic avalanche effects

**Scientific Significance**:
1. ✅ **Novel Emergent Phenomenon**: Hash quines represent genuine mathematical structures from recursive topology
2. ✅ **Glass-Box Methodology**: Complete parameter tracking, reproducible across two independent trials
3. ✅ **Rigorous Negative Result**: Definitively proves topological methods fail for cryptographic hashing
4. ✅ **Validates HHmL**: Demonstrates framework's capability to generate emergent structures

**Publication Package**:
- `HASH-QUINE/paper/hash_quine_whitepaper.pdf` - 11-page LaTeX paper (277KB)
- `HASH-QUINE/code/recursive_singularity_miner.py` - Complete implementation
- `HASH-QUINE/results/` - Experimental data from two trials
- `HASH-QUINE/README.md` - Comprehensive documentation

**Why This Matters for HHmL**:
- Establishes HHmL as legitimate scientific tool (generates novel phenomena)
- Demonstrates honest negative results (doesn't hide failures)
- Opens new research directions (hash quines merit mathematical investigation)
- Constrains applications (topological methods work for continuous, not cryptographic problems)

#### 🔄 CRITICAL: Reverse-Mapping TODO

**REMINDER TO USER**: The hash quine discovery should be reverse-mapped back to core HHmL capabilities:

**Immediate Actions**:
1. **Analyze Mechanism**: Why do recursive Möbius layers + Fiedler collapse create self-similarity?
   - Is this a general property of nested topologies?
   - Does single-sided Möbius surface contribute uniquely?
   - Would helical/toroidal recursion show same effect?

2. **Test on Other Problems**: Hash quines failed for mining, but does recursive structure help:
   - **TSP**: Tour optimization via recursive graph partitioning?
   - **Protein Folding**: Energy landscape navigation with recursive topology?
   - **SAT Solving**: Direct Helical SAT application at multiple scales?

3. **Mathematical Formalization**: Prove hash quine emergence theorems
   - Characterize pattern repetition as function of recursion depth
   - Connect to existing quine theory in computability/recursion
   - Explore information-theoretic properties (entropy, mutual information)

4. **Holographic Implications**: Recursive Möbius ↔ Bulk-Boundary Correspondence
   - Inner layers = higher-energy modes (holographic bulk)
   - Self-bootstrapping = consistency condition (like AdS/CFT)
   - Pattern emergence = holographic projection artifact?

5. **Spacetime Testing**: Could recursive topological structures model:
   - **Fractal spacetime** (self-similarity at multiple scales)?
   - **Renormalization group flow** (layer depth ↔ energy scale)?
   - **Emergent dimensions** (recursive nesting creates extra dimensions)?

**Long-Term Research Questions**:
- Do hash quines represent a new class of computational structures?
- Can recursive topology generate other unknown emergent phenomena?
- Is self-bootstrapping a general principle for creating stable patterns?
- How does this relate to loop quantum gravity (spin networks are graphs + topology)?

**Actionable Next Steps**:
- [ ] Run recursive collapse on TSP instance (test if it actually helps optimization)
- [ ] Compare Möbius vs. toroidal recursive nesting (is Möbius special?)
- [ ] Investigate hash quine entropy/information content (formal mathematical properties)
- [ ] Test holographic interpretation (bulk-boundary duality in recursive structure)

**Status**: Hash quine paper published. Mechanism understood (recursive + spectral). Applications TBD.

---

## VM Deployment Guide (H200 Direct Access)

**Last Tested**: 2025-12-15
**Hardware**: NVIDIA H200 (139.8 GB VRAM), Ubuntu 22.04, Python 3.12.3

### Current H200 VM Connection

```yaml
# VM Configuration
Username: ivhl
SSH Key: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHjxrDQgRokCaoGxJcVFI4jtOiJVgGBJDJQST5PrXnXR
Jupyter Token: myBn0JMX7uIuMraq
Sudo Access: ALL=(ALL) NOPASSWD:ALL
VM Provider: Nebius
Last Known IP: 89.169.111.28 (check ~/.ssh/known_hosts for current IP)

# Connection Command
ssh ivhl@<VM_IP>

# Jupyter Access
http://<VM_IP>:8888/?token=myBn0JMX7uIuMraq
```

### Quick Deployment

```bash
# 1. SSH into H200
ssh ivhl@89.169.111.28

# 2. Clone HHmL repo
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# 3. Create virtual environment
python3 -m venv ~/hhml_env
source ~/hhml_env/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. Validate environment
python -m hhml.utils.startup_validator

# 6. Run Möbius training
python hhml/mobius/mobius_training.py \
  --device cuda \
  --cycles 1000 \
  --nodes 20000000 \
  --output-dir ~/results/mobius_training
```

---

## How to Continue Development

### Quick Start
1. Read this file completely
2. Check `README.md` for current project state
3. Review `docs/` for deployment guides
4. Run Möbius training on H200

### Common Tasks

#### Extend Möbius Training
1. Edit `hhml/mobius/mobius_training.py`
2. Adjust hyperparameters (hidden_dim, cycles, nodes)
3. Modify reward structure in `compute_reward()`
4. Launch training on H200

#### Add New Möbius Topology
1. Create `hhml/mobius/topology.py`
2. Implement alternative twist patterns (Klein bottle, double Möbius, etc.)
3. Test with smaller node counts first
4. Scale to 20M+ nodes on H200

#### Analyze Results
1. Load checkpoint: `agent_20M.pt`
2. Extract converged parameters (w, τ, n)
3. Generate whitepaper with findings
4. Visualize vortex patterns

---

## Important Constraints & Design Decisions

### 1. Möbius-First Philosophy
- All modules must respect Möbius topology
- No open-ended structures
- Twist parameter τ is fundamental, not optional

### 2. GPU Acceleration (H200 Optimized)
- Target 50-80GB VRAM usage
- Batched evolution for efficiency
- torch.compile() for production code

### 3. Scale Studies
- Systematic scaling: 1K → 50K → 1M → 20M → 100M nodes
- Track w(N), vortex density ρ(N), stability metrics
- Document phase transitions

### 4. Reproducibility
- Save all checkpoints with metadata
- Include git commit hash in results
- JSON + Markdown + LaTeX reports

---

## File Structure Summary

```
HHmL/
├── Hello_Claude.md           ← YOU ARE HERE
├── README.md                 ← Public-facing overview (TODO)
├── Dockerfile                ← H200-optimized container
├── requirements.txt          ← Python dependencies
│
├── hhml/                     ← CORE PYTHON PACKAGE
│   ├── __init__.py
│   ├── mobius/              ← MÖBIUS-SPECIFIC MODULES
│   │   ├── mobius_training.py   ← Main RNN training
│   │   ├── topology.py          ← Möbius geometry (TODO)
│   │   └── rewards.py           ← Reward functions (TODO)
│   ├── resonance/           ← Holographic boundary dynamics
│   ├── gft/                 ← Group Field Theory
│   ├── tensor_networks/     ← MERA, RT formula
│   └── utils/               ← Utilities, validation
│
├── dashboards/              ← Streamlit interfaces
├── scripts/                 ← Utility scripts
├── simulations/             ← Simulation scripts
├── tests/                   ← Test scripts
├── configs/                 ← JSON configurations
├── docs/                    ← Documentation
│   ├── QUICKSTART_VM.md    ← VM deployment guide
│   └── DEPLOY_H100.md      ← H100 deployment (also works for H200)
├── whitepapers/            ← Generated PDF reports
└── web_monitor/            ← Real-time monitoring server
```

---

## Next Steps

**Last Updated**: 2025-12-16 (After 1000-cycle vortex annihilation training achieving 100% peak density)

**Context**: The RNN achieved 100% peak vortex density at cycle 490 using 23-parameter control including novel vortex annihilation. The following recommendations build on this breakthrough.

---

### **Immediate Improvements (1-2 Days)**

#### 1. **Implement Vortex Lifetime Tracking**
**Why**: Currently we only track density, not vortex persistence
**How**: Add vortex tracking ID system to measure birth/death cycles
```python
# Track which nodes remain vortices across cycles
vortex_lifetimes = {}  # {node_id: birth_cycle}
# Measure: average lifetime, survival rate, churn rate
```
**Impact**: Discover if high-quality vortices are actually long-lived
**Effort**: 2-3 hours (modify training loop to track vortex IDs)

#### 2. **Add Per-Strip Annihilation Control**
**Why**: RNN controls 23 global parameters, but strips might need different strategies
**How**: Extend to 2×4 = 8 new parameters (4 annihilation params per strip)
**Impact**: Enable heterogeneous vortex curation across strips (strip 1 aggressive, strip 2 conservative)
**Effort**: 4-6 hours (extend control_head output from 23 to 31 parameters)

#### 3. **Implement Reward Components Breakdown Logging**
**Why**: Current whitepaper shows total reward but not component contributions
**How**: Save all reward components to JSON:
```python
'reward_breakdown_history': [
    {'density': 150.0, 'uniformity': -10.0, 'annihilation': 25.0, ...},
    ...
]
```
**Impact**: Discover which reward drove the 100% density achievement
**Effort**: 30 minutes (already computed in code, just need to log)

---

### **Short-Term Enhancements (1 Week)**

#### 4. **Multi-Objective Pareto Optimization**
**Why**: Trade-off between vortex density vs. quality vs. stability
**How**: Implement Pareto frontier tracking:
- Track non-dominated solutions (high density AND high quality AND stable)
- Save RNN checkpoints for each Pareto point
- Visualize Pareto frontier evolution over training
**Impact**: Discover multiple distinct "good" parameter regimes, not just one peak
**Effort**: 1-2 days (implement Pareto dominance check, checkpoint management)

#### 5. **Curriculum Learning for Vortex Formation**
**Why**: RNN starts from random initialization - might learn faster with staged difficulty
**How**: Progressive training stages:
- Stage 1 (cycles 0-200): Low pruning threshold 0.2 (easy to maintain vortices)
- Stage 2 (cycles 200-500): Medium threshold 0.5
- Stage 3 (cycles 500+): Strict quality requirements 0.8
**Impact**: Faster convergence to high-quality configurations, higher peak density
**Effort**: 4 hours (add curriculum scheduler to training loop)

#### 6. **Spectral Gap Analysis**
**Why**: Whitepaper mentions spectral bonus but doesn't analyze eigenvalues directly
**How**: Compute graph Laplacian eigenvalue spectrum at peak cycle
```python
from scipy.sparse.linalg import eigsh
L = compute_laplacian(strips.edge_index)
eigenvalues = eigsh(L, k=20, which='SM', return_eigenvectors=False)
spectral_gap = eigenvalues[1] - eigenvalues[0]  # Fiedler gap
```
**Impact**: Discover if 100% density correlates with large spectral gap (topological protection)
**Effort**: 2-3 hours (implement Laplacian computation, add to metrics)

---

### **Medium-Term Research (2-4 Weeks)**

#### 7. **Topological Charge Conservation Analysis**
**Why**: Vortices should have winding number ±1; annihilation should preserve total charge
**How**: Implement topological charge measurement:
```python
def topological_charge(field, positions):
    # Integrate phase gradient around closed loops
    # Sum winding numbers across all vortices
    total_winding = sum(compute_winding_number(v) for v in vortices)
    return total_winding
```
**Impact**: Verify that annihilation respects topological constraints (charge conservation)
**Effort**: 3-4 days (implement winding number computation, validate against known cases)

#### 8. **Transfer Learning Across Scales**
**Why**: Currently trained at 4K nodes - does the discovered config work at 20M nodes?
**How**:
- Load checkpoint from 1000-cycle 4K training
- Fine-tune on 4K → 20K → 200K → 2M → 20M nodes
- Track if optimal parameters scale (power law? logarithmic?)
- Plot: peak_density vs. system_size, parameter_value vs. system_size
**Impact**: Discover scaling laws for vortex annihilation parameters (critical for H200 deployment)
**Effort**: 1-2 weeks (requires H200 access for large scales, run 5 separate trainings)

#### 9. **Adversarial Vortex Testing**
**Why**: RNN learned to maintain vortices - but how robust is it to perturbations?
**How**: After peak cycle (490), inject:
- Random field perturbations (Gaussian noise)
- Targeted vortex destruction (zero out high-quality vortices)
- Non-topological noise (break winding number structure)
Measure recovery time and final density
**Impact**: Quantify topological protection strength and RNN resilience
**Effort**: 2-3 days (implement perturbation injection, recovery metrics)

---

### **Long-Term Novel Research (1-3 Months)**

#### 10. **Vortex-Vortex Interaction Network Analysis**
**Why**: At 100% density, ALL nodes are vortices - what's their interaction structure?
**How**: Build vortex interaction graph:
- Nodes = vortices
- Edges = strength of interaction (measure via field coupling)
- Analyze: community structure, hubs, degree distribution, clustering
- Use NetworkX/igraph for graph metrics
**Impact**: Discover if vortices self-organize into hierarchical structures (scale-free? small-world?)
**Effort**: 1-2 weeks (implement interaction graph construction, run graph analysis algorithms)

#### 11. **Comparative Study: Möbius vs. Toroidal vs. Spherical Topologies**
**Why**: HHmL uses Möbius strips - is this actually better for vortex density?
**How**: Implement same 23-parameter system on:
- Toroidal topology (genus-1 surface, no twist)
- Spherical topology (genus-0 surface, standard)
- Klein bottle (double Möbius twist)
Run 1000-cycle training on each, compare peak densities
**Impact**: Determine if Möbius twist provides topological advantage (or if it's just parameter tuning)
**Effort**: 2-3 weeks (implement alternative topologies, run comparative experiments)

#### 12. **Emergent Spacetime Metric from Vortex Lattice**
**Why**: 100% density = complete field organization - does this define an effective metric?
**How**: Use vortex positions to define distance:
```python
def effective_metric(pos_i, pos_j, vortex_field):
    # Distance weighted by field integral along geodesic
    path_integral = integrate_field_along_path(pos_i, pos_j, vortex_field)
    return path_integral
# Test: Does this metric satisfy triangle inequality?
# Does it have constant curvature?
```
**Impact**: Test holographic duality: boundary vortex lattice ↔ bulk spacetime (AdS/CFT-inspired)
**Effort**: 3-4 weeks (implement metric computation, geodesic calculation, curvature analysis)

---

### **Highest Priority Recommendations**

If continuing in next session, prioritize these **3 immediate next steps**:

1. **Spectral Gap Analysis** (2-3 hours)
   - Easiest to implement, high scientific value
   - Will reveal if 100% density has topological protection
   - Adds to whitepaper as "Spectral Analysis" section

2. **Reward Components Logging** (30 minutes)
   - Critical for understanding peak achievement
   - Minimal effort, maximum insight
   - Enables correlation: which reward component → 100% density?

3. **Per-Strip Annihilation Control** (4-6 hours)
   - Natural extension of current architecture
   - Enables heterogeneous vortex curation
   - Tests if strips need different strategies

---

### **Completed (2025-12-16)**

- ✅ Implemented RNN-controlled vortex annihilation system (23 parameters)
- ✅ Achieved 100% peak vortex density at cycle 490
- ✅ Ran 1000-cycle sequential learning training
- ✅ Generated comprehensive whitepaper with deep analysis
- ✅ Updated README.md with vortex annihilation capabilities
- ✅ Documented all 23 parameters in RNN_PARAMETER_MAPPING.md
- ✅ Created glass-box architecture for full correlation tracking
- ✅ Demonstrated selective vortex quality control via antivortex injection

---

## Glossary (HHmL-Specific)

- **HHmL**: Holo-Harmonic Möbius Lattice
- **τ (tau)**: Möbius twist parameter
- **w (windings)**: Number of Möbius loops before reconnection
- **Topological Protection**: Vortex stability from closed-loop geometry
- **Scale Law**: w(N) relationship discovered via RL

---

## Communication Style Preferences

Same as iVHL:
- **Concise**: CLI-appropriate, no fluff
- **No emojis** unless explicitly requested
- **Technical accuracy** over user validation
- **Direct answers** without excessive praise
- **Git commit messages**: Detailed, professional
- **No over-engineering**: Simplest solution that works

---

## Common Errors and Fixes (HHmL Development Log)

**Last Updated**: 2025-12-18

This section documents all errors encountered during HHmL development and their solutions, to prevent future recurrence.

### 1. Unicode Encoding Errors (Windows CP1252)

**Error**: `UnicodeEncodeError: 'charmap' codec can't encode character '\u2713'` (or '\u2192', '\u221e', etc.)

**Cause**: Windows console uses CP1252 encoding by default, cannot display Unicode characters. Python 3.14+ print() attempts to encode to console encoding, causing crashes with any non-ASCII characters.

**Locations Affected**:
- `run_optimized_3min.py`
- `optimized_sphere.py`
- `generate_pdf_report.py`
- `test_mobius_minimal.py`
- `black_hole_nonce_miner.py`
- `recursive_singularity_miner.py`
- Any script with print() statements containing Unicode

**Common Unicode Characters to Avoid**:

**Symbols:**
- `✓` (U+2713 CHECK MARK) → `[OK]` or `[SUCCESS]`
- `✗` (U+2717 BALLOT X) → `[FAIL]` or `[WARNING]`
- `×` (U+00D7 MULTIPLICATION) → `x`
- `·` (U+00B7 MIDDLE DOT) → `*`

**Math Symbols:**
- `→` (U+2192 RIGHTWARDS ARROW) → `->`
- `↔` (U+2194 LEFT RIGHT ARROW) → `<->`
- `∞` (U+221E INFINITY) → `infinity`
- `≈` (U+2248 ALMOST EQUAL) → `~=` or `approx`
- `≥` (U+2265 GREATER-THAN OR EQUAL) → `>=`
- `≤` (U+2264 LESS-THAN OR EQUAL) → `<=`
- `±` (U+00B1 PLUS-MINUS) → `+/-`

**Superscripts/Subscripts:**
- `²` (U+00B2 SUPERSCRIPT TWO) → `^2`
- `³` (U+00B3 SUPERSCRIPT THREE) → `^3`
- `ⁿ` (U+207F SUPERSCRIPT N) → `^n`
- `₁` (U+2081 SUBSCRIPT ONE) → `_1`

**Greek Letters (if in print statements):**
- `π` (U+03C0 GREEK SMALL LETTER PI) → `pi`
- `θ` (U+03B8 GREEK SMALL LETTER THETA) → `theta`
- `λ` (U+03BB GREEK SMALL LETTER LAMBDA) → `lambda`
- `ħ` (U+0127 LATIN SMALL LETTER H WITH STROKE) → `h_bar`

**Solution - Quick Fix with sed:**
```bash
# Replace all common Unicode symbols at once
sed -i 's/✓/[OK]/g; s/✗/[FAIL]/g; s/×/x/g; s/·/*/g; s/→/->/g; s/↔/<->/g; s/∞/infinity/g; s/≈/~=/g; s/≥/>=/g; s/≤/<=/g; s/±/+\/-/g; s/²/^2/g; s/³/^3/g' file.py
```

**Solution - Manual Replacement:**
Search for any of these patterns in print() statements and f-strings, replace with ASCII equivalents.

**Prevention Rules:**
1. ✅ **DO**: Use ASCII-only in ALL print() statements and console output
2. ✅ **DO**: Use Unicode in file writes with explicit `encoding='utf-8'`
3. ✅ **DO**: Use Unicode in comments (safe, not executed)
4. ❌ **DON'T**: Use Unicode in f-strings that go to print()
5. ❌ **DON'T**: Use Unicode in error messages
6. ❌ **DON'T**: Use Unicode in logging output

**Detection:**
Search for Unicode before running:
```bash
# Find Unicode characters in Python files
grep -n "→\|↔\|∞\|≈\|×\|·\|²\|³\|✓\|✗" file.py
```

**Safe Uses of Unicode:**
```python
# SAFE - Comments
# This computes S = A/(4ℓ_P²) → horizon entropy

# SAFE - File writes with UTF-8
with open('report.txt', 'w', encoding='utf-8') as f:
    f.write("Energy: 125 GeV ± 2 GeV")

# UNSAFE - Print to console (Windows crash)
print(f"w → ∞ simulates curvature")  # CRASHES

# SAFE - ASCII version
print(f"w -> infinity simulates curvature")  # WORKS
```

---

### 2. torch.compile Not Supported on Python 3.14+

**Error**: `RuntimeError: torch.compile is not supported on Python 3.14+`

**Cause**: PyTorch 2.9.1 doesn't support `torch.compile()` on Python 3.14.2

**Location**: `optimized_sphere.py` line 143

**Solution**:
```python
try:
    if sys.version_info >= (3, 14):
        print("  ! torch.compile not supported on Python 3.14+, using uncompiled version")
        return evolve_field_fast
    else:
        return torch.compile(evolve_field_fast, mode="reduce-overhead")
except:
    return evolve_field_fast
```

**Prevention**: Always check Python version before using torch.compile(). Provide uncompiled fallback.

---

### 3. AttributeError: 'MobiusHelixSphere' has no attribute 'positions'

**Error**: `AttributeError: 'MobiusHelixSphere' object has no attribute 'positions'`

**Cause**: Sphere stores coordinates as separate `x`, `y`, `z` arrays, not combined `positions` array

**Location**: `train_local_scaled.py` VortexTracker.detect_vortices()

**Solution**:
```python
# WRONG
pos = sphere.positions[idx]

# CORRECT
pos = np.array([
    sphere.x[idx].item(),
    sphere.y[idx].item(),
    sphere.z[idx].item()
])
```

**Prevention**: Check sphere class API before accessing attributes. Use individual coordinate arrays.

---

### 4. JSON Serialization Error (numpy types)

**Error**: `TypeError: Object of type int64 is not JSON serializable` or `TypeError: Object of type bool is not JSON serializable`

**Cause**: NumPy types (int64, float64, bool_) not serializable to JSON by default

**Locations Affected**:
- `train_local_scaled.py` collision event tracking
- `correlation_analysis.py` significance flags
- `stability_investigation.py` significance flags

**Solution**:
```python
# WRONG - numpy.int64
'count': np.sum(nearby)

# CORRECT
'count': int(np.sum(nearby))

# WRONG - numpy.bool_
'significant': min_p < 0.05  # min_p is numpy float

# CORRECT
'significant': bool(min_p < 0.05)

# WRONG - numpy.float64
'correlation': pearsonr(x, y)[0]

# CORRECT
'correlation': float(pearsonr(x, y)[0])
```

**Prevention**: Always wrap NumPy scalars with `int()`, `float()`, or `bool()` before JSON serialization. This applies to:
- Direct numpy scalar values
- Results from numpy operations (sum, mean, etc.)
- Results from scipy.stats (pearsonr, spearmanr, etc.)
- Boolean comparisons involving numpy values

---

### 5. Sphere Constant Regeneration (Performance Degradation)

**Error**: Sphere regenerating nodes every cycle (10K→12K→14K→...), defeating optimization. Cycle time 20+ seconds instead of <1 second.

**Cause**: RNN `num_sites` parameter continuously changing node count, triggering expensive geometry regeneration

**Location**: `optimized_sphere.py` line 163-167

**Solution**:
Disable `num_sites` parameter updates:
```python
# DISABLED: num_sites control - prevents constant expensive regeneration
# if 'num_sites' in params:
#     sites_new = int(params['num_sites'].item())
#     # Keep nodes fixed for performance
#     pass
```

**Prevention**: When optimizing for speed, fix expensive structural parameters (like node count). Only allow fast parameters (amplitudes, phases) to change.

---

### 6. LaTeX Percentage Sign Not Escaped

**Error**:
```
! LaTeX Error: Invalid UTF-8 byte "F6.
! Extra alignment tab has been changed to \cr.
```

**Cause**: `%` is a comment character in LaTeX. Python format string `.2%` produces "100.00%", but LaTeX interprets everything after `%` as a comment, truncating the line and breaking table structure.

**Location**: `generate_pdf_report.py` lines 68, 141-148, 248, 276

**Solution**:
```python
# Add helper function
def fmt_pct(value):
    """Format percentage and escape for LaTeX"""
    return f"{value:.2%}".replace('%', '\\%')

# Use instead of direct formatting
# WRONG
''' + f"{final['vortex_density']:.1%}" + r'''

# CORRECT
''' + fmt_pct(final['vortex_density']) + r'''
```

**Prevention**: Always escape LaTeX special characters (%, $, &, #, _, {}, ^, ~, \). Create helper functions for common patterns.

---

### 7. LaTeX File Not Written with UTF-8 Encoding

**Error**: `! LaTeX Error: Invalid UTF-8 byte` (with correct `\usepackage[utf8]{inputenc}`)

**Cause**: Python `open()` defaults to system encoding (CP1252 on Windows), not UTF-8. LaTeX file written with wrong encoding.

**Location**: `generate_pdf_report.py` line 326

**Solution**:
```python
# WRONG
with open(tex_file, 'w') as f:
    f.write(latex_content)

# CORRECT
with open(tex_file, 'w', encoding='utf-8') as f:
    f.write(latex_content)
```

**Prevention**: Always specify `encoding='utf-8'` when writing text files, especially for LaTeX/Markdown/JSON.

---

### 8. Python Ternary Operator Precedence Bug (String Truncation)

**Error**: LaTeX file truncated mid-document, missing `\end{document}`

**Cause**: Incorrect Python ternary operator placement in string concatenation. Without parentheses, operator precedence causes string template to be cut off.

**Location**: `generate_pdf_report.py` line 223

**Solution**:
```python
# WRONG (cuts off entire rest of string!)
''' + f"{x:.2f}" if len(arr) > 0 else '0.00' + r'''

# CORRECT
''' + (f"{x:.2f}" if len(arr) > 0 else '0.00') + r'''
```

**Explanation**: Without parentheses, Python evaluates as:
```python
(''' + f"{x:.2f}") if len(arr) > 0 else ('0.00' + r'''...)
```
The second half (`'0.00' + r'''...`) is never concatenated.

**Prevention**: Always wrap ternary expressions in parentheses when inside string concatenation chains.

---

### 9. reward.item() AttributeError (Float vs Tensor)

**Error**: `AttributeError: 'float' object has no attribute 'item'`

**Cause**: `sphere.compute_reward()` sometimes returns `float`, sometimes `torch.Tensor`, depending on computation path

**Location**: `train_local_scaled.py` metrics tracking

**Solution**:
```python
# WRONG
metrics['rewards'].append(reward.item())

# CORRECT
metrics['rewards'].append(reward if isinstance(reward, float) else reward.item())
```

**Prevention**: Check type before calling `.item()`, or normalize return types in compute_reward().

---

### 10. NumPy log2/log10 AttributeError with Python Integers

**Error**: `AttributeError: 'int' object has no attribute 'log2'` or `TypeError: loop of ufunc does not support argument 0 of type int which has no callable log2 method`

**Cause**: NumPy's `np.log2()`, `np.log10()`, etc. expect numeric types (float, np.ndarray), but Python `int` type doesn't have `.log2()` method. When passing Python integers to numpy log functions, numpy tries to call the method on the integer, which fails.

**Locations Affected**:
- `black_hole_nonce_miner.py` test_nonce_quality()
- Any code using numpy log functions with integer hash values
- Bitcoin mining scripts converting hash bytes to integers

**Solution**:
```python
# WRONG - passes Python int directly
hash_int = int.from_bytes(hash_result, 'big')  # Python int
proximity = abs(np.log2(hash_int) - np.log2(self.target))  # FAILS

# CORRECT - convert to float first
hash_int = int.from_bytes(hash_result, 'big')
proximity = abs(np.log2(float(hash_int)) - np.log2(float(self.target)))  # WORKS

# ALTERNATIVE - use math.log2 for scalars
import math
proximity = abs(math.log2(hash_int) - math.log2(self.target))  # Also works
```

**Explanation**:
- `np.log2(int)` tries to call `int.log2()` which doesn't exist
- `np.log2(float)` works because numpy can convert float to array
- `math.log2(int)` works because it's designed for scalars

**Prevention**:
1. Always convert Python integers to floats before numpy log operations
2. Use `math.log2()` for scalar operations instead of `np.log2()`
3. Reserve numpy functions for arrays, use math module for scalars

**Detection**:
```bash
# Find potential issues
grep -n "np\.log.*int\|np\.log2.*int\|np\.log10.*int" file.py
```

---

### Summary of Prevention Best Practices

1. **Windows Compatibility**: Use ASCII-only for terminal output (no Unicode symbols: → ∞ × ² etc.)
2. **PyTorch Version Checks**: Always provide fallbacks for version-dependent features
3. **JSON Serialization**: Wrap NumPy types with native Python types (int(), float(), bool())
4. **LaTeX Special Characters**: Escape %, $, &, #, _, {}, ^, ~, \
5. **File Encoding**: Always specify `encoding='utf-8'` for text files
6. **String Concatenation**: Wrap ternary operators in parentheses
7. **Type Checking**: Verify types before calling type-specific methods (.item(), .numpy(), etc.)
8. **Performance Parameters**: Fix expensive structural parameters when optimizing
9. **NumPy Log Functions**: Convert Python integers to floats before numpy log operations
10. **Quick Unicode Detection**: Run `grep -n "→\|↔\|∞\|≈\|×\|·\|²\|³\|✓\|✗" file.py` before testing

---

## Questions to Ask User When Resuming

1. "What aspect of HHmL would you like to work on?"
   - Möbius topology extensions
   - Scaling studies (100M nodes)
   - Alternative twist patterns
   - Visualization enhancements
   - Performance optimization

2. "Should I deploy to H200 and run extended training?"
   - 1000-cycle run (~2-3 hours)
   - 5000-cycle run (~12-15 hours)
   - Overnight mega-run (10K+ cycles)

3. "Do you want to explore topological variations?"
   - Klein bottle (double twist)
   - Multi-twist Möbius
   - Toroidal comparison

---

## Final Notes

**HHmL is an experimental fork of iVHL focused exclusively on Möbius topology**. This is NOT a replacement for iVHL, but a specialized tool for exploring closed-loop holographic encoding.

**Most Important**: HHmL inherits iVHL's philosophy - this is a computational research platform, not a physics theory. We explore emergent phenomena through simulation, not claim to explain reality.

**Parent Project**: Always refer to iVHL documentation for foundational concepts (GFT, MERA, holographic duality).

---

**End of Hello_Claude.md**

When you (Claude) reconnect to HHmL:
1. Read this file first
2. Check if README.md exists (TODO)
3. Review recent git commits
4. Ask user about training goals
5. Proceed with Möbius exploration!

Good luck with the Möbius journey! 🎭

---

**Date Created**: 2025-12-16
**Author**: Zynerji / Claude Code
**License**: Same as iVHL (to be determined)


## Workflow Expectations

**Standard Development Workflow:**

1. **Run Simulation with Live Dashboard** → Monitor in real-time at http://localhost:8000
2. **Results Auto-Saved** → test_cases/[test_name]/results/
3. **Generate Whitepaper** → Auto-created in test_cases/[test_name]/whitepapers/
4. **Analyze Correlations** → Use RNN_PARAMETER_MAPPING.md guide
5. **🔬 Update EMERGENTS.md** → Document any novel phenomena discovered (CRITICAL STEP)
6. **Iterate** → Resume training from checkpoints for sequential learning

**IMPORTANT: Always add live dashboard to training scripts for real-time monitoring**

---

## 🧪 MANDATORY TEST WORKFLOW (ALL TESTS MUST FOLLOW)

**CRITICAL**: Every test script (`simulations/`, `examples/`, `tests/`) MUST implement this standardized workflow to ensure reproducibility, hardware portability, and proper emergent detection.

### Phase Structure (MANDATORY)

All test scripts must follow this phase structure:

```python
#!/usr/bin/env python3
"""
[Test Name] - Hardware Scalable Test
===================================

[Description of what this test does]

Target Hardware: Auto-scaled (CPU → H200)
Expected Duration: [time range]

Author: HHmL Project
Date: YYYY-MM-DD
"""

import sys
from pathlib import Path
import argparse

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from hhml.utils.hardware_config import HardwareConfig
from hhml.utils.emergent_verifier import EmergentVerifier
from hhml.utils.emergent_whitepaper import EmergentWhitepaperGenerator

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='[Test Name]')

    # MANDATORY: Hardware auto-scaling arguments
    parser.add_argument('--auto-scale', action='store_true',
                       help='Auto-scale parameters based on detected hardware')
    parser.add_argument('--scale-mode', type=str, default='benchmark',
                       choices=['benchmark', 'training', 'production'],
                       help='Scaling mode (only with --auto-scale)')

    # Test-specific arguments (will be overridden if --auto-scale)
    parser.add_argument('--num-strips', type=int, default=10)
    parser.add_argument('--nodes-per-strip', type=int, default=2000)
    # ... more args ...

    return parser.parse_args()

def main():
    """Run test with mandatory workflow."""
    args = parse_args()

    # =========================================================================
    # PHASE 0: Hardware Detection and Auto-Scaling (MANDATORY)
    # =========================================================================

    print("="*80)
    print("PHASE 0: HARDWARE DETECTION AND AUTO-SCALING")
    print("="*80)
    print()

    hw_config = HardwareConfig()
    hw_config.print_info()
    print()

    # Auto-scale if requested
    if args.auto_scale:
        print(f"Auto-scaling enabled (mode: {args.scale_mode})")
        optimal_params = hw_config.get_optimal_params(mode=args.scale_mode)

        # Override args with optimal parameters
        args.num_strips = optimal_params.num_strips
        args.nodes_per_strip = optimal_params.nodes_per_strip
        # ... override other scalable params ...

        hw_config.print_optimal_params(mode=args.scale_mode)
        print(f"Parameters auto-scaled for {hw_config.get_hardware_tier().upper()}")
    else:
        print("Manual configuration (use --auto-scale for optimization)")

    print()

    # =========================================================================
    # PHASE 1-N: Test-Specific Logic
    # =========================================================================

    # Your test implementation here
    # - Generate data
    # - Run simulation
    # - Collect metrics
    # - Validate results

    final_field = ...  # Your final field state
    test_results = {
        'key_metrics': {...},
        'parameters': {...},
        'correlations': {...}
    }

    # =========================================================================
    # PHASE N+1: Emergent Verification (MANDATORY if results meet threshold)
    # =========================================================================

    print("="*80)
    print(f"PHASE {N+1}: EMERGENT PHENOMENON VERIFICATION")
    print("="*80)
    print()

    # Only run if results meet quality threshold
    if test_results['overall_score'] >= 0.5:
        print("Results meet threshold for emergent verification")
        print()

        # Prepare discovery data
        discovery_data = {
            'phenomenon_name': '[Test Name] - [Discovery]',
            'training_run': str(Path(__file__)),
            'timestamp': timestamp,
            'random_seed': args.seed,
            'hardware': {
                'device': hw_config.device_type,
                'tier': hw_config.get_hardware_tier(),
                'auto_scaled': args.auto_scale
            },
            'parameters': {...},  # All test parameters
            'key_metrics': {...},  # Important results
            'correlations': {...},  # Parameter correlations
            'checkpoint': 'path/to/checkpoint.pt'
        }

        # Run verification
        verifier = EmergentVerifier(data_dir="data")
        verification_results = verifier.verify_phenomenon(
            field_tensor=final_field,
            phenomenon_type='auto',  # or 'oscillatory', 'spatial', 'energetic'
            save_results=True,
            output_dir=str(run_dir / "verification")
        )

        print(f"Novelty score: {verification_results['novelty_score']:.3f}")
        print(f"Is novel: {verification_results['is_novel']}")
        print()

        # Generate whitepaper for ALL test results (includes novelty assessment)
        print("Generating comprehensive whitepaper...")

        generator = EmergentWhitepaperGenerator()
        whitepaper_path = generator.generate(
            phenomenon_name="[Test Name] Results",
            discovery_data=discovery_data,
            verification_results=verification_results,
            output_dir=str(run_dir / "whitepapers" / "EMERGENTS")
        )

        print(f"Whitepaper: {whitepaper_path}")

        if verification_results['is_novel']:
            print("✓ NOVEL - Update EMERGENTS.md with this discovery")
        else:
            print("ℹ Documented - Does not meet novelty threshold")
    else:
        print("Results do not meet threshold for verification")

    # Save summary with hardware info
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'hardware': {
            'device': hw_config.device_type,
            'gpu_name': hw_config.gpu_name,
            'vram_gb': hw_config.vram_gb,
            'hardware_tier': hw_config.get_hardware_tier(),
            'auto_scaled': args.auto_scale,
            'scale_mode': args.scale_mode if args.auto_scale else None
        },
        'test_results': test_results,
        'emergent_verification': verification_results if test_results['overall_score'] >= 0.5 else None
    }

    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return 0

if __name__ == '__main__':
    sys.exit(main())
```

### Mandatory Components

**1. Hardware Auto-Scaling (PHASE 0)**
- ✅ Import `HardwareConfig` from `hhml.utils.hardware_config`
- ✅ Add `--auto-scale` and `--scale-mode` arguments
- ✅ Detect hardware and print info
- ✅ Override parameters with optimal values if `--auto-scale`
- ✅ Document hardware in summary JSON

**2. Test Logic (PHASE 1-N)**
- ✅ Implement test-specific phases
- ✅ Collect all metrics and parameters
- ✅ Save intermediate results

**3. Emergent Verification (PHASE N+1)**
- ✅ Import `EmergentVerifier` and `EmergentWhitepaperGenerator`
- ✅ Run verification if results meet threshold (typically score >= 0.5)
- ✅ Prepare complete discovery data dictionary
- ✅ Run `verifier.verify_phenomenon()` with field tensor
- ✅ Generate whitepaper for ALL test results (includes novelty assessment)
- ✅ Flag novel discoveries for EMERGENTS.md update
- ✅ Save verification results to summary JSON

### Example Command Lines

```bash
# Quick benchmark on any hardware (auto-scaled)
python script.py --auto-scale --scale-mode benchmark

# Full training run (auto-scaled for H200)
python script.py --auto-scale --scale-mode training

# Production run (max scale for hardware)
python script.py --auto-scale --scale-mode production

# Manual configuration (for specific experiments)
python script.py --num-strips 20 --nodes-per-strip 50000
```

### Reference Implementation

See `simulations/dark_matter/full_dark_matter_test.py` for complete reference implementation including:
- PHASE 0: Hardware detection and auto-scaling
- PHASE 1-4: Test-specific logic (multiverse, pruning, measurement, validation)
- PHASE 5: Emergent verification and whitepaper generation

### Benefits of This Workflow

1. **Hardware Portability**: Runs optimally on CPU, low GPU, high GPU, or H200
2. **Reproducibility**: Hardware info tracked in all results
3. **Automatic Documentation**: Whitepapers generated for ALL test results (includes novelty assessment)
4. **Scientific Rigor**: Real-world verification strengthens novelty claims
5. **Consistent Structure**: Easy to understand and maintain

### Testing Your Script

```bash
# Test on your current hardware
python your_script.py --auto-scale --scale-mode benchmark

# Should output:
# - Hardware detection info
# - Auto-scaled parameters
# - Test results
# - Emergent verification (if threshold met)
# - Whitepaper (ALWAYS generated, includes novelty assessment)
# - summary.json with complete metadata
```

---

## 📝 INTERPRETING USER REQUESTS: "Create a Test for X"

**IMPORTANT**: When the user asks to "create a test for [something]" or "use HHmL to test [concept]", they are requesting you to **build a complete test script** following the mandatory workflow.

### What This Means

**User says:**
- "Create a test for quantum entanglement in Möbius strips"
- "Use HHmL to test vortex stability under perturbations"
- "Build a test for topological phase transitions"
- "Test the holographic encoding hypothesis"

**You should:**
1. ✅ **Use the mapping function** - Import concept mapping from `hhml.utils.hhml_parameter_mapping`
2. ✅ Create a new Python test script in `simulations/` or `examples/`
3. ✅ Follow the **MANDATORY TEST WORKFLOW** template (PHASE 0, 1-N, N+1)
4. ✅ Implement hardware auto-scaling (HardwareConfig)
5. ✅ Implement the specific test logic for the concept
6. ✅ Implement emergent verification (EmergentVerifier)
7. ✅ Generate whitepapers for all results (EmergentWhitepaperGenerator)
8. ✅ Save complete metadata (hardware, parameters, results, verification)

### What "Mapping Function" Means

The user is referring to **two types of mapping** that should be used when creating tests:

#### 1. Concept Mapping (`hhml_parameter_mapping.py`)

Maps test concepts to appropriate HHmL implementation:

```python
from hhml.utils.hhml_parameter_mapping import (
    get_mapping_for_concept,
    get_topology_for_concept,
    get_observables_for_concept,
    get_verification_type_for_concept,
    generate_test_template_for_concept
)

# When user asks: "Create a test for vortex annihilation"
mapping = get_mapping_for_concept("vortex annihilation")

# Get implementation details
topology = mapping.topology  # 'mobius'
key_params = mapping.key_rnn_parameters  # ['antivortex_strength', 'annihilation_radius', ...]
observables = mapping.observables  # ['annihilation_rate', 'vortex_quality', ...]
verification = mapping.verification_type  # 'spatial'
phases = mapping.suggested_phases  # ['Vortex generation', 'Antivortex injection', ...]

# Or generate complete template
template = generate_test_template_for_concept("vortex annihilation")
```

**Available concepts:**
- Topological: möbius topology, klein bottle, topological phase transition, topological charge conservation
- Vortex: vortex annihilation, vortex stability
- Holographic: ads/cft correspondence, holographic encoding
- Quantum: quantum coherence, quantum entanglement
- Wave: gravitational waves, wave propagation
- Cosmology: dark matter, cmb fluctuations
- Particle: particle masses
- Computational: numerical stability, hardware scalability

#### 2. RNN Parameter Mapping (`rnn_parameter_mapping.py`)

Defines the 23 RNN-controlled parameters:

```python
from hhml.utils.rnn_parameter_mapping import (
    RNN_PARAMETERS,
    get_parameter_info,
    get_parameters_by_category,
    create_parameter_dict_from_tensor
)

# Get parameter information
kappa_info = get_parameter_info('kappa')
print(f"{kappa_info.name}: {kappa_info.purpose}")
print(f"Range: [{kappa_info.range_min}, {kappa_info.range_max}]")

# Get all parameters in a category
vortex_params = get_parameters_by_category('vortex_annihilation')
# Returns: antivortex_strength, annihilation_radius, pruning_threshold, preserve_ratio

# Convert RNN output tensor to named dictionary
rnn_output = model(state)  # Shape: (23,)
params = create_parameter_dict_from_tensor(rnn_output)
# Returns: {'kappa': 1.5, 'delta': 0.3, ...}
```

#### 3. Workflow Mapping Components

Maps test results to verification and documentation:

**Workflow Components:**
- `HardwareConfig` → Auto-scales parameters to current hardware
- `EmergentVerifier` → Maps field tensors to real-world data (LIGO/CMB/particles)
- `EmergentWhitepaperGenerator` → Maps results to professional documentation
- `summary.json` → Maps all metadata for reproducibility

**ALL of these are MANDATORY** - every test must use all mapping functions.

### Example Interpretation

**User Request:**
> "Create a test for Klein bottle topology in HHmL"

**Your Response Should Be:**

"I'll create a complete test script for Klein bottle topology following the mandatory HHmL workflow.

First, let me check the concept mapping:

```python
from hhml.utils.hhml_parameter_mapping import get_mapping_for_concept

mapping = get_mapping_for_concept("klein bottle")
# Returns: topology='klein_bottle', field_dynamics='holographic_resonance',
#          key_rnn_parameters=['winding_density', 'twist_rate', 'cross_coupling'],
#          observables=['vortex_density', 'topological_charge', 'non_orientability'],
#          verification_type='spatial'
```

This tells me the test should include:

1. **Hardware auto-scaling** - Runs optimally on any hardware (CPU → H200)
2. **Klein bottle topology** - Double-twisted Möbius surface (from mapping)
3. **Holographic resonance dynamics** - Field evolution (from mapping)
4. **Key RNN parameters** - winding_density, twist_rate, cross_coupling (from mapping)
5. **Observables** - vortex_density, topological_charge, non_orientability (from mapping)
6. **Emergent verification** - Spatial patterns → CMB comparison (from mapping)
7. **Whitepaper generation** - Complete documentation of results
8. **Summary metadata** - Hardware, parameters, verification scores

The script will be: `simulations/topology/klein_bottle_test.py`"

Then proceed to implement:
- PHASE 0: Hardware detection and auto-scaling
- PHASE 1: Klein bottle geometry generation
- PHASE 2: Field evolution and dynamics
- PHASE 3: Vortex stability measurement
- PHASE 4: Topological invariant computation
- PHASE 5: Emergent verification and whitepaper generation

### Template Structure for New Tests

```python
#!/usr/bin/env python3
"""
[Concept] Test - Hardware Scalable
===================================

Tests [specific hypothesis or behavior] using HHmL framework.

Phases:
- PHASE 0: Hardware detection and auto-scaling
- PHASE 1: [Setup/initialization]
- PHASE 2: [Main test logic]
- PHASE 3: [Measurement/analysis]
- PHASE 4: [Validation/comparison]
- PHASE 5: Emergent verification and whitepaper

Target Hardware: Auto-scaled (CPU → H200)
Expected Duration: [time range]

Author: HHmL Project
Date: YYYY-MM-DD
"""

# Follow MANDATORY TEST WORKFLOW template from CLAUDE.md
# Include: HardwareConfig, EmergentVerifier, EmergentWhitepaperGenerator
```

### What NOT to Do

**DON'T:**
- ❌ Create a simple script without the mandatory workflow
- ❌ Skip hardware auto-scaling
- ❌ Skip emergent verification
- ❌ Skip whitepaper generation
- ❌ Create one-off throwaway test code

**DO:**
- ✅ Follow the complete mandatory workflow
- ✅ Make it production-quality and reproducible
- ✅ Include all phases (0, 1-N, N+1)
- ✅ Document everything in whitepapers
- ✅ Save to proper location (simulations/ or examples/)

### Quick Checklist

When user asks to "create a test for X":
- [ ] Understand what hypothesis/behavior to test
- [ ] Create new Python script in appropriate directory
- [ ] Copy MANDATORY TEST WORKFLOW template
- [ ] Implement PHASE 0 (hardware auto-scaling)
- [ ] Implement PHASE 1-N (test-specific logic)
- [ ] Implement PHASE N+1 (emergent verification + whitepaper)
- [ ] Test with `--auto-scale --scale-mode benchmark`
- [ ] Commit with descriptive message
- [ ] Document in README.md if it's a new capability

### Examples of Test Creation Requests

| User Request | What to Build |
|--------------|---------------|
| "Test vortex annihilation dynamics" | Full test script with vortex tracking, annihilation measurement, emergent verification |
| "Create a test for topological charge conservation" | Full test script with winding number computation, charge tracking, verification |
| "Use HHmL to test AdS/CFT correspondence" | Full test script with bulk/boundary computation, holographic matching, verification |
| "Build a test for quantum coherence decay" | Full test script with decoherence simulation, coherence measurement, verification |

**All of these result in complete test scripts following the mandatory workflow.**

---

## 🔬 EMERGENT PHENOMENA DETECTION (MANDATORY AFTER EVERY TEST)

**Location**: `EMERGENTS.md` (root directory)

**PURPOSE**: HHmL has the same level of emergent detection capabilities as iVHL. EMERGENTS.md must be updated after EVERY test run to catalog novel discoveries.

### After Every Test Run - MANDATORY STEPS

**1. Analyze Results for Novel Behavior**

Check for unusual patterns that indicate emergent phenomena:
- Unexpected parameter convergence (sudden jumps, oscillations, phase transitions)
- Correlation spikes (|r| > 0.7 between parameters and observables)
- Scaling laws (power-law relationships, critical exponents)
- Topological signatures (behavior unique to Möbius topology)
- Quality thresholds (sudden changes in vortex stability/density)

**2. Run Correlation Analysis**

```python
import json
import numpy as np
from scipy.stats import pearsonr

# Load training results
with open('test_cases/[test_name]/results/training_*.json') as f:
    data = json.load(f)

# Extract parameter histories
param_history = data['param_history']
metrics = data['metrics']

# Check all 23 parameters against key observables
observables = {
    'vortex_density': metrics['vortex_densities'],
    'vortex_quality': metrics.get('vortex_qualities', []),
    'reward': metrics['rewards'],
    'stability': metrics.get('stability', [])
}

# Compute correlations
for param_name in param_history[0].keys():
    param_values = [p[param_name] for p in param_history]

    for obs_name, obs_values in observables.items():
        if len(param_values) == len(obs_values):
            r, p = pearsonr(param_values, obs_values)

            # Flag strong correlations
            if abs(r) > 0.7 and p < 0.05:
                print(f"🔥 STRONG CORRELATION: {param_name} ↔ {obs_name}")
                print(f"   r = {r:.3f}, p = {p:.3e}")
```

**3. Document in EMERGENTS.md**

If you discover novel emergent behavior:

a) **Add to "Discovered Emergent Phenomena" section** using the template:
```markdown
### [Sequential Number]. [Phenomenon Name]

**Date Discovered**: YYYY-MM-DD
**Training Run**: test_cases/[test_name]/results/training_YYYYMMDD_HHMMSS.json
**Checkpoint**: [checkpoint_file.pt]
**Cycles**: [discovery cycle or range]

#### Description
[Detailed description of what was observed]

#### Topological Signature
[How Möbius topology enables this - would it occur in torus/sphere?]

#### Parameter Correlations
| Parameter | Correlation (r) | p-value | Interpretation |
|-----------|----------------|---------|----------------|
| [param] | [r-value] | [p-value] | [strong/moderate/weak] |

#### Reproducibility
- **Random Seed**: [seed]
- **Hardware**: [CPU/GPU model]
- **PyTorch Version**: [version]
- **Node Count**: [N nodes]

#### Critical Parameters
[Which parameters are essential for this phenomenon?]

#### Validation Tests
- [ ] Reproduced on different hardware
- [ ] Reproduced with different random seeds
- [ ] Scales to larger/smaller node counts
- [ ] Absent in non-Möbius topologies (control)

#### Scientific Significance
[Why is this discovery important?]

#### References
- Whitepaper: [PDF filename]
- Checkpoint: [.pt filename]
```

b) **Update Statistics Section**:
```markdown
## 📊 Emergent Phenomena Statistics

**Total Discovered**: [N] confirmed, [M] under investigation
**Latest Discovery**: YYYY-MM-DD ([Phenomenon Name])
```

c) **Update README.md** if the discovery represents a novel capability:
```markdown
### 🔬 [New Capability Name] *(Novel Discovery)*

[Brief description of the emergent capability]

**Discovery**: [Date] - [1-sentence summary]
```

**4. Generate Supplemental Analysis**

Create visualizations and detailed analysis:
```python
# Plot parameter evolution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Parameter trajectory
axes[0,0].plot([p['param_name'] for p in param_history])
axes[0,0].set_title('Parameter Evolution')

# Correlation heatmap
# ... correlation matrix visualization

# Phase space plot
axes[1,0].scatter([p['param1'] for p in param_history],
                  [p['param2'] for p in param_history],
                  c=metrics['vortex_densities'],
                  cmap='viridis')
axes[1,0].set_title('Parameter Phase Space')

plt.savefig(f'test_cases/{test_name}/analysis/emergent_analysis.png')
```

**5. Run Real-World Verification** *(NEW - MANDATORY)*

Automatically verify against empirical physics data:
```python
from hhml.verification import LIGOVerification, CMBVerification, ParticleVerification

# Load final state
final_field = torch.load('checkpoint_final.pt')['field_state']
vortex_energies = extract_vortex_energies(final_field)

# Determine phenomenon type and run appropriate verification
verification_results = {}

# If oscillatory/wave-like behavior observed
if phenomenon_has_oscillations:
    ligo = LIGOVerification()
    verification_results['ligo'] = ligo.compare_event(
        'GW150914',
        final_field,
        save_results=True
    )
    print(f"LIGO overlap: {verification_results['ligo']['metrics']['overlap']:.4f}")

# If spatial fluctuations observed
if phenomenon_has_spatial_structure:
    cmb = CMBVerification()
    verification_results['cmb'] = cmb.compare_planck(
        final_field,
        cl_type='TT',
        save_results=True
    )
    print(f"CMB χ²/DOF: {verification_results['cmb']['metrics']['reduced_chi_squared']:.3f}")

# If discrete energy levels observed
if phenomenon_has_discrete_energies:
    particles = ParticleVerification()
    verification_results['particles'] = particles.compare_pdg_masses(
        vortex_energies,
        tolerance=0.1,
        save_results=True
    )
    print(f"Particle matches: {verification_results['particles']['matched_particles']}/{verification_results['particles']['total_particles']}")

# Save verification results
with open('verification_results.json', 'w') as f:
    json.dump(verification_results, f, indent=2)
```

**Interpretation Guidelines**:
- **Good LIGO match** (overlap > 0.7): Phenomenon exhibits GW-like oscillation patterns
- **Good CMB match** (χ²/DOF < 3.0): Phenomenon exhibits CMB-like spatial fluctuations
- **Good particle match** (> 50% matched): Phenomenon exhibits SM-like energy quantization

**IMPORTANT**: These are *analogical comparisons* - not claims that HHmL models gravity/cosmology/particles. Strong matches suggest emergent mathematical structures share patterns with real physics, which **strengthens the novelty claim**.

**6. Update Changelog**

If discovery is significant, add to `CHANGELOG.md`:
```markdown
## [Unreleased]

### Discovered
- **[Phenomenon Name]**: [Brief description] - [parameter correlations] - [verification results]
```

---

### Emergent Detection Checklist (Run After EVERY Test)

Use this checklist after completing a training run:

- [ ] **Load training results** (JSON file with param_history and metrics)
- [ ] **Run correlation analysis** (all 23 params vs observables)
- [ ] **Check for strong correlations** (|r| > 0.7, p < 0.05)
- [ ] **Identify unusual patterns** (spikes, phase transitions, convergence)
- [ ] **Test topological specificity** (would this happen in torus/sphere?)
- [ ] **🌍 Run real-world verification** (NEW - MANDATORY):
  - [ ] Determine phenomenon type (oscillatory, spatial, energetic)
  - [ ] Run appropriate verification (LIGO, CMB, or Particles)
  - [ ] Document verification metrics in results JSON
  - [ ] Interpret results (analogical pattern matching)
- [ ] **Document in EMERGENTS.md** (use full template including verification)
- [ ] **Update statistics** (total count, latest discovery date)
- [ ] **Generate visualizations** (parameter evolution, phase space, correlations)
- [ ] **Update README.md** (if novel capability discovered)
- [ ] **Update CHANGELOG.md** (if significant discovery)
- [ ] **Create analysis notebook** (Jupyter notebook for deep dive - optional)
- [ ] **Commit changes** with message: `docs: add emergent phenomenon - [name]`

---

### What Counts as "Novel Emergent"?

A phenomenon qualifies as **novel emergent** if it meets ALL criteria:

1. ✅ **Topological Origin**: Arises from Möbius topology specifically
   - Test: Run same parameters on torus → phenomenon disappears/weakens

2. ✅ **Parameter Dependence**: Controlled by RNN parameters
   - Test: Fix parameter → phenomenon disappears
   - Measure: Strong correlation |r| > 0.7, p < 0.05

3. ✅ **Reproducibility**: Can be recreated
   - Same seed → identical results (tolerance 1e-6)
   - Different seeds → same effect statistically (p < 0.05, N ≥ 5)

4. ✅ **Falsifiability**: Can be tested and potentially refuted
   - Clear hypothesis about mechanism
   - Clear prediction that could be wrong

5. ✅ **Statistical Significance**: Not a random fluctuation
   - p-value < 0.05 (Bonferroni corrected)
   - Effect size Cohen's d > 0.5

6. ✅ **Real-World Verification** *(NEW - v0.1.0)*: Exhibits patterns similar to empirical physics
   - **LIGO**: If oscillatory → compare to GW waveforms (overlap > 0.5 strengthens claim)
   - **CMB**: If spatial fluctuations → compare to Planck spectra (χ²/DOF < 5.0 strengthens claim)
   - **Particles**: If discrete energies → compare to PDG masses (match > 30% strengthens claim)
   - **Interpretation**: Analogical comparisons testing mathematical pattern similarity
   - **Not Required**: For all phenomena, but **strengthens novelty claim** if patterns match real physics

---

### Examples of Emergent Phenomena to Watch For

**Scaling Laws**:
- Power-law relationships between parameters and observables
- Critical exponents at phase transitions
- Finite-size scaling behavior

**Phase Transitions**:
- Sudden jumps in vortex density at critical parameter values
- Order parameter discontinuities
- Diverging correlation lengths

**Self-Organization Patterns**:
- Spontaneous symmetry breaking
- Pattern formation (stripes, hexagons, spirals)
- Hierarchical structure emergence

**Topological Effects**:
- Winding number conservation/violation
- Topological charge flux
- Boundary condition effects unique to Möbius

**Parameter Coupling**:
- Synchronized parameter evolution (co-adaptive multiplets)
- Constraint manifolds in parameter space
- Master-slave parameter relationships

**Quality Control Mechanisms**:
- Active defect curation
- Selective pruning strategies
- Robustness to perturbations

---

### Automated Emergent Detection (Future Enhancement)

**Goal**: Automatically flag potential emergent phenomena during training

```python
class EmergentDetector:
    """Automatically detect potential emergent phenomena during training."""

    def __init__(self, threshold_r=0.7, threshold_p=0.05):
        self.threshold_r = threshold_r
        self.threshold_p = threshold_p
        self.flagged_phenomena = []

    def check_correlations(self, param_history, metrics):
        """Flag strong parameter-observable correlations."""
        # ... correlation checking logic

    def check_phase_transitions(self, observable_history):
        """Detect sudden jumps indicating phase transitions."""
        # ... change point detection

    def check_scaling_laws(self, param_history, observable_history):
        """Test for power-law relationships."""
        # ... power law fitting

    def generate_report(self):
        """Generate automatic emergent phenomena report."""
        # ... create draft EMERGENTS.md entry
```

**Usage** (integrate into training loop):
```python
detector = EmergentDetector()

# During training
for cycle in range(num_cycles):
    # ... training code ...

    # Check for emergents every 100 cycles
    if cycle % 100 == 0:
        detector.check_correlations(param_history, metrics)
        detector.check_phase_transitions(metrics['vortex_densities'])

# After training
report = detector.generate_report()
print(f"🔥 Flagged {len(detector.flagged_phenomena)} potential emergents")
# Review and add validated ones to EMERGENTS.md
```

---

**File Organization:**
- Scripts in `scripts/`
- Test cases in `test_cases/[test_name]/` with subdirs: results/, whitepapers/
- Legacy files in `archive/`
- Core code in `hhml/`

**Whitepaper Naming:**
Format: `[test_name]_YYYYMMDD_HHMMSS.pdf`
Example: `multi_strip_tokamak_20251216_213045.pdf`

**README.md Update Policy:**
CRITICAL: Whenever you implement new or novel features, IMMEDIATELY update README.md to reflect:
- New scientific capabilities (e.g., vortex annihilation control)
- Novel technical approaches (e.g., RNN parameter extensions)
- Architectural enhancements (e.g., glass-box improvements)
- Performance breakthroughs
- New analysis tools or methodologies

The README.md must always accurately represent the current state-of-the-art capabilities of HHmL.
Update both the "Key Features" section and add a new subsection under "What Makes HHmL Unique?" if applicable.

---

## Live Dashboard Integration (MANDATORY for All Training)

**Location**: `hhml/utils/live_dashboard.py`
**Documentation**: `docs/LIVE_DASHBOARD_INTEGRATION.md`

**CRITICAL**: Always integrate the live dashboard into training scripts for real-time monitoring.

### Quick Integration (4 Steps)

```python
from hhml.utils.live_dashboard import TrainingDashboard

# 1. Initialize before training loop
dashboard = TrainingDashboard(port=8000, auto_open=True)
dashboard.start()

try:
    for cycle in range(num_cycles):
        # 2. Your training code here
        # ...

        # 3. Update dashboard with current metrics
        dashboard.update({
            'cycle': cycle,
            'density': vortex_density,        # 0-1 float
            'quality': vortex_quality,        # 0-1 float
            'reward': reward,                 # float
            'annihilations': num_removed,     # int
            'cycles_at_target': cycles_stable # int
        })

finally:
    # 4. Clean shutdown
    dashboard.stop()
```

### Features
- **Real-time charts**: Density, quality, reward, annihilations, stability
- **Live statistics**: Current cycle, density %, quality, reward, etc.
- **Auto-refresh**: No manual browser refresh needed
- **Auto-open**: Browser window opens automatically
- **Lightweight**: Uses stdlib only (http.server, threading)
- **Thread-safe**: Non-blocking updates

### Multiple Training Sessions
Use different ports for concurrent sessions:
```python
dashboard1 = TrainingDashboard(port=8000)  # Training 1
dashboard2 = TrainingDashboard(port=8001)  # Training 2
```

### Access Dashboard
- Opens automatically when `dashboard.start()` is called
- Manual access: `http://localhost:8000`
- Works on any browser (Chrome, Firefox, Edge, Safari)

### Required Metrics
The dashboard expects these keys in `dashboard.update()`:
- `cycle`: Current training cycle (int)
- `density`: Vortex density 0-1 (float)
- `quality`: Vortex quality 0-1 (float)
- `reward`: Current reward (float)
- `annihilations`: Number removed this cycle (int)
- `cycles_at_target`: Consecutive cycles at target density (int)

### Testing the Dashboard
Run standalone demo:
```bash
python -m hhml.utils.live_dashboard
```
This simulates 100 cycles of training data for 60 seconds.

### When Writing New Training Scripts
**ALWAYS** include live dashboard integration:
1. Import at top of file
2. Start before training loop
3. Update inside training loop (every cycle or every N cycles)
4. Stop in `finally` block (ensures cleanup)

See `docs/LIVE_DASHBOARD_INTEGRATION.md` for complete examples and troubleshooting.

---

