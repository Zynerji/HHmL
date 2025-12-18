# HHmL Production Refactoring - Migration Guide

**Date:** December 17, 2025
**Version:** 0.1.0 → Production-Ready

---

## Overview

This document describes the complete production-ready refactoring of the HHmL project. The repository has been transformed from a research prototype into a professional, production-ready Python package with modern infrastructure.

---

## What Changed

### 1. Repository Structure

**Before:**
```
HHmL/
├── hhml/                    # Flat package structure
├── scripts/                 # Mixed scripts
├── test_cases/             # Mixed test outputs
├── docs/                   # Scattered documentation
└── [Many loose files in root]
```

**After:**
```
HHmL/
├── src/hhml/               # Modular package structure
│   ├── core/              # Physics modules (mobius, resonance, gft, tensor_networks)
│   ├── ml/                # ML components (rl, training)
│   ├── analysis/          # Analysis tools (dark_matter)
│   ├── monitoring/        # Web dashboard & monitoring
│   └── utils/             # Shared utilities
├── tests/                  # Organized test suite (unit, integration, benchmarks)
├── examples/              # Example usage scripts (training, analysis)
├── docker/                # Complete Docker infrastructure
├── docs/                  # Organized documentation (guides, deployment, theory)
├── tools/                 # Development tools (whitepaper, benchmarking)
├── data/                  # Data directory (gitignored)
└── [Clean root with production files]
```

---

## Import Changes

### Old Imports (Still Work Temporarily)

```python
from hhml.mobius.mobius_training import MobiusStrip
from hhml.resonance.holographic_resonance import HolographicResonance
from hhml.gft.condensate_dynamics import GFTCondensate
```

### New Imports (Recommended)

```python
from hhml.core.mobius.mobius_training import MobiusStrip
from hhml.core.resonance.holographic_resonance import HolographicResonance
from hhml.core.gft.condensate_dynamics import GFTCondensate
from hhml.ml.rl.td3_agent import TD3Agent
from hhml.monitoring.live_dashboard import TrainingDashboard
```

---

## File Locations

### Python Package (`src/hhml/`)

| Old Location | New Location |
|:-------------|:-------------|
| `hhml/mobius/*` | `src/hhml/core/mobius/*` |
| `hhml/resonance/*` | `src/hhml/core/resonance/*` |
| `hhml/gft/*` | `src/hhml/core/gft/*` |
| `hhml/tensor_networks/*` | `src/hhml/core/tensor_networks/*` |
| `hhml/rl/*` | `src/hhml/ml/rl/*` |
| `hhml/dark_matter/*` | `src/hhml/analysis/dark_matter/*` |
| `hhml/utils/*` | `src/hhml/utils/*` (unchanged) |
| `web_monitor/*` | `src/hhml/monitoring/*` |

### Scripts & Examples

| Old Location | New Location |
|:-------------|:-------------|
| `scripts/train_*.py` | `examples/training/train_*.py` |
| `scripts/run_*.py` | `examples/training/run_*.py` |
| `simulations/*` | `examples/analysis/*` |

### Documentation

| Old Location | New Location |
|:-------------|:-------------|
| `H200_DEPLOYMENT.md` | `docs/deployment/h200.md` |
| `MULTI_STRIP_TOPOLOGY.md` | `docs/guides/multi_strip_topology.md` |
| `RNN_PARAMETER_MAPPING.md` | `docs/guides/RNN_PARAMETER_MAPPING.md` |
| `CLAUDE.md` | `CLAUDE.md` (root, for AI context) |

### Tools

| Old Location | New Location |
|:-------------|:-------------|
| `web_monitor/whitepaper_generator.py` | `tools/whitepaper/whitepaper_generator.py` |

### Data & Outputs

| Old Location | New Location |
|:-------------|:-------------|
| `checkpoints/` | `data/checkpoints/` (gitignored) |
| `results/` | `data/results/` (gitignored) |
| `whitepapers/` | `data/outputs/whitepapers/` (gitignored) |

---

## New Features

### 1. Docker Integration

**Complete containerization infrastructure:**

```bash
# Build all images
cd docker && ./scripts/build.sh all

# Run production training
./scripts/run.sh production

# Run development environment (JupyterLab)
./scripts/run.sh development

# Stop all containers
./scripts/run.sh stop
```

**Available Images:**
- `hhml:cpu-latest` - CPU-only lightweight (~2GB)
- `hhml:cuda-latest` - H100/H200 GPU support (~8GB)
- `hhml:dev-latest` - Full development environment (~10GB)

**Docker Compose Services:**
- `hhml-training` - Main training with GPU
- `hhml-monitor` - Live web dashboard
- `hhml-whitepaper` - On-demand report generation

### 2. Modern Python Packaging

**Installation:**

```bash
# Development install (editable)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# With all optional dependencies
pip install -e ".[dev,viz,docs]"
```

**Package configuration in `pyproject.toml`:**
- Modern build system (setuptools>=65)
- Dependency management
- Code quality tools (Black, Flake8, MyPy)
- Testing configuration (Pytest)

### 3. Production Files

| File | Purpose |
|:-----|:--------|
| `LICENSE` | MIT License |
| `CONTRIBUTING.md` | Contribution guidelines |
| `CHANGELOG.md` | Version history |
| `.gitignore` | Comprehensive ignore rules |
| `.dockerignore` | Docker-specific ignores |
| `.editorconfig` | Editor configuration |
| `pyproject.toml` | Modern Python packaging |
| `setup.py` | Backwards compatibility |

### 4. Professional README

**New README includes:**
- LaTeX-formatted mathematical framework
- Complete Docker deployment guide
- Architecture diagrams
- Installation instructions
- Usage examples
- Citation information
- Contact: @Conceptual1

---

## Migration Steps

### For Users

1. **Pull latest changes:**
   ```bash
   git pull origin main
   ```

2. **Reinstall package:**
   ```bash
   pip uninstall hhml
   pip install -e .
   ```

3. **Update imports in your code** (see [Import Changes](#import-changes))

4. **Test your scripts:**
   ```bash
   python your_script.py
   ```

### For Contributors

1. **Pull and reinstall:**
   ```bash
   git pull origin main
   pip install -e ".[dev]"
   ```

2. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run tests to verify:**
   ```bash
   pytest tests/
   ```

4. **Update your development workflow** (see [CONTRIBUTING.md](CONTRIBUTING.md))

### For Deployment

1. **For Docker deployment:**
   ```bash
   cd docker
   ./scripts/build.sh cuda
   ./scripts/run.sh production
   ```

2. **For traditional deployment:**
   ```bash
   pip install -e .
   python examples/training/train_mobius_basic.py --cycles 1000
   ```

3. **For H200 VM:**
   ```bash
   ssh ivhl@89.169.111.28
   cd HHmL
   git pull
   docker-compose up -d
   ```

---

## Breaking Changes

### Import Paths

All import paths have changed due to reorganization. Update your code:

**Old:**
```python
from hhml.mobius.mobius_training import MobiusStrip
```

**New:**
```python
from hhml.core.mobius.mobius_training import MobiusStrip
```

### File Paths

If your code references file paths directly, update them:

**Old:**
```python
checkpoint_path = "checkpoints/agent.pt"
```

**New:**
```python
checkpoint_path = "data/checkpoints/agent.pt"
```

### Configuration Files

Configuration files moved to `configs/` directory:

**Old:**
```python
config = yaml.load(open("multiscale_config.yaml"))
```

**New:**
```python
config = yaml.load(open("configs/multiscale_config.yaml"))
```

---

## Benefits

### 1. Professional Structure
- Clean separation of concerns
- Modular architecture
- Industry-standard layout

### 2. Easier Contribution
- Clear contribution guidelines
- Pre-commit hooks for code quality
- Comprehensive testing setup

### 3. Better Deployment
- Docker containers for reproducibility
- Easy scaling from laptop to H200
- Multiple deployment options

### 4. Improved Documentation
- Organized documentation structure
- LaTeX-formatted README
- Complete migration guide

### 5. Production Ready
- Proper Python packaging
- CI/CD ready
- Version controlled
- License and contribution guidelines

---

## Troubleshooting

### Import Errors

**Error:**
```
ImportError: cannot import name 'MobiusStrip' from 'hhml.mobius'
```

**Solution:**
```bash
# Reinstall package
pip uninstall hhml
pip install -e .

# Update imports
from hhml.core.mobius.mobius_training import MobiusStrip
```

### Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'hhml'
```

**Solution:**
```bash
# Ensure you're in the HHmL directory
cd HHmL

# Install in editable mode
pip install -e .
```

### Docker Build Fails

**Error:**
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solution:**
```bash
# Clean Docker cache
docker system prune -a

# Rebuild
cd docker
./scripts/build.sh cuda
```

### Tests Fail

**Error:**
```
ImportError in test files
```

**Solution:**
```bash
# Reinstall with dev dependencies
pip install -e ".[dev]"

# Run tests from root directory
pytest tests/
```

---

## Rollback (if needed)

If you need to rollback to the old structure temporarily:

```bash
# Checkout previous commit (before refactoring)
git log --oneline | head -20
git checkout <commit-hash-before-refactoring>

# Or use tag if available
git checkout v0.0.1
```

**Note:** It's recommended to update your code instead of rolling back, as the new structure is production-ready and future updates will use this organization.

---

## Timeline

- **December 17, 2025**: Production refactoring completed
- **December 18-20, 2025**: Migration period (old imports still work)
- **December 21+, 2025**: New structure fully enforced

---

## Support

If you encounter issues during migration:

1. **Check this guide** for common problems
2. **Review CONTRIBUTING.md** for development workflow
3. **Open a GitHub Issue** with the `migration` label
4. **Contact:** [@Conceptual1](https://twitter.com/Conceptual1)

---

## Acknowledgments

This refactoring transforms HHmL from a research prototype into a production-ready framework, enabling:
- Easier collaboration
- Better reproducibility
- Scalable deployment
- Professional presentation

Thank you for adapting to the new structure! The benefits will be worth the migration effort.

---

**Questions?** Open a GitHub Discussion or reach out to @Conceptual1 on Twitter.
