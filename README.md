# Holo-Harmonic Möbius Lattice (HHmL)

**A Glass-Box Framework for Emergent Topological Phenomena Discovery**

[![License](https://img.shields.io/badge/license-TBD-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

The Holo-Harmonic Möbius Lattice (HHmL) is a computational framework for investigating emergent phenomena in topologically non-trivial field configurations. By combining **Möbius strip topology** with **RNN-controlled parameter spaces**, HHmL enables systematic exploration of correlations between topological configurations and emergent vortex dynamics.

### Key Features

- 🎭 **Möbius Topology**: Closed-loop, boundary-free geometric structures
- 🧠 **RNN Control**: 19 parameters autonomously optimized via reinforcement learning
- 🔍 **Glass-Box Architecture**: Complete transparency for correlation tracking
- 📊 **Reproducible Science**: Full parameter trajectories saved for peer review
- ⚡ **Auto-Scaling**: CPU (2K nodes) → H200 GPU (20M+ nodes)
- 📄 **Automated Reporting**: LaTeX whitepaper generation from results

---

## Quick Start

\`\`\`bash
# Clone repository
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# Install dependencies
pip install -r requirements.txt

# Run training (auto-detects hardware)
python scripts/train_multi_strip.py --cycles 100

# Generate whitepaper
python web_monitor/whitepaper_generator.py
\`\`\`

---

## What Makes HHmL Unique?

### 1. Möbius Strip Topology

Unlike traditional approaches using flat space or simple spheres, HHmL exploits Möbius strips:

- **No Boundary Discontinuities**: 180° twist eliminates endpoint artifacts
- **Topological Protection**: Single-sided surface stabilizes resonance modes
- **Novel Harmonic Modes**: Unique to Möbius geometry

### 2. Glass-Box RNN Control

The RNN controls **19 parameters** across 6 categories:

| Category | Parameters | Examples |
|----------|-----------|----------|
| **Geometry (4)** | Shape & structure | κ (elongation), δ (triangularity), QEC layers |
| **Physics (4)** | Field dynamics | Damping, nonlinearity, amplitude variance |
| **Spectral (3)** | Graph methods | ω (helical frequency), diffusion timestep |
| **Sampling (3)** | Computational | Sample ratio, neighbors, sparsity |
| **Mode (2)** | Method selection | Sparse density, spectral weight |
| **Extended (3)** | Topology | Winding density, twist rate, coupling |

**Every parameter is tracked every cycle** → Full correlation analysis possible.

### 3. Reproducible & Peer-Reviewable

- ✅ Complete parameter trajectories saved
- ✅ Random seeds and hardware specs logged
- ✅ No hidden hyperparameters
- ✅ Automated whitepaper generation
- ✅ Open-source codebase

---

## Scientific Workflow

\`\`\`
1. Run Simulation
   └─> python scripts/train_multi_strip.py --cycles 100

2. Results Saved
   └─> test_cases/multi_strip/results/training_YYYYMMDD_HHMMSS.json

3. Generate Whitepaper
   └─> python web_monitor/whitepaper_generator.py

4. Whitepaper Created
   └─> test_cases/multi_strip/whitepapers/multi_strip_YYYYMMDD_HHMMSS.pdf

5. Analyze Correlations
   └─> See RNN_PARAMETER_MAPPING.md for correlation analysis methods
\`\`\`

---

## Repository Structure

\`\`\`
HHmL/
├── hhml/                      # Core Python package
│   ├── mobius/               # Möbius-specific modules
│   ├── resonance/            # Field dynamics
│   ├── tensor_networks/      # MERA holography
│   └── utils/                # Hardware config, validation
├── scripts/                   # Training scripts
│   └── train_multi_strip.py  # Main RNN training
├── test_cases/               # Test configurations & results
│   ├── multi_strip/
│   │   ├── results/         # JSON simulation outputs
│   │   └── whitepapers/     # Generated PDFs
│   └── benchmarks/
├── web_monitor/              # Whitepaper generation
│   └── whitepaper_generator.py
├── docs/                     # Documentation
├── RNN_PARAMETER_MAPPING.md  # Parameter correlation guide
├── CLAUDE.md                 # AI assistant context
├── README.tex                # Full mathematical documentation
└── README.md                 # This file
\`\`\`

---

## Documentation

- **[README.tex](README.tex)**: Comprehensive mathematical framework (compile with LaTeX)
- **[RNN_PARAMETER_MAPPING.md](RNN_PARAMETER_MAPPING.md)**: Complete guide to correlation tracking
- **[CLAUDE.md](CLAUDE.md)**: Workflow expectations and development guide
- **[H200_DEPLOYMENT.md](H200_DEPLOYMENT.md)**: Large-scale deployment guide

---

## Scientific Merit

### What HHmL Is

- ✅ Computational research tool for emergent phenomena
- ✅ Glass-box system for correlation discovery
- ✅ Platform for reproducible topological field experiments

### What HHmL Is NOT

- ❌ Theory of fundamental physics
- ❌ Model of quantum gravity or cosmology
- ❌ Replacement for established physical theories

**This is a mathematical and computational research platform, not a physical theory.**

---

## Citation

If you use HHmL in your research, please cite:

\`\`\`bibtex
@software{hhml2025,
  title = {Holo-Harmonic Möbius Lattice (HHmL): A Glass-Box Framework
           for Emergent Topological Phenomena Discovery},
  author = {HHmL Research Collective},
  year = {2025},
  url = {https://github.com/Zynerji/HHmL},
  note = {Computational research platform for investigating emergent
          phenomena in Möbius strip topologies}
}
\`\`\`

---

## Contact

- **GitHub**: [https://github.com/Zynerji/HHmL](https://github.com/Zynerji/HHmL)
- **Issues**: [https://github.com/Zynerji/HHmL/issues](https://github.com/Zynerji/HHmL/issues)

---

<div align="center">

**HHmL: Exploring emergent phenomena through topological field dynamics**

*Mathematical research platform — not a physical theory*

</div>
