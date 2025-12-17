# HHmL: Holo-Harmonic Möbius Lattice

**Computational exploration of holographic resonance on Möbius strip topology**

[![Status](https://img.shields.io/badge/status-development-yellow)]()
[![Python](https://img.shields.io/badge/python-3.11+-blue)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.1+-green)]()
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()

---

## What is HHmL?

HHmL (Holo-Harmonic Möbius Lattice) is a **fork of the iVHL framework** focused exclusively on **Möbius strip topology** for holographic resonance simulations.

### Key Innovation: Closed-Loop Topology

Unlike iVHL's open helical structure, HHmL uses a **Möbius strip** (single-sided surface with 180° twist) that provides:

- ✅ **No endpoints** → No phase discontinuities
- ✅ **Topological protection** → Enhanced vortex stability
- ✅ **Harmonic richness** → Unique resonance modes
- ✅ **Higher vortex density** → 82% achieved (vs. 0.03% collapse in open helix)

---

## Quick Start

### Prerequisites
- NVIDIA GPU (H200/H100 recommended, 50-140GB VRAM)
- Python 3.11+
- CUDA 12.1+

### Installation

```bash
# Clone repository
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# Create virtual environment
python3 -m venv hhml_env
source hhml_env/bin/activate  # On Windows: hhml_env\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Run Möbius Training

```bash
# Quick test (100 cycles, ~1 minute on H200)
python hhml/mobius/mobius_training.py --device cuda --cycles 100 --nodes 100000

# Full training (1000 cycles, ~2-3 hours)
python hhml/mobius/mobius_training.py --device cuda --cycles 1000 --nodes 20000000
```

---

## Scientific Context

### ⚠️ Important Disclaimer

HHmL is **NOT**:
- A theory of everything
- A replacement for established physics
- Claiming to explain real physical phenomena

HHmL **IS**:
- A computational research platform
- An exploration tool for holographic duality concepts (AdS/CFT-inspired)
- A framework for studying topological effects on emergent spacetime
- An RL discovery engine for Möbius-specific phenomena

---

## Architecture

### Möbius Strip Holographic Boundary

```
   Standard Helix          Möbius Strip
   (iVHL - open)          (HHmL - closed)

   ╭─────────╮           ╭──────╮
   │  ┌───┐  │           │  ╱╲  │
   │  └───┘  │    →      │ ╱  ╲ │  (180° twist)
   │ endpoints│           │╱____╲│
   ╰─────────╯           ╰──────╯
   Discontinuous         Continuous
```

### 11-Dimensional Framework

**Boundary (2D+1)**: θ, φ, t
**Emergent Bulk (3D)**: x, y, z
**Internal Fields (5D)**: c₁, c₂, c₃, s, r
**Möbius-Specific**: τ (twist), w (windings)

### RNN-Controlled Parameters

The 4-layer LSTM (4096 hidden dim) autonomously discovers optimal configurations:

- **w (windings)**: Discovered optimum ≈ 109-110 at 20M nodes
- **τ (twist)**: Möbius twist rate modulation
- **n (sampling)**: Adaptive node density (500-5000)

---

## Training Results (500 Cycles @ 20M Nodes)

**Hardware**: NVIDIA H200 (140GB VRAM)
**Duration**: 72.5 minutes
**VRAM Usage**: 50.6GB peak (36% utilization)

### Converged Parameters

| Parameter | Initial | Final | Change |
|-----------|---------|-------|--------|
| w (windings) | 3.8 | **109.63** | 28.9× |
| L (QEC layers) | 7.0 | **9.7** | 1.4× |
| n (sampling) | 2.0 | **4.99** | 2.5× |
| Vortex density | - | **82%** | - |
| RNN value | 0 | **3,599.5** | - |

### Key Discovery

**Möbius topology prevents vortex collapse at scale**
At 20M nodes, HHmL maintains 82% vortex density where iVHL helical runs experienced catastrophic collapse (0.03%).

This demonstrates:
1. **Scale compensation**: w(N) scaling relationship
2. **Topological stability**: Closed-loop protects vortices
3. **Harmonic optimization**: RNN discovers rich resonance modes

---

## Repository Structure

```
HHmL/
├── CLAUDE.md                 # Context file for Claude Code
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── Dockerfile                # H200-optimized container
│
├── hhml/                     # Core package
│   ├── mobius/              # Möbius-specific modules
│   │   └── mobius_training.py   # Main RNN training
│   ├── resonance/           # Holographic resonance
│   ├── gft/                 # Group Field Theory
│   ├── tensor_networks/     # MERA holography
│   └── utils/               # Utilities
│
├── web_monitor/             # Real-time 3D visualization
├── dashboards/              # Streamlit interfaces
├── configs/                 # Configuration files
├── docs/                    # Documentation
│   ├── QUICKSTART_VM.md    # H200 VM deployment
│   └── DEPLOY_H100.md      # H100 deployment guide
└── whitepapers/            # Generated reports
```

---

## Roadmap

### ✅ Completed
- [x] Fork iVHL codebase
- [x] Implement Möbius RNN training
- [x] 500-cycle H200 validation
- [x] Discover w ≈ 109 optimal windings
- [x] Achieve 82% vortex density

### 🚧 In Progress
- [ ] Create comprehensive README (this file)
- [ ] Deploy to GitHub
- [ ] 1000-cycle extended training

### 📋 Planned
- [ ] Implement `topology.py` for alternative Möbius patterns
- [ ] Möbius-specific visualization dashboard
- [ ] Scale to 100M nodes (~130GB VRAM)
- [ ] Klein bottle topology (double Möbius)
- [ ] Comparative whitepaper: Helix vs. Möbius vs. Toroidal

---

## Performance Benchmarks

### H200 (140GB VRAM)
- **20M nodes**: 0.11 cycles/sec, 50.6GB VRAM
- **Expected 100M nodes**: ~130GB VRAM, ~0.05 cycles/sec

### Comparison to iVHL
| Metric | iVHL (50K helix) | HHmL (20M Möbius) | Improvement |
|--------|------------------|-------------------|-------------|
| Vortex density | 0.03% (collapse) | 82% | **2733×** |
| Scale | 50K nodes | 20M nodes | **400×** |
| Stability | Unstable | Stable | Topological |

---

## Citation & Attribution

HHmL is a fork of the [iVHL framework](https://github.com/Zynerji/iVHL) by Zynerji.

**Parent Project**: iVHL (Vibrational Helical Lattice)
**Möbius Extension**: HHmL development team

If you use HHmL in your research, please cite both projects:
```
@software{hhml2025,
  title={HHmL: Holo-Harmonic Möbius Lattice Framework},
  author={Zynerji and contributors},
  year={2025},
  url={https://github.com/Zynerji/HHmL}
}

@software{ivhl2025,
  title={iVHL: Vibrational Helical Lattice Framework},
  author={Zynerji},
  year={2025},
  url={https://github.com/Zynerji/iVHL}
}
```

---

## Documentation

- **CLAUDE.md**: Full context for Claude Code AI assistant
- **docs/QUICKSTART_VM.md**: H200 VM deployment guide
- **docs/DEPLOY_H100.md**: H100 container deployment

For conceptual background, see parent project [iVHL documentation](https://github.com/Zynerji/iVHL/tree/main/docs).

---

## Community & Support

**Issues**: Report bugs or request features via GitHub Issues
**Discussions**: Share findings and ask questions in GitHub Discussions
**License**: TBD (to be determined)

---

## Acknowledgments

- **iVHL Framework**: Foundation for all holographic resonance work
- **NVIDIA**: H200 GPU enabling large-scale simulations
- **PyTorch Team**: Deep learning framework
- **Research Inspirations**: Maldacena (AdS/CFT), Ryu-Takayanagi (holographic entanglement), Oriti (GFT)

---

**Status**: Development
**Last Updated**: 2025-12-16
**Next Milestone**: 1000-cycle H200 training run

For detailed context, read [CLAUDE.md](CLAUDE.md) ← **Start here if using Claude Code**
