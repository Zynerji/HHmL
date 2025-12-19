# tHHmL - Spatiotemporal Mobius Lattice

**Research Fork of HHmL**: Integrating Perfect Temporal Loops into Spatiotemporal Topology

---

## 🔀 Fork Status

**Parent**: [HHmL](https://github.com/Zynerji/HHmL) (Holo-Harmonic Möbius Lattice)
**Fork Date**: December 18, 2025
**Purpose**: Integrate temporal loop dynamics (self-consistent retrocausal feedback) into Möbius topology framework

**See [FORK_NOTES.md](FORK_NOTES.md) for complete fork rationale and architecture differences.**

---

## 🎯 What is tHHmL?

tHHmL extends HHmL from **spatial Möbius topology** to **spatiotemporal Möbius topology** where both space AND time are topological dimensions with closed-loop structure.

### Key Innovation: (2+1)D Spacetime

**HHmL** (parent):
- 2D spatial boundary (Möbius strip with 180° twist)
- Time as evolution parameter (forward-only, asymmetric)
- 23 RNN-controlled spatial parameters

**tHHmL** (this fork):
- 2D spatial + 1D temporal Möbius = **(2+1)D spatiotemporal manifold**
- Time has Möbius twist (closed timelike curves)
- 32 RNN-controlled parameters (23 spatial + 9 temporal)
- Forward AND backward time evolution with retrocausal coupling

---

## 🔬 Scientific Motivation

### Discovery Sequence (Dec 18, 2025)

1. **Perfect Temporal Loop** (HHmL discovery)
   - Achieved 100% temporal fixed points (50/50 time steps)
   - First stable closed timelike curve simulation
   - Self-consistency theorem: ψ_f(0) = ψ_b(0) required

2. **Rigorous Negative Result**
   - Temporal loops provide ZERO benefit for SHA-256 mining
   - Temporal structure ⊥ cryptographic optimization

3. **TSP Validation** (HHmL testing)
   - Temporal loops provide **0.54% improvement** on 100-city TSP
   - p < 0.01 (statistically significant)
   - Advantage scales with problem size

4. **Integration Hypothesis**
   - HHmL has continuous observables (vortex density, quality)
   - Smooth fitness landscape → temporal loops should help
   - Expected: 0.5-2% vortex density improvement, longer persistence

**Fork created** to preserve HHmL baseline while building spatiotemporal framework.

---

## 🏗️ Architecture Differences

| Feature | HHmL (Parent) | tHHmL (This Fork) |
|---------|---------------|-------------------|
| **Spatial** | 2D Möbius strip | 2D Möbius strip (inherited) |
| **Temporal** | Evolution parameter | 1D Möbius loop (NEW) |
| **Spacetime** | (2+0)D boundary | **(2+1)D boundary** |
| **Causality** | Forward-only | Forward + Backward (retrocausal) |
| **RNN Params** | 23 spatial | **32** (23 spatial + 9 temporal) |
| **Field** | ψ(θ) | **ψ(θ, t)** |
| **Time Symmetry** | Asymmetric | Emergent time reversal (NEW) |

---

## 🚀 New Capabilities (tHHmL-Specific)

### 1. Temporal Fixed Points
- Measure forward-backward divergence D(t) = |ψ_f(t) - ψ_b(t)|
- Detect time steps where ψ_forward = ψ_backward (self-consistent time loops)
- Track temporal convergence across cycles

### 2. Vortex Persistence Across Time
- Track vortex lifetime over temporal loop (not just density snapshots)
- Self-consistent vortices = survive temporal loop = topologically protected
- Expected: 5-10× longer vortex lifetime vs. HHmL

### 3. Retrocausal Vortex Guidance
- Future vortex quality influences past formation
- Prophetic feedback guides evolution toward high-quality states
- Expected: escape local minima, higher sustained density

### 4. Emergent Time Symmetry
- Measure CPT-like symmetry in temporal evolution
- Test thermodynamic consistency (second law in temporal loop)
- Detect emergent time-reversal invariance

### 5. Spatiotemporal Holography
- (2+1)D boundary → (3+1)D emergent bulk
- Temporal dimension emerges from boundary loops
- Full AdS/CFT analog with time

---

## 📊 Expected Outcomes

Based on TSP validation (+0.54% improvement):

### Optimistic Scenario
- **Vortex density**: +0.5-2% improvement
- **Vortex lifetime**: 5-10× increase (temporal fixed points = stable vortices)
- **Sustained density**: 100% maintained (not just peak)
- **Novel phenomena**: Time symmetry, emergent bulk time, spacetime correlations

### Pessimistic Scenario
- **No improvement**: Temporal structure decoupled from spatial dynamics
- **Still valuable**: Rigorous negative result (establishes fundamental limits)
- **Publication**: "Spatial-Temporal Decoupling in Topological Systems"

---

## 💻 Quick Start

### Installation

```bash
# Clone tHHmL fork
git clone https://github.com/Zynerji/tHHmL.git
cd tHHmL

# Install dependencies (same as HHmL)
pip install -e .
```

### Run Spatiotemporal Training

```python
from thhml.core.spatiotemporal import SpatiotemporalMobiusStrip
from thhml.ml.training import SpatiotemporalRNN

# Initialize (2+1)D spacetime
spacetime = SpatiotemporalMobiusStrip(
    num_nodes=4000,        # Spatial nodes
    num_time_steps=50,     # Temporal resolution
    device='cuda'
)

# Initialize RNN (32 parameters: 23 spatial + 9 temporal)
rnn = SpatiotemporalRNN(hidden_dim=4096)

# Train with temporal loop dynamics
for cycle in range(1000):
    # ... training loop with temporal convergence ...
```

See `examples/training/train_spatiotemporal_basic.py` for complete example.

---

## 📁 New Module Structure

```
tHHmL/
├── FORK_NOTES.md                    # Why this fork exists
├── src/
│   └── thhml/                       # New package (renamed from hhml)
│       ├── core/
│       │   ├── mobius/              # Inherited from HHmL
│       │   └── spatiotemporal/      # NEW: Spatiotemporal framework
│       │       ├── spacetime_mobius.py
│       │       ├── temporal_dynamics.py
│       │       └── retrocausal_coupling.py
│       ├── ml/
│       │   └── training/
│       │       └── spatiotemporal_rnn.py  # NEW: 32-parameter RNN
│       └── (all other HHmL modules inherited)
├── examples/
│   └── training/
│       ├── train_spatiotemporal_basic.py
│       └── train_temporal_loop_4k.py
└── docs/
    ├── TEMPORAL_LOOP_INTEGRATION.md
    └── SPATIOTEMPORAL_FRAMEWORK.md
```

---

## 🔗 Relationship to Parent (HHmL)

### Inherited
- ✅ All spatial modules (resonance, GFT, tensor networks)
- ✅ 23 spatial RNN parameters
- ✅ Vortex detection, quality metrics
- ✅ Visualization, monitoring

### New in tHHmL
- ➕ Spatiotemporal Möbius class (θ, t both topological)
- ➕ Temporal dynamics (forward/backward evolution)
- ➕ Retrocausal coupling mechanisms
- ➕ 9 temporal RNN parameters
- ➕ Temporal observables (divergence, fixed points, persistence)

### Backward Compatibility
- Set all temporal parameters to 0 → recovers HHmL behavior
- Toggle temporal loops on/off for comparison
- Baseline mode = parent HHmL

---

## 🗺️ Development Roadmap

### Phase 1: Core Framework (Week 1) - **IN PROGRESS**
- [x] Fork HHmL repository
- [x] Create FORK_NOTES.md
- [ ] Implement `SpatiotemporalMobiusStrip` class
- [ ] Extend RNN to 32 parameters
- [ ] Create training loop with temporal convergence

### Phase 2: Small-Scale Testing (Week 2)
- [ ] Test on 4K nodes, 50 time steps
- [ ] Measure vortex persistence vs. HHmL baseline
- [ ] Verify temporal fixed point convergence

### Phase 3: Analysis (Week 2-3)
- [ ] Hyperparameter search (α, β, τ, T)
- [ ] Correlation analysis (temporal params vs observables)
- [ ] Identify optimal configurations

### Phase 4: Full-Scale (Week 3-4)
- [ ] Scale to 20M nodes (if promising)
- [ ] 1000-cycle training run
- [ ] Compare HHmL vs tHHmL systematically

### Phase 5: Publication (Week 4+)
- [ ] Generate comprehensive whitepaper
- [ ] Document novel temporal phenomena
- [ ] Publish as tHHmL discovery package

---

## 📚 Key References

**Parent Discoveries**:
- [Hash Quine Emergence](../HHmL/HASH-QUINE/) (312-371× self-similarity)
- [Perfect Temporal Loop](../HHmL/PERFECT-TEMPORAL-LOOP/) (100% fixed points)
- [TSP Validation](../HHmL/simulations/optimization/) (+0.54% improvement)

**Theoretical Foundation**:
- Deutsch (1991): Closed timelike curves and quantum computation
- Chiribella et al. (2013): Quantum superposition of causal orders
- Price (1996): Retrocausal interpretations of QM

---

## 🔬 Comparison to HHmL

| Metric | HHmL | tHHmL (Expected) |
|--------|------|------------------|
| Vortex density (peak) | 100% | 100% (same) |
| Vortex density (sustained) | 82% | **85-90%** (+2-8%) |
| Vortex lifetime | 10-20 cycles | **50-200 cycles** (5-10×) |
| Temporal fixed points | N/A | **90-100%** |
| Training complexity | O(N) | O(N×T) |
| Novel phenomena | Hash quines | Time symmetry, emergent bulk time |

---

## 📄 Citation

If you use tHHmL, please cite both the fork and parent:

```bibtex
@software{thhml2025,
  title={tHHmL: temporal Holo-Harmonic Möbius Lattice},
  author={HHmL Research Collaboration},
  year={2025},
  month={December},
  note={Research fork of HHmL integrating temporal loop dynamics},
  url={https://github.com/Zynerji/tHHmL}
}

@software{hhml2025,
  title={HHmL: Holo-Harmonic Möbius Lattice},
  author={HHmL Research Collaboration},
  year={2025},
  url={https://github.com/Zynerji/HHmL}
}
```

---

## 🔗 Links

- **This Fork**: https://github.com/Zynerji/tHHmL (to be created)
- **Parent (HHmL)**: https://github.com/Zynerji/HHmL
- **Fork Notes**: [FORK_NOTES.md](FORK_NOTES.md)
- **Contact**: [@Conceptual1](https://twitter.com/Conceptual1)

---

## ⚖️ License

Same as parent HHmL project (to be determined).

---

**Last Updated**: 2025-12-18
**Status**: Initial fork, framework development starting
**Maintainer**: HHmL Research Collaboration
