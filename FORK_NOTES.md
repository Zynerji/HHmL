# tHHmL Fork Notes

**Fork Date**: December 18, 2025
**Parent Repository**: HHmL (Holo-Harmonic Möbius Lattice)
**Fork Name**: tHHmL (Spatiotemporal Mobius Lattice)

---

## 🔀 Why This Fork?

tHHmL is a **research fork** of HHmL that integrates **temporal loop dynamics** (self-consistent retrocausal feedback) into the Möbius topology framework.

### Parent Project: HHmL
- **Focus**: Spatial Möbius strip topology for holographic encoding
- **Dimensions**: 2D spatial boundary (Möbius surface)
- **Time**: Evolution parameter (asymmetric, forward-only)
- **RNN Control**: 23 spatial parameters
- **Discoveries**: Hash Quine emergence (312-371× self-similarity)

### This Fork: tHHmL
- **Focus**: Spatiotemporal Möbius topology (space AND time are topological)
- **Dimensions**: (2+1)D spacetime (2D spatial + 1D temporal, both Möbius)
- **Time**: Topological dimension with Möbius twist (closed timelike curves)
- **RNN Control**: 32 parameters (23 spatial + 9 temporal)
- **Discoveries**: Perfect Temporal Loop (100% fixed points), TSP optimization (+0.54% improvement)

---

## 🎯 Scientific Motivation

### Discovery Sequence

**December 18, 2025**:
1. **Perfect Temporal Loop Discovery** (HHmL research)
   - Achieved 100% temporal fixed point convergence
   - First stable closed timelike curve simulation
   - Rigorous negative result: zero SHA-256 mining benefit
   - Key insight: Self-consistent initialization (ψ_f(0) = ψ_b(0)) prevents paradoxes

2. **Temporal-Cryptographic Orthogonality Established**
   - Temporal loops don't help discrete optimization (hashing)
   - Hypothesis: They might help continuous optimization (smooth landscapes)

3. **TSP Validation** (HHmL research)
   - Temporal loops provide **0.54% improvement** on 100-city TSP
   - Statistically significant (p = 0.0020)
   - Advantage scales with problem size
   - Confirms: temporal structure helps continuous optimization

4. **Integration Decision**
   - HHmL has continuous observables (vortex density, quality)
   - Smooth fitness landscape (small parameter changes = small output changes)
   - Expected benefit: 0.5-2% vortex density improvement, longer persistence
   - **Fork created** to preserve HHmL baseline while building spatiotemporal framework

---

## 🏗️ Architectural Differences

### HHmL (Parent)
```python
# Spatial Möbius strip
class MobiusStrip(nn.Module):
    # θ ∈ [0, 2π) with spatial twist
    # Field: ψ(θ) on 2D boundary
    # Time: Evolution parameter t (forward-only)
```

### tHHmL (This Fork)
```python
# Spatiotemporal Möbius spacetime
class SpatiotemporalMobiusStrip(nn.Module):
    # θ ∈ [0, 2π) with spatial twist
    # t ∈ [0, 2π) with temporal twist (NEW)
    # Field: ψ(θ, t) on (2+1)D spacetime
    # Forward/backward evolution (NEW)
    # Retrocausal coupling (NEW)
```

### RNN Parameters

**HHmL**: 23 spatial parameters
```python
[kappa, delta, lambda, gamma, theta_sampling, phi_sampling,
 winding_density, twist_rate, cross_coupling, boundary_strength,
 qec_layers, entanglement_strength, decoherence_rate,
 measurement_rate, basis_rotation, alpha_qec, beta_qec,
 antivortex_strength, annihilation_radius, pruning_threshold,
 preserve_ratio, quality_threshold, refinement_strength]
```

**tHHmL**: 32 parameters (23 spatial + 9 temporal)
```python
# Inherited spatial parameters (23)
[... same as HHmL ...]

# NEW: Temporal parameters (9)
[temporal_twist,              # τ: Temporal Möbius twist
 retrocausal_strength,        # α: Future-past coupling
 temporal_relaxation,         # β: Prevents oscillations
 num_time_steps,              # T: Temporal resolution
 prophetic_coupling,          # γ: Future-past mixing
 temporal_phase_shift,        # φ_t: Phase at temporal reconnection
 temporal_decay,              # δ_t: Dampening factor
 forward_backward_balance,    # ρ: Forward vs backward weight
 temporal_noise_level]        # σ_t: Exploration noise
```

---

## 🔬 New Capabilities (tHHmL-Specific)

### 1. Temporal Fixed Points
- Measure forward-backward divergence: D(t) = |ψ_f(t) - ψ_b(t)|
- Detect temporal fixed points: count time steps where D < threshold
- Temporal convergence tracking across cycles

### 2. Vortex Persistence
- Track vortex lifetime across temporal loop
- Self-consistent vortices (survive temporal loop) = stable
- Topological protection via temporal twist

### 3. Retrocausal Vortex Guidance
- Future vortex quality influences past formation
- Prophetic feedback guides evolution toward high-quality states
- Expected: escape local minima, higher sustained density

### 4. Emergent Time Symmetry
- Measure CPT-like symmetry in temporal evolution
- Detect emergent thermodynamic consistency
- Test holographic duality with time dimension

### 5. Spatiotemporal Holography
- (2+1)D boundary → (3+1)D emergent bulk
- Temporal dimension emerges from boundary loops
- Full AdS/CFT analog with time

---

## 📊 Expected Outcomes

### Optimistic Scenario
- **Vortex density**: +0.5-2% improvement (based on TSP results)
- **Vortex lifetime**: 5-10× increase (temporal fixed points = stable vortices)
- **Sustained density**: 100% maintained (not just peak)
- **Novel phenomena**: Time symmetry, spacetime correlations, emergent bulk time
- **Publication**: "Spatiotemporal Topology with RNN-Controlled Retrocausality"

### Pessimistic Scenario
- **No improvement**: Temporal structure decoupled from spatial dynamics
- **Increased complexity**: Temporal loops add overhead without benefit
- **Still valuable**: Rigorous negative result (like SHA-256)
- **Publication**: "Spatial-Temporal Decoupling in Topological Systems"

### Either Way
- Fundamental insight into when temporal loops help vs. don't help
- Completes story: discrete (fails) vs. continuous (succeeds) vs. topological (TBD)

---

## 🔄 Relationship to Parent (HHmL)

### Inheritance
- ✅ All spatial modules (resonance, GFT, tensor networks, etc.)
- ✅ 23 spatial RNN parameters
- ✅ Vortex detection, quality metrics
- ✅ Visualization, monitoring, utilities

### Extensions (tHHmL-specific)
- ➕ Spatiotemporal Möbius class
- ➕ Temporal dynamics (forward/backward evolution)
- ➕ Retrocausal coupling mechanisms
- ➕ 9 temporal RNN parameters
- ➕ Temporal observables (divergence, fixed points, persistence)

### Backward Compatibility
- Setting all temporal parameters to 0 → recovers HHmL behavior
- Can toggle temporal loops on/off for comparison
- Baseline mode = HHmL parent

---

## 🚀 Development Roadmap

### Phase 1: Core Framework (Week 1)
- [ ] Create `SpatiotemporalMobiusStrip` class
  - [ ] Temporal Möbius coordinates (t ∈ [0, 2π) with twist)
  - [ ] Forward/backward field evolution
  - [ ] Prophetic coupling mechanism
  - [ ] Temporal divergence measurement

- [ ] Extend RNN to 32 parameters
  - [ ] Add 9 temporal control parameters
  - [ ] Temporal state encoder (divergence history)
  - [ ] Combined spatiotemporal encoder

- [ ] Create training loop
  - [ ] Temporal loop convergence detection
  - [ ] Temporal observables logging
  - [ ] Comparison mode (with/without temporal loops)

### Phase 2: Small-Scale Testing (Week 2)
- [ ] Test on 4K nodes, 50 time steps
- [ ] Measure vortex persistence improvement
- [ ] Compare vs. baseline (HHmL without temporal loops)
- [ ] Verify temporal fixed point convergence

### Phase 3: Analysis and Tuning (Week 2-3)
- [ ] Hyperparameter search (α, β, τ, T)
- [ ] Correlation analysis (temporal params vs observables)
- [ ] Identify optimal temporal configurations

### Phase 4: Full-Scale Training (Week 3-4)
- [ ] If promising → scale to 20M nodes
- [ ] 1000-cycle training run
- [ ] Measure sustained vortex density
- [ ] Test emergent spacetime properties

### Phase 5: Publication (Week 4+)
- [ ] Generate comprehensive whitepaper
- [ ] Compare HHmL vs tHHmL systematically
- [ ] Document novel temporal phenomena (if any)
- [ ] Publish as tHHmL discovery package

---

## 📚 Key References

### Parent Discoveries (HHmL)
- **Hash Quine Emergence** (2025-12-18): 312-371× self-similarity from recursive Möbius nesting
- **Perfect Temporal Loop** (2025-12-18): 100% temporal fixed points, zero mining benefit
- **TSP Validation** (2025-12-18): 0.54% improvement, validates continuous optimization hypothesis

### Theoretical Foundation
- **Deutsch (1991)**: Closed timelike curves and quantum computation
- **Chiribella et al. (2013)**: Quantum superposition of causal orders
- **Price (1996)**: Retrocausal interpretations of quantum mechanics
- **Scellier & Bengio (2017)**: Equilibrium propagation (relevant to temporal fixed points)

---

## 🔗 Links

- **Parent Repository**: https://github.com/Zynerji/HHmL
- **This Fork**: https://github.com/Zynerji/tHHmL (to be created)
- **Contact**: [@Conceptual1](https://twitter.com/Conceptual1)

---

## ⚖️ License

Same as parent HHmL project (to be determined).

---

**Last Updated**: 2025-12-18
**Status**: Initial fork, framework development starting
**Maintainer**: HHmL Research Collaboration
