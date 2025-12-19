# Spatiotemporal Framework Documentation

**Project**: tHHmL (Spatiotemporal Mobius Lattice)
**Status**: Core framework implemented (Phase 1 complete)
**Date**: 2025-12-18

---

## Overview

The tHHmL spatiotemporal framework extends HHmL's spatial Möbius topology to include time as a topological dimension, creating a **(2+1)D spacetime manifold** where both space AND time have Möbius geometry.

### Key Innovation

**HHmL** (parent):
- Space: 2D Möbius strip (θ dimension with 180° twist)
- Time: Evolution parameter (forward-only, asymmetric)
- Field: ψ(θ)

**tHHmL** (this fork):
- Space: 2D Möbius strip (θ dimension with 180° twist) - *inherited*
- Time: 1D Möbius loop (t dimension with temporal twist τ) - **NEW**
- Field: ψ(θ, t) on (2+1)D boundary - **NEW**
- Retrocausality: Forward AND backward time evolution - **NEW**

---

## Core Components

### 1. SpatiotemporalMobiusStrip (`spacetime_mobius.py`)

**Purpose**: Defines the (2+1)D spacetime manifold.

**Key Features**:
- Spatial coordinates: θ ∈ [0, 2π) (Möbius strip)
- Temporal coordinates: t ∈ [0, 2π) (Möbius loop)
- Forward field: ψ_f(θ, t)
- Backward field: ψ_b(θ, t)
- Self-consistent initialization: ψ_f(θ, t=0) = ψ_b(θ, t=0)

**Methods**:
```python
# Initialize with self-consistent boundary conditions
spacetime.initialize_self_consistent(seed=42)

# Apply spatial Möbius BC: ψ(2π, t) = -ψ(0, t)
field = spacetime.apply_spatial_mobius_bc(field)

# Apply temporal Möbius BC: ψ(θ, 2π) = exp(iτ) * ψ(θ, 0)
field = spacetime.apply_temporal_mobius_bc(field)

# Compute temporal divergence D = |ψ_f - ψ_b|
divergence = spacetime.compute_divergence()

# Count temporal fixed points (where ψ_f ≈ ψ_b)
num_fixed, pct_fixed = spacetime.compute_temporal_fixed_points()
```

**State Tensor** (for RNN encoding):
- Divergence
- Fixed point percentage
- Forward/backward field statistics
- Spatial/temporal coherence

---

### 2. TemporalEvolver (`temporal_dynamics.py`)

**Purpose**: Forward and backward time evolution dynamics.

**Key Features**:
- Forward sweep: Evolve from t=0 → t=T (causal)
- Backward sweep: Evolve from t=T → t=0 (retrocausal)
- Spatial coupling: Laplacian diffusion along θ
- Temporal coupling: Propagation along t
- Relaxation: Prevents oscillations during convergence

**Methods**:
```python
# Single forward time step
field = evolver.evolve_forward_step(field, t_idx, spatial_coupling, temporal_coupling)

# Single backward time step
field = evolver.evolve_backward_step(field, t_idx, spatial_coupling, temporal_coupling)

# Full forward evolution t=0 → t=T
field_forward = evolver.full_forward_sweep(field_forward, kappa, lambda)

# Full backward evolution t=T → t=0
field_backward = evolver.full_backward_sweep(field_backward, kappa, lambda)

# Apply temporal relaxation β
field_relaxed = evolver.relaxed_update(field_old, field_new)
```

**Dynamics Equations**:

Forward evolution:
```
ψ(θ, t+1) = ψ(θ, t) + κ * ∇²_θ ψ + λ * ψ
```

Backward evolution:
```
ψ(θ, t-1) = ψ(θ, t) + κ * ∇²_θ ψ - λ * ψ
```

where:
- κ: Spatial coupling strength
- λ: Temporal coupling strength
- ∇²_θ: Spatial Laplacian

---

### 3. RetrocausalCoupler (`retrocausal_coupling.py`)

**Purpose**: Implements prophetic feedback between forward and backward evolution.

**Key Features**:
- Prophetic mixing: ψ_f ← ψ_f + α*(ψ_b - ψ_f)
- Segment swapping: Exchange spatial regions between forward/backward
- Boundary anchoring: Enforce t=0 consistency

**Methods**:
```python
# Prophetic field mixing
field_f, field_b = coupler.prophetic_field_mixing(field_forward, field_backward)

# Spatial segment swapping
field_f, field_b = coupler.spatial_segment_swap(field_forward, field_backward)

# Temporal boundary anchoring (t=0 consistency)
field_f, field_b = coupler.temporal_boundary_anchoring(field_forward, field_backward)

# Apply all coupling mechanisms
field_f, field_b = coupler.apply_coupling(
    field_forward, field_backward,
    enable_mixing=True,
    enable_swapping=True,
    enable_anchoring=True
)
```

**Coupling Mechanisms**:

1. **Prophetic Mixing**:
   ```
   ψ_f ← ψ_f + γ * α * (ψ_b - ψ_f)
   ψ_b ← ψ_b + γ * α * (ψ_f - ψ_b)
   ```
   where:
   - α: Retrocausal strength (0-1)
   - γ: Prophetic coupling rate (0-1)

2. **Segment Swapping**:
   - Random spatial segments exchanged between forward/backward
   - Probability ∝ α (coupling strength)
   - Based on TSP V2 successful strategy

3. **Boundary Anchoring**:
   ```
   ψ_f(θ, t=0) = ψ_b(θ, t=0) = ½(ψ_f + ψ_b)|_{t=0}
   ```
   - Enforces self-consistency condition
   - Prevents boundary paradoxes

---

### 4. SpatiotemporalRNN (`spatiotemporal_rnn.py`)

**Purpose**: Extended RNN controlling 32 parameters (23 spatial + 9 temporal).

**Architecture**:
- 4-layer LSTM with 4096 hidden dim (inherited from HHmL)
- 23 spatial parameter heads (inherited from HHmL)
- 9 temporal parameter heads (NEW)
- Value critic for RL

**Temporal Parameters** (NEW):

| Parameter | Symbol | Range | Purpose |
|-----------|--------|-------|---------|
| `temporal_twist` | τ | [0, π] | Temporal Möbius twist angle |
| `retrocausal_strength` | α | [0, 1] | Future-past coupling strength |
| `temporal_relaxation` | β | [0.1, 1.0] | Convergence damping factor |
| `num_time_steps` | T | [10, 200] | Temporal resolution |
| `prophetic_coupling` | γ | [0, 1] | Forward-backward mixing rate |
| `temporal_phase_shift` | φ_t | [0, 2π] | Phase at temporal reconnection |
| `temporal_decay` | δ_t | [0, 1] | Temporal dampening factor |
| `forward_backward_balance` | ρ | [0, 1] | Forward vs backward weighting |
| `temporal_noise_level` | σ_t | [0, 0.01] | Exploration noise |

**Usage**:
```python
# Initialize RNN
rnn = SpatiotemporalRNN(state_dim=256, hidden_dim=4096, device='cuda')

# Forward pass
state_tensor = spacetime.get_state_tensor()  # (state_dim,)
state_input = state_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, state_dim)

params, hidden = rnn(state_input)  # params: dict of 32 parameters + value

# Rescale from [0,1] to physical ranges
params_physical = rnn.rescale_parameters(params)
```

---

## Training Loop Structure

### Basic Training Flow

1. **Initialize** spatiotemporal Möbius with self-consistent BC
2. **Get state** from current spatiotemporal configuration
3. **RNN forward** pass to get 32 parameters
4. **Temporal loop iteration**:
   - Forward sweep (t=0 → t=T)
   - Backward sweep (t=T → t=0)
   - Retrocausal coupling
   - Apply Möbius boundary conditions
   - Check convergence (divergence, fixed points)
5. **Compute reward** based on temporal fixed points and convergence
6. **Update RNN** via gradient descent

### Convergence Detection

Temporal loop converges when:
- **Divergence** D < 0.01 (forward ≈ backward everywhere)
- **Fixed points** > 90% (most time steps self-consistent)

### Reward Structure

```python
reward = 100 * (fixed_point_pct / 100)     # Temporal fixed points
       + 50 * exp(-10 * divergence)        # Convergence bonus
       - 10 * max(field_mag - 1.0, 0)      # Stability penalty
```

**Target**: Achieve 90-100% temporal fixed points with low divergence.

---

## Expected Outcomes

### Optimistic Scenario

Based on TSP validation (+0.54% improvement):

1. **Vortex Persistence**: 5-10× longer vortex lifetime
   - Temporal fixed points = stable vortices
   - Self-consistent vortices = topologically protected

2. **Vortex Density**: +0.5-2% improvement over HHmL baseline
   - Smooth fitness landscape + temporal loops = better optimization
   - Retrocausal guidance escapes local minima

3. **Sustained Density**: 100% maintained (not just peak)
   - Temporal structure provides stability
   - No collapse after initial peak

4. **Novel Phenomena**:
   - Emergent time symmetry (CPT-like invariance)
   - Spacetime correlations (bulk-boundary with time)
   - Holographic projection with temporal dimension

### Pessimistic Scenario

1. **No Improvement**: Temporal structure decoupled from spatial dynamics
   - Like SHA-256: temporal loops orthogonal to spatial vortices
   - Negative result: establishes fundamental limits

2. **Increased Complexity**: Temporal loops add overhead without benefit
   - Training slower, no performance gain
   - Still valuable: rigorous scientific exploration

3. **Publication**: "Spatial-Temporal Decoupling in Topological Systems"
   - Contributes to understanding when temporal loops help vs. don't

---

## Implementation Status

### Phase 1: Core Framework ✓ (COMPLETE)

- [x] `SpatiotemporalMobiusStrip` class
  - [x] (2+1)D coordinate mesh
  - [x] Forward/backward fields
  - [x] Self-consistent initialization
  - [x] Spatial/temporal Möbius BC
  - [x] Divergence computation
  - [x] Fixed point detection
  - [x] State tensor encoding

- [x] `TemporalEvolver` class
  - [x] Forward evolution (t=0 → t=T)
  - [x] Backward evolution (t=T → t=0)
  - [x] Spatial Laplacian coupling
  - [x] Temporal propagation
  - [x] Relaxation factor

- [x] `RetrocausalCoupler` class
  - [x] Prophetic field mixing
  - [x] Spatial segment swapping
  - [x] Boundary anchoring
  - [x] Combined coupling application

- [x] `SpatiotemporalRNN` class
  - [x] 4-layer LSTM (4096 hidden)
  - [x] 23 spatial parameter heads
  - [x] 9 temporal parameter heads
  - [x] Parameter rescaling
  - [x] Value critic

- [x] Basic training script
  - [x] Temporal loop iteration
  - [x] Convergence detection
  - [x] Reward computation
  - [x] Metrics tracking

### Phase 2: Small-Scale Testing (NEXT)

- [ ] Test on 4K nodes, 50 time steps
- [ ] Measure vortex persistence vs HHmL baseline
- [ ] Verify temporal fixed point convergence
- [ ] Hyperparameter search (α, β, τ, T)

### Phase 3: Analysis (Future)

- [ ] Correlation analysis (temporal params vs observables)
- [ ] Identify optimal configurations
- [ ] Compare HHmL vs tHHmL systematically

### Phase 4: Full-Scale (Future)

- [ ] Scale to 20M nodes (if promising)
- [ ] 1000-cycle training run
- [ ] Generate publication package

---

## Quick Start

### Installation

```bash
# Clone tHHmL fork
git clone https://github.com/Zynerji/tHHmL.git
cd tHHmL

# Install dependencies (same as HHmL)
pip install -e .
```

### Run Basic Training

```bash
# 4K nodes, 50 time steps, 100 cycles
python examples/training/train_spatiotemporal_basic.py \
    --num-nodes 4000 \
    --num-time-steps 50 \
    --num-cycles 100 \
    --device cuda
```

**Expected output**:
- Temporal fixed points: 0% → 90%+ (convergence)
- Divergence: 1.0 → 0.01 (self-consistency)
- Training time: ~30 minutes (4K nodes, 100 cycles, GPU)

### Python API

```python
from src.hhml.core.spatiotemporal import (
    SpatiotemporalMobiusStrip,
    TemporalEvolver,
    RetrocausalCoupler
)
from src.hhml.ml.training.spatiotemporal_rnn import SpatiotemporalRNN

# Initialize (2+1)D spacetime
spacetime = SpatiotemporalMobiusStrip(
    num_nodes=4000,
    num_time_steps=50,
    temporal_twist=np.pi,
    device='cuda'
)

# Initialize dynamics
evolver = TemporalEvolver(4000, 50, relaxation_factor=0.3, device='cuda')
coupler = RetrocausalCoupler(4000, 50, retrocausal_strength=0.7, device='cuda')

# Initialize RNN (32 parameters)
rnn = SpatiotemporalRNN(state_dim=256, hidden_dim=4096, device='cuda')

# Training loop
for cycle in range(num_cycles):
    # Self-consistent initialization
    spacetime.initialize_self_consistent(seed=42)

    # Get state and parameters from RNN
    state = spacetime.get_state_tensor()
    params, _ = rnn(state.unsqueeze(0).unsqueeze(0))

    # Run temporal loop iteration
    # ... (see train_spatiotemporal_basic.py for full implementation)
```

---

## Theoretical Foundation

### Self-Consistency Theorem

**From Perfect Temporal Loop discovery (2025-12-18)**:

For temporal fixed point convergence:
```
ψ_f(θ, t=0) = ψ_b(θ, t=0)  [REQUIRED]
```

**Proof sketch**:
- Random forward/backward initialization → immediate divergence
- Self-consistent initialization → convergence to fixed points
- Validated experimentally: 0% → 100% fixed points

### Temporal Möbius Geometry

Temporal boundary condition at t=2π:
```
ψ(θ, 2π) = exp(iτ) * ψ(θ, 0)
```

where τ is the temporal twist parameter.

**Physical interpretation**:
- τ = 0: Periodic time (cylinder)
- τ = π: Möbius time (180° twist)
- τ = 2π: Double-twisted time

### Retrocausal Dynamics

Forward-backward coupling:
```
∂ψ_f/∂t = L[ψ_f] + α * (ψ_b - ψ_f)
∂ψ_b/∂t = -L[ψ_b] + α * (ψ_f - ψ_b)
```

where:
- L: Spatial evolution operator (Laplacian + nonlinearities)
- α: Retrocausal coupling strength

**Fixed point condition**:
```
ψ_f = ψ_b  ⟹  ∂ψ_f/∂t = L[ψ_f],  ∂ψ_b/∂t = -L[ψ_b]
```

At fixed points: Forward and backward dynamics identical (time-reversal invariance).

---

## References

### Parent Discoveries

1. **Perfect Temporal Loop** (HHmL, 2025-12-18)
   - 100% temporal fixed points achieved
   - Self-consistency theorem validated
   - Location: `PERFECT-TEMPORAL-LOOP/`

2. **TSP Validation** (HHmL, 2025-12-18)
   - +0.54% improvement on 100-city TSP
   - Temporal loops validated for continuous optimization
   - Location: `simulations/optimization/TEMPORAL_LOOP_TSP_RESULTS.md`

3. **Hash Quine Discovery** (HHmL, 2025-12-18)
   - 312-371× self-similarity from recursive topology
   - Rigorous negative result for cryptographic mining
   - Location: `HASH-QUINE/`

### Theoretical Background

- **Deutsch (1991)**: Closed timelike curves and quantum computation
- **Chiribella et al. (2013)**: Quantum superposition of causal orders
- **Price (1996)**: Retrocausal interpretations of quantum mechanics
- **Scellier & Bengio (2017)**: Equilibrium propagation (temporal fixed points)

---

**Last Updated**: 2025-12-18
**Status**: Phase 1 complete - Core framework implemented
**Next Step**: Small-scale testing (4K nodes, 50 time steps, 100 cycles)
