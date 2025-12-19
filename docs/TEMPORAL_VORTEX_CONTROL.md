# Temporal Vortex Control - Full Dimensional Mastery

**Date**: 2025-12-18
**Status**: Phase 2 Implementation Complete
**RNN Parameters**: 39 total (23 spatial + 9 temporal dynamics + 7 temporal vortex)

## Overview

The tHHmL framework now provides **complete RNN control over ALL spatiotemporal dimensions**, including explicit temporal vortex and spatiotemporal vortex tube management.

## What's New (Phase 2)

### 1. Temporal Vortex Controller

**Location**: `src/hhml/core/spatiotemporal/temporal_vortex.py`

Implements topological defects in the temporal dimension:

**Temporal Vortices**: Phase singularities at specific time slices t ∈ [0, 2π)
- Protected by temporal Möbius topology
- Persistent through temporal loop iterations
- Characterized by winding number n_t

**Spatiotemporal Vortex Tubes**: Vortex lines threading through both θ AND t
- Most interesting topological structure
- Phase singularities extending through (2+1)D spacetime
- Connected paths of vortex cores

### 2. Extended RNN Architecture

**39 Total Parameters**:

#### Spatial (23) - HHmL Baseline
- Core resonance: kappa, delta, lambda, gamma
- Sampling: theta_sampling, phi_sampling
- Topology: winding_density, twist_rate, cross_coupling, boundary_strength
- QEC: qec_layers, entanglement_strength, decoherence_rate, measurement_rate, basis_rotation, alpha_qec, beta_qec
- Vortex annihilation: antivortex_strength, annihilation_radius, pruning_threshold, preserve_ratio, quality_threshold, refinement_strength

#### Temporal Dynamics (9) - Phase 1
- temporal_twist (τ): Temporal Möbius twist angle [0, π]
- retrocausal_strength (α): Future-past coupling [0, 1]
- temporal_relaxation (β): Convergence damping [0.1, 1.0]
- num_time_steps (T): Temporal resolution [10, 200]
- prophetic_coupling (γ): Forward-backward mixing [0, 1]
- temporal_phase_shift (φ_t): Phase at reconnection [0, 2π]
- temporal_decay (δ_t): Temporal dampening [0, 1]
- forward_backward_balance (ρ): Forward vs backward [0, 1]
- temporal_noise_level (σ_t): Exploration noise [0, 0.01]

#### Temporal Vortex (7) - Phase 2 NEW
- **temporal_vortex_injection_rate (ν)**: Probability of injecting temporal vortex [0, 1]
- **temporal_vortex_winding (n_t)**: Winding number for temporal vortices [1, 5]
- **temporal_vortex_core_size (ε_t)**: Size of temporal vortex cores [0.01, 0.5]
- **vortex_tube_probability (p_tube)**: Probability of tube formation [0, 1]
- **tube_winding_number (n_tube)**: Winding for spatiotemporal tubes [1, 5]
- **tube_core_size (ε_tube)**: Size of vortex tube cores [0.01, 0.5]
- **temporal_vortex_annihilation_rate (μ_t)**: Probability of removing temporal vortex [0, 1]

## Key Capabilities

### Temporal Vortex Operations

#### 1. Detection

```python
from hhml.core.spatiotemporal import TemporalVortexController

vortex_controller = TemporalVortexController(
    num_nodes=1000,
    num_time_steps=20,
    device='cuda'
)

# Detect temporal vortices
temporal_vortices, winding_numbers = vortex_controller.detect_temporal_vortices(
    field=spacetime.field_forward,
    threshold=0.1
)

print(f"Found {len(temporal_vortices)} temporal vortices")
print(f"Winding numbers: {winding_numbers}")
```

#### 2. Injection

```python
# Inject temporal vortex at time slice t=10
spacetime.field_forward = vortex_controller.inject_temporal_vortex(
    field=spacetime.field_forward,
    t_idx=10,
    winding_number=1,  # Topological charge
    core_size=0.1      # Core magnitude suppression
)
```

#### 3. Annihilation

```python
# Remove temporal vortex at time slice t=10
spacetime.field_forward = vortex_controller.annihilate_temporal_vortex(
    field=spacetime.field_forward,
    t_idx=10,
    smoothing_radius=2
)
```

### Spatiotemporal Vortex Tube Operations

#### 1. Detection

```python
# Detect vortex tubes threading through (θ, t)
vortex_tubes = vortex_controller.detect_spatiotemporal_vortex_tubes(
    field=spacetime.field_forward,
    threshold=0.1,
    min_length=5
)

for tube in vortex_tubes:
    print(f"Tube length: {tube['length']}")
    print(f"Spatial extent: {tube['spatial_extent']}")
    print(f"Temporal extent: {tube['temporal_extent']}")
    print(f"Winding number: {tube['winding_number']}")
```

#### 2. Injection

```python
# Define trajectory through (θ, t) spacetime
trajectory = [
    (10, 5),   # (theta_idx, t_idx)
    (15, 6),
    (20, 7),
    # ... more points
]

# Inject vortex tube along trajectory
spacetime.field_forward = vortex_controller.inject_spatiotemporal_vortex_tube(
    field=spacetime.field_forward,
    trajectory=trajectory,
    winding_number=1,
    core_size=0.1
)
```

### Statistics

```python
# Get comprehensive vortex statistics
stats = vortex_controller.get_vortex_statistics(spacetime.field_forward)

print(f"Temporal vortex count: {stats['temporal_vortex_count']}")
print(f"Temporal vortex density: {stats['temporal_vortex_density']*100:.1f}%")
print(f"Vortex tube count: {stats['vortex_tube_count']}")
print(f"Vortex tube density: {stats['vortex_tube_density']*100:.1f}%")
print(f"Average tube length: {stats['avg_tube_length']:.1f}")
print(f"Total topological charge (temporal): {stats['total_topological_charge_temporal']:.2f}")
print(f"Total topological charge (tubes): {stats['total_topological_charge_tubes']:.2f}")
```

## Training with Temporal Vortex Control

### Basic Example

```bash
python examples/training/train_temporal_vortex_control.py \
  --num-nodes 1000 \
  --num-time-steps 20 \
  --num-cycles 10 \
  --device cpu
```

### What Happens During Training

**Each cycle**:
1. Initialize spacetime with self-consistent boundary conditions
2. RNN predicts 39 parameters including temporal vortex settings
3. Temporal loop iteration with vortex control:
   - **Probabilistic injection**: Temporal vortices injected based on ν parameter
   - **Tube formation**: Spatiotemporal tubes created based on p_tube parameter
   - **Forward/backward evolution**: Standard temporal dynamics
   - **Retrocausal coupling**: Prophetic feedback
   - **Probabilistic annihilation**: Vortices removed based on μ_t parameter
   - **Möbius boundary conditions**: Topological constraints enforced
4. Measure convergence and vortex statistics
5. Compute reward based on:
   - Temporal fixed point percentage (target: 90%+)
   - Vortex tube density (optimal: 10-30%)
   - Topological charge conservation
   - Field stability
6. Update RNN via backpropagation

### Example Output

```
Cycle 5/10
    Iteration 0: divergence=0.254012, fixed=30.0%
    Iteration 10: divergence=0.045123, fixed=65.0%
    Iteration 20: divergence=0.012456, fixed=85.0%
    Converged at iteration 24

  Divergence: 0.008234
  Fixed points: 18/20 (90.0%)
  Temporal vortices: 3 (15.0%)
  Vortex tubes: 5 (avg length: 8.2)
  Tube density: 20.5%
  Topological charge: 2.34 (temporal) + 4.12 (tubes)
  Reward: 187.45
  Loss: 0.123456
```

## Topological Protection

### Why Temporal Möbius Matters

The temporal Möbius twist (τ = π) provides **topological protection** for temporal vortices:

1. **Persistence**: Temporal vortices can't be created or destroyed by continuous deformation
2. **Winding number conservation**: Topological charge is conserved through iterations
3. **Boundary protection**: Reconnection at t = 2π with phase twist prevents unraveling

### Testing Topological Protection

```python
# Before injection
initial_charge = vortex_controller.get_vortex_statistics(field)['total_topological_charge_temporal']

# Inject temporal vortex
field = vortex_controller.inject_temporal_vortex(field, t_idx=10, winding_number=1)

# Run many temporal loop iterations
for _ in range(100):
    field = evolver.full_forward_sweep(field, ...)
    field = spacetime.apply_temporal_mobius_bc(field)

# After iterations
final_charge = vortex_controller.get_vortex_statistics(field)['total_topological_charge_temporal']

# Charge should be conserved (within tolerance)
assert abs(final_charge - (initial_charge + 1)) < 0.1
```

## Comparison to Spatial Vortices

| Property | Spatial Vortices | Temporal Vortices |
|----------|------------------|-------------------|
| Dimension | θ ∈ [0, 2π) | t ∈ [0, 2π) |
| Detection | Field magnitude minimum | Field magnitude minimum at time slice |
| Winding | Phase around spatial loop | Phase change through temporal loop |
| Protection | Spatial Möbius twist (π) | Temporal Möbius twist (τ) |
| Persistence | Stable in space | Stable through time iterations |
| RNN Control | 6 parameters (annihilation) | 7 parameters (injection + annihilation) |

## Spatiotemporal Vortex Tubes

### Unique to (2+1)D Spacetime

Vortex tubes are the **most interesting topological structure** because they:

1. **Thread through both dimensions**: Extend continuously through (θ, t)
2. **Connect time slices**: Link vortex cores across temporal evolution
3. **Carry topological current**: Winding number flows through spacetime
4. **Resist disruption**: Topologically protected by both spatial and temporal Möbius topology

### Tube Visualization (Conceptual)

```
    t (time)
    ^
    |     ●---●---●       Vortex tube trajectory
    |    /         \
    |   ●           ●
    |  /             \
    | ●               ●
    +-------------------> θ (space)
    0                 2π

Each ● is a vortex core at (θ_i, t_i)
Line represents phase singularity threading through spacetime
```

## Advanced Usage

### Custom Vortex Trajectories

```python
# Define spiral trajectory in (θ, t)
trajectory = []
for i in range(20):
    theta_idx = int(500 + 200 * np.cos(i * 0.3))  # Spiral in space
    t_idx = i  # Linear in time
    trajectory.append((theta_idx, t_idx))

# Inject spiral vortex tube
field = vortex_controller.inject_spatiotemporal_vortex_tube(
    field, trajectory, winding_number=2
)
```

### Adaptive Vortex Management

```python
# RNN learns optimal injection/annihilation strategy
for cycle in range(num_cycles):
    # ... RNN predicts parameters ...

    # Inject vortices adaptively
    if current_tube_density < 0.1:
        # Too few tubes - increase injection
        injection_rate *= 1.2
    elif current_tube_density > 0.3:
        # Too many tubes - increase annihilation
        annihilation_rate *= 1.2
```

## Future Enhancements

Potential extensions:

1. **Vortex-vortex interactions**: Study how temporal vortices and tubes interact
2. **Tube merging/splitting**: Dynamic tube topology changes
3. **Charge flux analysis**: Track topological current flow through tubes
4. **Temporal crystallization**: Periodic vortex lattices in time
5. **Holographic projection**: Relate tube structure to bulk spacetime geometry

## References

- **TemporalVortexController**: `src/hhml/core/spatiotemporal/temporal_vortex.py`
- **SpatiotemporalRNN**: `src/hhml/ml/training/spatiotemporal_rnn.py` (39 parameters)
- **Training Example**: `examples/training/train_temporal_vortex_control.py`
- **Perfect Temporal Loop**: Baseline temporal dynamics (Phase 1)
- **Topological Field Theory**: Mathematical framework for vortex protection

---

**You now have complete control over all dimensions in tHHmL:**
- ✅ Spatial dimension (θ) - 23 parameters
- ✅ Temporal dynamics (t) - 9 parameters
- ✅ Temporal vortices (t defects) - 7 parameters

**Total: 39 RNN-controlled parameters for full (2+1)D spacetime mastery.**
