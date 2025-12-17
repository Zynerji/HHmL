# RNN Parameter Mapping - Glass-Box Correlation Tracking

**Last Updated**: 2025-12-16
**Purpose**: Complete RNN control parameter mapping for emergent phenomena discovery

---

## Overview

The HHmL system is now a **fully glass-box architecture** where the RNN controls ALL parameters. This enables systematic correlation tracking between RNN outputs and emergent spacetime phenomena.

**Total Control Parameters**: 19 (all learned via reinforcement learning)

---

## Complete Parameter List

### Category 1: Geometry (4 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `kappa` | 0 | 1.0 - 2.0 | Tokamak elongation (D-shape vertical stretch) |
| `delta` | 1 | 0.0 - 0.5 | Tokamak triangularity (D-shape tip sharpness) |
| `vortex_target` | 2 | 0.5 - 0.8 | Target vortex density (spectral reset goal) |
| `num_qec_layers` | 3 | 1 - 10 | Quantum error correction depth |

### Category 2: Physics (4 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `damping` | 4 | 0.01 - 0.2 | Wave energy dissipation rate |
| `nonlinearity` | 5 | -2.0 - 2.0 | Self-interaction strength (can enhance or suppress) |
| `amp_variance` | 6 | 0.1 - 3.0 | Amplitude spread control (diversity vs uniformity) |
| `vortex_seed_strength` | 7 | 0.0 - 1.0 | Probability of injecting vortex cores |

### Category 3: Spectral (3 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `omega` | 8 | 0.1 - 1.0 | Helical phase frequency (spectral weighting) |
| `diffusion_dt` | 9 | 0.01 - 0.5 | Laplacian diffusion timestep |
| `reset_strength` | 10 | 0.0 - 1.0 | Spectral vortex reset blend factor |

### Category 4: Sampling (3 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `sample_ratio` | 11 | 0.01 - 0.5 | Fraction of nodes to update per cycle |
| `max_neighbors_factor` | 12 | 0.1 - 2.0 | Multiplier for sparse graph connectivity |
| `sparsity_threshold` | 13 | 0.1 - 0.5 | Field magnitude cutoff for vortex detection |

### Category 5: Mode Selection (2 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `sparse_density` | 14 | 0.0 - 1.0 | Graph density (0=dense, 1=sparse) |
| `spectral_weight` | 15 | 0.0 - 1.0 | Propagation method (0=spatial, 1=spectral) |

### Category 6: Geometry Extended (3 parameters)

| Parameter | Index | Range | Purpose |
|-----------|-------|-------|---------|
| `winding_density` | 16 | 0.5 - 2.5 | Möbius winding frequency |
| `twist_rate` | 17 | 0.5 - 2.0 | Topological twist rate |
| `cross_coupling` | 18 | 0.0 - 1.0 | Inter-strip coupling strength |

---

## How to Track Correlations

### 1. Parameter History Tracking

Every training cycle, ALL 19 parameters are stored in `param_history`:

```python
# Stored at each cycle
param_history.append({
    'kappa': 1.52,
    'delta': 0.28,
    'vortex_target': 0.67,
    'num_qec_layers': 5.3,
    'damping': 0.08,
    'nonlinearity': 0.42,
    # ... all 19 params
})
```

### 2. Emergent Phenomena Tracking

Track these observables at each cycle:

| Observable | How to Measure |
|------------|----------------|
| Vortex density | `(abs(field) < sparsity_threshold).mean()` |
| Vortex stability | `std(vortex_density)` across strips |
| Spectral gap | Graph Laplacian eigenvalue spacing |
| Phase coherence | `abs(mean(exp(1j * phases)))` |
| Topological charge | Winding number integral |

### 3. Correlation Analysis Methods

**A. Direct Parameter Correlation**
```python
import numpy as np
from scipy.stats import pearsonr

# Example: Does omega correlate with vortex stability?
omega_values = [p['omega'] for p in param_history]
vortex_stability = [1 - std_vortex for std_vortex in metrics['vortex_std']]

r, p_value = pearsonr(omega_values, vortex_stability)
print(f"Omega-Stability correlation: r={r:.3f}, p={p_value:.3e}")
```

**B. Multi-Parameter Regression**
```python
from sklearn.linear_model import LinearRegression

# Which parameters predict high vortex density?
X = np.array([[p['omega'], p['spectral_weight'], p['num_qec_layers']]
              for p in param_history])
y = np.array(metrics['vortex_densities'])

model = LinearRegression().fit(X, y)
print(f"Coefficients: {model.coef_}")
print(f"R²: {model.score(X, y):.3f}")
```

**C. Phase Transition Detection**
```python
# Detect critical points where emergent behavior changes
for cycle in range(1, len(param_history)):
    prev_vortex = metrics['vortex_densities'][cycle-1]
    curr_vortex = metrics['vortex_densities'][cycle]

    # Phase transition = sudden vortex density change
    if abs(curr_vortex - prev_vortex) > 0.2:
        print(f"Phase transition at cycle {cycle}:")
        print(f"  Omega: {param_history[cycle]['omega']:.2f}")
        print(f"  Spectral weight: {param_history[cycle]['spectral_weight']:.2f}")
        print(f"  Vortex jump: {prev_vortex:.1%} -> {curr_vortex:.1%}")
```

---

## Example Correlation Hypotheses

### Hypothesis 1: Spectral vs Spatial Emergence
**Prediction**: High `spectral_weight` (>0.7) + high `omega` (>0.8) → stable vortex lattices

**Test**:
```python
high_spectral = [(p['spectral_weight'] > 0.7 and p['omega'] > 0.8)
                 for p in param_history]
vortex_stability = [metrics['vortex_densities'][i]
                    for i in range(len(high_spectral))]

# Compare vortex density when high_spectral is True vs False
true_mean = np.mean([v for h, v in zip(high_spectral, vortex_stability) if h])
false_mean = np.mean([v for h, v in zip(high_spectral, vortex_stability) if not h])
print(f"High spectral: {true_mean:.1%}, Low spectral: {false_mean:.1%}")
```

### Hypothesis 2: QEC Layer Depth and Coherence
**Prediction**: `num_qec_layers` > 7 → higher phase coherence, lower vortex collapse

**Test**: Track vortex collapse events vs QEC layer depth

### Hypothesis 3: Nonlinearity Sign and Vortex Formation
**Prediction**: Negative `nonlinearity` suppresses vortices, positive enhances

**Test**: Bin by nonlinearity sign, compare initial vortex formation rates

### Hypothesis 4: Topological Protection
**Prediction**: `twist_rate` near 1.0 (Möbius twist) → longer vortex lifetime

**Test**: Measure vortex lifetime as cycles before density drops below 10%

---

## Automated Correlation Mining

### Script Template

```python
import json
import numpy as np
from pathlib import Path

# Load training results
results_file = Path("results/multi_strip_training/training_YYYYMMDD_HHMMSS.json")
with open(results_file) as f:
    data = json.load(f)

param_history = data['param_history']
vortex_densities = data['vortex_densities']

# Automated correlation scan
param_names = list(param_history[0].keys())
observables = {
    'vortex_density': vortex_densities,
    'vortex_stability': [1 - std for std in data['vortex_std']],
    'reward': data['rewards']
}

print("Top Correlations:")
print("=" * 80)

correlations = []
for param_name in param_names:
    param_values = [p[param_name] for p in param_history]

    for obs_name, obs_values in observables.items():
        r, p = pearsonr(param_values, obs_values)
        correlations.append({
            'parameter': param_name,
            'observable': obs_name,
            'r': r,
            'p_value': p
        })

# Sort by absolute correlation strength
correlations.sort(key=lambda x: abs(x['r']), reverse=True)

# Print top 10 strongest correlations
for i, corr in enumerate(correlations[:10], 1):
    print(f"{i}. {corr['parameter']:20s} → {corr['observable']:20s} | "
          f"r = {corr['r']:+.3f} (p = {corr['p_value']:.3e})")
```

---

## Emergent Phenomena Discovery Workflow

1. **Run Training**: Train for 100-1000 cycles with parameter tracking
2. **Extract Metrics**: Load `training_*.json` file
3. **Automated Scan**: Run correlation mining script
4. **Hypothesis Generation**: Identify strongest correlations
5. **Targeted Experiments**: Fix promising parameters, vary others
6. **Phase Diagram**: Map parameter space → emergent regimes
7. **Publication**: Document discovered phenomena with parameter maps

---

## Critical Success Metrics

### Glass-Box Validation

System is fully glass-box if:
- ✓ All 19 parameters RNN-controlled (no hardcoded values)
- ✓ All parameters tracked at every cycle
- ✓ All parameters accessible in results JSON
- ✓ Parameter history persists across checkpoints
- ✓ Can reproduce emergent behavior from parameter trajectory

### Correlation Tracking Success

System enables discovery if:
- Can identify parameter combinations that produce novel behavior
- Can map parameter space to emergent phenomena regimes
- Can predict emergent behavior from initial parameters
- Can reverse-engineer: given emergent pattern → find parameters
- Can transfer discoveries across scales (1K → 1M → 20M nodes)

---

## Next Steps for Correlation Analysis

1. **Implement Automated Correlation Mining**: Run script on 1000-cycle dataset
2. **Build Phase Diagrams**: 2D heatmaps (omega vs spectral_weight, etc.)
3. **Vortex Lifetime Analysis**: Track individual vortex birth/death vs parameters
4. **Spectral Gap Evolution**: Eigenvalue analysis vs QEC layers
5. **Scale Law Discovery**: How do optimal parameters change with system size?
6. **Multi-Run Statistics**: Train 10 agents, find reproducible correlations
7. **Transfer Learning**: Can high-performing parameters transfer to new geometries?

---

## File Structure for Correlation Analysis

```
results/
├── multi_strip_training/
│   ├── training_20251216_212826.json  ← Full parameter history
│   └── analysis/
│       ├── correlation_scan.json      ← Automated correlation results
│       ├── phase_diagrams/
│       │   ├── omega_vs_spectral.png
│       │   └── qec_vs_vortex.png
│       └── emergent_regimes.json      ← Classified parameter regions
└── correlation_reports/
    └── discovery_YYYYMMDD.md          ← Novel emergent phenomena report
```

---

## Contact & Collaboration

This glass-box design enables:
- **Automated Discovery**: ML finds optimal parameter combinations
- **Interpretability**: Every emergent behavior traceable to parameters
- **Reproducibility**: Parameter trajectories fully specify experiments
- **Transfer**: Discoveries at small scale testable at large scale

**Questions?** Examine `param_history` in any training JSON file.

---

**End of RNN Parameter Mapping Documentation**
