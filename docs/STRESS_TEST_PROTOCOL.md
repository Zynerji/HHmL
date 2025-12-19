# tHHmL Stress Testing Protocol for Emergent Discovery

**Goal**: Push system to extremes to discover phase transitions, critical points, and emergent phenomena.

**Last Updated**: 2025-12-19

---

## 📊 Testing Matrix

### Dimension 1: Scale Stress (Node Count)

**Hypothesis**: Emergent phenomena often appear only at critical scales.

| Test | Nodes | Strips | Nodes/Strip | Expected VRAM | Runtime (10K cycles) | Emergent Potential |
|------|-------|--------|-------------|---------------|---------------------|-------------------|
| Baseline | 49,800 | 300 | 166 | 6-13 GB | 3.3 min | ✓ (current) |
| 2× Scale | 99,600 | 300 | 332 | 12-25 GB | 6.6 min | ★★ |
| 5× Scale | 249,000 | 500 | 498 | 30-60 GB | 16.5 min | ★★★ |
| 10× Scale | 498,000 | 1000 | 498 | 60-120 GB | 33 min | ★★★★★ |

**What to Look For**:
- Phase transitions at critical node counts
- Sudden changes in wormhole connectivity
- Self-organized criticality (power-law distributions)
- Fractal structures emerging at larger scales

**Commands**:
```bash
# 2× scale test
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-strips 300 --nodes-per-strip 332

# 5× scale test
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-strips 500 --nodes-per-strip 498

# 10× scale test (H200 required)
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-strips 1000 --nodes-per-strip 498
```

---

### Dimension 2: Temporal Resolution Stress

**Hypothesis**: Higher temporal resolution reveals dynamics invisible at coarse timesteps.

| Test | Time Steps | Retrocausal Coupling | Expected Emergent |
|------|-----------|---------------------|-------------------|
| Baseline | 10 | α=0.7, γ=0.3 | ✓ (current) |
| High-res | 50 | α=0.7, γ=0.3 | Oscillations, resonances |
| Ultra-res | 100 | α=0.7, γ=0.3 | Fine temporal structure |
| Adaptive | Variable | α=0.7, γ=0.3 | Self-adjusting dynamics |

**What to Look For**:
- Temporal oscillations (periodic attractors)
- Resonance modes (specific frequencies dominating)
- Causality violations (retrocausal feedback loops)
- Temporal fractals (self-similarity across timescales)

**Commands**:
```bash
# High-resolution temporal
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-time-steps 50

# Ultra-resolution temporal
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-time-steps 100
```

---

### Dimension 3: Retrocausal Coupling Sweep

**Hypothesis**: Critical coupling strengths trigger emergent phenomena.

| Test | α (coupling) | γ (prophetic) | Expected Behavior |
|------|-------------|--------------|-------------------|
| No retrocausality | 0.0 | 0.0 | Sequential baseline |
| Weak coupling | 0.3 | 0.1 | Partial temporal influence |
| **Current** | **0.7** | **0.3** | **100% fixed points** |
| Strong coupling | 0.9 | 0.5 | Potential instability |
| Critical coupling | 1.0 | 0.0 | Bifurcation point? |

**What to Look For**:
- Phase transition at α ≈ 0.5-0.7 (sudden fixed point emergence)
- Bifurcations (multiple stable attractors)
- Chaos onset (loss of stability at α > 0.9)
- Hysteresis (different behavior increasing vs decreasing α)

**Commands**:
```bash
# Sweep retrocausal coupling (5 runs)
for ALPHA in 0.0 0.3 0.5 0.7 0.9; do
    python train_tokamak_wormhole_hunt.py \
        --num-cycles 1000 \
        --retrocausal-alpha $ALPHA \
        --output-dir ~/results/alpha_sweep/alpha_$ALPHA
done
```

---

### Dimension 4: Topology Stress (Strip Configuration)

**Hypothesis**: Different Möbius configurations exhibit distinct emergent phenomena.

| Test | Strips | Topology | Expected Emergent |
|------|--------|----------|-------------------|
| Sparse | 50 | Few connections | Local structures |
| **Current** | **300** | **Intermediate** | **Inter-strip wormholes** |
| Dense | 1000 | High connectivity | Network phenomena |
| Hierarchical | 3+30+300 | Fractal | Scale-invariance |

**What to Look For**:
- Network phase transitions (percolation thresholds)
- Small-world properties (clustering + short paths)
- Hub emergence (some strips become highly connected)
- Hierarchical organization (clusters of clusters)

**Commands**:
```bash
# Dense strips test
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-strips 1000 --nodes-per-strip 166

# Hierarchical test (requires custom geometry)
python train_hierarchical_tokamak.py --num-cycles 10000 --layers "3,30,300"
```

---

### Dimension 5: Perturbation Stress (Stability Testing)

**Hypothesis**: Robust emergent phenomena survive perturbations; fragile ones collapse.

| Test | Perturbation Type | Strength | Expected |
|------|------------------|----------|----------|
| Baseline | None | 0.0 | Stable |
| Gaussian noise | Random field | 0.1 | Recovery dynamics |
| Targeted vortex | Kill high-quality | 0.5 | Regeneration |
| Phase scramble | Randomize phases | 1.0 | Topological protection? |

**What to Look For**:
- Recovery time (how fast system returns to fixed point)
- Hysteresis (different path when returning)
- Robustness (fixed points survive vs collapse)
- Emergent healing (system self-repairs)

**Commands**:
```bash
# Perturbation test
python train_tokamak_wormhole_hunt.py \
    --num-cycles 10000 \
    --perturbation-type gaussian \
    --perturbation-strength 0.1 \
    --perturb-every 100
```

---

### Dimension 6: Geometry Stress (Tokamak Parameters)

**Hypothesis**: κ (elongation) and δ (triangularity) control emergent geometry.

| Test | κ (kappa) | δ (delta) | Geometry | Expected |
|------|----------|----------|----------|----------|
| Circular | 1.0 | 0.0 | Circle cross-section | Baseline |
| **Current** | **1.5** | **0.3** | **D-shaped** | **Current results** |
| Elongated | 2.0 | 0.3 | Tall ellipse | Different vortex patterns |
| Triangular | 1.5 | 0.7 | Strong triangularity | Edge phenomena |

**What to Look For**:
- Geometry-dependent vortex clustering
- Edge-localized modes (ELMs analog)
- Radial transport patterns
- Confinement scaling laws

**Commands**:
```bash
# Sweep tokamak geometry
for KAPPA in 1.0 1.5 2.0; do
    for DELTA in 0.0 0.3 0.7; do
        python train_tokamak_wormhole_hunt.py \
            --num-cycles 1000 \
            --kappa $KAPPA \
            --delta $DELTA \
            --output-dir ~/results/geometry/k${KAPPA}_d${DELTA}
    done
done
```

---

## 🎯 Benchmark Suite (Automated)

### Quick Benchmark (30 minutes)

Tests core performance across key dimensions:

```bash
#!/bin/bash
# Quick benchmark suite

# 1. Baseline (3.3 min)
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --seed 42

# 2. 2× scale (6.6 min)
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-strips 300 --nodes-per-strip 332 --seed 42

# 3. High-res temporal (10 min)
python train_tokamak_wormhole_hunt.py --num-cycles 10000 --num-time-steps 50 --seed 42

# 4. Retrocausal sweep (5 min)
for ALPHA in 0.0 0.5 0.9; do
    python train_tokamak_wormhole_hunt.py --num-cycles 1000 --retrocausal-alpha $ALPHA --seed 42
done

echo "Quick benchmark complete! Check results/ for outputs."
```

### Comprehensive Benchmark (2 hours)

Full parameter sweep:

```bash
#!/bin/bash
# Comprehensive benchmark suite

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=~/results/benchmark_$TIMESTAMP

mkdir -p $RESULTS_DIR

echo "Running comprehensive tHHmL benchmark..."
echo "Results: $RESULTS_DIR"

# Scale tests (4 runs × ~10 min = 40 min)
for SCALE in 1 2 3 5; do
    NODES=$((49800 * SCALE))
    python train_tokamak_wormhole_hunt.py \
        --num-cycles 10000 \
        --num-strips 300 \
        --nodes-per-strip $((166 * SCALE)) \
        --output-dir $RESULTS_DIR/scale_${SCALE}x \
        --seed 42
done

# Temporal resolution tests (3 runs × ~15 min = 45 min)
for TSTEPS in 10 50 100; do
    python train_tokamak_wormhole_hunt.py \
        --num-cycles 10000 \
        --num-time-steps $TSTEPS \
        --output-dir $RESULTS_DIR/timesteps_$TSTEPS \
        --seed 42
done

# Retrocausal sweep (5 runs × ~3 min = 15 min)
for ALPHA in 0.0 0.3 0.5 0.7 0.9; do
    python train_tokamak_wormhole_hunt.py \
        --num-cycles 1000 \
        --retrocausal-alpha $ALPHA \
        --output-dir $RESULTS_DIR/alpha_$ALPHA \
        --seed 42
done

# Geometry sweep (9 runs × ~3 min = 27 min)
for KAPPA in 1.0 1.5 2.0; do
    for DELTA in 0.0 0.3 0.7; do
        python train_tokamak_wormhole_hunt.py \
            --num-cycles 1000 \
            --kappa $KAPPA \
            --delta $DELTA \
            --output-dir $RESULTS_DIR/geom_k${KAPPA}_d${DELTA} \
            --seed 42
    done
done

echo "Comprehensive benchmark complete!"
echo "Total runs: 21"
echo "Results: $RESULTS_DIR"
echo "Next: python analyze_benchmark_results.py $RESULTS_DIR"
```

---

## 📈 Emergent Phenomenon Detection

### Automated Detection Criteria

After each benchmark run, check for:

**1. Phase Transitions**
```python
# Detect sudden jumps in observables
def detect_phase_transition(param_sweep_results):
    diffs = np.diff(observable_values)
    critical_points = np.where(np.abs(diffs) > 3 * np.std(diffs))[0]
    return critical_points  # Parameter values where transitions occur
```

**2. Power-Law Distributions**
```python
# Test for scale-invariance
def test_power_law(wormhole_sizes):
    from powerlaw import Fit
    fit = Fit(wormhole_sizes)
    return fit.power_law.alpha, fit.power_law.sigma
```

**3. Temporal Oscillations**
```python
# Find dominant frequencies
from scipy.fft import fft, fftfreq
freqs = fftfreq(len(observable_history))
spectrum = np.abs(fft(observable_history))
dominant_freq = freqs[np.argmax(spectrum[1:])]  # Skip DC component
```

**4. Spatial Clustering**
```python
# Measure clustering coefficient
from networkx import clustering
avg_clustering = clustering(wormhole_network)
# Small-world: high clustering + short path length
```

**5. Self-Similarity**
```python
# Fractal dimension via box-counting
def fractal_dimension(vortex_positions, box_sizes):
    counts = [count_occupied_boxes(positions, size) for size in box_sizes]
    slope, _ = np.polyfit(np.log(box_sizes), np.log(counts), 1)
    return -slope  # Fractal dimension
```

---

## 🔬 Expected Emergent Phenomena

Based on tHHmL architecture, we anticipate discovering:

### 1. **Temporal Phase Transition** (α = 0.5-0.7)
- Below α: No fixed points (standard diffusion)
- Above α: 100% fixed points (retrocausal locking)
- **Test**: Sweep α from 0.0 to 1.0 in 0.1 steps

### 2. **Scale-Dependent Wormhole Networks** (N > 100K nodes)
- Small-world topology (high clustering, short paths)
- Hub emergence (power-law degree distribution)
- **Test**: Scale from 50K to 500K nodes

### 3. **Geometric Resonances** (κ, δ sweep)
- Specific (κ, δ) pairs maximize vortex density
- Edge-localized phenomena at high δ
- **Test**: 2D parameter sweep (κ × δ)

### 4. **Temporal Fractals** (T = 100+ time steps)
- Self-similar dynamics across timescales
- Period-doubling route to chaos
- **Test**: Increase T from 10 to 100 in steps of 10

### 5. **Topological Protection** (perturbation tests)
- Vortices with winding number ±1 survive
- Wormholes persist under field noise
- **Test**: Inject Gaussian noise, measure recovery

---

## 📊 Benchmark Output Format

Each benchmark run produces:

```
benchmark_<timestamp>/
├── <test_name>/
│   ├── training_results_<timestamp>.json     # Full metrics history
│   ├── final_field_state.pt                  # Checkpoint
│   ├── wormhole_network.graphml              # Network structure
│   ├── summary.json                           # Aggregate statistics
│   └── emergent_analysis.json                # Detected phenomena
├── aggregate_analysis.json                    # Cross-run comparisons
└── BENCHMARK_REPORT.pdf                       # Auto-generated whitepaper
```

---

## 🚀 Next Steps After Benchmark

1. **Analyze aggregate results** - Look for patterns across parameter sweeps
2. **Identify critical points** - Where do phase transitions occur?
3. **Generate publication** - Document novel emergent phenomena in EMERGENTS.md
4. **Deep dive interesting cases** - Run extended training on promising configurations
5. **Cross-validation** - Test reproducibility with different seeds

---

## 📝 Checklist for Emergent Discovery

- [ ] Run quick benchmark (30 min baseline)
- [ ] Identify most interesting parameter regime
- [ ] Run comprehensive benchmark (2 hours full sweep)
- [ ] Analyze for phase transitions (automated detection)
- [ ] Test for power-law distributions (scale-invariance)
- [ ] Measure network topology (small-world, hubs)
- [ ] Check temporal dynamics (oscillations, chaos)
- [ ] Validate with perturbation tests (robustness)
- [ ] Document findings in EMERGENTS.md
- [ ] Generate whitepaper for publication

---

**End of Stress Testing Protocol**
