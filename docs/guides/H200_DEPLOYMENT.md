# H200 Deployment Guide for Multi-Strip Flux Tube Simulations

**Last Updated**: 2025-12-16
**Target Hardware**: NVIDIA H200 GPU (140GB VRAM)
**Framework**: HHmL Multi-Strip Flux Tubes

---

## Overview

This guide covers deploying the multi-strip tokamak-style Möbius simulations to the H200 GPU for maximum scale.

**Auto-Scaling Features**:
- Hardware auto-detection (CPU → H200)
- Sparse (CPU/low-memory) → Dense (H200) mode switching
- Parameter auto-tuning based on available VRAM
- Production configs: 20 strips × 500K nodes = 10M total nodes

---

## Quick Start (H200)

```bash
# SSH into H200 VM
ssh ivhl@<H200_IP>

# Clone repository
cd ~
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# Create environment
python3 -m venv ~/hhml_h200_env
source ~/hhml_h200_env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Test hardware detection
python hhml/utils/hardware_config.py

# Run quick benchmark (5 minutes)
python benchmark_multi_strip.py --mode quick

# Run full training (auto-scales to H200 production parameters)
python train_multi_strip.py --mode production --cycles 1000
```

---

## Expected H200 Performance

### Hardware Detection
```
================================================================================
HARDWARE CONFIGURATION
================================================================================
Device: CUDA
GPU: NVIDIA H200
VRAM: 140.0 GB
CUDA Compute Capability: (9, 0)
H200 Detected: YES
Hardware Tier: H200
================================================================================
```

### Auto-Scaled Parameters (Production Mode)

| Parameter | Value | Memory |
|-----------|-------|--------|
| **Number of Strips** | 20 | - |
| **Nodes per Strip** | 500,000 | - |
| **Total Nodes** | 10,000,000 | ~120 MB |
| **Mode** | **DENSE** | ~380 GB (distance matrix!) |
| **Hidden Dim** | 4096 | ~268 MB |
| **Precision** | float16 (mixed) | 50% savings |

**IMPORTANT**: Dense mode with 10M nodes requires:
- Distance matrix: `[10M, 10M] float32 = 400 GB` (exceeds H200 VRAM!)
- **Solution**: Use float16 mixed precision OR reduce to 6M nodes

### Recommended H200 Configs

**Config 1: Maximum Accuracy (Dense, float32)**
```python
python train_multi_strip.py --strips 15 --nodes 400000 --cycles 5000
# Total: 6M nodes
# Memory: ~144 GB dense distance matrix
# Mode: DENSE (all-to-all interactions)
# Expected: 0.2-0.5 cycles/sec
```

**Config 2: Maximum Scale (Sparse, float16)**
```python
python train_multi_strip.py --strips 20 --nodes 1000000 --cycles 10000
# Total: 20M nodes
# Memory: ~80 GB (sparse graph only)
# Mode: SPARSE (100-200 neighbors/node)
# Expected: 2-5 cycles/sec
```

**Config 3: Balanced (Dense, float16)**
```python
python train_multi_strip.py --strips 18 --nodes 500000 --cycles 5000
# Total: 9M nodes
# Memory: ~162 GB (float16 distance matrix)
# Mode: DENSE with mixed precision
# Expected: 0.3-0.8 cycles/sec
```

---

## Architecture Differences: CPU vs H200

### Local CPU (Low-RAM)
```
Strips: 2
Nodes per Strip: 1,000
Total: 2,000 nodes
Mode: SPARSE (max_neighbors=20)
Memory: ~4 MB
Speed: ~15K nodes/sec
```

### H200 GPU (Production)
```
Strips: 15
Nodes per Strip: 400,000
Total: 6,000,000 nodes
Mode: DENSE (all-to-all)
Memory: ~144 GB
Speed: ~2-5M nodes/sec (100-300× faster)
```

**Key Difference**: DENSE mode on H200!
- CPU: Only nearby nodes interact (sparse graph)
- H200: **Every node interacts with every other node** (maximum accuracy)
- Distance matrix precomputed: `[N, N]` all pairwise distances
- Wave propagation uses matrix operations (GPU-optimized)

---

## Monitoring and Optimization

### Real-Time Monitoring

```bash
# Terminal 1: Run training
python train_multi_strip.py --mode production --cycles 10000

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

### Expected GPU Metrics (H200, 6M nodes dense)
```
GPU Memory Usage: 120-140 GB / 140 GB (85-100%)
GPU Utilization: 95-100%
Temperature: 65-75°C
Power: 600-700W
```

### Performance Targets

| Hardware | Nodes | Mode | Throughput | Memory |
|----------|-------|------|------------|--------|
| CPU (8GB RAM) | 2K | Sparse | 15K nodes/sec | 4 MB |
| Mid GPU (16GB) | 50K | Sparse | 200K nodes/sec | 100 MB |
| High GPU (80GB) | 400K | Dense | 2M nodes/sec | 6 GB |
| **H200 (140GB)** | **6M** | **Dense** | **3-5M nodes/sec** | **144 GB** |

---

## Troubleshooting

### Out of Memory (DENSE mode)

**Symptom**: `CUDA out of memory` during dense graph construction

**Causes**:
1. Too many nodes for dense mode
2. float32 precision (8× more memory than float16)

**Solutions**:
```python
# Solution 1: Reduce nodes
python train_multi_strip.py --strips 12 --nodes 300000  # 3.6M nodes

# Solution 2: Force sparse mode
python train_multi_strip.py --strips 20 --nodes 500000  # 10M nodes, sparse

# Solution 3: Mixed precision (if implemented)
# Edit sparse_tokamak_strips.py, use torch.float16 for distance matrix
```

### Slow Initialization

**Symptom**: Graph construction takes 10+ minutes

**Expected**:
- Sparse graph (6M nodes): 30-60 seconds
- Dense graph (6M nodes): 5-10 minutes (computing 36 trillion distances!)

**This is normal for dense mode**. The distance matrix is huge:
- 6M nodes: `[6M, 6M] = 36,000,000,000,000 elements`
- Even at 1 billion/sec, that's 36,000 seconds = 10 hours!

**Optimization**: Precompute and cache distance matrix to disk
```python
# In sparse_tokamak_strips.py, add caching:
cache_file = f"distance_matrix_{total_nodes}.pt"
if Path(cache_file).exists():
    self.distance_matrix = torch.load(cache_file)
else:
    self.distance_matrix = compute_distances()
    torch.save(self.distance_matrix, cache_file)
```

### Vortex Collapse

**Symptom**: Vortex density drops from 100% → 0% during training

**Cause**: Reward function may be inverted or RNN learned wrong policy

**Fix**: Check `compute_multi_strip_reward()` in `train_multi_strip.py`
- Ensure higher vortex density → higher reward
- Target range: 50-80% vortex density
- Add exploration bonus if stuck

---

## Advanced: Custom Configs

### Override Auto-Detection

Force sparse even on H200 (for testing):
```python
# In train_multi_strip.py
strips = SparseTokamakMobiusStrips(
    num_strips=20,
    nodes_per_strip=500000,
    device='cuda',
    force_sparse=True  # Force sparse mode
)
```

Force dense on mid-range GPU (may OOM!):
```python
strips = SparseTokamakMobiusStrips(
    num_strips=4,
    nodes_per_strip=50000,
    device='cuda',
    force_dense=True  # Force dense mode
)
```

### Custom Tokamak Parameters

```python
strips = SparseTokamakMobiusStrips(
    num_strips=15,
    nodes_per_strip=400000,
    device='cuda',
    kappa=1.7,      # Elongation (1.2-1.8)
    delta=0.4,      # Triangularity (0.1-0.5)
    r_minor=0.08,   # Minor radius (cross-section thickness)
    sparse_threshold=0.25,  # Interaction distance (sparse only)
    max_neighbors=150       # Max edges/node (sparse only)
)
```

**Tokamak Physics**:
- **kappa (elongation)**: ITER uses ~1.7 (tall D-shape)
- **delta (triangularity)**: JET uses ~0.4 (asymmetric D)
- Higher kappa/delta → more surface area per strip → more nodes can fit

---

## Benchmarking

### Quick Benchmark (5 minutes)
```bash
python benchmark_multi_strip.py --mode quick
```

Tests:
1. Geometry generation (N=1,2,3 strips)
2. Wave propagation throughput
3. Scaling analysis (1K, 2K, 5K nodes)

### Full Benchmark (15 minutes)
```bash
python benchmark_multi_strip.py --mode full
```

Additional tests:
4. Scaling to 50K nodes
5. Sparse vs Dense comparison

### Production Benchmark (30+ minutes, H200 only)
```bash
python benchmark_multi_strip.py --mode production
```

H200-specific:
- Scaling to 500K nodes/strip
- 20-strip configurations
- Dense mode stress test

Results saved to: `results/benchmarks/benchmark_h200_<timestamp>.json`

---

## Expected Training Results (H200, 1000 cycles)

**Configuration**: 15 strips × 400K nodes = 6M nodes, DENSE mode

**Timeline**:
- Initialization: ~8 minutes (distance matrix computation)
- Training: ~30-60 minutes (0.3-0.6 cycles/sec)
- Total: ~40-70 minutes

**Expected Outcomes**:
- Final vortex density: 60-75% (target range)
- Vortex uniformity (std): <5% (balanced across strips)
- RNN convergence: κ≈1.6-1.7, δ≈0.35-0.45
- Reward trend: Increasing then plateau

**Deliverables**:
- `results/multi_strip_training/training_<timestamp>.json`
- RNN checkpoint: `agent_6M.pt` (if implemented)
- Visualization data: vortex positions, field snapshots

---

## Comparison: Single-Strip vs Multi-Strip

| Metric | Single Strip (Original) | Multi-Strip (New) |
|--------|-------------------------|-------------------|
| Nodes | 20M | 6M (15×400K) |
| Topology | 1 Möbius strip | 15 nested tokamak strips |
| Vortex Density | 82% (achieved) | Target: 70-80% |
| Cross-Strip Coupling | N/A | **NEW: Learned by RNN** |
| Poles | 2 (N/S) | **2 shared, NO singularity!** |
| Memory (H200) | ~800 MB | ~144 GB (dense) |
| Training Time | 72 min (500 cycles) | ~60 min (1000 cycles est.) |
| Unique Features | High single-strip density | **Multi-scale flux tube structure** |

**Key Advantage**: Multi-strip enables cross-scale interactions and tokamak-inspired nested geometry.

---

## Next Steps

1. **Run quick benchmark** to verify H200 detection
2. **Start with Config 1** (6M nodes dense) for maximum accuracy
3. **Monitor vortex density** - aim for 60-80% target
4. **If successful**: Scale to Config 2 (20M nodes sparse) for production
5. **Save checkpoints** every 100 cycles for analysis

---

## Contact / Issues

- **Repository**: https://github.com/Zynerji/HHmL
- **Issues**: Report H200-specific problems with tag `[H200]`
- **Performance**: Share benchmark results in discussions

---

**End of H200 Deployment Guide**

Ready to deploy to the most powerful GPU available! 🚀
