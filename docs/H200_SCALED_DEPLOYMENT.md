# H200 Scaled Training Deployment Guide

**Date**: 2025-12-17
**Script**: `scripts/train_h200_scaled.py`
**Target Hardware**: NVIDIA H200 (140GB VRAM)
**Configuration**: 28-32 Möbius strips, 4096 hidden dimensions, Quality-Guided Learning

---

## Executive Summary

This guide covers deployment of the **maximum Möbius coverage** training run on the H200 GPU. The configuration uses 30 strips (default) with 4096 hidden dimensions, resulting in ~170M trainable parameters and an estimated 60-80GB VRAM usage.

**Expected Performance**:
- Training speed: ~30 seconds per cycle (at 140K node scale)
- Total training time: 60 cycles in ~30 minutes (time-limited validation run)
- Target outcome: Early validation of quality-guided learning at massive scale
- **Note**: 60 cycles may not reach full convergence (baseline needed 505 cycles), but should show clear learning trends

---

## 30-Minute Validation Strategy

### Why 60 Cycles?

The baseline training (2 strips, 512 hidden_dim) converged in **505 cycles**, but at 140K nodes and 6144 hidden dimensions, we're testing:

1. **System Stability**: Does the architecture run without OOM errors at 90-95% VRAM utilization?
2. **Learning Trajectory**: Are reward, density, and quality trending upward?
3. **Transfer Learning**: If bootstrapped from baseline checkpoint, does it converge faster?
4. **Scale Performance**: Can we process 35× more nodes in reasonable time?

### What to Look For

**Success Indicators (60 cycles)**:
- ✅ Vortex density trending upward (10% → 30% → 50%+)
- ✅ Reward increasing steadily (not oscillating or collapsing)
- ✅ Quality score improving (0.3 → 0.5 → 0.7+)
- ✅ Annihilations decreasing (starting high, trending down)
- ✅ System stable (no OOM, consistent cycle times)

**If Successful**: Run extended training (500-1000 cycles, 4-8 hours) to achieve 100% convergence

**If Issues**: Fall back to conservative config (30 strips, 60K nodes) or debug performance bottlenecks

---

## Configuration Details

### System Scale (Option 4: Balanced Maximum)

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Möbius Strips** | 40 | Dense sphere coverage (1.33× increase) |
| **Nodes per Strip** | 3,500 | High-resolution per strip |
| **Total Nodes** | 140,000 | 35× increase from baseline |
| **Hidden Dimensions** | 6,144 | 12× increase from baseline |
| **RNN Parameters** | ~340M | Maximum capacity for complex behaviors |
| **Estimated VRAM** | 125-135 GB | 90-95% H200 utilization |

### Training Parameters

```yaml
Cycles: 1000 (default)
Learning Rate: 1e-4
Optimizer: Adam
Gradient Clipping: 1.0
Checkpoint Frequency: Every 50 cycles
Exploration Noise: 0.05 → 0.0 (linear decay)
```

### Quality-Guided Learning Rewards

```python
Density Reward:      +100 to +300 (target 95-100%)
Neighborhood:        +150
Core Depth:          +150
Stability:           +150
Quality × Density:   +200
Annihilation Penalty: -50 per removal
Stability Bonus:     up to +200
Zero-Ann Bonus:      +500 (when removed=0 AND density≥95%)

Maximum Possible:    ~1650
```

---

## H200 VM Access

### Connection Details

```bash
# SSH into H200
ssh ivhl@<VM_IP>

# Check current VM IP (from local machine)
cat ~/.ssh/known_hosts | grep ivhl
# Or check Nebius dashboard

# Jupyter Access (if needed)
http://<VM_IP>:8888/?token=myBn0JMX7uIuMraq
```

### VM Configuration

```yaml
Username: ivhl
Provider: Nebius
GPU: NVIDIA H200 (139.8 GB VRAM)
OS: Ubuntu 22.04
Python: 3.12.3
Sudo: NOPASSWD:ALL
```

---

## Deployment Steps

### 1. Connect to H200

```bash
ssh ivhl@<VM_IP>
```

### 2. Clone/Update Repository

```bash
# If first time
git clone https://github.com/Zynerji/HHmL.git
cd HHmL

# If updating existing
cd HHmL
git pull origin master
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv ~/hhml_h200_env
source ~/hhml_h200_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HHmL dependencies
pip install -r requirements.txt
```

### 4. Validate Environment

```bash
python -m hhml.utils.startup_validator
```

Expected output:
```
[OK] PyTorch installed
[OK] CUDA available
[OK] GPU: NVIDIA H200
[OK] VRAM: 139.8 GB
```

### 5. Run Scaled Training

#### Default Configuration (Option 4: Balanced Maximum - 30-Minute Validation)

```bash
# Maximum utilization: 40 strips, 140K nodes, 6144 hidden_dim
# 60 cycles in ~30 minutes (time-limited validation)
python scripts/train_h200_scaled.py \
  --device cuda
```

This uses:
- **40 Möbius strips** (dense coverage)
- **3,500 nodes per strip** (140,000 total)
- **6,144 hidden dimensions** (340M parameters)
- **125-135 GB VRAM** (90-95% utilization)
- **60 cycles** (auto-stops at 30 minutes)
- **Checkpoints every 10 cycles**

#### Conservative Configuration

```bash
# Conservative: 30 strips, 60K nodes, 4096 hidden_dim
python scripts/train_h200_scaled.py \
  --strips 30 \
  --nodes 2000 \
  --hidden-dim 4096 \
  --cycles 1000 \
  --device cuda
```

#### Resume from Checkpoint

```bash
python scripts/train_h200_scaled.py \
  --resume checkpoints/h200_scaled/checkpoint_cycle_500.pt \
  --cycles 1000 \
  --device cuda
```

### 6. Monitor Training

The script automatically opens a live dashboard at `http://localhost:8000`. To access from your local machine:

```bash
# On local machine, create SSH tunnel
ssh -L 8000:localhost:8000 ivhl@<VM_IP>

# Then open in browser
http://localhost:8000
```

Alternatively, check logs:
```bash
tail -f checkpoints/h200_scaled/training_*.log
```

---

## Output Files

### Checkpoints

Location: `checkpoints/h200_scaled/`

```
checkpoint_cycle_50.pt
checkpoint_cycle_100.pt
...
checkpoint_final_cycle_999.pt
```

Checkpoint contents:
- RNN state_dict
- Optimizer state_dict
- Training metrics (rewards, densities, quality scores)
- Configuration metadata

### Results

Location: `results/h200_scaled/`

```
training_20251217_153045.json
```

JSON structure:
```json
{
  "configuration": {
    "strips": 30,
    "nodes": 2000,
    "hidden_dim": 4096,
    "cycles": 1000
  },
  "metrics": {
    "rewards": [...],
    "vortex_densities": [...],
    "quality_scores": [...]
  },
  "final_state": {
    "cycles": 1000,
    "final_density": 1.0,
    "final_reward": 1650.0,
    "peak_density": 1.0
  }
}
```

### Logs

Location: `checkpoints/h200_scaled/training_*.log`

Contains timestamped output from all training cycles.

---

## Performance Monitoring

### GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Detailed memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv -l 1
```

Expected during training:
```
GPU Utilization: 90-100%
VRAM Usage: 60-80 GB
Temperature: <80°C
```

### Python Memory Profiling

The script automatically logs VRAM usage every 100 cycles:
```
[VRAM] Allocated: 65.3 GB | Reserved: 68.1 GB
```

---

## Expected Results

### 30-Minute Validation Run (60 cycles)

**Baseline Reference** (2 strips, 512 hidden_dim, fully converged):
- **Cycles to convergence**: ~505
- **Final density**: 100.0%
- **Final quality**: 1.00
- **Annihilations**: 0
- **Final reward**: 1650

**Expected After 60 Cycles** (40 strips, 6144 hidden_dim, early validation):
- **Vortex density**: 30-60% (trending upward)
- **Quality score**: 0.4-0.7 (improving)
- **Annihilations**: 20-50 per cycle (decreasing)
- **Reward**: 200-800 (increasing steadily)
- **System stability**: No OOM, consistent 20-40s cycle times

**Success Criteria**:
- ✅ All metrics trending in correct direction
- ✅ No crashes or divergence
- ✅ VRAM usage stable at 125-135 GB
- ✅ Quality-guided learning showing same patterns as baseline (just slower convergence due to scale)

**If Successful → Extended Run**:
- Run additional 500+ cycles (8-10 hours)
- Expected full convergence: 100% density, 1.00 quality, 0 annihilations
- Target: Replicate baseline breakthrough at 35× scale

---

## Troubleshooting

### OOM (Out of Memory) Error

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
1. Reduce number of strips:
   ```bash
   python scripts/train_h200_scaled.py --strips 24 --cycles 1000
   ```

2. Reduce hidden dimensions:
   ```bash
   python scripts/train_h200_scaled.py --hidden-dim 3072 --cycles 1000
   ```

3. Reduce nodes per strip:
   ```bash
   python scripts/train_h200_scaled.py --nodes 1500 --cycles 1000
   ```

### Slow Training Speed

**Symptoms**: >30 seconds per cycle

**Diagnosis**:
```bash
# Check if quality metrics sampling is bottleneck
# Script already samples max 5000 vortices for performance
```

**Solutions**:
1. Reduce quality metrics sampling in `compute_detailed_quality_metrics()`:
   ```python
   max_sample = min(2000, len(vortex_indices))  # Reduce from 5000
   ```

2. Check if annihilation is expensive:
   ```bash
   # If many annihilations, quality-guided learning hasn't converged yet
   # Let it continue training
   ```

### Dashboard Not Accessible

**Symptoms**: `http://localhost:8000` not loading

**Solutions**:
1. Verify SSH tunnel:
   ```bash
   ssh -L 8000:localhost:8000 ivhl@<VM_IP>
   ```

2. Check if port is in use:
   ```bash
   lsof -i :8000
   ```

3. Use different port:
   ```bash
   python scripts/train_h200_scaled.py --cycles 1000 --device cuda
   # Then manually start dashboard on different port
   ```

### Training Divergence

**Symptoms**: Reward becomes negative, density drops to 0%

**Solutions**:
1. Resume from earlier checkpoint:
   ```bash
   python scripts/train_h200_scaled.py \
     --resume checkpoints/h200_scaled/checkpoint_cycle_450.pt \
     --cycles 1000
   ```

2. Reduce learning rate:
   ```python
   optimizer = torch.optim.Adam(agent.parameters(), lr=5e-5)  # Reduce from 1e-4
   ```

---

## Post-Training Analysis

### Generate Whitepaper

```bash
# TODO: Create whitepaper generation script
python scripts/generate_whitepaper.py \
  --results results/h200_scaled/training_*.json \
  --output whitepapers/h200_scaled_quality_guided.pdf
```

### Visualize Results

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('results/h200_scaled/training_20251217_153045.json') as f:
    data = json.load(f)

# Plot density over time
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(data['metrics']['vortex_densities'])
plt.xlabel('Cycle')
plt.ylabel('Vortex Density')
plt.title('Vortex Density Evolution')

plt.subplot(1, 3, 2)
plt.plot(data['metrics']['quality_scores'])
plt.xlabel('Cycle')
plt.ylabel('Quality Score')
plt.title('Vortex Quality Evolution')

plt.subplot(1, 3, 3)
plt.plot(data['metrics']['rewards'])
plt.xlabel('Cycle')
plt.ylabel('Reward')
plt.title('Reward Evolution')

plt.tight_layout()
plt.savefig('h200_scaled_results.png', dpi=300)
```

### Compare to Baseline

```python
# Load baseline (2 strips, 512 hidden_dim)
with open('results/quality_guided/training_20251217_*.json') as f:
    baseline = json.load(f)

# Load scaled (30 strips, 4096 hidden_dim)
with open('results/h200_scaled/training_20251217_*.json') as f:
    scaled = json.load(f)

# Compare convergence speed
baseline_convergence = next(i for i, d in enumerate(baseline['metrics']['vortex_densities']) if d >= 0.95)
scaled_convergence = next(i for i, d in enumerate(scaled['metrics']['vortex_densities']) if d >= 0.95)

print(f"Baseline convergence: cycle {baseline_convergence}")
print(f"Scaled convergence: cycle {scaled_convergence}")
print(f"Convergence ratio: {scaled_convergence / baseline_convergence:.2f}x")
```

---

## Scientific Questions

This scaled training run aims to answer:

1. **Scalability**: Does quality-guided learning work at 30× scale (2 strips → 30 strips)?
2. **Capacity**: Does 4096 hidden_dim provide better inter-strip coordination than 512?
3. **Convergence**: Does the approach converge in similar time (~500 cycles) at scale?
4. **Stability**: Can 60K total nodes maintain 100% density with zero annihilations?
5. **Topology**: Do 30 Möbius strips create unique vortex interaction patterns?

---

## Next Steps After Training

1. **Validate Results**: Confirm 100% density, 1.00 quality, 0 annihilations
2. **Generate Whitepaper**: Document scaled training breakthrough
3. **Push to GitHub**: Commit results, checkpoints (if small enough), whitepaper
4. **Scale Further**: If successful, try 50 strips or 100K nodes
5. **Production Deployment**: Use trained checkpoint for HHmL production runs

---

## Expected Timeline

### 30-Minute Validation Run

| Task | Duration |
|------|----------|
| VM Setup | 5 minutes |
| Environment Setup | 10 minutes |
| **Training (60 cycles)** | **30 minutes** |
| Quick Analysis | 5 minutes |
| **Total** | **50 minutes** |

### Full Convergence Run (if validation successful)

| Task | Duration |
|------|----------|
| Extended Training (500 cycles) | 4-5 hours |
| Post-analysis | 30 minutes |
| Whitepaper Generation | 1 hour |
| **Total** | **6-7 hours** |

---

## Contact & Support

**Repository**: https://github.com/Zynerji/HHmL
**Issues**: https://github.com/Zynerji/HHmL/issues
**Author**: HHmL Project / Claude Code
**Date**: 2025-12-17

---

**End of Deployment Guide**

Good luck with the H200 scaled training! 🚀
