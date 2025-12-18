# H200 Quick Start - 30-Minute Validation Run

**Date**: 2025-12-17
**Configuration**: Option 4 (Balanced Maximum)
**Time**: 30 minutes
**Status**: Ready to deploy

---

## What Was Built

**Maximum H200 Resource Utilization**:
- **40 Möbius strips** (dense sphere coverage)
- **140,000 nodes** (3,500 per strip)
- **6,144 hidden dimensions** (340M parameters)
- **125-135 GB VRAM** (90-95% H200 utilization)

**Quality-Guided Learning**:
- Proven approach from baseline (100% density, 0 annihilations)
- Direct reward for neighborhood density, core depth, stability
- Zero-annihilation bonus (+500)
- Quality × Density product

**30-Minute Validation Strategy**:
- **60 cycles** in ~30 minutes
- Auto-stop at time limit
- Checkpoint every 10 cycles
- Early validation before extended training

---

## Deploy on H200

**VM Details**:
- IP: `89.169.97.59`
- User: `ivhl`
- Jupyter Token: `lCHdHxTLUoWfTkIL`

### Step 1: Connect

```bash
ssh ivhl@89.169.97.59
```

### Step 2: Clone Repository

```bash
cd ~
git clone https://github.com/Zynerji/HHmL.git
cd HHmL
```

### Step 3: Setup Environment

```bash
python3 -m venv ~/hhml_h200_env
source ~/hhml_h200_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Step 4: Validate

```bash
python -m hhml.utils.startup_validator
```

### Step 5: Run Training

```bash
python scripts/train_h200_scaled.py --device cuda
```

That's it! Training will:
- Run for 60 cycles (~30 minutes)
- Auto-stop at 30-minute time limit
- Save checkpoints every 10 cycles
- Open live dashboard at http://localhost:8000

---

## Access Dashboard from Local Machine

```bash
# On local machine (open new terminal)
ssh -L 8000:localhost:8000 ivhl@89.169.97.59

# Then open browser
http://localhost:8000
```

**Alternatively**, access Jupyter:
```
http://89.169.97.59:8888/?token=lCHdHxTLUoWfTkIL
```

---

## What to Expect

### Real-Time Metrics

```
Cycle   10/60 | Density:  15.3%               | Quality: 0.42 | Reward:  285.3 | Removed:  67 | Stable:  0 | Time: 28.3s
Cycle   20/60 | Density:  28.7%               | Quality: 0.51 | Reward:  412.8 | Removed:  52 | Stable:  0 | Time: 31.2s
Cycle   30/60 | Density:  42.1%               | Quality: 0.63 | Reward:  638.5 | Removed:  38 | Stable:  0 | Time: 29.7s
Cycle   40/60 | Density:  56.4%               | Quality: 0.74 | Reward:  891.2 | Removed:  24 | Stable:  0 | Time: 30.5s
Cycle   50/60 | Density:  68.9%               | Quality: 0.82 | Reward: 1124.6 | Removed:  12 | Stable:  0 | Time: 32.1s
Cycle   60/60 | Density:  79.3%               | Quality: 0.88 | Reward: 1342.7 | Removed:   5 | Stable:  0 | Time: 31.8s
```

### Success Indicators

✅ **Density trending upward**: 15% → 30% → 50% → 70%+
✅ **Quality improving**: 0.4 → 0.6 → 0.8+
✅ **Annihilations decreasing**: 67 → 52 → 38 → 24 → 12 → 5
✅ **Reward increasing**: 285 → 412 → 638 → 891 → 1124 → 1342
✅ **Stable cycle times**: ~30 seconds per cycle
✅ **No OOM errors**: VRAM stable at 125-135 GB

---

## After 30 Minutes

### If Successful (All Metrics Trending Up)

```bash
# Run extended training for full convergence
python scripts/train_h200_scaled.py \
  --resume checkpoints/h200_scaled/checkpoint_cycle_60.pt \
  --cycles 500 \
  --max-time 28800 \
  --device cuda
```

This will run for:
- **500 additional cycles** (~4-5 hours)
- **8-hour time limit** (28800 seconds)
- Target: 100% density, 1.00 quality, 0 annihilations

### If Transfer Learning Desired

```bash
# First, transfer baseline checkpoint
python scripts/transfer_to_h200.py \
  --baseline checkpoints/quality_guided/checkpoint_final_cycle_599.pt \
  --output checkpoints/h200_scaled/transferred_baseline.pt

# Then train with transferred weights (may converge 2-3× faster)
python scripts/train_h200_scaled.py \
  --resume checkpoints/h200_scaled/transferred_baseline.pt \
  --cycles 300 \
  --max-time 10800 \
  --device cuda
```

---

## Output Files

### Checkpoints (saved automatically)

```
checkpoints/h200_scaled/
  checkpoint_cycle_10.pt
  checkpoint_cycle_20.pt
  ...
  checkpoint_cycle_60.pt
  checkpoint_final_cycle_59.pt
```

### Results

```
results/h200_scaled/
  training_20251217_153045.json
```

### Logs

```
checkpoints/h200_scaled/
  training_20251217_153045.log
```

---

## Key Numbers

| Metric | Baseline (2 strips) | Scaled (40 strips) | Factor |
|--------|--------------------|--------------------|--------|
| **Strips** | 2 | 40 | 20× |
| **Nodes** | 4,000 | 140,000 | 35× |
| **Hidden Dim** | 512 | 6,144 | 12× |
| **Parameters** | 10.3M | 340M | 33× |
| **VRAM** | 5-10 GB | 125-135 GB | ~20× |
| **Cycle Time** | 5-10s | 30s | 3-6× |
| **Convergence** | 505 cycles | TBD (expect 500-700) | ~1× |

---

## Troubleshooting

### OOM Error

**Solution**: Use conservative config
```bash
python scripts/train_h200_scaled.py \
  --strips 30 \
  --nodes 2000 \
  --hidden-dim 4096 \
  --device cuda
```

### Slow Performance (>60s per cycle)

**Check**: GPU utilization
```bash
nvidia-smi
```

Expected: 90-100% utilization

### Dashboard Not Loading

**Solution**: Create SSH tunnel
```bash
ssh -L 8000:localhost:8000 ivhl@<VM_IP>
```

---

## Scientific Questions

This 30-minute validation run tests:

1. **Scale Invariance**: Does quality-guided learning work at 35× scale?
2. **Resource Utilization**: Can we use 90-95% of H200 VRAM safely?
3. **Performance**: Is 30s per cycle acceptable at 140K nodes?
4. **Stability**: Does the system remain stable over 60 cycles?
5. **Learning Trajectory**: Are we on track for 100% convergence?

**If successful**: Validates HHmL framework for massive-scale deployment (1M+ nodes)

---

## Commands Reference

```bash
# Default run (60 cycles, 30 min)
python scripts/train_h200_scaled.py --device cuda

# Conservative (30 strips, 60K nodes)
python scripts/train_h200_scaled.py --strips 30 --nodes 2000 --hidden-dim 4096 --device cuda

# Extended (500 cycles, 8 hours)
python scripts/train_h200_scaled.py --cycles 500 --max-time 28800 --device cuda

# Resume from checkpoint
python scripts/train_h200_scaled.py --resume checkpoints/h200_scaled/checkpoint_cycle_60.pt --device cuda

# Transfer learning
python scripts/transfer_to_h200.py --baseline checkpoints/quality_guided/checkpoint_final_cycle_599.pt
python scripts/train_h200_scaled.py --resume checkpoints/h200_scaled/transferred_baseline.pt --device cuda
```

---

**Status**: Ready to deploy
**Next Step**: SSH into H200 and run Step 2-5
**Time**: ~50 minutes total (10 min setup + 30 min training + 10 min analysis)

Good luck! 🚀
