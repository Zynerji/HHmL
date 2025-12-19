# H200 Tokamak Wormhole Hunt - Deployment Guide

**Date**: 2025-12-19
**Configuration**: 300 strips × 166 nodes/strip = 49,800 nodes
**Test Run**: 20 cycles (~1-2 minutes)
**Full Run**: 800 cycles (~40-60 minutes)

---

## Quick Start (Test Run)

### 1. SSH to H200

```bash
ssh ivhl@89.169.102.45
```

### 2. Navigate and Pull Latest Code

```bash
cd ~/tHHmL
git pull origin master
source ~/hhml_env/bin/activate
```

### 3. Run 20-Cycle Test

```bash
python examples/training/train_tokamak_wormhole_hunt.py \
    --num-strips 300 \
    --nodes-per-strip 166 \
    --num-time-steps 100 \
    --num-cycles 20 \
    --seed 42 \
    --output-dir scratch/results/tokamak_wormhole
```

**Expected output:**
```
================================================================================
TOKAMAK WORMHOLE HUNT - 300 STRIP TRAINING
================================================================================
Device: cuda
Strips: 300
Nodes per strip: 166
Total nodes: 49,800
Time steps: 100
Target cycles: 20
Random seed: 42
Output: scratch/results/tokamak_wormhole/tokamak_run_YYYYMMDD_HHMMSS

Initializing INTERMEDIATE sparse mode tokamak geometry...
  Strips: 300
  Nodes per strip: 166
  Total nodes: 49,800
  Tokamak params: kappa=1.5, delta=0.3
  Sparse threshold: 0.6
  Max neighbors: 894
  Geometry generation: X.XXs
  Graph construction: X.XXs
  Estimated memory: XXX.X MB

Initializing temporal dynamics...
  Field shape: (49800, 100)
  Total DOF: 4,980,000

Initializing RNN controller...
  Input dim: 10
  Hidden dim: 4096
  Output dim: 39 (control parameters)
  Total RNN parameters: 160K

================================================================================
TRAINING START
================================================================================

Cycle 0/20
  Reward: XX.XX (fp=XX.X, vx=XX.X, wh=XX.X, uni=XX.X)
  Fixed points: XX.X% (XXXXX/4980000)
  Divergence: X.XXXXXX
  Vortices: XXX
  Wormholes: X
  Radial transport: max_gradient=X.XXXX
  Speed: X.XXs/cycle, ETA: X.X min
...
```

### 4. Check Test Results

```bash
# List output files
ls -lh scratch/results/tokamak_wormhole/tokamak_run_*/

# Should see:
# - training_results_YYYYMMDD_HHMMSS.json (~1-2 MB)
# - best_checkpoint_YYYYMMDD_HHMMSS.pt (~2-3 GB)

# Quick inspection
python << 'EOF'
import json
from pathlib import Path

runs = sorted(Path('scratch/results/tokamak_wormhole').glob('tokamak_run_*'))
latest = runs[-1]

with open(latest / f'training_results_{latest.name.split("_")[-2]}_{latest.name.split("_")[-1]}.json') as f:
    results = json.load(f)

print(f"Test run summary:")
print(f"  Cycles completed: {len(results['metrics']['rewards'])}")
print(f"  Best reward: {max(results['metrics']['rewards']):.2f}")
print(f"  Total wormholes detected: {sum(results['metrics']['wormhole_counts'])}")
print(f"  Peak wormholes: {max(results['metrics']['wormhole_counts'])}")
print(f"  Training time: {results['total_time_sec']:.1f}s ({results['total_time_sec']/60:.2f} min)")
EOF
```

**If test succeeds** (completes without errors), proceed to full 800-cycle run.

---

## Full 800-Cycle Run

### Option A: Background with nohup (Recommended)

```bash
nohup python examples/training/train_tokamak_wormhole_hunt.py \
    --num-strips 300 \
    --nodes-per-strip 166 \
    --num-time-steps 100 \
    --num-cycles 800 \
    --seed 42 \
    --output-dir scratch/results/tokamak_wormhole \
    > tokamak_800cycle_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > tokamak_800.pid

# Monitor progress
tail -f tokamak_800cycle_*.log

# Check for wormholes
grep "Wormholes:" tokamak_800cycle_*.log | tail -20

# Check completion
grep "TRAINING COMPLETE" tokamak_800cycle_*.log
```

### Option B: tmux Session

```bash
# Create session
tmux new -s tokamak800

# Inside tmux
python examples/training/train_tokamak_wormhole_hunt.py \
    --num-strips 300 \
    --nodes-per-strip 166 \
    --num-time-steps 100 \
    --num-cycles 800 \
    --seed 42 \
    --output-dir scratch/results/tokamak_wormhole

# Detach: Ctrl+B then D
# Reattach: tmux attach -t tokamak800
```

---

## Expected Behavior

### During Training

**Every 5 cycles** you should see:
```
Cycle X/800
  Reward: XX.XX (fp=XX.X, vx=XX.X, wh=XX.X, uni=XX.X)
  Fixed points: XX.X% (XXXXX/4980000)
  Divergence: X.XXXXXX
  Vortices: XXX
  Wormholes: X
    Avg strip separation: XX.X
  Radial transport: max_gradient=X.XXXX
  Speed: X.XXs/cycle, ETA: XX.X min
```

**Wormhole Detection Indicators:**
- **Wormholes: 0** - No wormholes detected (early training)
- **Wormholes: 1-5** - Few wormholes (interesting!)
- **Wormholes: 10+** - Many wormholes (DISCOVERY!)
- **Avg strip separation: 50-150** - Long-range wormholes (best!)

**Healthy Training Signs:**
- Fixed points gradually increase (0% → 70%+)
- Divergence decreases (1.0 → 0.001)
- Vortices appear (100-500 range)
- Wormholes occasionally detected (>0)
- Reward increases over cycles

**Potential Issues:**
- **Reward stuck negative** - Normal early training, should improve
- **NaN gradients** - Should auto-recover with gradient clipping
- **Wormholes: 0 throughout** - Still valuable (proves they're rare/absent)
- **OOM error** - Reduce nodes_per_strip to 150

### After Training

**Output files in** `scratch/results/tokamak_wormhole/tokamak_run_YYYYMMDD_HHMMSS/`:
- `training_results_*.json` - All metrics, wormhole detections
- `best_checkpoint_*.pt` - Best model weights + wormhole data

**Expected metrics** (800 cycles):
- Best reward: 100-200 (depends on wormhole formation)
- Best fixed points: 70-80%
- Total wormholes detected: 10-100+ (DISCOVERY if >50)
- Training time: 40-60 minutes

---

## Post-Training Analysis

### 1. Copy Results to Local

```bash
# On local machine (Windows)
mkdir -p /c/Users/cknop/.local/bin/tHHmL/results/tokamak_wormhole_20251219

# Copy training results
scp -i /c/Users/cknop/.ssh/id_ed25519 \
    ivhl@89.169.102.45:~/tHHmL/scratch/results/tokamak_wormhole/tokamak_run_*/training_results_*.json \
    /c/Users/cknop/.local/bin/tHHmL/results/tokamak_wormhole_20251219/

# Copy checkpoint
scp -i /c/Users/cknop/.ssh/id_ed25519 \
    ivhl@89.169.102.45:~/tHHmL/scratch/results/tokamak_wormhole/tokamak_run_*/best_checkpoint_*.pt \
    /c/Users/cknop/.local/bin/tHHmL/results/tokamak_wormhole_20251219/
```

### 2. Run Wormhole Analysis

```bash
cd /c/Users/cknop/.local/bin/tHHmL
python examples/analysis/analyze_tokamak_wormholes.py \
    --results-file results/tokamak_wormhole_20251219/training_results_*.json \
    --output-dir results/tokamak_wormhole_20251219/analysis
```

This will generate:
- Wormhole statistics (count, distribution, separation)
- Radial transport analysis (diffusion vs. tunneling)
- Strip correlation matrix (which strips couple?)
- Charge flow visualization
- Emergent wormhole findings document

---

## Key Wormhole Signatures to Look For

### 1. **Long-Range Wormholes** (Strip separation > 50)
- Indicates true tunneling across tokamak layers
- Strongest evidence for emergent spacetime shortcuts

### 2. **Charge-Conserving Wormholes**
- Opposite topological charges on endpoints
- Suggests true topological wormhole (not random alignment)

### 3. **Persistent Wormholes**
- Same wormhole appears across multiple cycles
- Indicates stable geometry (not transient)

### 4. **Radial Transport Acceleration**
- Field propagates faster than diffusion predicts
- Direct evidence of wormhole-mediated transport

### 5. **Strip Synchronization**
- Distant strips show phase coherence
- Suggests wormhole-enabled communication

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'src.hhml'"

```bash
cd ~/tHHmL
pip install -e .
```

### Issue: CUDA Out of Memory

**Reduce node count:**
```bash
python examples/training/train_tokamak_wormhole_hunt.py \
    --num-strips 300 \
    --nodes-per-strip 150 \  # Reduced from 166
    --num-cycles 20
```

### Issue: Training crashes with NaN

**Should auto-recover with gradient clipping, but if persistent:**
- Reduce learning rate in script (line 382: `lr=1e-5` instead of `1e-4`)
- Check log for repeated NaN warnings

### Issue: No wormholes detected

**This is scientifically valuable!** It means:
- Wormholes are genuinely rare/absent (important negative result)
- May need different parameters to induce wormholes
- Still publish as "Absence of Wormholes in Standard Tokamak Geometry"

---

## Scientific Outcomes

### If wormholes detected (>10 total):
✅ **Novel Discovery**: "Topologically-Mediated Radial Transport via Inter-Strip Vortex Wormholes"
✅ Update EMERGENTS.md with wormhole phenomenon
✅ Generate whitepaper with wormhole statistics
✅ Test reproducibility with different seeds

### If no wormholes detected:
✅ **Rigorous Negative Result**: "Absence of Inter-Strip Wormholes in 300-Strip Tokamak Geometry"
✅ Document null result (equally valuable)
✅ Explore parameter space (different coupling strengths, etc.)

---

## Next Steps After 800-Cycle Run

1. **Analyze wormhole statistics**
2. **Measure radial transport speed**
3. **Test reproducibility** (run with seeds 43, 44, 45)
4. **Parameter scan** (vary coupling strengths to induce wormholes)
5. **Scaling study** (100 strips vs. 300 strips vs. 500 strips)
6. **Compare to single-strip** (does multi-strip enable new phenomena?)

---

## Quick Reference Commands

**SSH**: `ssh ivhl@89.169.102.45`
**Activate env**: `source ~/hhml_env/bin/activate`
**Pull code**: `cd ~/tHHmL && git pull origin master`
**Test run**: `python examples/training/train_tokamak_wormhole_hunt.py --num-cycles 20`
**Full run**: Add `nohup ... &` and monitor with `tail -f *.log`
**Check wormholes**: `grep "Wormholes:" *.log | grep -v "Wormholes: 0"`
**Kill training**: `cat tokamak_800.pid | xargs kill`

---

**Ready to hunt wormholes!** 🚀🌀

Run 20-cycle test first to verify everything works, then launch 800-cycle full run.
