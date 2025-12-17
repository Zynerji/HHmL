# Live Dashboard Integration Guide

## Quick Start - Add to Any Training Script

The live dashboard provides real-time monitoring of training metrics through a web browser.

### 1. Import the Dashboard

```python
from hhml.utils.live_dashboard import TrainingDashboard
```

### 2. Initialize Before Training

```python
# Initialize dashboard (will auto-open browser)
dashboard = TrainingDashboard(port=8000, auto_open=True)
dashboard.start()

print("Dashboard running at http://localhost:8000")
```

### 3. Update in Training Loop

```python
for cycle in range(num_cycles):
    # ... your training code ...

    # Update dashboard with current metrics
    dashboard.update({
        'cycle': cycle,
        'density': vortex_density,           # 0-1 float
        'quality': vortex_quality,           # 0-1 float
        'reward': reward,                    # float
        'annihilations': num_removed,        # int
        'cycles_at_target': cycles_stable    # int
    })
```

### 4. Stop After Training

```python
# Clean shutdown
dashboard.stop()
```

## Complete Example

```python
from hhml.utils.live_dashboard import TrainingDashboard

def train_with_dashboard():
    # Start dashboard
    dashboard = TrainingDashboard(port=8000)
    dashboard.start()

    try:
        for cycle in range(500):
            # Your training logic
            vortex_density = compute_vortex_density()
            vortex_quality = compute_vortex_quality()
            reward = compute_reward()
            num_removed = count_annihilations()

            # Update dashboard
            dashboard.update({
                'cycle': cycle,
                'density': vortex_density,
                'quality': vortex_quality,
                'reward': reward,
                'annihilations': num_removed,
                'cycles_at_target': 0  # or your stability counter
            })

    finally:
        # Always stop dashboard
        dashboard.stop()
```

## Features

### Real-Time Charts
- **Vortex Density & Quality**: Dual line chart showing both metrics
- **Reward Over Time**: Training reward progression
- **Annihilation Events**: Bar chart of vortex removals per cycle
- **Stability**: Cycles spent at target density

### Live Statistics Cards
- Current Cycle
- Vortex Density (%)
- Vortex Quality (0-1)
- Current Reward
- Annihilations This Cycle
- Consecutive Cycles at Target

### Auto-Features
- Auto-opens browser window when started
- Auto-refreshes with new data (no manual refresh needed)
- Keeps last 100 data points visible
- Responsive design (works on any screen size)

## Configuration Options

```python
# Custom port
dashboard = TrainingDashboard(port=9000)

# Disable auto-open browser
dashboard = TrainingDashboard(port=8000, auto_open=False)
```

## Multiple Training Runs

If running multiple training sessions simultaneously, use different ports:

```python
# Training 1
dashboard1 = TrainingDashboard(port=8000)

# Training 2
dashboard2 = TrainingDashboard(port=8001)
```

## Testing the Dashboard

Run the standalone test:

```python
python -m hhml.utils.live_dashboard
```

This will simulate 100 cycles of training data for 60 seconds.

## Troubleshooting

**Browser doesn't open automatically?**
- Manually navigate to `http://localhost:8000`

**Port already in use?**
- Choose a different port: `TrainingDashboard(port=8001)`

**Dashboard shows "Disconnected"?**
- Training may have finished - normal behavior
- Check console for errors

**No data appearing?**
- Ensure `dashboard.update()` is being called in your training loop
- Check that metrics dict has correct keys

## Integration Checklist

- [ ] Import `TrainingDashboard`
- [ ] Call `dashboard.start()` before training loop
- [ ] Call `dashboard.update(metrics)` inside training loop
- [ ] Call `dashboard.stop()` after training (in `finally` block)
- [ ] Pass all required metrics: cycle, density, quality, reward

## Example: Integrating into train_multi_strip.py

```python
# At top of file
from hhml.utils.live_dashboard import TrainingDashboard

# In train_multi_strip() function, after initialization:
dashboard = TrainingDashboard(port=8000)
dashboard.start()

try:
    for cycle in range(start_cycle, total_cycles):
        # ... existing training code ...

        # After computing metrics, add:
        dashboard.update({
            'cycle': cycle,
            'density': reward_breakdown['vortex_density_mean'],
            'quality': reward_breakdown.get('avg_vortex_quality', 0.0),
            'reward': reward,
            'annihilations': annihilation_stats['num_removed'],
            'cycles_at_target': cycles_at_target if 'cycles_at_target' in locals() else 0
        })

finally:
    dashboard.stop()
```

That's it! Your training now has live monitoring.
