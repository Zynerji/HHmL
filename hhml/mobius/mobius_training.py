#!/usr/bin/env python3
"""
Möbius Helix Training - Enhanced Architecture
============================================
New features:
- RNN controls helical sites (num_nodes)
- Hidden dim: 4096 (4× increase)
- Möbius strip topology (no open ends)
- FIXED REWARD: Removed coherence component causing vortex collapse
- Exponential penalties for collapse
- 60-90 minute extended run

Author: iVHL Framework
Date: 2025-12-16
"""

import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from collections import deque

print("Initializing Möbius Helix Training...")
print("=" * 80)

# ============================================================================
# Enhanced RNN Agent with Helical Sites Control
# ============================================================================

class MobiusRNNAgent(nn.Module):
    """Enhanced RNN with helical sites control and 4096 hidden dim"""

    def __init__(self, state_dim=256, action_dim=128, hidden_dim=4096, device='cpu'):
        super().__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim

        print(f"Initializing Möbius RNN Agent:")
        print(f"  Hidden dim: {hidden_dim} (4× baseline)")
        print(f"  Device: {device}")

        # 4-layer LSTM with 4096 hidden
        self.rnn = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
            dropout=0.15
        ).to(device)

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Tanh()
        ).to(device)

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

        # Structural parameter heads
        self.w_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        self.tau_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        # NEW: Helical sites control
        self.sites_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # [0, 1] → [10K, 100K]
        ).to(device)

        # uVHL parameter heads
        self.uvhl_w_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        self.uvhl_L_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        self.uvhl_n_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        ).to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        self.hidden = None
        self.episode_count = 0

        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total parameters: {total_params:,}")

    def reset_hidden(self, batch_size=1):
        h0 = torch.zeros(4, batch_size, self.hidden_dim).to(self.device)
        c0 = torch.zeros(4, batch_size, self.hidden_dim).to(self.device)
        self.hidden = (h0, c0)

    def forward(self, state):
        if self.hidden is None:
            self.reset_hidden()

        rnn_out, new_hidden = self.rnn(state.unsqueeze(0).unsqueeze(0), self.hidden)
        self.hidden = tuple(h.detach() for h in new_hidden)

        features = rnn_out.squeeze(0).squeeze(0)
        action = self.actor(features)
        value = self.critic(features)

        return action, value, features

    def compute_structure_params(self, features):
        """Compute all structural parameters including helical sites"""
        w_normalized = self.w_head(features)
        tau_normalized = self.tau_head(features)
        sites_normalized = self.sites_head(features)  # NEW
        uvhl_w_normalized = self.uvhl_w_head(features)
        uvhl_L_normalized = self.uvhl_L_head(features)
        uvhl_n_normalized = self.uvhl_n_head(features)

        # Scale to ranges
        w_windings = 3.8 + w_normalized * (150.0 - 3.8)  # [3.8, 150]
        tau_torsion = tau_normalized * 3.0  # [0, 3] - even lower to avoid collapse
        num_sites = 10000 + sites_normalized * 90000  # [10K, 100K] helical sites
        uvhl_w = 0.1 + uvhl_w_normalized * 49.9  # [0.1, 50]
        uvhl_L = 0.1 + uvhl_L_normalized * 49.9  # [0.1, 50]
        uvhl_n = 0.05 + uvhl_n_normalized * 19.95  # [0.05, 20]

        # Compute ratios
        w_L_ratio = uvhl_w / (uvhl_L + 1e-6)
        w_n_ratio = uvhl_w / (uvhl_n + 1e-6)
        L_n_ratio = uvhl_L / (uvhl_n + 1e-6)
        tau_w_ratio = tau_torsion / (w_windings + 1e-6)
        sites_w_ratio = num_sites / (w_windings + 1e-6)  # Sites per winding

        return {
            'windings': w_windings,
            'tau_torsion': tau_torsion,
            'num_sites': num_sites,  # NEW
            'uvhl_w': uvhl_w,
            'uvhl_L': uvhl_L,
            'uvhl_n': uvhl_n,
            'w_L_ratio': w_L_ratio,
            'w_n_ratio': w_n_ratio,
            'L_n_ratio': L_n_ratio,
            'tau_w_ratio': tau_w_ratio,
            'sites_w_ratio': sites_w_ratio  # NEW
        }


# ============================================================================
# Möbius Helix Sphere
# ============================================================================

class MobiusHelixSphere:
    """Holographic sphere with Möbius strip topology (no open ends)"""

    def __init__(self, num_nodes=50000, radius=1.0, device='cpu',
                 w_windings=3.8, tau_torsion=0.0,
                 uvhl_w=3.2, uvhl_L=7.5, uvhl_n=3.8):
        self.device = device
        self.num_nodes = num_nodes
        self.radius = radius
        self.w_windings = w_windings
        self.tau_torsion = tau_torsion
        self.uvhl_w = uvhl_w
        self.uvhl_L = uvhl_L
        self.uvhl_n = uvhl_n

        print(f"\nInitializing Möbius Helix Sphere:")
        print(f"  Topology: Möbius strip (no open ends)")
        print(f"  Nodes: {num_nodes:,}")
        print(f"  w={w_windings:.2f}, tau={tau_torsion:.2f}")
        print(f"  uVHL: w={uvhl_w:.2f}, L={uvhl_L:.2f}, n={uvhl_n:.2f}")

        # Generate Möbius helix lattice
        self.theta, self.phi = self._generate_mobius_helix(w_windings, tau_torsion)

        # Cartesian
        self.x = radius * torch.sin(self.theta) * torch.cos(self.phi)
        self.y = radius * torch.sin(self.theta) * torch.sin(self.phi)
        self.z = radius * torch.cos(self.theta)

        # Wave properties
        self.amplitudes = torch.ones(num_nodes, device=device) * 2.0
        self.phases = torch.rand(num_nodes, device=device) * 2 * np.pi
        self.frequencies = torch.randn(num_nodes, device=device) * 0.3 + 1.5

        # Field
        self.field = torch.zeros(num_nodes, dtype=torch.complex64, device=device)

        # Tracking
        self.vortex_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)

        mem_mb = (self.x.element_size() * self.x.nelement() * 3 +
                  self.field.element_size() * self.field.nelement()) / 1e6
        print(f"  Memory: {mem_mb:.1f} MB")

    def _generate_mobius_helix(self, w_windings, tau_torsion=0.0):
        """
        Generate Möbius helix: helix twisted into Möbius strip (no open ends)

        Key difference from regular helix:
        - Regular: theta ∈ [0, π], has north/south poles (open ends)
        - Möbius: theta ∈ [0, 2π], wraps back with half-twist (closed loop)
        """
        indices = torch.arange(self.num_nodes, device=self.device, dtype=torch.float32)

        pi = torch.tensor(3.14159265359, device=self.device)

        # Möbius parameter: u ∈ [0, 2π] (full loop around the band)
        u = 2.0 * pi * indices / self.num_nodes

        # For Möbius strip, we need:
        # 1. Main circular path (around the central circle)
        # 2. Half-twist as we go around
        # 3. Helical winding superimposed

        # Theta: oscillates with half-twist for Möbius topology
        # As u goes 0→2π, theta goes 0→π→0 (half-twist)
        theta_mobius = pi * (1.0 + torch.cos(u))  # [0, 2π] with half oscillation

        # Phi: helical winding around the Möbius band
        phi = w_windings * u  # Helical winding

        # Add torsion coupling
        phi = phi + tau_torsion * theta_mobius

        # Add Möbius twist: half-twist as we complete the loop
        mobius_twist = 0.5 * u  # 180° twist over full loop
        phi = phi + mobius_twist

        # Use theta_mobius as the "latitude" on Möbius band
        theta = theta_mobius

        return theta, phi

    def apply_structure_params(self, params):
        """Apply RNN-discovered parameters including num_sites"""
        regenerate = False

        if 'windings' in params:
            w_new = params['windings'].item()
            self.w_windings = 0.95 * self.w_windings + 0.05 * w_new
            regenerate = True

        if 'tau_torsion' in params:
            tau_new = params['tau_torsion'].item()
            self.tau_torsion = 0.9 * self.tau_torsion + 0.1 * tau_new
            regenerate = True

        # NEW: Update num_nodes (helical sites)
        if 'num_sites' in params:
            sites_new = int(params['num_sites'].item())
            # Smooth transition: gradually adjust node count
            target_nodes = int(0.95 * self.num_nodes + 0.05 * sites_new)
            target_nodes = max(10000, min(target_nodes, 100000))  # Clamp [10K, 100K]

            if abs(target_nodes - self.num_nodes) > 1000:  # Only regenerate if significant change
                self.num_nodes = target_nodes
                regenerate = True
                print(f"  [Adjusting helical sites: {self.num_nodes:,}]")

        if regenerate:
            with torch.no_grad():
                self.theta, self.phi = self._generate_mobius_helix(
                    self.w_windings, self.tau_torsion
                )
                self.x = self.radius * torch.sin(self.theta) * torch.cos(self.phi)
                self.y = self.radius * torch.sin(self.theta) * torch.sin(self.phi)
                self.z = self.radius * torch.cos(self.theta)

                # Resize arrays if node count changed
                if self.amplitudes.shape[0] != self.num_nodes:
                    self.amplitudes = torch.ones(self.num_nodes, device=self.device) * 2.0
                    self.phases = torch.rand(self.num_nodes, device=self.device) * 2 * np.pi
                    self.frequencies = torch.randn(self.num_nodes, device=self.device) * 0.3 + 1.5
                    self.field = torch.zeros(self.num_nodes, dtype=torch.complex64, device=self.device)

        # Update uVHL parameters
        if 'uvhl_w' in params:
            self.uvhl_w = 0.9 * self.uvhl_w + 0.1 * params['uvhl_w'].item()
        if 'uvhl_L' in params:
            self.uvhl_L = 0.9 * self.uvhl_L + 0.1 * params['uvhl_L'].item()
        if 'uvhl_n' in params:
            self.uvhl_n = 0.9 * self.uvhl_n + 0.1 * params['uvhl_n'].item()

    def evolve(self, action=None, structure_params=None, dt=0.01):
        """Evolution with adaptive sampling"""
        if structure_params is not None:
            self.apply_structure_params(structure_params)

        if action is not None:
            action_expanded = torch.zeros(self.num_nodes, device=self.device)
            action_len = min(len(action), self.num_nodes)
            action_expanded[:action_len] = action[:action_len]
            self.amplitudes += action_expanded * 0.02
            self.amplitudes = torch.clamp(self.amplitudes, 0.5, 5.0)

        t_now = torch.tensor(time.time(), device=self.device)

        # Adaptive sampling
        base_samples = 1000
        sample_size = int(base_samples * (self.uvhl_n / 3.8))
        sample_size = min(max(sample_size, 500), 2000)

        sample_indices = torch.randperm(self.num_nodes, device=self.device)[:sample_size]

        # Batched computation
        batch_size = 100
        field_updates = torch.zeros(sample_size, dtype=torch.complex64, device=self.device)

        for batch_start in range(0, sample_size, batch_size):
            batch_end = min(batch_start + batch_size, sample_size)
            batch_idx = sample_indices[batch_start:batch_end]

            sample_x = self.x[batch_idx].unsqueeze(1)
            sample_y = self.y[batch_idx].unsqueeze(1)
            sample_z = self.z[batch_idx].unsqueeze(1)

            distances = torch.sqrt(
                (sample_x - self.x.unsqueeze(0))**2 +
                (sample_y - self.y.unsqueeze(0))**2 +
                (sample_z - self.z.unsqueeze(0))**2
            ) + 0.05

            wave = self.amplitudes.unsqueeze(0) * torch.sin(
                self.frequencies.unsqueeze(0) * t_now - 3.0 * distances
            ) / distances

            field_magnitudes = torch.sum(wave, dim=1)
            field_updates[batch_start:batch_end] = field_magnitudes * torch.exp(
                1j * self.phases[batch_idx]
            )

        self.field[sample_indices] = field_updates
        self._detect_vortices()

    def _detect_vortices(self):
        """Detect vortices"""
        stride = max(1, self.num_nodes // 500)
        sample = self.field[::stride]
        field_mag = torch.abs(sample)
        vortex_count = torch.sum((field_mag < 0.3).float()).item() * stride
        self.vortex_history.append(int(vortex_count))

    def get_state(self):
        """Get 256-dim state"""
        stride = max(1, self.num_nodes // 256)
        sample_indices = torch.arange(
            0, min(256 * stride, self.num_nodes), stride,
            device=self.device, dtype=torch.long
        )[:256]

        field_mag = torch.abs(self.field[sample_indices])
        field_phase = torch.angle(self.field[sample_indices])

        state = torch.cat([field_mag[:128], field_phase[:128]])
        return state.float()

    def compute_reward(self):
        """FIXED REWARD: Strongly maximize vortex density"""
        stride = max(1, self.num_nodes // 1000)
        sample_field = torch.abs(self.field[::stride])

        # Vortex density (PRIMARY GOAL - maximize this!)
        vortex_count = torch.sum((sample_field < 0.3).float()) * stride
        vortex_density = vortex_count / self.num_nodes

        # PRIMARY: Strong reward for vortex density
        vortex_reward = vortex_density * 20.0

        # EXPONENTIAL BONUS for high density
        if vortex_density > 0.80:
            bonus = 100.0 * (vortex_density - 0.80)  # Massive bonus
        elif vortex_density > 0.70:
            bonus = 50.0 * (vortex_density - 0.70)  # Strong bonus
        else:
            bonus = 0.0

        # EXPONENTIAL PENALTY for collapse
        if vortex_density < 0.30:
            penalty = -100.0 * (0.30 - vortex_density)  # Severe penalty
        elif vortex_density < 0.50:
            penalty = -30.0 * (0.50 - vortex_density)  # Strong penalty
        else:
            penalty = 0.0

        total_reward = vortex_reward + bonus + penalty
        # REMOVED: coherence_reward (was causing collapse!)

        self.reward_history.append(total_reward.item())

        return total_reward.item()


# ============================================================================
# Training Loop
# ============================================================================

def run_mobius_training(num_cycles=1500, num_nodes=50000, device='cpu', checkpoint_interval=100):
    """Run Möbius helix training"""

    print("\n" + "=" * 80)
    print("MÖBIUS HELIX TRAINING - ENHANCED ARCHITECTURE")
    print("=" * 80)
    print(f"Features:")
    print(f"  - Möbius strip topology (no open ends)")
    print(f"  - RNN controls helical sites [10K, 100K]")
    print(f"  - Hidden dim: 4096 (4× baseline)")
    print(f"  - FIXED reward: removed coherence bug")
    print(f"  - Exponential penalties for collapse")
    print(f"  - Target: 60-90 minutes")
    print(f"  - Cycles: {num_cycles}")
    print(f"  - Nodes: {num_nodes:,}")
    print()

    # Initialize
    agent = MobiusRNNAgent(device=device, hidden_dim=4096)
    sphere = MobiusHelixSphere(num_nodes=num_nodes, device=device)

    # Create output directory
    output_dir = Path("results/mobius_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Metrics tracking
    metrics_history = []
    best_reward = -float('inf')
    best_checkpoint = None

    # Performance tracking
    cycle_times = deque(maxlen=50)

    print("Starting training...")
    print()

    start_time = time.time()

    for cycle in range(num_cycles):
        cycle_start = time.time()

        # Get state
        state = sphere.get_state()

        # RNN forward pass
        action, value, features = agent.forward(state)

        # Compute structural parameters
        structure_params = agent.compute_structure_params(features)

        # Evolve sphere
        with torch.no_grad():
            sphere.evolve(action.detach(), structure_params)

        # Compute reward
        reward = sphere.compute_reward()

        # Policy gradient update
        loss = -value * reward
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        agent.optimizer.step()

        # Track metrics
        w = structure_params['windings'].item()
        tau = structure_params['tau_torsion'].item()
        sites = structure_params['num_sites'].item()
        uvhl_w = structure_params['uvhl_w'].item()
        uvhl_L = structure_params['uvhl_L'].item()
        uvhl_n = structure_params['uvhl_n'].item()
        sites_w = structure_params['sites_w_ratio'].item()

        vortices = sphere.vortex_history[-1] if sphere.vortex_history else 0
        vortex_density = vortices / sphere.num_nodes

        metrics = {
            'cycle': cycle,
            'w': w,
            'tau': tau,
            'num_sites': sites,
            'sites_w_ratio': sites_w,
            'uvhl_w': uvhl_w,
            'uvhl_L': uvhl_L,
            'uvhl_n': uvhl_n,
            'vortices': vortices,
            'vortex_density': vortex_density,
            'reward': reward,
            'rnn_value': value.item(),
            'actual_nodes': sphere.num_nodes
        }
        metrics_history.append(metrics)

        # Update best
        if reward > best_reward:
            best_reward = reward
            best_checkpoint = metrics.copy()

        # Cycle timing
        cycle_time = time.time() - cycle_start
        cycle_times.append(cycle_time)

        # Progress print
        if cycle % 10 == 0 or cycle == num_cycles - 1:
            elapsed = time.time() - start_time
            avg_cycle_time = np.mean(cycle_times)
            eta = avg_cycle_time * (num_cycles - cycle - 1)

            print(f"[{cycle:4d}/{num_cycles}] "
                  f"Sites={sphere.num_nodes/1000:.0f}K w={w:5.1f} tau={tau:4.2f} | "
                  f"Vortex: {vortex_density*100:5.1f}% ({vortices:,}) | "
                  f"Reward={reward:7.2f} | "
                  f"Time: {elapsed/60:.1f}m ETA: {eta/60:.1f}m")

        # Save checkpoint
        if (cycle + 1) % checkpoint_interval == 0:
            checkpoint_file = output_dir / f"mobius_checkpoint_{timestamp}_cycle{cycle+1}.json"
            checkpoint_data = {
                'cycle': cycle + 1,
                'metrics': metrics,
                'best_reward': best_reward,
                'elapsed_time': time.time() - start_time
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"  [Checkpoint: cycle {cycle+1}, best_reward={best_reward:.2f}]")

    total_time = time.time() - start_time

    print()
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)

    # Final analysis
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    final = metrics_history[-1]
    print(f"\nFinal Parameters:")
    print(f"  Helical sites: {final['actual_nodes']:,}")
    print(f"  w (windings): {final['w']:.2f}")
    print(f"  tau (torsion): {final['tau']:.3f}")
    print(f"  sites/w ratio: {final['sites_w_ratio']:.0f}")
    print(f"  Vortex density: {final['vortex_density']*100:.2f}%")
    print(f"  Final reward: {final['reward']:.2f}")
    print(f"  Best reward: {best_reward:.2f}")

    # Save final results
    results_file = output_dir / f"mobius_training_{timestamp}_FINAL.json"
    results = {
        'timestamp': timestamp,
        'device': str(device),
        'num_cycles': num_cycles,
        'total_time_seconds': total_time,
        'final_metrics': final,
        'best_checkpoint': best_checkpoint,
        'best_reward': best_reward,
        'metrics_history': metrics_history
    }

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Full training mode: 1200 cycles (~60-90 minutes)
    num_cycles = 1200

    print("=" * 80)
    print("FULL TRAINING MODE: 1200 cycles (~60-90 minutes)")
    print("=" * 80)
    print("ENHANCEMENTS:")
    print("  1. FIXED REWARD: Removed coherence component")
    print("  2. Hidden dim: 4096 (4× baseline)")
    print("  3. Exponential penalties for vortex collapse")
    print("=" * 80)
    print("Starting in 3 seconds...")
    time.sleep(3)

    results = run_mobius_training(
        num_cycles=num_cycles,
        device=device,
        checkpoint_interval=100  # Checkpoint every 100 cycles
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
