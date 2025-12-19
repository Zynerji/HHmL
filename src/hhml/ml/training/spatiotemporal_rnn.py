#!/usr/bin/env python3
"""
Spatiotemporal RNN - 39-Parameter Control
==========================================

Extends HHmL RNN from 23 → 39 parameters for full spatiotemporal control.

Parameter Breakdown:
- 23 spatial parameters (inherited from HHmL)
- 9 temporal dynamics parameters (tHHmL)
- 7 temporal vortex parameters (NEW - Phase 2)

Total: 39 parameters controlling (2+1)D spacetime dynamics including
temporal vortices and spatiotemporal vortex tubes.

Author: tHHmL Project (Spatiotemporal Mobius Lattice)
Date: 2025-12-18
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict


class SpatiotemporalRNN(nn.Module):
    """
    Extended RNN for full spatiotemporal Möbius control.

    Architecture:
    - 4-layer LSTM with 4096 hidden dim (inherited from HHmL)
    - 39 parameter heads:
      * 23 spatial (HHmL parameters)
      * 9 temporal dynamics (tHHmL Phase 1)
      * 7 temporal vortex (tHHmL Phase 2 - NEW)

    Temporal Dynamics Parameters (Phase 1):
    1. temporal_twist (τ): Temporal Möbius twist angle
    2. retrocausal_strength (α): Future-past coupling strength
    3. temporal_relaxation (β): Convergence damping factor
    4. num_time_steps (T): Temporal resolution (discretization)
    5. prophetic_coupling (γ): Forward-backward mixing rate
    6. temporal_phase_shift (φ_t): Phase at temporal reconnection
    7. temporal_decay (δ_t): Temporal dampening factor
    8. forward_backward_balance (ρ): Weighting between forward/backward
    9. temporal_noise_level (σ_t): Exploration noise in temporal evolution

    Temporal Vortex Parameters (Phase 2 - NEW):
    10. temporal_vortex_injection_rate (ν): Rate of temporal vortex injection
    11. temporal_vortex_winding (n_t): Winding number for temporal vortices
    12. temporal_vortex_core_size (ε_t): Size of temporal vortex cores
    13. vortex_tube_probability (p_tube): Probability of tube formation
    14. tube_winding_number (n_tube): Winding for spatiotemporal tubes
    15. tube_core_size (ε_tube): Size of vortex tube cores
    16. temporal_vortex_annihilation_rate (μ_t): Rate of temporal vortex removal
    """

    def __init__(
        self,
        state_dim: int = 256,
        hidden_dim: int = 4096,
        device: str = 'cuda'
    ):
        super().__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        print(f"Initializing Spatiotemporal RNN:")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  State dim: {state_dim}")
        print(f"  Total parameters: 39 (23 spatial + 9 temporal + 7 vortex)")
        print(f"  Device: {device}")

        # 4-layer LSTM (inherited from HHmL)
        self.rnn = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=4,
            batch_first=True,
            dropout=0.15
        ).to(device)

        # Value critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ).to(device)

        # =====================================================================
        # Spatial Parameter Heads (23 parameters - inherited from HHmL)
        # =====================================================================

        # Core resonance
        self.kappa_head = self._make_param_head()    # Wave coupling
        self.delta_head = self._make_param_head()    # Damping
        self.lambda_head = self._make_param_head()   # Wavelength scale
        self.gamma_head = self._make_param_head()    # Nonlinearity

        # Sampling
        self.theta_sampling_head = self._make_param_head()
        self.phi_sampling_head = self._make_param_head()

        # Topology
        self.winding_density_head = self._make_param_head()
        self.twist_rate_head = self._make_param_head()
        self.cross_coupling_head = self._make_param_head()
        self.boundary_strength_head = self._make_param_head()

        # Quantum error correction
        self.qec_layers_head = self._make_param_head()
        self.entanglement_strength_head = self._make_param_head()
        self.decoherence_rate_head = self._make_param_head()
        self.measurement_rate_head = self._make_param_head()
        self.basis_rotation_head = self._make_param_head()
        self.alpha_qec_head = self._make_param_head()
        self.beta_qec_head = self._make_param_head()

        # Vortex annihilation
        self.antivortex_strength_head = self._make_param_head()
        self.annihilation_radius_head = self._make_param_head()
        self.pruning_threshold_head = self._make_param_head()
        self.preserve_ratio_head = self._make_param_head()
        self.quality_threshold_head = self._make_param_head()
        self.refinement_strength_head = self._make_param_head()

        # =====================================================================
        # Temporal Parameter Heads (9 parameters - NEW for tHHmL)
        # =====================================================================

        self.temporal_twist_head = self._make_param_head()           # τ
        self.retrocausal_strength_head = self._make_param_head()     # α
        self.temporal_relaxation_head = self._make_param_head()      # β
        self.num_time_steps_head = self._make_param_head()           # T
        self.prophetic_coupling_head = self._make_param_head()       # γ
        self.temporal_phase_shift_head = self._make_param_head()     # φ_t
        self.temporal_decay_head = self._make_param_head()           # δ_t
        self.forward_backward_balance_head = self._make_param_head() # ρ
        self.temporal_noise_level_head = self._make_param_head()     # σ_t

        # =====================================================================
        # Temporal Vortex Parameter Heads (7 parameters - NEW Phase 2)
        # =====================================================================

        self.temporal_vortex_injection_rate_head = self._make_param_head()      # ν
        self.temporal_vortex_winding_head = self._make_param_head()             # n_t
        self.temporal_vortex_core_size_head = self._make_param_head()           # ε_t
        self.vortex_tube_probability_head = self._make_param_head()             # p_tube
        self.tube_winding_number_head = self._make_param_head()                 # n_tube
        self.tube_core_size_head = self._make_param_head()                      # ε_tube
        self.temporal_vortex_annihilation_rate_head = self._make_param_head()   # μ_t

        print("  Parameter heads initialized (39 total)")

    def _make_param_head(self) -> nn.Module:
        """Create a parameter head network."""
        return nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # All parameters in [0, 1], rescale later
        ).to(self.device)

    def forward(self, state: torch.Tensor, hidden=None) -> Dict[str, torch.Tensor]:
        """
        Forward pass: state → 39 parameters + value.

        Args:
            state: Input state tensor (batch, seq_len, state_dim)
            hidden: Optional LSTM hidden state

        Returns:
            parameters: Dictionary of 39 parameters + value
        """
        # RNN encoding
        if hidden is None:
            rnn_out, hidden = self.rnn(state)
        else:
            rnn_out, hidden = self.rnn(state, hidden)

        # Get last hidden state
        h_last = rnn_out[:, -1, :]  # (batch, hidden_dim)

        # Value prediction
        value = self.critic(h_last)

        # =====================================================================
        # Spatial Parameters (23)
        # =====================================================================

        spatial_params = {
            'kappa': self.kappa_head(h_last).squeeze(),
            'delta': self.delta_head(h_last).squeeze(),
            'lambda': self.lambda_head(h_last).squeeze(),
            'gamma': self.gamma_head(h_last).squeeze(),

            'theta_sampling': self.theta_sampling_head(h_last).squeeze(),
            'phi_sampling': self.phi_sampling_head(h_last).squeeze(),

            'winding_density': self.winding_density_head(h_last).squeeze(),
            'twist_rate': self.twist_rate_head(h_last).squeeze(),
            'cross_coupling': self.cross_coupling_head(h_last).squeeze(),
            'boundary_strength': self.boundary_strength_head(h_last).squeeze(),

            'qec_layers': self.qec_layers_head(h_last).squeeze(),
            'entanglement_strength': self.entanglement_strength_head(h_last).squeeze(),
            'decoherence_rate': self.decoherence_rate_head(h_last).squeeze(),
            'measurement_rate': self.measurement_rate_head(h_last).squeeze(),
            'basis_rotation': self.basis_rotation_head(h_last).squeeze(),
            'alpha_qec': self.alpha_qec_head(h_last).squeeze(),
            'beta_qec': self.beta_qec_head(h_last).squeeze(),

            'antivortex_strength': self.antivortex_strength_head(h_last).squeeze(),
            'annihilation_radius': self.annihilation_radius_head(h_last).squeeze(),
            'pruning_threshold': self.pruning_threshold_head(h_last).squeeze(),
            'preserve_ratio': self.preserve_ratio_head(h_last).squeeze(),
            'quality_threshold': self.quality_threshold_head(h_last).squeeze(),
            'refinement_strength': self.refinement_strength_head(h_last).squeeze(),
        }

        # =====================================================================
        # Temporal Dynamics Parameters (9 - Phase 1)
        # =====================================================================

        temporal_params = {
            'temporal_twist': self.temporal_twist_head(h_last).squeeze(),
            'retrocausal_strength': self.retrocausal_strength_head(h_last).squeeze(),
            'temporal_relaxation': self.temporal_relaxation_head(h_last).squeeze(),
            'num_time_steps': self.num_time_steps_head(h_last).squeeze(),
            'prophetic_coupling': self.prophetic_coupling_head(h_last).squeeze(),
            'temporal_phase_shift': self.temporal_phase_shift_head(h_last).squeeze(),
            'temporal_decay': self.temporal_decay_head(h_last).squeeze(),
            'forward_backward_balance': self.forward_backward_balance_head(h_last).squeeze(),
            'temporal_noise_level': self.temporal_noise_level_head(h_last).squeeze(),
        }

        # =====================================================================
        # Temporal Vortex Parameters (7 - Phase 2 NEW)
        # =====================================================================

        vortex_params = {
            'temporal_vortex_injection_rate': self.temporal_vortex_injection_rate_head(h_last).squeeze(),
            'temporal_vortex_winding': self.temporal_vortex_winding_head(h_last).squeeze(),
            'temporal_vortex_core_size': self.temporal_vortex_core_size_head(h_last).squeeze(),
            'vortex_tube_probability': self.vortex_tube_probability_head(h_last).squeeze(),
            'tube_winding_number': self.tube_winding_number_head(h_last).squeeze(),
            'tube_core_size': self.tube_core_size_head(h_last).squeeze(),
            'temporal_vortex_annihilation_rate': self.temporal_vortex_annihilation_rate_head(h_last).squeeze(),
        }

        # Combine all parameters
        all_params = {**spatial_params, **temporal_params, **vortex_params, 'value': value.squeeze()}

        return all_params, hidden

    def rescale_parameters(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Rescale parameters from [0, 1] to physical ranges.

        Spatial parameters: Use HHmL ranges
        Temporal parameters: Define appropriate ranges for temporal dynamics

        Args:
            params: Raw parameters in [0, 1]

        Returns:
            rescaled: Parameters in physical ranges
        """
        rescaled = params.copy()

        # Spatial parameters (use HHmL ranges)
        rescaled['kappa'] = 0.5 + 3.5 * params['kappa']  # [0.5, 4.0]
        rescaled['delta'] = 0.1 + 0.9 * params['delta']  # [0.1, 1.0]
        rescaled['lambda'] = 0.5 + 4.5 * params['lambda']  # [0.5, 5.0]
        # ... (add all 23 spatial parameters rescaling)

        # Temporal dynamics parameters (Phase 1)
        rescaled['temporal_twist'] = np.pi * params['temporal_twist']  # [0, π]
        rescaled['retrocausal_strength'] = params['retrocausal_strength']  # [0, 1]
        rescaled['temporal_relaxation'] = 0.1 + 0.9 * params['temporal_relaxation']  # [0.1, 1.0]
        rescaled['num_time_steps'] = 10 + int(190 * params['num_time_steps'])  # [10, 200]
        rescaled['prophetic_coupling'] = params['prophetic_coupling']  # [0, 1]
        rescaled['temporal_phase_shift'] = 2*np.pi * params['temporal_phase_shift']  # [0, 2π]
        rescaled['temporal_decay'] = params['temporal_decay']  # [0, 1]
        rescaled['forward_backward_balance'] = params['forward_backward_balance']  # [0, 1]
        rescaled['temporal_noise_level'] = 0.01 * params['temporal_noise_level']  # [0, 0.01]

        # Temporal vortex parameters (Phase 2 - NEW)
        rescaled['temporal_vortex_injection_rate'] = params['temporal_vortex_injection_rate']  # [0, 1] probability
        rescaled['temporal_vortex_winding'] = 1 + int(4 * params['temporal_vortex_winding'])  # [1, 5] integer winding
        rescaled['temporal_vortex_core_size'] = 0.01 + 0.49 * params['temporal_vortex_core_size']  # [0.01, 0.5]
        rescaled['vortex_tube_probability'] = params['vortex_tube_probability']  # [0, 1] probability
        rescaled['tube_winding_number'] = 1 + int(4 * params['tube_winding_number'])  # [1, 5] integer winding
        rescaled['tube_core_size'] = 0.01 + 0.49 * params['tube_core_size']  # [0.01, 0.5]
        rescaled['temporal_vortex_annihilation_rate'] = params['temporal_vortex_annihilation_rate']  # [0, 1] probability

        return rescaled
