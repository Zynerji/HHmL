#!/usr/bin/env python3
"""
Transfer Learning: Bootstrap H200 Training from Successful Baseline
====================================================================
This script initializes the H200 scaled model (30 strips, 4096 hidden_dim)
using the proven quality-guided checkpoint (2 strips, 512 hidden_dim).

Transfer Strategy:
1. Load successful baseline checkpoint (100% density, 0 annihilations)
2. Expand RNN hidden dimensions: 512 → 4096 (8× scale)
3. Expand strip handling: 2 → 30 strips (15× scale)
4. Initialize new parameters with small random values
5. Resume training with quality-guided learning

Benefits:
- Faster convergence (starts with proven vortex generation knowledge)
- Lower risk of early collapse
- Scientific comparison: transfer learning vs. from-scratch

Author: HHmL Framework
Date: 2025-12-17
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
from train_multi_strip import MultiStripRNNAgent
import argparse


def expand_lstm_state(small_state_dict, small_hidden_dim, large_hidden_dim, num_layers=4):
    """
    Expand LSTM hidden dimensions by initializing new dimensions with small random values.

    Strategy:
    - Copy existing 512 dimensions
    - Initialize new (4096-512) dimensions with N(0, 0.01)
    - This preserves learned patterns while allowing exploration
    """
    large_state_dict = {}

    for key, value in small_state_dict.items():
        if 'lstm' not in key:
            # Non-LSTM parameters - copy directly or skip if incompatible
            large_state_dict[key] = value
            continue

        # LSTM parameters need expansion
        if 'weight_ih' in key or 'weight_hh' in key:
            # Weight matrices: expand hidden dimension
            if 'weight_ih' in key:
                # Input-to-hidden: [4*hidden, input]
                # Expand hidden dimension (rows)
                small_tensor = value  # Shape: [4*512, input]
                input_dim = small_tensor.shape[1]
                large_tensor = torch.randn(4 * large_hidden_dim, input_dim) * 0.01
                # Copy existing weights
                large_tensor[:4*small_hidden_dim, :] = small_tensor
            else:
                # Hidden-to-hidden: [4*hidden, hidden]
                # Expand both dimensions
                small_tensor = value  # Shape: [4*512, 512]
                large_tensor = torch.randn(4 * large_hidden_dim, large_hidden_dim) * 0.01
                # Copy existing weights (upper-left block)
                large_tensor[:4*small_hidden_dim, :small_hidden_dim] = small_tensor

            large_state_dict[key] = large_tensor

        elif 'bias' in key:
            # Bias vectors: expand hidden dimension
            small_tensor = value  # Shape: [4*512]
            large_tensor = torch.randn(4 * large_hidden_dim) * 0.01
            # Copy existing biases
            large_tensor[:4*small_hidden_dim] = small_tensor
            large_state_dict[key] = large_tensor
        else:
            # Unknown LSTM parameter - skip
            print(f"Warning: Skipping unknown LSTM parameter {key}")

    return large_state_dict


def transfer_checkpoint(baseline_checkpoint_path, output_path,
                        source_strips=2, target_strips=30,
                        source_hidden_dim=512, target_hidden_dim=4096,
                        nodes_per_strip=2000):
    """
    Transfer learning from baseline to scaled configuration.

    Args:
        baseline_checkpoint_path: Path to successful baseline checkpoint
        output_path: Path to save transferred checkpoint
        source_strips: Baseline strip count (2)
        target_strips: Scaled strip count (30)
        source_hidden_dim: Baseline hidden dim (512)
        target_hidden_dim: Scaled hidden dim (4096)
        nodes_per_strip: Nodes per strip (same for both)
    """
    print("="*80)
    print("TRANSFER LEARNING: BASELINE → H200 SCALED")
    print("="*80)
    print()
    print(f"Source Configuration:")
    print(f"  Strips: {source_strips}")
    print(f"  Hidden dim: {source_hidden_dim}")
    print(f"  Nodes per strip: {nodes_per_strip}")
    print()
    print(f"Target Configuration:")
    print(f"  Strips: {target_strips}")
    print(f"  Hidden dim: {target_hidden_dim}")
    print(f"  Nodes per strip: {nodes_per_strip}")
    print()

    # Load baseline checkpoint
    print(f"Loading baseline checkpoint: {baseline_checkpoint_path}")
    baseline = torch.load(baseline_checkpoint_path, map_location='cpu')

    baseline_state = baseline['model_state_dict']
    baseline_metrics = baseline.get('metrics', {})

    print(f"  Baseline cycle: {baseline.get('cycle', 'unknown')}")
    if baseline_metrics.get('vortex_densities'):
        print(f"  Baseline final density: {baseline_metrics['vortex_densities'][-1]:.1%}")
    if baseline_metrics.get('rewards'):
        print(f"  Baseline final reward: {baseline_metrics['rewards'][-1]:.1f}")
    print()

    # Create target model
    print("Creating scaled model architecture...")
    target_agent = MultiStripRNNAgent(
        num_strips=target_strips,
        nodes_per_strip=nodes_per_strip,
        hidden_dim=target_hidden_dim,
        device='cpu'
    )

    target_params = sum(p.numel() for p in target_agent.parameters())
    print(f"  Target parameters: {target_params:,}")
    print()

    # Transfer weights
    print("Transferring weights with expansion...")

    # Expand LSTM state
    expanded_state = expand_lstm_state(
        baseline_state,
        source_hidden_dim,
        target_hidden_dim,
        num_layers=4
    )

    # Load expanded state (partial - only LSTM layers)
    target_agent.load_state_dict(expanded_state, strict=False)
    print("  [OK] LSTM weights transferred and expanded")

    # Re-initialize output heads (strip-specific, cannot transfer)
    print("  [Note] Reinitializing output heads (strip count changed)")
    target_agent.action_head.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    target_agent.control_head.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    target_agent.value_head.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
    print()

    # Save transferred checkpoint
    print(f"Saving transferred checkpoint: {output_path}")
    torch.save({
        'model_state_dict': target_agent.state_dict(),
        'cycle': 0,  # Reset cycle counter
        'metrics': {'rewards': [], 'vortex_densities': [], 'quality_scores': []},
        'transfer_info': {
            'source_checkpoint': str(baseline_checkpoint_path),
            'source_strips': source_strips,
            'source_hidden_dim': source_hidden_dim,
            'target_strips': target_strips,
            'target_hidden_dim': target_hidden_dim,
            'baseline_final_density': baseline_metrics.get('vortex_densities', [0])[-1],
            'baseline_final_reward': baseline_metrics.get('rewards', [0])[-1]
        }
    }, output_path)

    print("  [OK] Checkpoint saved")
    print()
    print("="*80)
    print("TRANSFER COMPLETE")
    print("="*80)
    print()
    print("Next step: Run H200 training with --resume flag:")
    print(f"  python scripts/train_h200_scaled.py \\")
    print(f"    --resume {output_path} \\")
    print(f"    --strips {target_strips} \\")
    print(f"    --hidden-dim {target_hidden_dim} \\")
    print(f"    --cycles 1000 \\")
    print(f"    --device cuda")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer learning from baseline to H200 scaled')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to successful baseline checkpoint')
    parser.add_argument('--output', type=str,
                        default='checkpoints/h200_scaled/transferred_baseline.pt',
                        help='Output path for transferred checkpoint')
    parser.add_argument('--source-strips', type=int, default=2,
                        help='Baseline strip count')
    parser.add_argument('--target-strips', type=int, default=30,
                        help='Scaled strip count')
    parser.add_argument('--source-hidden-dim', type=int, default=512,
                        help='Baseline hidden dimensions')
    parser.add_argument('--target-hidden-dim', type=int, default=4096,
                        help='Scaled hidden dimensions')
    parser.add_argument('--nodes', type=int, default=2000,
                        help='Nodes per strip (same for both)')
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run transfer
    transfer_checkpoint(
        baseline_checkpoint_path=args.baseline,
        output_path=output_path,
        source_strips=args.source_strips,
        target_strips=args.target_strips,
        source_hidden_dim=args.source_hidden_dim,
        target_hidden_dim=args.target_hidden_dim,
        nodes_per_strip=args.nodes
    )
