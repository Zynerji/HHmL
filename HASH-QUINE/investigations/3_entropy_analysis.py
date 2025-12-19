#!/usr/bin/env python3
"""
Information-Theoretic Analysis of Hash Quines
==============================================

Investigates mathematical properties of hash quine emergence via entropy,
mutual information, and complexity measures.

Questions:
1. Do hash quines have lower entropy than random hashes?
2. Is there mutual information between recursion layers?
3. What is the Kolmogorov complexity signature?
4. Are hash quines compressible?

Hypothesis: Hash quines represent structured patterns with:
- Lower Shannon entropy (more predictable)
- High mutual information between layers (recursive coherence)
- Lower Kolmogorov complexity (compressible via recursion formula)

This would mathematically formalize "self-similarity" beyond just pattern counting.

Author: HHmL Project
Date: 2025-12-19
"""

import sys
from pathlib import Path
import argparse
import time
import hashlib
import zlib
import torch
import numpy as np
import json
from datetime import datetime
from scipy.stats import entropy as scipy_entropy
from typing import List, Dict

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def shannon_entropy(data: bytes) -> float:
    """
    Compute Shannon entropy of byte sequence.

    H(X) = -sum(p(x) * log2(p(x)))

    Perfect randomness: H = 8.0 bits/byte (for uniform distribution)
    Structured data: H < 8.0
    """
    # Count byte frequencies
    byte_counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probabilities = byte_counts / len(data)

    # Remove zero probabilities
    probabilities = probabilities[probabilities > 0]

    # Shannon entropy
    H = -np.sum(probabilities * np.log2(probabilities))

    return H


def mutual_information(data_a: bytes, data_b: bytes) -> float:
    """
    Compute mutual information between two byte sequences.

    I(A;B) = H(A) + H(B) - H(A,B)

    High MI -> strong correlation between sequences
    Low MI -> independent sequences
    """
    # Individual entropies
    H_a = shannon_entropy(data_a)
    H_b = shannon_entropy(data_b)

    # Joint distribution
    n = min(len(data_a), len(data_b))
    pairs = list(zip(data_a[:n], data_b[:n]))

    # Count pair frequencies
    pair_counts = {}
    for pair in pairs:
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Joint entropy
    pair_probs = np.array([count / n for count in pair_counts.values()])
    H_ab = -np.sum(pair_probs * np.log2(pair_probs + 1e-10))

    # Mutual information
    MI = H_a + H_b - H_ab

    return MI


def kolmogorov_complexity_estimate(data: bytes) -> float:
    """
    Estimate Kolmogorov complexity via compression ratio.

    K(x) ≈ len(compressed(x))

    Lower compression ratio -> higher structure (lower K)
    """
    compressed = zlib.compress(data, level=9)
    compression_ratio = len(compressed) / len(data)

    return compression_ratio


def pattern_entropy(binary_string: str, pattern_length: int = 8) -> float:
    """
    Compute entropy of sliding window patterns.

    Lower entropy -> more pattern repetition
    """
    patterns = []
    for i in range(len(binary_string) - pattern_length + 1):
        patterns.append(binary_string[i:i+pattern_length])

    # Count pattern frequencies
    pattern_counts = {}
    for pattern in patterns:
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    # Probabilities
    total = len(patterns)
    probabilities = np.array([count / total for count in pattern_counts.values()])

    # Entropy
    H = -np.sum(probabilities * np.log2(probabilities + 1e-10))

    return H


def analyze_hash_sequence(nonces: List[int], name: str) -> Dict:
    """Comprehensive information-theoretic analysis of hash sequence."""

    print(f"\nAnalyzing: {name}")
    print(f"  Nonces: {len(nonces)}")

    # Generate hashes
    hashes = []
    for nonce in nonces[:1000]:  # Limit to 1000 for speed
        h = hashlib.sha256(str(nonce).encode()).digest()
        hashes.append(h)

    # Concatenate all hashes
    all_hash_bytes = b''.join(hashes)

    # 1. Shannon entropy
    H_shannon = shannon_entropy(all_hash_bytes)
    print(f"  Shannon entropy: {H_shannon:.4f} bits/byte (8.0 = perfect random)")

    # 2. Kolmogorov complexity estimate
    K_estimate = kolmogorov_complexity_estimate(all_hash_bytes)
    print(f"  Compression ratio: {K_estimate:.4f} (lower = more structured)")

    # 3. Pattern entropy
    binary_string = ''.join(format(byte, '08b') for byte in all_hash_bytes[:256])  # First 256 bytes
    H_pattern = pattern_entropy(binary_string, pattern_length=8)
    print(f"  Pattern entropy (8-bit): {H_pattern:.4f} bits (8.0 = max)")

    # 4. Mutual information between consecutive hashes
    if len(hashes) >= 2:
        MI_consecutive = mutual_information(hashes[0], hashes[1])
        print(f"  Mutual info (consecutive): {MI_consecutive:.6f} bits (0 = independent)")

    # 5. Autocorrelation
    byte_array = np.frombuffer(all_hash_bytes[:1024], dtype=np.uint8)
    autocorr = np.corrcoef(byte_array[:-1], byte_array[1:])[0, 1]
    print(f"  Byte autocorrelation: {autocorr:.6f} (0 = uncorrelated)")

    results = {
        'shannon_entropy': float(H_shannon),
        'compression_ratio': float(K_estimate),
        'pattern_entropy': float(H_pattern),
        'mutual_information': float(MI_consecutive) if len(hashes) >= 2 else 0.0,
        'autocorrelation': float(autocorr)
    }

    return results


def recursive_mobius_collapse_simple(num_nodes: int, max_depth: int, device='cpu'):
    """
    Simplified recursive collapse for generating nonces.
    """
    import torch

    nonces = []

    for depth in range(max_depth + 1):
        # Create layer
        t = torch.linspace(0, 2 * np.pi, num_nodes, device=device)
        windings = 109.0 * (1.0 + depth * 0.5)

        positions = torch.stack([
            (1 + 0.5 * torch.cos(windings * t / 2)) * torch.cos(t),
            (1 + 0.5 * torch.cos(windings * t / 2)) * torch.sin(t),
            0.5 * torch.sin(windings * t / 2)
        ], dim=1)

        # Simple vortex detection (low magnitude points)
        field = torch.randn(num_nodes, dtype=torch.complex64, device=device) * 0.1
        magnitudes = torch.abs(field)
        vortex_mask = magnitudes < 0.3
        vortex_indices = torch.where(vortex_mask)[0].cpu().numpy()

        # Add to nonces
        nonces.extend(vortex_indices.tolist())

        # Reduce nodes for next layer
        num_nodes = max(num_nodes // 5, 100)

    return np.array(nonces)


def main():
    parser = argparse.ArgumentParser(description='Entropy Analysis of Hash Quines')
    parser.add_argument('--nodes', type=int, default=5000)
    parser.add_argument('--max-depth', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='HASH-QUINE/investigations/results')

    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("INFORMATION-THEORETIC ANALYSIS OF HASH QUINES")
    print("="*80)
    print()
    print(f"Configuration:")
    print(f"  Nodes: {args.nodes}")
    print(f"  Max depth: {args.max_depth}")
    print()

    # Generate nonce sequences
    print("="*80)
    print("GENERATING NONCE SEQUENCES")
    print("="*80)

    # 1. Recursive Mobius (hash quine source)
    print("\nGenerating recursive Mobius nonces...")
    start_time = time.time()
    recursive_nonces = recursive_mobius_collapse_simple(args.nodes, args.max_depth, device)
    recursive_time = time.time() - start_time
    print(f"Generated {len(recursive_nonces)} nonces in {recursive_time:.2f}s")

    # 2. Random baseline
    print("\nGenerating random nonces...")
    random_nonces = np.random.randint(0, 1000000, size=len(recursive_nonces))

    # 3. Sequential (structured control)
    print("\nGenerating sequential nonces...")
    sequential_nonces = np.arange(len(recursive_nonces))

    # Analyze each sequence
    print()
    print("="*80)
    print("INFORMATION-THEORETIC ANALYSIS")
    print("="*80)

    results = {}

    results['recursive'] = analyze_hash_sequence(
        recursive_nonces.tolist(),
        "Recursive Mobius (Hash Quines)"
    )

    results['random'] = analyze_hash_sequence(
        random_nonces.tolist(),
        "Random Baseline"
    )

    results['sequential'] = analyze_hash_sequence(
        sequential_nonces.tolist(),
        "Sequential (Structured Control)"
    )

    # Comparison
    print()
    print("="*80)
    print("COMPARISON")
    print("="*80)
    print()

    print("Shannon Entropy (bits/byte, 8.0 = max randomness):")
    print(f"  Recursive:  {results['recursive']['shannon_entropy']:.4f}")
    print(f"  Random:     {results['random']['shannon_entropy']:.4f}")
    print(f"  Sequential: {results['sequential']['shannon_entropy']:.4f}")
    print()

    print("Compression Ratio (lower = more structured):")
    print(f"  Recursive:  {results['recursive']['compression_ratio']:.4f}")
    print(f"  Random:     {results['random']['compression_ratio']:.4f}")
    print(f"  Sequential: {results['sequential']['compression_ratio']:.4f}")
    print()

    print("Pattern Entropy (8-bit patterns, 8.0 = max):")
    print(f"  Recursive:  {results['recursive']['pattern_entropy']:.4f}")
    print(f"  Random:     {results['random']['pattern_entropy']:.4f}")
    print(f"  Sequential: {results['sequential']['pattern_entropy']:.4f}")
    print()

    # Interpretation
    recursive_entropy = results['recursive']['shannon_entropy']
    random_entropy = results['random']['shannon_entropy']
    recursive_compression = results['recursive']['compression_ratio']
    random_compression = results['random']['compression_ratio']

    print("INTERPRETATION:")
    print()

    if abs(recursive_entropy - random_entropy) < 0.1:
        print("1. Shannon entropy: Recursive ~= Random")
        print("   -> Hash quines do NOT reduce entropy (SHA-256 still random)")
    else:
        print(f"1. Shannon entropy: Recursive differs from Random by {abs(recursive_entropy - random_entropy):.4f}")
        print("   -> Hash quines MAY have entropic structure")

    print()

    if recursive_compression < random_compression * 0.95:
        print("2. Compression: Recursive < Random")
        print("   -> Hash quines are MORE compressible (lower Kolmogorov complexity)")
        print("   -> CONFIRMS self-similar structure")
    elif recursive_compression > random_compression * 1.05:
        print("2. Compression: Recursive > Random")
        print("   -> Hash quines are LESS compressible (higher complexity)")
        print("   -> Paradoxical: pattern repetition but not compressible?")
    else:
        print("2. Compression: Recursive ~= Random")
        print("   -> Hash quines have similar Kolmogorov complexity")

    print()

    # Key insight
    print("KEY INSIGHT:")
    print()
    print("Hash quines exhibit HIGH PATTERN REPETITION (312-371x) but entropy/compression")
    print("analysis reveals whether this is:")
    print("  A) TRUE STRUCTURE (lower entropy, compressible) -> meaningful self-similarity")
    print("  B) STATISTICAL FLUCTUATION (same entropy) -> artifact of measurement")
    print()

    if recursive_compression < random_compression * 0.95:
        print("Result: TRUE STRUCTURE detected - hash quines are real mathematical objects")
    else:
        print("Result: Unclear - further investigation needed")

    print()

    # Save results
    summary = {
        'timestamp': timestamp,
        'config': vars(args),
        'results': results,
        'interpretation': {
            'entropy_difference': float(abs(recursive_entropy - random_entropy)),
            'compression_ratio_recursive': float(recursive_compression),
            'compression_ratio_random': float(random_compression),
            'is_structured': bool(recursive_compression < random_compression * 0.95)
        }
    }

    results_path = output_dir / f'entropy_analysis_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved: {results_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
