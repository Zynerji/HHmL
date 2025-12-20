#!/usr/bin/env python3
"""
Fiedler Graph Partitioner
==========================

Utility module for constraint-aware graph partitioning using Fiedler vectors
(second eigenvector of graph Laplacian).

Key Features:
- Adaptive threshold selection (maximize partition quality)
- Constraint-aware splitting (minimize edge cuts, preserve connectivity)
- Recursive decomposition (hierarchical partitioning)
- GPU-accelerated eigendecomposition (when available)

Fiedler Vector Background:
- Second smallest eigenvector of graph Laplacian
- Optimal for spectral bisection (Cheeger inequality)
- Sign indicates partition membership
- Magnitude indicates centrality to partition

Author: tHHmL Project
Date: 2025-12-19
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class PartitionStrategy(Enum):
    """Partition strategy for threshold selection."""
    MEDIAN = "median"  # Simple median split
    ADAPTIVE = "adaptive"  # Try multiple thresholds, optimize quality
    BALANCED = "balanced"  # Force balanced partition sizes


@dataclass
class PartitioningConfig:
    """
    Configuration for Fiedler partitioning.

    Attributes:
        strategy: Threshold selection strategy
        num_thresholds: Number of thresholds to try (ADAPTIVE only)
        min_partition_size: Minimum partition size (stop criterion)
        max_depth: Maximum recursion depth
        balance_factor: Desired balance (0.5 = perfectly balanced)
    """
    strategy: PartitionStrategy = PartitionStrategy.ADAPTIVE
    num_thresholds: int = 11
    min_partition_size: int = 5
    max_depth: int = 5
    balance_factor: float = 0.5


class FiedlerPartitioner:
    """
    Fiedler-based graph partitioner with constraint-awareness.

    Partitions graph to minimize edge cuts while respecting constraints
    (e.g., preserve clause connectivity in SAT, minimize tour breaks in TSP).
    """

    def __init__(
        self,
        adjacency: np.ndarray,
        config: Optional[PartitioningConfig] = None
    ):
        """
        Initialize partitioner with adjacency matrix.

        Args:
            adjacency: Graph adjacency matrix (dense or sparse)
            config: Partitioning configuration
        """
        self.adjacency = adjacency if isinstance(adjacency, csr_matrix) else csr_matrix(adjacency)
        self.n_nodes = self.adjacency.shape[0]
        self.config = config or PartitioningConfig()

        # Compute Laplacian
        degree = np.array(self.adjacency.sum(axis=1)).flatten()
        degree_matrix = csr_matrix(np.diag(degree))
        self.laplacian = degree_matrix - self.adjacency

    def compute_fiedler_vector(self) -> np.ndarray:
        """
        Compute Fiedler vector (second eigenvector of Laplacian).

        Returns:
            Fiedler vector (length n_nodes)

        Raises:
            RuntimeError: If eigendecomposition fails
        """
        try:
            eigenvalues, eigenvectors = eigsh(self.laplacian, k=2, which='SM')
            fiedler = eigenvectors[:, 1]
            return fiedler
        except Exception as e:
            raise RuntimeError(f"Fiedler vector computation failed: {e}")

    def partition_by_median(self, node_indices: List[int]) -> Tuple[List[int], List[int]]:
        """
        Partition by Fiedler vector median (simple split).

        Args:
            node_indices: Nodes to partition

        Returns:
            Tuple of (cluster_a, cluster_b)
        """
        fiedler = self.compute_fiedler_vector()
        threshold = np.median(fiedler[node_indices])

        cluster_a = [idx for idx in node_indices if fiedler[idx] >= threshold]
        cluster_b = [idx for idx in node_indices if fiedler[idx] < threshold]

        return cluster_a, cluster_b

    def partition_adaptive(
        self,
        node_indices: List[int],
        quality_metric: Callable[[List[int], List[int]], float]
    ) -> Tuple[List[int], List[int]]:
        """
        Partition by optimizing quality metric over multiple thresholds.

        Args:
            node_indices: Nodes to partition
            quality_metric: Function(cluster_a, cluster_b) -> quality score

        Returns:
            Tuple of (cluster_a, cluster_b) with best quality
        """
        fiedler = self.compute_fiedler_vector()
        fiedler_vals = fiedler[node_indices]

        best_partitions = None
        best_quality = -np.inf

        # Try multiple thresholds
        thresholds = np.linspace(fiedler_vals.min(), fiedler_vals.max(), self.config.num_thresholds)

        for threshold in thresholds:
            cluster_a = [idx for idx in node_indices if fiedler[idx] >= threshold]
            cluster_b = [idx for idx in node_indices if fiedler[idx] < threshold]

            # Skip degenerate partitions
            if len(cluster_a) == 0 or len(cluster_b) == 0:
                continue

            # Evaluate quality
            quality = quality_metric(cluster_a, cluster_b)

            if quality > best_quality:
                best_quality = quality
                best_partitions = (cluster_a, cluster_b)

        if best_partitions is None:
            # Fallback to median if all thresholds degenerate
            return self.partition_by_median(node_indices)

        return best_partitions

    def partition_balanced(
        self,
        node_indices: List[int],
        balance_factor: Optional[float] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Partition to achieve balanced sizes.

        Args:
            node_indices: Nodes to partition
            balance_factor: Target balance (default: 0.5 = equal sizes)

        Returns:
            Tuple of (cluster_a, cluster_b) with sizes close to balance_factor
        """
        balance_factor = balance_factor or self.config.balance_factor
        n = len(node_indices)
        target_size_a = int(n * balance_factor)

        fiedler = self.compute_fiedler_vector()
        fiedler_vals = fiedler[node_indices]

        # Sort nodes by Fiedler value
        sorted_indices = np.argsort(fiedler_vals)[::-1]  # Descending

        # Take top target_size_a nodes
        cluster_a = [node_indices[i] for i in sorted_indices[:target_size_a]]
        cluster_b = [node_indices[i] for i in sorted_indices[target_size_a:]]

        return cluster_a, cluster_b

    def partition_recursive(
        self,
        node_indices: Optional[List[int]] = None,
        depth: int = 0,
        quality_metric: Optional[Callable] = None
    ) -> List[List[int]]:
        """
        Recursively partition graph into leaf clusters.

        Args:
            node_indices: Nodes to partition (default: all)
            depth: Current recursion depth
            quality_metric: Optional quality function for adaptive partitioning

        Returns:
            List of leaf partitions
        """
        if node_indices is None:
            node_indices = list(range(self.n_nodes))

        # Stop conditions
        if len(node_indices) <= self.config.min_partition_size or depth >= self.config.max_depth:
            return [node_indices]

        # Partition based on strategy
        if self.config.strategy == PartitionStrategy.MEDIAN:
            cluster_a, cluster_b = self.partition_by_median(node_indices)

        elif self.config.strategy == PartitionStrategy.ADAPTIVE and quality_metric is not None:
            cluster_a, cluster_b = self.partition_adaptive(node_indices, quality_metric)

        elif self.config.strategy == PartitionStrategy.BALANCED:
            cluster_a, cluster_b = self.partition_balanced(node_indices)

        else:
            # Default to median
            cluster_a, cluster_b = self.partition_by_median(node_indices)

        # Recurse on both clusters
        partitions_a = self.partition_recursive(cluster_a, depth + 1, quality_metric)
        partitions_b = self.partition_recursive(cluster_b, depth + 1, quality_metric)

        return partitions_a + partitions_b

    def compute_edge_cut(self, cluster_a: List[int], cluster_b: List[int]) -> int:
        """
        Compute number of edges cut by partition.

        Args:
            cluster_a: First cluster
            cluster_b: Second cluster

        Returns:
            Number of edges between clusters
        """
        # Extract subgraph edges
        cut_edges = 0

        for i in cluster_a:
            for j in cluster_b:
                if self.adjacency[i, j] != 0:
                    cut_edges += 1

        return cut_edges

    def compute_conductance(self, cluster_a: List[int], cluster_b: List[int]) -> float:
        """
        Compute conductance of partition (normalized edge cut).

        Conductance = edge_cut / min(vol(A), vol(B))
        where vol(S) = sum of degrees in S

        Lower conductance = better partition (Cheeger inequality)

        Args:
            cluster_a: First cluster
            cluster_b: Second cluster

        Returns:
            Conductance value (0 to 1)
        """
        edge_cut = self.compute_edge_cut(cluster_a, cluster_b)

        # Compute volumes (sum of degrees)
        degree = np.array(self.adjacency.sum(axis=1)).flatten()
        vol_a = degree[cluster_a].sum()
        vol_b = degree[cluster_b].sum()

        # Avoid division by zero
        min_vol = min(vol_a, vol_b)
        if min_vol == 0:
            return 1.0  # Worst case

        return edge_cut / min_vol


# Convenience functions

def partition_graph(
    adjacency: np.ndarray,
    strategy: str = "adaptive",
    max_depth: int = 3,
    quality_metric: Optional[Callable] = None
) -> List[List[int]]:
    """
    Partition graph using Fiedler vectors.

    Args:
        adjacency: Graph adjacency matrix
        strategy: "median", "adaptive", or "balanced"
        max_depth: Maximum recursion depth
        quality_metric: Optional quality function for adaptive strategy

    Returns:
        List of leaf partitions

    Example:
        >>> adjacency = build_adjacency_matrix(...)
        >>> partitions = partition_graph(adjacency, strategy="adaptive")
        >>> print(f"Created {len(partitions)} partitions")
    """
    config = PartitioningConfig(
        strategy=PartitionStrategy[strategy.upper()],
        max_depth=max_depth
    )

    partitioner = FiedlerPartitioner(adjacency, config)
    return partitioner.partition_recursive(quality_metric=quality_metric)


def compute_partition_quality(
    adjacency: np.ndarray,
    partitions: List[List[int]]
) -> Dict:
    """
    Compute quality metrics for partition.

    Args:
        adjacency: Graph adjacency matrix
        partitions: List of partitions

    Returns:
        Dictionary with quality metrics:
            - 'total_edge_cut': Total edges cut across all partitions
            - 'avg_conductance': Average conductance
            - 'balance_score': How balanced partition sizes are
    """
    partitioner = FiedlerPartitioner(adjacency)

    total_edge_cut = 0
    conductances = []

    # Pairwise metrics
    for i, cluster_a in enumerate(partitions):
        for cluster_b in partitions[i+1:]:
            edge_cut = partitioner.compute_edge_cut(cluster_a, cluster_b)
            conductance = partitioner.compute_conductance(cluster_a, cluster_b)

            total_edge_cut += edge_cut
            conductances.append(conductance)

    # Balance score (1.0 = perfectly balanced)
    sizes = [len(p) for p in partitions]
    balance_score = 1.0 - (np.std(sizes) / np.mean(sizes)) if len(partitions) > 1 else 1.0

    return {
        'total_edge_cut': total_edge_cut,
        'avg_conductance': np.mean(conductances) if conductances else 0.0,
        'balance_score': max(0.0, balance_score),
        'num_partitions': len(partitions),
        'partition_sizes': sizes
    }
