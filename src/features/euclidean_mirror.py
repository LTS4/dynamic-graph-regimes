"""Iso-mirror embedding from Chen et al. (2024).

Implementation of the method from:
    Chen, T., Lubberts, Z., Athreya, A., Park, Y., & Priebe, C. E. (2024).
    "Euclidean mirrors and first-order changepoints in network time series."
    arXiv:2405.11111

The algorithm detects first-order changepoints in network time series by:
1. Computing Adjacency Spectral Embeddings (ASE) for each graph
2. Computing pairwise distances between embeddings (with optional Procrustes alignment)
3. Applying Classical Multidimensional Scaling (CMDS)
4. Applying ISOMAP to obtain a 1D "iso-mirror" representation

The original work also localizes changepoints via L-infinity piecewise linear regression, while we
delegate that to Ruptures.

Reference implementation:
https://github.com/TianyiChen97/Euclidean-mirrors-and-first-order-changepoints-in-network-time-series
"""

import warnings
from pathlib import Path

import numpy as np
import sparse
from numpy.linalg import eigh, norm, svd
from numpy.typing import NDArray
from sklearn.manifold import Isomap

################################################################################
# Helper functions


def adjacency_spectral_embedding(
    adjs: NDArray[np.float64],
    d: int,
    diagaug: bool = False,
    eig_filter: str = "abs",
) -> NDArray[np.float64]:
    """Compute Adjacency Spectral Embedding (ASE) for a single graph.

    Args:
        adj (sparse.COO): Array of adjacency matrices of shape (..., n_nodes, n_nodes)
        d (int): Embedding dimension
        diagaug (bool, optional): Whether to use diagonal augmentation. Defaults to True.

    Returns:
        np.NDArray embedding of shape (..., n_nodes, d)
    """

    if diagaug:
        raise NotImplementedError
        # n = adjs.shape[-1]
        # np.fill_diagonal(adjs, adjs.sum(axis=1) / (n - 1))

    # For symmetric matrices, use eigendecomposition
    if np.allclose(adjs, adjs.swapaxes(-2, -1)):
        evals, evecs = eigh(adjs)
        evals = evals[..., : -d - 1 : -1]
        evecs = evecs[..., : -d - 1 : -1]
        match eig_filter:
            case "abs":
                evals = np.abs(evals)
            case "relu":
                evals[evals < 0] = 0
            case _:
                raise ValueError(f"Unknown eig_filter {eig_filter}")
        evecs *= np.expand_dims(np.sqrt(evals), -2)
        return evecs
    else:
        raise NotImplementedError
        # # For asymmetric matrices, use SVD
        # U, s, Vt = svd(adjs, full_matrices=False)
        # U = U[..., :d]
        # s = s[..., :d]
        # V = Vt[..., :d, :].T

        # sqrt_s = np.sqrt(s)
        # Xhat = U * sqrt_s
        # Xhat_R = V * sqrt_s
        # return np.stack([Xhat, Xhat_R], axis=-1)


def procrustes_align(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute Procrustes transformation to align X to Y.

    Finds orthogonal matrix W that minimizes ||X @ W - Y||_F

    Args:
        X (NDArray): Source matrix of shape (n, d)
        Y (NDArray): Target matrix of shape (n, d)

    Returns:
        NDArray: Orthogonal transformation matrix W of shape (d, d)
    """
    M = x.T @ y
    U, _, Vt = svd(M)
    W = U @ Vt
    return W


def compute_embedding_distance_matrix(
    X: NDArray[np.float64],
    procrustes: bool = True,
) -> NDArray[np.float64]:
    """Compute pairwise distance matrix between graph embeddings.

    Args:
        embeddings (NDArray): Embedding matrices of shape (..., n_samples, n_nodes, d)
        procrustes (bool, optional): Whether to use Procrustes alignment. Defaults to True.

    Returns:
        NDArray: Distance matrices of shape (..., n_samples, n_samples)
    """
    # Extract dimensions

    if not procrustes:
        # This correspond to the sqrt of the mean squared distances between node embeddings
        out = norm(
            np.expand_dims(X, -4) - np.expand_dims(X, -3),
            ord="fro",
            axis=(-2, -1),
        ) / np.sqrt(X.shape[-2])
    else:
        raise NotImplementedError
        # # With Procrustes, need to compute alignment for each pair in each batch

        # for i in range(n_samples):
        #     for j in range(i + 1, n_samples):
        #         W = procrustes_align(X[b, i], X[b, j])
        #         Xi_aligned = X[b, i] @ W

        #         # Normalized Frobenius distance
        #         dist = norm(Xi_aligned - X[b, j], ord="fro") ** 2 / n_nodes
        #         D[b, i, j] = np.sqrt(dist)
        #         D[b, j, i] = D[b, i, j]
    assert out.shape == (X.shape[-3], X.shape[-3])
    return out


def double_center(x: NDArray[np.float64]) -> NDArray:
    m = x.shape[-1]
    centering = np.eye(m) - np.ones((m, m)) / m
    return -0.5 * centering @ x @ centering


################################################################################
# Euclidean iso-mirror


def isomirror(
    adjs: sparse.COO,
    embedding_dim: int = -1,
    mds_dim: int = -1,
    use_procrustes: bool = False,
    diagaug: bool = False,
) -> NDArray:
    """Iso-Mirror representation of dynamic graphs from Chen et al. (2024).

    Args:
        adjs (sparse.COO): Time series of adjacency matrices of shape (n_graphs, n_nodes, n_nodes)
        embedding_dim (int, optional): Dimension for Adjacency Spectral Embedding.
            Defaults to -1, which uses `n_nodes`.
        mds_dim (int, optional): Number of CMDS dimensions to use for ISOMAP.
            Defaults to -1, which uses `n_graphs`.
        use_procrustes (bool, optional): Whether to use Procrustes alignment when computing
            distances. Defaults to True.
        diagaug (bool, optional): Whether to use diagonal augmentation in ASE. Defaults to True.

    References:
        Chen, T., Lubberts, Z., Athreya, A., Park, Y., & Priebe, C. E. (2024).
        "Euclidean mirrors and first-order changepoints in network time series."
        arXiv:2405.11111
    """
    n_samples, n_nodes, *_ = adjs.shape
    if embedding_dim < 1:
        embedding_dim = n_nodes
    if mds_dim < 1:
        mds_dim = n_samples

    # Step 1: Compute ASE for each graph
    # Step 2: Compute distance matrix
    distance_matrix = compute_embedding_distance_matrix(
        adjacency_spectral_embedding(adjs.todense(), embedding_dim, diagaug),
        procrustes=use_procrustes,
    )

    # Step 3: Apply CMDS
    mds_coords = adjacency_spectral_embedding(
        double_center(distance_matrix**2),
        mds_dim,
        diagaug=False,
        eig_filter="relu",
    )

    # Step 4: Apply ISOMAP
    for k in range(2, n_samples):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            iso_mirror = Isomap(n_neighbors=k, n_components=1).fit_transform(mds_coords)
            if not w:
                break
    else:
        raise ValueError("Isomap could not find a connected NN-graph")

    # Ensure iso-mirror starts negative (convention from paper)
    if iso_mirror[0] > 0:
        iso_mirror *= -1

    return iso_mirror


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from src.experiments.io import load_file

    snapshots = load_file(
        Path("data/synthetic/72c0b2cb/00/graph_bw_threshold-edge-uniform_full.npz")
    )
    y = isomirror(
        snapshots,
        embedding_dim=-1,
        mds_dim=-1,
        use_procrustes=False,
        diagaug=False,
    )
    print(y.shape)

    plt.plot(y)
    plt.show()
