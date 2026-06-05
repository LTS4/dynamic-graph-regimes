"""Graph invariant features"""

import numpy as np
import pandas as pd
import sparse
from numpy.typing import NDArray
from scipy.signal import convolve

from src.utils import square_to_vec


def vec_weights(adjs: sparse.COO):
    return sparse.as_coo(square_to_vec(adjs.todense()))


def graph_invariants(adjs: sparse.COO | list[sparse.COO]) -> NDArray[np.float64]:
    """Compute size of the graph,  number of triangles,  scan,  max degree, avg degree.

    Idea from: M. Tang, Y. Park, N. H. Lee, and C. E. Priebe, “Attribute Fusion in a Latent Process
    Model for Time Series of Graphs,” IEEE Transactions on Signal Processing, vol. 61, no. 7, pp.
    1721-1732, Apr. 2013, doi: 10.1109/TSP.2013.2243445.

    Args:
        adjs (NDArray[np.float64]): Sequence of adjacency matrices
            of shape (n_samples, ..., n_nodes, n_nodes).

    Returns:
        NDArray[np.float64]: Design matrix of shape (n_samples, n_feats)
    """
    if isinstance(adjs, list):
        return np.concatenate([graph_invariants(adj) for adj in adjs], axis=-1)

    out = np.empty(adjs.shape[:-2] + (5,))

    # size
    out[..., 0] = adjs.sum((-2, -1)).todense()

    # NOTE: sparse have trouble handling negative axes
    n_axes = len(adjs.shape)
    tri_diag = sparse.diagonal(adjs @ adjs @ adjs, axis1=n_axes - 2, axis2=n_axes - 1)
    # triangles
    out[..., 1] = tri_diag.sum(axis=-1).todense() / 6

    # Scan sums the edges between neighbors of a node, then take the max
    out[..., 2] = tri_diag.max(-1).todense()

    degrees = adjs.sum(axis=-1)
    # max_degree
    out[..., 3] = degrees.max(axis=-1).todense()
    # avg_edge_prob
    out[..., 4] = degrees.mean(axis=-1).todense()

    return out.reshape(adjs.shape[0], -1)


def normalized_fusion(x: NDArray[np.float64], window_len: int) -> NDArray[np.float64]:
    """Compute normalized fusion statistic of x.


    Formula from [1]: M. Tang, Y. Park, N. H. Lee, and C. E. Priebe, “Attribute Fusion in a Latent
    Process Model for Time Series of Graphs,” IEEE Transactions on Signal Processing, vol. 61, no.
    7, pp.  1721-1732, Apr. 2013, doi: 10.1109/TSP.2013.2243445.

    Args:
        x (NDArray[np.float64]): Input data of shape (n_samples, n_features).
        l (int): Window length.

    Returns:
        NDArray[np.float64]: Array of shape (n_samples, n_features).
    """
    # Compute moving average of width l on first axis of x
    ma = convolve(x, np.ones((window_len, 1)) / window_len, mode="full")

    fluctuation = np.sqrt(
        np.sum(
            (
                convolve(
                    x[:, np.newaxis, :],
                    np.eye(window_len)[::-1, :, np.newaxis],
                    "full",
                )  # type: ignore
                - ma[:, np.newaxis, :]
            )
            ** 2,
            1,
        )
        / (window_len - 1)
    )

    # We shift MA and fluctuations by 1 as per definition from [1]
    return (x - ma[1 : x.shape[0] + 1]) / fluctuation[1 : x.shape[0] + 1]


def cusum(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute CUSUM of x.

    Args:
        x (NDArray[np.float64]): Input data of shape (n_samples, n_features).
        l (int): Window length.
    """
    cumulative = pd.DataFrame(np.cumsum(x, axis=0))
    return (cumulative - cumulative.expanding(1).min()).values
