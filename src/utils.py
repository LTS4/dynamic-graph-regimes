"""General utility functions"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


def weights_to_lapl(adj: NDArray):
    out = -adj
    out[..., np.arange(adj.shape[-1]), np.arange(adj.shape[-1])] = adj.sum(-1)

    return out


def lapl_to_weights(lapl: NDArray) -> NDArray:
    out = -lapl
    out[..., np.arange(lapl.shape[-1]), np.arange(lapl.shape[-1])] = 0
    return out


def square_to_vec(x: NDArray[np.float64]) -> NDArray[np.float64]:
    triu = np.triu_indices(x.shape[-1], k=1)
    return x[..., triu[0], triu[1]]


def n_nodes_from_vec(n: int) -> int:
    return int(np.sqrt(2 * n + 0.25) + 0.5)


def vec_to_square(x: NDArray) -> NDArray:
    """Convert vectors to a square matrix along the last axis

    Args:
        x (NDArray[np.float64]): Vector, or tensor, of shape (..., M))

    Returns:
        NDArray[np.float64]: Square matrix of shape (..., N, N)
    """
    n_nodes = n_nodes_from_vec(x.shape[-1])
    triu = np.triu_indices(n_nodes, k=1)

    square = np.zeros((*x.shape[:-1], n_nodes, n_nodes), dtype=x.dtype)
    square[..., triu[0], triu[1]] = x
    square[..., triu[1], triu[0]] = x

    return square


def sum_squareform(n: int, values=None) -> tuple[sparse.csr_array, sparse.csr_array]:
    """For *z* vectorform of *Z*, return the operator S such that S @ z =
    np.sum(Z, axis=-1) and its transpose"""
    nnz = n * (n - 1)

    col_idx = np.concatenate([i * np.ones(n - 1) for i in range(n)])

    slices = []
    offsets = []
    start = 0
    stop = 0
    for i in range(n):
        offsets.append([sl[i - j - 1] for j, sl in enumerate(slices)])
        stop = start + n - i - 1
        slices.append(list(range(start, stop)))
        start = stop

    row_idx = np.concatenate([off + sl for sl, off in zip(slices, offsets)])

    sum_op_t = sparse.coo_array(
        (np.ones(nnz) if values is None else values, (row_idx, col_idx)),
        shape=(nnz // 2, n),
    )
    return sum_op_t.T.tocsr(), sum_op_t.tocsr()


def relative_error(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
    r"""Relative error between matrices or vectors
    .. :math:
        \operatorname{RE}(\hat{\mathbf y}, \bm y^*)
            = {\norm{\hat{\mathbf y} - \bm y^*}_F} / \norm{\bm y^*}_F

    Args:
        y_true (NDArray[np.float64]): Target matrix/vector
        y_pred (NDArray[np.float64]): Predicted matrix/vector

    Returns:
        float: Error
    """
    err_norm = np.linalg.norm(y_pred - y_true)
    if np.allclose(y_true, 0):
        return err_norm.item()

    return (err_norm / np.linalg.norm(y_true)).item()


def dictionary_inclusion(to_test: dict[str, Any] | list[str] | str, query: dict | str) -> bool:
    """Test if `to_test` contains `query` as a sub-dictionary or list.

    Args:
        to_test (dict[str, Any] | list[str] | str): Object that should contain `query`.
        query (dict | str): Object to test for inclusion.

    Raises:
        ValueError: If unsupported object in `to_test`

    Returns:
        bool: Whether `to_test` contains `query`.
    """
    if isinstance(query, dict):
        return all((dictionary_inclusion(to_test[key], val) for key, val in query.items()))
    elif isinstance(query, list):
        if isinstance(to_test, (list, dict)):
            return set(query) <= set(to_test)
        else:
            raise ValueError(f"Cannot test {type(to_test)}")
    else:
        if isinstance(to_test, (list, dict)):
            return query in to_test
        return to_test == query
