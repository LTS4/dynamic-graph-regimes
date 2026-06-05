"""Interpolarion and barycenters of graphs"""

from logging import info
from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from numpy.linalg import eigh
from numpy.typing import NDArray
from scipy.linalg import pinvh, sqrtm

from src.linalg import inv_pos, pinv_spd, sqrtm_spd
from src.utils import relative_error

# INTERPOLATION ####################################################################################


def interpolate_bureswasserstein_inv(inv0: NDArray, inv1: NDArray, t: float | NDArray):
    """Interpolate between matrices 0 and 1 in the sense of Bures-Wasserstein mean"""
    if isinstance(t, np.ndarray):
        t = t[:, np.newaxis, np.newaxis]

    inn = np.real(sqrtm(inv0 @ inv1))
    return (1 - t) ** 2 * inv0 + t**2 * inv1 + (t - t**2) * (inn + inn.T)


def interpolate_bureswasserstein(lapl0: NDArray, lapl1: NDArray, t: float | NDArray, rcond=1e-8):
    """Interpolate between laplacians 0 and 1 in the sense of Bures-Wasserstein mean of Graphs

    Args:
        lapl0 (NDArray): Laplacian of graph 0
        lapl1 (NDArray): Laplacian of graph 1
        t (float | NDArray): Interpolation parameter, or array of parameters
        rcond (float, optional): Absolute cutoff for small singular values in pinv.
            Defaults to 1e-8.

    Returns:
        NDArray: Interpolated Laplacian (s)
    """
    # We add n_nodes as it passes through and we can easily remove it.
    n_nodes = lapl0.shape[-1]
    inv0, inv1 = pinvh(np.stack([lapl0, lapl1]) + 1 / n_nodes, atol=rcond)

    return (
        pinvh(interpolate_bureswasserstein_inv(inv0, inv1, t), atol=rcond) - 1 / n_nodes
    )  # type: ignore


# BARYCENTERS ######################################################################################


# NOTE: somehow this is still faster if run sequentially
def bureswasserstein_mean(
    laplacians: NDArray,
    weights: Optional[NDArray] = None,
    inv_laplacians: Optional[NDArray] = None,
    s_0: Optional[NDArray] = None,
    max_iter=100,
    tol=1e-8,
    rcond=1e-5,
    verbose=0,
    return_pinv=False,
) -> NDArray:
    """Implementation of Bures-Wasserstein Mean of graphs from Haasler and Frossard (2024)

    Args:
        laplacians (NDArray): Tensor of Laplacians of shape (n_graphs, n_nodes, n_nodes)
        weights (NDArray, optional): Weights for interpolation of shape (n_graphs,). Defaults to
            uniform average.
        s_0 (NDArray, optional): Initial SPD matrix. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        rcond (float, optional): Relative cutoff for small singular values in pinv.
            Defaults to 1e-3.

    Returns:
        NDArray: Laplacian of barycenter graph

    References:
        I. Haasler and P. Frossard, “Bures-Wasserstein Means of Graphs,” in AIStats, PMLR, Apr.
            2024, pp. 1873-1881. Available: https://proceedings.mlr.press/v238/haasler24a.html

    """

    # **Given** $L_1, \dots, L_m \in \R^{N \times N}$ SPD Laplacians with only one zero eigenvalue,
    # and initial SPD matrix $S \in \R^{N \times N}$.
    # 1. $\Sigma_j \gets ( L_j + \frac{1}{N} \bm 1_{N \times N} )^{-1}$ for $j = 1, \dots, m$;
    # 2. **while** Not converged **do**
    #   1. $$S \gets S^{-1/2} \left(
    #      \sum_{j=1}^m \lambda_j \left( S^{1/2} \Sigma_j S^{1/2} \right)^{1/2}
    #      \right)^2 S^{-1/2}$$
    # 3. **return** $L \gets S^{-1} - \frac{1}{N} \bm 1_{N \times N}$

    if (inv_laplacians is None) == (laplacians is None):
        raise ValueError("Either inv_laplacians or laplacians must be provided")

    if inv_laplacians is None:
        inv_laplacians = pinv_spd(laplacians + 1.0 / laplacians.shape[1])

    n_graphs, n_nodes, _ = inv_laplacians.shape

    if weights is None:
        weights = np.ones(n_graphs, dtype=float) / n_graphs

    if s_0 is None:
        out = np.eye(n_nodes)
    else:
        out = s_0

    for i in range(max_iter):
        sqrt = sqrtm_spd(out, rcond=rcond)

        out_new = np.tensordot(
            weights, sqrtm_spd(sqrt @ inv_laplacians @ sqrt, rcond=rcond), axes=(0, 0)
        )

        evals, evecs = eigh(out)
        sqrtinv = evecs @ (
            np.sqrt(inv_pos(evals, rcond=rcond))[..., np.newaxis] * evecs.swapaxes(-1, -2)
        )

        out_new = sqrtinv @ out_new @ out_new @ sqrtinv

        if relative_error(out, out_new) < tol:
            if verbose > 0:
                info("BW mean of graphs conveged after %d iterations", i)
            break

        out = out_new

    if return_pinv:
        return out

    return pinv_spd(out, rcond=rcond) - 1.0 / n_nodes


def bureswasserstein_mean_vec(
    laplacians: NDArray,
    weights: Optional[NDArray] = None,
    inv_laplacians: Optional[NDArray] = None,
    s_0: Optional[NDArray] = None,
    max_iter=100,
    tol=1e-8,
    rcond=1e-5,
    return_pinv=False,
    verbose=0,
) -> NDArray:
    """Implementation of Bures-Wasserstein Mean of graphs from Haasler and Frossard (2024)

    Args:
        laplacians (NDArray): Tensor of Laplacians of shape (..., n_graphs, n_nodes, n_nodes)
        weights (NDArray, optional): Weights for interpolation of shape (..., n_graphs). Defaults to
            uniform average.
        s_0 (NDArray, optional): Initial SPD matrix. Defaults to None.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        rcond (float, optional): Relative cutoff for small singular values in pinv.
            Defaults to 1e-3.

    Returns:
        NDArray: Laplacian of barycenter graph

    References:
        I. Haasler and P. Frossard, “Bures-Wasserstein Means of Graphs,” in AIStats, PMLR, Apr.
            2024, pp. 1873-1881. Available: https://proceedings.mlr.press/v238/haasler24a.html

    """

    # **Given** $L_1, \dots, L_m \in \R^{N \times N}$ SPD Laplacians with only one zero eigenvalue,
    # and initial SPD matrix $S \in \R^{N \times N}$.
    # 1. $\Sigma_j \gets ( L_j + \frac{1}{N} \bm 1_{N \times N} )^{-1}$ for $j = 1, \dots, m$;
    # 2. **while** Not converged **do**
    #   1. $$S \gets S^{-1/2} \left(
    #      \sum_{j=1}^m \lambda_j \left( S^{1/2} \Sigma_j S^{1/2} \right)^{1/2}
    #      \right)^2 S^{-1/2}$$
    # 3. **return** $L \gets S^{-1} - \frac{1}{N} \bm 1_{N \times N}$

    if (inv_laplacians is None) == (laplacians is None):
        raise ValueError("Either inv_laplacians or laplacians must be provided")

    if inv_laplacians is None:
        inv_laplacians = pinvh(laplacians + 1.0 / laplacians.shape[-1], atol=rcond)  # type:ignore
    assert isinstance(inv_laplacians, np.ndarray)

    n_nodes = inv_laplacians.shape[-1]

    if weights is None:
        weights = np.ones(inv_laplacians.shape[:-2], dtype=float)
        weights /= weights.shape[-1]
    if len(weights.shape) == 1:
        weights = np.expand_dims(weights, 0)

    if s_0 is None:
        out = np.tile(np.eye(n_nodes), (*inv_laplacians.shape[:-3], 1, 1))
    else:
        out = s_0

    weights = np.expand_dims(weights, (-2, -1))

    for i in range(max_iter):
        sqrt = np.expand_dims(sqrtm_spd(out, rcond=rcond), 1)

        inter = sqrt @ inv_laplacians @ sqrt
        inter = sqrtm_spd(inter, rcond=rcond)
        out_new = np.sum(weights * inter, axis=-3)
        assert (out_new.shape) == out.shape

        evals, evecs = eigh(out)
        sqrtinv = evecs @ (
            np.sqrt(inv_pos(evals, rcond=rcond))[..., np.newaxis] * evecs.swapaxes(-1, -2)
        )

        out_new = sqrtinv @ out_new @ out_new @ sqrtinv

        if relative_error(out, out_new) < tol:
            if verbose > 0:
                info("BW mean of graphs conveged after %d iterations", i)
            break

        out = out_new

    if return_pinv:
        return out

    return pinv_spd(out, rcond=rcond) - 1.0 / n_nodes


def bw_moving_average(
    ma_size: int,
    *,
    laplacians=None,
    inv_laplacians=None,
    return_pinv=False,
    rcond=1e-8,
    execution="vectorized",
) -> NDArray:
    """Compute BM means on rolling windows of laplacians, or inverse laplacians

    Args:
        ma_size (int): Moving average size
        laplacians (NDArray, optional): Laplacian matrices. Defaults to None.
        inv_laplacians (_type_, optional): Pre-inverted Laplacians, make computation faster.
            Defaults to None.
        return_pinv (bool, optional): Wether to return pseudoinverse of MA. Defaults to False.
        rcond (float, optional): Inversion tolerance. Defaults to 1e-8.

    Raises:
        ValueError: _description_

    Returns:
        NDArray: Laplacians of moving average, or their pinv
    """
    if (laplacians is None) == (inv_laplacians is None):
        raise ValueError("Either laplacians or inv_laplacians must be provided")

    if inv_laplacians is None:
        n_nodes = laplacians.shape[-1]  # type: ignore
        inv_laplacians = pinvh(
            laplacians + 1.0 / n_nodes,
            atol=rcond,
        )
    else:
        n_nodes = inv_laplacians.shape[-1]

    match execution:
        case "parallel":
            return np.stack(
                Parallel(n_jobs=64, prefer="threads")(
                    delayed(bureswasserstein_mean)(
                        laplacians=None,
                        inv_laplacians=inv_lapl,
                        rcond=rcond,
                        return_pinv=return_pinv,
                    )
                    for inv_lapl in sliding_window_view(
                        inv_laplacians,
                        window_shape=(ma_size, n_nodes, n_nodes),
                    ).squeeze()
                )
            )
        case "vectorized":
            return bureswasserstein_mean_vec(
                laplacians=None,
                inv_laplacians=sliding_window_view(
                    inv_laplacians,
                    window_shape=(ma_size, n_nodes, n_nodes),
                ).squeeze(),
                rcond=rcond,
                return_pinv=return_pinv,
            )
        case _:
            raise ValueError(f"Execution mode not recognized: {execution}")
