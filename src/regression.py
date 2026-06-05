"""Implementation of regression methods for graph sequences"""

from typing import Optional

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import pinv
from numpy.typing import NDArray

from src.interpolation import bureswasserstein_mean, bureswasserstein_mean_vec


def get_kernel_weights(x_in: NDArray, x_out: NDArray, kernel_choice: str, kernel_pars=None):
    """Get weights for points corressponding to x_in wrt x_out.

    Args:
        x_in (NDArray): Regressor array of shape (n_samples_in, n_features)
        x_out (NDArray): Regressed array of shape (n_samples_out, n_features)
        kernel_choice (str): Kernel choice, either "zalles" or "rbf"
        kernel_pars (_type_, optional): Parameters for the kernel, e.g., `gamma` for RBF kernel.
            Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    kernel_pars = kernel_pars or {}

    if len(x_in.shape) < 2:
        x_in = x_in.reshape(-1, 1, 1)
    if len(x_out.shape) < 2:
        x_out = x_out.reshape(-1, 1, 1)

    # shape: (n_samples_in, n_samples_out)
    match kernel_choice:
        case "zalles":
            x_mean = x_in.mean(axis=0, keepdims=True)
            x_in = x_in - x_mean
            x_cov = x_in.T @ x_in / (x_in.shape[0] - 1)  # shape: (n_features, n_features)
            x_kernel = 1 + x_in @ np.linalg.solve(x_cov, (x_out - x_mean).T)

        case "rbf":
            x_kernel = np.exp(
                -np.sum(
                    (x_in[:, np.newaxis, :] - x_out[np.newaxis, :, :]) ** 2,
                    axis=-1,
                )
                * kernel_pars.get("gamma", 1)
            )
        case _:
            raise ValueError(f"Unknown kernel: {kernel_choice}")

    # NOTE: Kernel normalization is not in Zalles work
    x_kernel /= x_kernel.sum(0, keepdims=True)
    return np.moveaxis(x_kernel, -1, 0)


def ot_network_regression(
    x_in: NDArray,
    laplacians: NDArray,
    x_out: NDArray,
    kernel: str,
    kernel_pars: Optional[dict] = None,
    rcond=1e-8,
    retun_kernel=False,
    inv_laplacians: Optional[NDArray] = None,
    return_pinv=False,
    execution="vectorized",
) -> NDArray | tuple[NDArray, NDArray]:
    """Regression between Laplacians in graph space as function of `x_(in|out)`.

    The `zalles` kernel implementats the method from "An Optimal Transport Approach for Network
    Regression", Zalles et al. 2024.

    Args:
        x_in: Input data, shape (n_samples_in, [n_features])
        laplacians: Graph Laplacians, shape (n_samples_in, n_nodes, n_nodes)
        x_out: Output data, shape (n_samples_out, [n_features])
        kernel: Kernel type, either "zalles" or "rbf"
        rcond: Cutoff for small singular values in pinv
        kernel_pars: Parameters for the kernel, e.g., `gamma` for RBF kernel
        retun_kernel: If True, return the kernel matrix as well
        inv_laplacians: Precomputed inverse Laplacians, shape (n_samples_in, n_nodes, n_nodes)
            If not provided, they will be computed from `laplacians`.

    Raises:
        ValueError: If the kernel is not recognized.

    Returns:
        NDArray | tuple[NDArray, NDArray]: The regression result, either as a single array of shape
            (n_samples_out, n_nodes, n_nodes) or as a tuple containing the result and the kernel
            matrix of shape (n_samples_in, n_samples_out).

    References:

    - A. G. Zalles, K. M. Hung, A. E. Finneran, L. Beaudrot, and C. A. Uribe, "An Optimal Transport
    Approach for Network Regression," Jun. 17, 2024, https://doi.org/10.48550/arXiv.2406.12204.
    """
    if inv_laplacians is None:
        inv_laplacians = pinv(laplacians + 1.0 / laplacians.shape[1], hermitian=True)

    x_kernel = get_kernel_weights(
        x_in=x_in, x_out=x_out, kernel_choice=kernel, kernel_pars=kernel_pars
    )

    match execution:
        case "parallel":
            out = np.stack(
                Parallel(n_jobs=-1, prefer="threads")(
                    delayed(bureswasserstein_mean)(
                        laplacians=None,
                        weights=x_kernel_i,
                        inv_laplacians=inv_laplacians,
                        rcond=rcond,
                        return_pinv=return_pinv,
                    )
                    for x_kernel_i in x_kernel
                )
            )
        case "comprehension":
            out = np.stack(
                [
                    bureswasserstein_mean(
                        laplacians=None,
                        weights=x_kernel_i,
                        inv_laplacians=inv_laplacians,
                        rcond=rcond,
                        return_pinv=return_pinv,
                    )
                    for x_kernel_i in x_kernel
                ]
            )
        case "vectorized":
            out = bureswasserstein_mean_vec(
                laplacians=None,
                weights=x_kernel,
                inv_laplacians=inv_laplacians,
                rcond=rcond,
                return_pinv=return_pinv,
            )

    if retun_kernel:
        return out, x_kernel
    else:
        return out
