"""Changepoint costs based on distances from barycenters"""

from abc import ABC
from typing import Any, Callable

import numpy as np
from numpy.linalg import eigh
from numpy.typing import NDArray
from ruptures.base import BaseCost

from src.distance import bw_dist_decomposed
from src.interpolation import bureswasserstein_mean
from src.linalg import inv_pos, pinv_spd
from src.regression import ot_network_regression
from src.utils import vec_to_square, weights_to_lapl


class LinearBarycenterCost(BaseCost):
    """Compute linear barycenter of weights and compute distance with Frobenious norm"""

    @property
    def model(self):
        return "linear-barycenter"

    def __init__(self, context_size=2, n_center=3, rcond=0.00001) -> None:
        self.context_size = context_size
        self.n_center = n_center or self.context_size
        self.min_size = 2 * self.context_size + self.n_center
        self.rcond = rcond

        self.signal: NDArray[np.float64]  # shape: (time, n_edges)
        self.errors: dict[tuple[int, int], float] = {}

    def fit(self, vec_weights):
        # pylint: disable=arguments-differ
        """Set the internal parameter."""
        self.signal = vec_weights

    def error(self, start, end):
        if (start, end) not in self.errors:
            ref_idxs = list(range(start, start + self.context_size)) + list(
                range(end - self.context_size, end)
            )

            c_start = (end - start - self.n_center) / 2
            hyp_idxs = list(
                range(np.floor(c_start).astype(int), np.ceil(c_start + self.n_center).astype(int))
            )

            self.errors[(start, end)] = np.linalg.norm(
                (np.mean(self.signal[ref_idxs], axis=0) - np.mean(self.signal[hyp_idxs], axis=0)),
                ord=2,
            )

        return self.errors[(start, end)]


class BWBaseCost(BaseCost, ABC):
    """Base class for BW costs"""

    def __init__(self, context_size=2, rcond=1e-5) -> None:
        super().__init__()
        self.context_size = context_size
        self.min_size = 2 * self.context_size + 1
        self.rcond = rcond

        self.signal: NDArray[np.float64]  # shape: (time, n_edges)
        self.inv_laplacians: NDArray[np.float64]  # shape: (time, n_nodes, n_nodes)
        self.evecs: NDArray[np.float64]  # shape: (time, n_nodes, n_nodes)
        self.inv_evals: NDArray[np.float64]  # shape: (time, n_nodes)
        self.t: NDArray[np.int64]  # shape: (time)
        self.errors_: dict[tuple[int, int], float] = {}

    def fit(self, vec_snapshots):
        # pylint: disable=arguments-differ
        """Set the internal parameter."""
        self.signal = vec_snapshots
        lapls = weights_to_lapl(vec_to_square(vec_snapshots))
        self.inv_laplacians = pinv_spd(lapls + 1.0 / lapls.shape[1], rcond=self.rcond)
        evals, self.evecs = eigh(lapls)
        self.inv_evals = inv_pos(evals, rcond=self.rcond)

        duration = lapls.shape[0]
        self.t = np.arange(duration)

        return self


class BWMovingAverageCost(BWBaseCost):
    """Smooth graphs with a moving BW average, then compute trajectories as interpolation between
    pairs"""

    def error(self, start, end):
        raise NotImplementedError()


class BWBarycenterCost(BWBaseCost):
    """For each interval compute barycenter and compare to central snapshots"""

    @property
    def model(self):
        return "bw-barycenter"

    # IDEA: two possibilities:
    # 1. Compute barycenter of endpoints and of middle-points and compute their distance
    # [2]. Compute barycenter of endpoints and compute sum of distances to middle-points

    def __init__(
        self,
        context_size=2,
        n_center=None,
        center_choice="barycenter",
        kernel="rbf",
        kernel_pars: None | dict[str, Callable[[int, int], Any]] = None,
        distance_choice="sum",
        rcond=0.00001,
    ) -> None:
        super().__init__(context_size, rcond)

        self.n_center = n_center or self.context_size
        self.min_size = 2 * self.context_size + self.n_center

        self.center_choice = center_choice

        self.kernel = kernel
        if kernel_pars is None:
            match kernel:
                case "rbf":
                    self.kernel_pars = {"gamma": lambda start, end: 1 / (end - start)}
                case _:
                    raise ValueError("Kernel parameters must be provided for non-RBF kernels.")

        self.distance_choice = distance_choice

    def error(self, start, end):
        if self.errors_.get((start, end)) is None:
            ref_idxs = list(range(start, start + self.context_size)) + list(
                range(end - self.context_size, end)
            )

            c_start = (end - start - self.n_center) / 2
            hyp_idxs = list(
                range(np.floor(c_start).astype(int), np.ceil(c_start + self.n_center).astype(int))
            )

            match self.center_choice:
                case "barycenter":
                    inv_center = bureswasserstein_mean(
                        laplacians=None,
                        inv_laplacians=self.inv_laplacians[ref_idxs],
                        rcond=self.rcond,
                        return_pinv=True,
                    )
                case "regression":
                    inv_center = ot_network_regression(
                        self.t[ref_idxs],
                        laplacians=None,
                        inv_laplacians=self.inv_laplacians[ref_idxs],
                        x_out=self.t[hyp_idxs].mean(),
                        kernel=self.kernel,
                        kernel_pars={
                            key: func(start, end) for key, func in self.kernel_pars.items()
                        },
                        return_pinv=True,
                        rcond=self.rcond,
                    )
                case _:
                    raise ValueError("Invalid center type")

            match self.distance_choice:
                case "sum":
                    self.errors_[(start, end)] = np.sum(
                        bw_dist_decomposed(
                            self.inv_laplacians[hyp_idxs],
                            inv_center,
                            rcond=self.rcond,
                        )
                    ).item()
                case "barycenter":
                    self.errors_[(start, end)] = bw_dist_decomposed(
                        *eigh(
                            bureswasserstein_mean(
                                laplacians=None,
                                inv_laplacians=self.inv_laplacians[hyp_idxs],
                                rcond=self.rcond,
                                return_pinv=True,
                            )
                        ),
                        inv_center,
                        rcond=self.rcond,
                    ).item()
                case _:
                    raise ValueError("Invalid distance_choice")

        return self.errors_[(start, end)]
