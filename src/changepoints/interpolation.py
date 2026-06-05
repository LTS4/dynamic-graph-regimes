"""Costs based on distance from interpolations"""

################################################################################
# Interpolation

import numpy as np
from numpy.typing import NDArray
from ruptures.base import BaseCost
from scipy.linalg import pinvh
from scipy.signal import convolve

from src.changepoints.barycenter import BWBaseCost
from src.distance import bw_dist_decomposed
from src.interpolation import bw_moving_average, interpolate_bureswasserstein_inv
from src.regression import ot_network_regression
from src.utils import n_nodes_from_vec, vec_to_square, weights_to_lapl


class BWRegressionCost(BWBaseCost):
    """Segment cost based on sum of distances from BW regression"""

    @property
    def model(self):
        return "bw-regression"

    def error(self, start, end):
        """Return the approximation cost on the segment [start:end].

        Args:
            start (int): start of the segment
            end (int): end of the segment

        Returns:
            float: segment cost
        """
        if self.errors_.get((start, end)) is None:
            idxs = list(range(start, start + self.context_size)) + list(
                range(end - self.context_size, end)
            )
            inv_interpolation = ot_network_regression(
                self.t[idxs],
                laplacians=None,
                inv_laplacians=self.inv_laplacians[idxs],
                x_out=self.t[start + self.context_size : end - self.context_size],
                kernel="rbf",
                kernel_pars={"gamma": 1 / (end - start)},
                rcond=self.rcond,
                return_pinv=True,
            )

            self.errors_[(start, end)] = np.sum(
                bw_dist_decomposed(
                    self.inv_evals[start + self.context_size : end - self.context_size],
                    self.evecs[start + self.context_size : end - self.context_size],
                    inv_interpolation,
                    rcond=self.rcond,
                )
            ).item()

        return self.errors_[(start, end)]


class GraphVariationCost(BaseCost):
    """Mean variation cost"""

    @property
    def model(self):
        return "variation"

    def __init__(
        self,
        ma_size: int = 0,
        interpolation: str = "bw",
        distance: str = "bw",
        stat: str = "mean",
        min_size=3,
        rcond=1e-5,
    ) -> None:

        self.ma_size: int = ma_size

        if interpolation != distance:
            raise ValueError("Different interpolation and distances are not supported")
        self.interpolation: str = interpolation
        self.distance: str = distance

        self.stat = stat

        self.min_size: int = min_size
        self.rcond: float = rcond

        self.signal: NDArray
        self.n_samples_: int
        self.n_nodes_: int
        self.offset_: int
        self.errors_: dict[tuple[int, int], float] = {}

    def _moving_average(self, vec_snapshots: NDArray) -> NDArray:
        match self.interpolation:
            case "bw":
                inv_laplacians = pinvh(
                    weights_to_lapl(vec_to_square(vec_snapshots))
                    + 1.0 / n_nodes_from_vec(vec_snapshots.shape[-1]),
                    atol=self.rcond,
                )
                if self.ma_size > 0:
                    return bw_moving_average(
                        self.ma_size,
                        inv_laplacians=inv_laplacians,
                        return_pinv=True,
                        rcond=self.rcond,
                    )
                return inv_laplacians  # type: ignore

            case "l2" | "l1":
                if self.ma_size > 0:
                    return convolve(
                        vec_snapshots,
                        np.ones((self.ma_size, 1), dtype=float) / self.ma_size,
                        mode="valid",
                    )
                return vec_snapshots

            case _:
                raise ValueError("Invalid ma_distance")

    def fit(self, vec_snapshots):
        # pylint: disable=arguments-differ
        self.signal = self._moving_average(vec_snapshots)

        self.n_samples_ = vec_snapshots.shape[0]
        self.n_nodes_ = n_nodes_from_vec(vec_snapshots.shape[-1])

        self.offset_ = max(0, (self.ma_size - 1) // 2)

    def error(self, start: int, end: int):
        if self.errors_.get((start, end)) is None:
            if (end - start < self.min_size) or (start > self.n_samples_ - self.ma_size):
                return np.inf

            if self.ma_size > 0:
                end_shift = end - self.ma_size + 1
            else:
                end_shift = end

            t = np.linspace(0, 1, num=end_shift - start)
            match self.interpolation:
                case "bw":
                    geodesic = interpolate_bureswasserstein_inv(
                        self.signal[start], self.signal[end_shift - 1], t=t
                    )
                case "l2" | "l1":
                    geodesic = (
                        t[:, np.newaxis] * self.signal[[end_shift - 1]]
                        + (1 - t)[:, np.newaxis] * self.signal[[start]]
                    )

            match self.distance:
                case "bw":
                    errors = bw_dist_decomposed(geodesic, self.signal[start:end_shift]) ** 2
                case "l2":
                    errors = (geodesic - self.signal[start:end_shift]) ** 2
                case "l1":
                    errors = np.abs(geodesic - self.signal[start:end_shift])

            self.errors_[(start, end)] = getattr(np, self.stat)(errors).item()

        return self.errors_[(start, end)]


class MeanVariationCost(GraphVariationCost):
    def __init__(self, **kwargs) -> None:
        super().__init__(stat="mean", **kwargs)

    @property
    def model(self):
        return "mean-variation"
