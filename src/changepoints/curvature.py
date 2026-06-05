"""Changepoint detection methods based on curvature"""

from abc import ABC, abstractmethod

import numpy as np
from joblib import Parallel, delayed
from numpy.linalg import eigh
from numpy.typing import NDArray
from scipy.signal import convolve, find_peaks

from src.distance import bw_dist, bw_dist_decomposed
from src.interpolation import bureswasserstein_mean, bw_moving_average
from src.utils import vec_to_square, weights_to_lapl


class BaseCurvatureModel(ABC):
    def __init__(
        self,
        ma_size: int,
        window_size: int,
        window: str = "forward",
        curvature_smoothing=1,
        rcond=1e-5,
    ) -> None:
        self.rcond = rcond
        self.ma_size = ma_size
        self.window_size = window_size
        self.window = window
        self.curvature_smoothing = curvature_smoothing

        match self.window:
            case "forward":
                self.offset = 0
            case "center":
                self.offset = (self.window_size + 1) // 2
            case "backward":
                self.offset = self.window_size - 1
            case _:
                raise ValueError("Invalid window direction")

        self.curvature_: NDArray
        self.n_samples_: int

    @abstractmethod
    def fit(self, vec_weights: NDArray) -> "BaseCurvatureModel":
        raise NotImplementedError()

    def predict(self, width_coeff: float) -> list[int]:

        if self.curvature_smoothing > 1:
            ec: NDArray = convolve(
                self.curvature_,
                np.ones(self.curvature_smoothing) / self.curvature_smoothing,
                mode="same",
            )
        else:
            ec = self.curvature_

        # FIXME: should add half ma_size
        cpts = find_peaks(-ec, width=width_coeff * 3 * self.ma_size)[0] + self.offset

        # We add a changepoint at the end for compatibility with ruptures
        return cpts.tolist() + [self.n_samples_]


class BWCurvaturePeaks(BaseCurvatureModel):
    def __init__(
        self,
        ma_size: int,
        window_size: int,
        window: str = "forward",
        curvature_smoothing=1,
        rcond=0.00001,
    ) -> None:
        super().__init__(ma_size, window_size, window, curvature_smoothing, rcond)

        self.ma_inv_evals_: NDArray
        self.ma_inv_evecs_: NDArray
        self.inv_centers_: NDArray

    def fit(self, vec_weights: NDArray) -> "BWCurvaturePeaks":
        self.n_samples_ = vec_weights.shape[0]
        ts = np.arange(self.n_samples_)

        # compute moving average on laplacians
        ma_inv_lapls = bw_moving_average(
            self.ma_size,
            laplacians=weights_to_lapl(vec_to_square(vec_weights)),
            return_pinv=True,
            rcond=self.rcond,
        )
        self.ma_inv_evals_, self.ma_inv_evecs_ = eigh(ma_inv_lapls)

        # Rolling centers of [(0,w),...,(T-w, T)]
        self.inv_centers_ = np.stack(
            Parallel(n_jobs=64, prefer="threads")(
                delayed(bureswasserstein_mean)(
                    laplacians=None,
                    inv_laplacians=ma_inv_lapls[[t, t + self.window_size]],
                    rcond=self.rcond,
                    return_pinv=True,
                )
                for t in ts[: -self.window_size - self.ma_size + 1]
            )
        )

        # FIXME: use self.rcond
        self.curvature_ = np.stack(
            Parallel(n_jobs=64, prefer="threads")(
                delayed(bw_dist_decomposed)(ma_inv_eval, ma_inv_evec, inv_center)
                for ma_inv_eval, ma_inv_evec, inv_center in zip(
                    self.ma_inv_evals_[self.offset :],
                    self.ma_inv_evecs_[self.offset :],
                    self.inv_centers_,
                )
            )
        )

        return self


class LinearCurvaturePeaks(BaseCurvatureModel):
    def __init__(
        self,
        ma_size: int,
        window_size: int,
        window: str = "forward",
        curvature_smoothing=1,
        distance: str = "l2",
        rcond=0.00001,
    ) -> None:
        super().__init__(ma_size, window_size, window, curvature_smoothing, rcond)
        self.distance = distance

    def fit(self, vec_weights: NDArray) -> BaseCurvatureModel:
        self.n_samples_ = vec_weights.shape[0]

        # compute moving average on vec_weights
        ma_weights: NDArray = convolve(
            vec_weights, np.ones((self.ma_size, 1)) / self.ma_size, "valid"
        )

        # Rolling centers of [(0,w),...,(T-w, T)]
        rolling_centers = convolve(
            ma_weights,
            np.expand_dims(
                np.array([0.5] + (self.window_size - 2) * [0.0] + [0.5], dtype=float), -1
            ),
            "valid",
        )

        match self.distance:
            case "l2":
                self.curvature_ = np.linalg.norm(
                    ma_weights[self.offset : self.offset + rolling_centers.shape[0]]
                    - rolling_centers,
                    ord=2,
                    axis=1,
                )
            case "bw":
                self.curvature_ = np.stack(
                    Parallel(n_jobs=64, prefer="threads")(
                        delayed(bw_dist)(lapl_ma, lapl_center, self.rcond)
                        for lapl_ma, lapl_center, in zip(
                            weights_to_lapl(vec_to_square(ma_weights[self.offset :])),
                            weights_to_lapl(vec_to_square(rolling_centers[self.offset :])),
                        )
                    )
                )

        return self
