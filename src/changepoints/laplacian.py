"""Implementation of the CPD algorithm from Huang et al. (2020)

[1] S. Huang, Y. Hitti, G. Rabusseau, and R. Rabbany, “Laplacian Change Point Detection for Dynamic
    Graphs,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery &
    Data Mining, Virtual Event CA USA: ACM, Aug. 2020, pp. 349-358. doi: 10.1145/3394486.3403077.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from ruptures.base import BaseEstimator

from src.utils import n_nodes_from_vec, vec_to_square, weights_to_lapl


class LAD(BaseEstimator):
    # pylint: disable=arguments-differ
    def __init__(
        self,
        short_window: int,
        long_window: int,
        num_k: int = None,
        typical_choice: str = "principal",
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.num_k = num_k
        self.typical_choice = typical_choice

        self.n_samples_: int
        self.zscores_: NDArray[np.float64]
        self.argsorted_: NDArray[np.int64]

    def _compute_typical_vec(self, svals: NDArray) -> tuple[NDArray, NDArray]:
        match self.typical_choice:
            case "principal":
                # SWV already transpose the view, as axes are preserved
                short_norm = np.linalg.svd(
                    sliding_window_view(
                        svals, self.short_window, axis=0
                    ),  # shape: (n_samples - short_window + 1, n_nodes, short_window)
                    full_matrices=False,
                )[0][..., 0]
                long_norm = np.linalg.svd(
                    sliding_window_view(
                        svals, self.long_window, axis=0
                    ),  # shape: (n_samples - long_window + 1, n_nodes, long_window)
                    full_matrices=False,
                )[0][..., 0]
            case "average":
                short_norm = sliding_window_view(svals, self.short_window, axis=0).mean(-1)
                long_norm = sliding_window_view(svals, self.long_window, axis=0).mean(-1)
            case _:
                raise ValueError("Invalid choice of typical vector computation")

        return short_norm, long_norm

    def initialize(self, vec_weights: NDArray):
        """Init missing parameters and validate"""
        self.n_samples_ = vec_weights.shape[0]

        n_nodes = n_nodes_from_vec(vec_weights.shape[-1])
        if not self.num_k:
            self.num_k = n_nodes
        elif self.num_k > n_nodes:
            raise ValueError("num_k cannot be larger than number of nodes")

        if self.short_window >= self.long_window:
            raise ValueError("short_window must be smaller than long_window")

    def fit(self, vec_weights: NDArray) -> "LAD":
        """Follow algorithm from Huang et al. (2020):
        1. Compute SVD of each graph
        2. Compute "normal behaviors" vectors of long and short term
        3. Compute cosine similarities and take max
        """
        self.initialize(vec_weights)
        lapls = weights_to_lapl(vec_to_square(vec_weights))

        # Compute singular values for each Laplacian
        # shape: (n_graphs, num_k)
        svals: NDArray[np.float64] = np.linalg.svd(lapls, compute_uv=False, hermitian=True)[
            :, : self.num_k
        ]
        # Normalize first dim to unit
        svals /= np.linalg.norm(svals, ord=2, axis=-1, keepdims=True)

        # Compute normal behaviors from sliding windows
        # shape: (n_graphs - window + 1, num_k)
        short_norm, long_norm = self._compute_typical_vec(svals)

        # Compute Z scores (i.e. cosine similarity) between svals and corresponding normal vectors
        # We set initial values to 0
        zscores = np.zeros(shape=(svals.shape[0], 2), dtype=float)
        zscores[self.short_window - 1 :, 0] = 1 - np.abs(
            np.einsum("tk,tk->t", svals[self.short_window - 1 :], short_norm)
        )
        zscores[self.long_window - 1 :, 1] = 1 - np.abs(
            np.einsum("tk,tk->t", svals[self.long_window - 1 :], long_norm)
        )
        self.zscores_ = np.abs(np.diff(zscores, 1, axis=0, prepend=0)).max(1)

        # We suppose that differences are centered at zero and normalize Z score
        self.zscores_ /= np.sqrt(np.sum(self.zscores_**2) / (self.n_samples_ - 1))

        # Higher scores are more likely to be changepoints
        self.argsorted_ = np.argsort(self.zscores_)[::-1]

        return self

    def predict(self, z_thr: float = 2, n_bkps: int = None) -> list[int]:
        """Identify peaks in anomaly score increase from previous step.

        Original model uses fixed :var:`n_changes`, while we allow for anomaly detection based on
        number of std deviations through :var:`z_thr`.
        """
        if n_bkps:
            cpts = np.sort(self.argsorted_[: int(n_bkps)])
        else:
            cpts = np.nonzero(self.zscores_ > z_thr)[0]

        return cpts.tolist() + [self.n_samples_]

    def fit_predict(self, vec_weights: NDArray, z_thr: float = 2, n_changes: int = None):
        self.fit(vec_weights)
        return self.predict(z_thr, n_changes)
