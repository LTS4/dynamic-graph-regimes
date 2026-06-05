"""Prototype dissimilarity features from Zambon et al. (2018)

[1] D. Zambon, C. Alippi, and L. Livi, “Concept Drift and Anomaly Detection in Graph Streams,” IEEE
    Trans. Neural Netw. Learning Syst., vol. 29, no. 11, pp. 5592-5605, Nov. 2018, doi:
    10.1109/TNNLS.2018.2804443.
"""

import numpy as np
import sparse
from cdg.embedding import DissimilarityRepresentation
from numpy.typing import NDArray

from src.distance import bw_dist
from src.utils import weights_to_lapl

################################################################################
# PROTOTYPE DISCREPANCY


def prototype_discrepancy(
    adjs: sparse.COO | NDArray, n_prototypes: int, dist_choice: str
) -> NDArray:
    """Compute distance to most relevant samples, i.e. prototypes, in x

    Args:
        adjs (NDArray): Sequence of sparse adjacency matrices of shape
            (..., n_samples, n_nodes, n_nodes)
        n_prototypes (int): Number of prototypes
        dist_choice (str): Choice of distance function

    Raises:
        ValueError: If unknown distance choice

    Returns:
        NDArray: Array of shape (..., n_samples, n_prototypes)
    """
    # Compute distances
    match dist_choice:
        # pylint: disable=function-redefined
        case "l1":

            def dist_fun(x0: sparse.COO | NDArray, x1: sparse.COO | NDArray) -> NDArray:
                out = (
                    np.abs(np.expand_dims(x0, axis=-4) - np.expand_dims(x1, axis=-3)).sum((-2, -1))
                    / 2
                )
                if isinstance(out, sparse.SparseArray):
                    out = out.todense()
                return out  # type: ignore

        case "bw":

            def dist_fun(x0: sparse.COO | NDArray, x1: sparse.COO | NDArray) -> NDArray:
                x0 = x0.todense() if isinstance(x0, sparse.SparseArray) else x0
                x1 = x1.todense() if isinstance(x1, sparse.SparseArray) else x1
                return bw_dist(
                    weights_to_lapl(np.expand_dims(x0, -4)),
                    weights_to_lapl(np.expand_dims(x1, -3)),
                )

        case _:
            raise ValueError(f"Unknown distance choice {dist_choice}")

    return DissimilarityRepresentation(nprot=n_prototypes).fit_transform(
        graphs=adjs, dist_fun=dist_fun
    )
