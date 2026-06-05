from itertools import pairwise

import numpy as np
import sparse
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from src.interpolation import interpolate_bureswasserstein
from src.sampling.graphs import sample_graphs
from src.sampling.structures import sample_sbm_structures
from src.utils import lapl_to_weights, weights_to_lapl

################################################################################
# SEQUENCE SAMPLING


def nonzero_multinomial(rng: Generator, *args, **kwargs) -> NDArray[np.int64]:
    while np.any(np.isclose(out := rng.multinomial(*args, **kwargs), 0)):
        pass
    return out


def sample_changepoints(
    change_prob: float, n_samples: int, min_change_distance: int = 1, seed: int | Generator = None
) -> NDArray[np.int64]:
    """Sample changepoints with a geometric distribution.

    Args:
        change_prob (float): Probability of change at each sample.
        n_samples (int): Number of samples.
        min_change_distance (int, optional): Minimum distance between changepoints. Defaults to 1.
        seed (int | Generator, optional): Random seed or generator. Defaults to None.

    Returns:
        NDArray[np.int64]: _description_
    """
    rng = default_rng(seed)
    # Fix this side so that last changepoint is after n_samples with prob> 95%
    size = int(1.5 * n_samples * change_prob)
    changepoints = np.cumsum(
        np.max(
            [
                rng.geometric(p=change_prob, size=size),
                min_change_distance * np.ones(size, dtype=int),
            ],
            axis=0,
        )
    )
    return np.concatenate([[0], changepoints[changepoints < n_samples], [n_samples]])


def sample_static_regimes(
    changepoints: NDArray[np.int64],
    sbm_endpoints: NDArray[np.float64],
    return_sparse=True,
    seed: int | Generator = None,
) -> NDArray[np.float64] | sparse.COO:
    """Sample iid graphs from SBM models with abrupt regime changes

    Args:
        changepoints (NDArray[np.int64]): Timestamps of regime changes.
        sbm_endpoints (NDArray[np.float64]): SBM targets at changpoints.
        speeds (NDArray[np.float64], optional): Speed of evolution between changepoints.
            Defaults to None, which means full evolution. If smaller than 1 there is only partial
            evolution between endpoints.
        return_sparse: whether to return sparse array. Defaults to True.
        seed (int | Generator, optional): Random seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Sequence of graphs of shape (changepoints[-1], n_nodes, n_nodes)
    """
    n_nodes = sbm_endpoints.shape[-1]

    snapshots = np.empty((changepoints[-1], n_nodes, n_nodes), dtype=float)

    for (t0, t1), sbm in zip(pairwise(changepoints), sbm_endpoints):
        snapshots[t0:t1] = sample_graphs(
            np.tile(sbm, (t1 - t0, 1, 1)), sampling_strategy="threshold-iid", seed=seed
        )

    if return_sparse:
        return sparse.COO.from_numpy(snapshots)
    return snapshots


def sample_geodesic_varying_speed(
    changepoints: NDArray,
    graph_endpoints: NDArray[np.float64],
    interpolation_strategy: str,
    sampling_strategy: str,
    speeds: NDArray[np.float64],
    mask_low=0.0,
    mask_high=1.0,
    rcond=1e-8,
    return_sparse: bool = True,
    seed: int | Generator = None,
):
    """Sample a sequence of graphs along the Bures-Wasserstein interpolation of the two graph endpoints.

    Args:
        changepoints (NDArray[np.int64]): Timestamps of regime changes.
        sbm_endpoints (NDArray[np.float64]): SBM targets at changpoints.
        sampling_strategy (str): Strategy for sampling the sequence.
            Options are 'iid' or 'threshold'.
        speeds (NDArray[np.float64], optional): Percentage of evolution between changepoints, cumulates to 1.
        return_sparse (bool, optional): Whether to return sparse array. Defaults to True.
        seed (int | Generator, optional): Random seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Sequence of graphs of shape (changepoints[-1], n_nodes, n_nodes)
    """
    # We compute sampling timesteps based on speeds. The min/max logic on cpt0 allows to avoid
    # stalling on the same progress after changeoint
    t = np.hstack(
        [
            np.linspace(t0, t1, num=cpt1 - max(0, cpt0 - 1))[min(cpt0, 1) :]
            for (t0, cpt0), (t1, cpt1) in pairwise(zip(np.cumsum(speeds), changepoints))
        ]
    )

    snapshots = sample_graphs(
        structures=graph_geodesic(
            start_graph=graph_endpoints[0],
            end_graph=graph_endpoints[1],
            interpolation_strategy=interpolation_strategy,
            num=None,
            speed=None,
            rcond=rcond,
            t=t,
        ),
        sampling_strategy=sampling_strategy,
        mask_low=mask_low,
        mask_high=mask_high,
        seed=seed,
    )

    if return_sparse:
        return sparse.COO.from_numpy(snapshots)
    return snapshots


def sample_geodesic_sequences(
    changepoints: NDArray,
    graph_endpoints: NDArray[np.float64],
    interpolation_strategy: str,
    sampling_strategy: str,
    speeds: NDArray[np.float64] = None,
    mask_low=0.0,
    mask_high=1.0,
    rcond=1e-8,
    return_sparse: bool = True,
    seed: int | Generator = None,
) -> NDArray[np.float64] | sparse.COO:
    """Sample a sequence of graphs along the Bures-Wasserstein interpolation.

    Args:
        changepoints (NDArray[np.int64]): Timestamps of regime changes.
        sbm_endpoints (NDArray[np.float64]): SBM targets at changpoints.
        sampling_strategy (str): Strategy for sampling the sequence.
            Options are 'iid' or 'threshold'.
        speeds (NDArray[np.float64], optional): Speed of evolution between changepoints.
            Defaults to None, which means full evolution. If smaller than 1 there is only partial
            evolution between endpoints.
        return_sparse (bool, optional): Whether to return sparse array. Defaults to True.
        seed (int | Generator, optional): Random seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Sequence of graphs of shape (changepoints[-1], n_nodes, n_nodes)
    """
    rng = default_rng(seed)

    n_targets, n_nodes, _ = graph_endpoints.shape
    if speeds is None:
        speeds = np.ones(n_targets)

    snapshots = np.empty((changepoints[-1], n_nodes, n_nodes), dtype=float)
    # FIXME: apply same logic as in sample_geodesic_varying_speed to avoid stalling
    for i, (t0, t1) in enumerate(pairwise(changepoints)):
        t1 = min(t1 + 1, changepoints[-1])

        snapshots[t0:t1] = sample_graphs(
            structures=graph_geodesic(
                start_graph=snapshots[t0] if t0 > 0 else graph_endpoints[i],
                end_graph=graph_endpoints[i + 1],
                interpolation_strategy=interpolation_strategy,
                speed=speeds[i],
                num=t1 - t0,
                rcond=rcond,
            ),
            sampling_strategy=sampling_strategy,
            mask_low=mask_low,
            mask_high=mask_high,
            seed=rng,
        )

    if return_sparse:
        return sparse.COO.from_numpy(snapshots)
    return snapshots


def graph_geodesic(
    start_graph: NDArray,
    end_graph: NDArray,
    interpolation_strategy: str,
    num: int,
    speed=1.0,
    rcond=1e-8,
    t: NDArray | None = None,
):
    if t is None:
        t = np.linspace(0, speed, num=num)

    match interpolation_strategy:
        case "bw" | "bures-wasserstein":
            trajectory = lapl_to_weights(
                interpolate_bureswasserstein(
                    lapl0=weights_to_lapl(
                        start_graph
                    ),  # We start with the first target in t0, o/w we use current graph
                    lapl1=weights_to_lapl(end_graph),
                    t=t,
                    rcond=rcond,
                )
            )
        case "linear":
            t = t[:, np.newaxis, np.newaxis]
            trajectory = (1 - t) * start_graph[np.newaxis, ...] + t * end_graph[np.newaxis, ...]
        case _:
            raise ValueError(f"Invalid interpolation_strategy {interpolation_strategy}")
    return trajectory


def sample_bw_trajectory(
    changepoints: NDArray,
    sbm_endpoints: NDArray[np.float64],
    sampling_strategy: str,
    speeds: NDArray[np.float64] = None,
    return_sparse: bool = True,
    seed: int | Generator = None,
) -> NDArray[np.float64] | sparse.COO:
    return sample_geodesic_sequences(
        changepoints=changepoints,
        graph_endpoints=sbm_endpoints,
        interpolation_strategy="bw",
        sampling_strategy=sampling_strategy,
        speeds=speeds,
        return_sparse=return_sparse,
        seed=seed,
    )


def sample_latent_linear(
    changepoints: NDArray[np.int64],
    sbm_endpoints: NDArray[np.float64],
    speeds: NDArray[np.float64] = None,
    return_sparse: bool = True,
    seed: int | Generator = None,
) -> NDArray[np.float64] | sparse.COO:
    """_summary_

    Args:
        changepoints (NDArray[np.int64]): Timestamps of regime changes.
        sbm_endpoints (NDArray[np.float64]): SBM targets at changpoints.
        speed (NDArray[np.float64], optional): Speed of evolution between changepoints.
            Defaults to None, which means full evolution. If smaller than 1 there is only partial
            evolution between endpoints.
        return_sparse (bool, optional): Whether to return sparse array. Defaults to True.
        seed (int | Generator, optional): Random seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Sequence of graphs of shape (changepoints[-1], n_nodes, n_nodes)
    """
    n_targets, n_nodes, _ = sbm_endpoints.shape
    if speeds is None:
        speeds = np.ones(n_targets)

    snapshots = np.empty((changepoints[-1], n_nodes, n_nodes), dtype=float)
    for (t0, t1), speed, (sbm0, sbm1) in zip(
        pairwise(changepoints), speeds, pairwise(sbm_endpoints)
    ):
        lspace = np.linspace(0, speed, num=t1 - t0 + 1)[:-1, np.newaxis, np.newaxis]
        snapshots[t0:t1] = lspace * sbm1 + (1 - lspace) * sbm0

    snapshots = sample_graphs(snapshots, sampling_strategy="iid", seed=seed)

    if return_sparse:
        return sparse.COO.from_numpy(snapshots)
    return snapshots


if __name__ == "__main__":
    from src.visual import print_sequence

    RNG = default_rng(11)
    N_NODES = 15

    sbm_epts_, sbm_blocks_ = sample_sbm_structures(
        3,
        N_NODES,
        n_blocks=[2, 3, 2],
        avg_edge_prob=0.3,
        within_mean=0.6,
        within_scale=0.00,
        seed=RNG,
        return_block_size=True,
        permuted_sbm=False,
    )

    print("blocks:", sbm_blocks_)

    # _print_sequence(
    #     sample_iid_regimes(10, [0, 5, 10], sbm_endpoints=sbm_epts_, seed=RNG),
    #     title="IID",
    # )

    # _print_sequence(
    #     sample_latent_linear(50, [0, 25, 50], sbm_endpoints=sbm_epts_, seed=RNG),
    #     title="Latent linear",
    # )

    SAMPLING_STRATEGY = "threshold-edge-uniform"
    print_sequence(
        sample_bw_trajectory(
            [0, 25, 49],
            sbm_endpoints=sbm_epts_,
            sampling_strategy=SAMPLING_STRATEGY,
            seed=RNG,
        ),
        title=f"BW {SAMPLING_STRATEGY}",
    )
