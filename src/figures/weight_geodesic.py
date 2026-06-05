from itertools import product

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.random import default_rng
from numpy.typing import NDArray

from src.constants import PATHS
from src.interpolation import interpolate_bureswasserstein
from src.sampling.graphs import sample_graphs
from src.sampling.structures import sample_sbm_structures
from src.utils import lapl_to_weights, square_to_vec, weights_to_lapl

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
    }
)


def plot_curves(t: NDArray, g0: NDArray, g1: NDArray, file_name: str):
    inter_bw = square_to_vec(
        lapl_to_weights(interpolate_bureswasserstein(weights_to_lapl(g0), weights_to_lapl(g1), t))
    )
    g0 = square_to_vec(g0)
    g1 = square_to_vec(g1)
    inter_l2 = t[:, None] * g1[None, ...] - (t - 1)[:, None] * g0[None, ...]

    group_masks = [(g0 == s) & (g1 == e) for s, e in product(np.unique(g0), np.unique(g1))]
    curves_bw = [inter_bw[:, g_mask].mean(1) for g_mask in group_masks]

    _fig, ax = plt.subplots(figsize=(2.5, 2.5), dpi=200)
    ax.plot(t, inter_bw, color="tab:blue", alpha=0.1, linewidth=0.1)
    for cmean in curves_bw:
        l_bw_mean, *_ = ax.plot(t, cmean, c="tab:blue")
    l_l2, *_ = ax.plot(t, inter_l2, c="tab:orange")
    ax.set(xlabel=r"$\tau$", ylabel="Edge weight", xlim=(0, 1))
    ax.legend((l_l2, l_bw_mean), ("Linear", "BW"), loc="upper right")
    plt.savefig(PATHS.figures / file_name, bbox_inches="tight")


def main(
    n_nodes: int,
    avg_edge_prob: float,
    within_mean: float,
    n_blocks: list[int],
    seed=250827,
):
    rng = default_rng(seed)

    g0 = nx.adjacency_matrix(
        nx.barabasi_albert_graph(
            n_nodes,
            int(np.floor((n_nodes - np.sqrt(n_nodes**2 * (1 - 2 * avg_edge_prob))) / 2)),
            seed=rng,
        )
    ).todense()[perm := rng.permutation(n_nodes)[:, np.newaxis], perm.T]

    # Sample SBM endpoints
    g1 = sample_graphs(
        sample_sbm_structures(
            n_targets=1,
            n_nodes=n_nodes,
            n_blocks=n_blocks,
            avg_edge_prob=avg_edge_prob,
            within_mean=within_mean,
            within_scale=0.0,
            seed=rng,
            permuted_sbm=False,
        ),
        sampling_strategy="threshold-iid",
        seed=rng,
    ).squeeze()  # type:ignore

    t = np.linspace(0, 1, num=25)
    plot_curves(t, g0, g1, "sample-geodesic-bw.pdf")


if __name__ == "__main__":
    main(
        n_nodes=50,
        avg_edge_prob=0.3,
        within_mean=0.4,
        n_blocks=[2],
    )
