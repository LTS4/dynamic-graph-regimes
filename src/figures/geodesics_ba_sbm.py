import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.constants import PATHS
from src.sampling.graphs import sample_graphs
from src.sampling.sequences import graph_geodesic
from src.sampling.structures import sample_sbm_structures
from src.visual import plot_adj, plot_graph

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
    }
)


def main(
    n_samples: int,
    n_nodes: int,
    avg_edge_prob: float,
    within_mean: float,
    n_blocks: list[int],
    fig_height,
    seed=250827,
):
    rng = np.random.default_rng(seed)

    start_graph = nx.adjacency_matrix(
        nx.barabasi_albert_graph(
            n_nodes,
            int(np.floor((n_nodes - np.sqrt(n_nodes**2 * (1 - 2 * avg_edge_prob))) / 2)),
            seed=rng,
        )
    ).todense()[perm := rng.permutation(n_nodes)[:, np.newaxis], perm.T]

    # Sample SBM endpoints
    end_graph = sample_graphs(
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

    common_args = dict(
        start_graph=start_graph,
        end_graph=end_graph,
        num=n_samples,
        rcond=1e-10,
    )
    snapshots_lin = graph_geodesic(interpolation_strategy="linear", **common_args)
    snapshots_bw = graph_geodesic(interpolation_strategy="bw", **common_args)

    fig, axes = plt.subplots(
        2,
        n_samples,
        figsize=(fig_height / 2.1 * n_samples, fig_height),
        dpi=200,
        sharex=True,
        sharey=True,
    )
    cmap = "grey_r"
    plot_adj(snapshots_lin, axes=axes[0], cmap=cmap)
    axes[0, 0].set_ylabel("Linear")
    plot_adj(snapshots_bw, axes=axes[1], cmap=cmap)
    for ax in axes.ravel():
        # ax.spines[:].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
    for i, ax in enumerate(axes[1]):
        if i == 0:
            ax.set_ylabel("BW")
            ax.set_xlabel("$G_0$")
        elif i == n_samples - 1:
            ax.set_xlabel("$G_1$")
        else:
            ax.set_xlabel(f"$\\Gamma(G_0, G_1, {i/(n_samples-1):.2f})$")

    fig.tight_layout()
    plt.savefig(PATHS.figures / "geodesics-ba-sbm-adj.pdf", bbox_inches="tight")

    fig, axes = plt.subplots(
        2,
        n_samples,
        figsize=(fig_height / 2.1 * n_samples, fig_height),
        dpi=200,
        sharex=True,
        sharey=True,
    )
    plot_graph(snapshots_lin, axes=axes[0], eps=1e-3)
    axes[0, 0].axis("on")
    axes[0, 0].spines[:].set_visible(False)
    axes[0, 0].set_ylabel("Linear")

    plot_graph(snapshots_bw, axes=axes[1], eps=1e-3)
    for i, ax in enumerate(axes[1]):
        ax.axis("on")
        ax.spines[:].set_visible(False)
        if i == 0:
            ax.set_ylabel("BW")
            ax.set_xlabel("$G_0$")
        elif i == n_samples - 1:
            ax.set_xlabel("$G_1$")
        else:
            ax.set_xlabel(f"$\\Gamma(G_0, G_1, {i/(n_samples-1):.2f})$")

    fig.tight_layout()
    plt.savefig(PATHS.figures / "geodesics-ba-sbm.png", bbox_inches="tight")
    plt.savefig(PATHS.figures / "geodesics-ba-sbm.pdf", bbox_inches="tight")


if __name__ == "__main__":
    N_NODES = 20
    main(
        n_samples=5,
        n_nodes=N_NODES,
        avg_edge_prob=0.3,
        within_mean=0.45,
        n_blocks=[2],
        fig_height=2.5,
    )
