import matplotlib.pyplot as plt
import numpy as np

from src.constants import PATHS
from src.interpolation import interpolate_bureswasserstein
from src.sampling.graphs import sample_graphs
from src.sampling.structures import sbm_structure
from src.utils import lapl_to_weights, weights_to_lapl
from src.visual import plot_graph


def main():
    rng = np.random.default_rng(1640)

    structs = np.stack(
        [
            sbm_structure([3, 6], probs=[[1, 0.05], [0.05, 0.6]]),
            sbm_structure([6, 3], probs=[[0.6, 0.05], [0.05, 1]]),
            sbm_structure([3, 3, 3], probs=[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]),
        ]
    )

    endpoints = sample_graphs(structs, "threshold-iid", seed=rng)
    plot_graph(endpoints, height=1.5)

    geodesic0 = lapl_to_weights(
        interpolate_bureswasserstein(
            weights_to_lapl(endpoints[0]), weights_to_lapl(endpoints[1]), t=np.linspace(0, 1, num=7)
        )
    )
    samples0 = sample_graphs(geodesic0, sampling_strategy="threshold-edge-uniform", seed=rng)

    geodesic1 = lapl_to_weights(
        interpolate_bureswasserstein(
            weights_to_lapl(samples0[3]), weights_to_lapl(endpoints[2]), t=np.linspace(0, 1, num=3)
        )
    )
    samples1 = sample_graphs(geodesic1, sampling_strategy="threshold-edge-uniform", seed=rng)

    fig_dir = PATHS.figures / "geodesic-example"
    fig_dir.mkdir(exist_ok=True)

    for i, adj in enumerate(samples0):
        plot_graph(adj[np.newaxis, ...], height=0.5)
        plt.savefig(fig_dir / f"sample_{i}.pdf", transparent=True)
    for i, adj in enumerate(samples1):
        plot_graph(adj[np.newaxis, ...], height=0.5)
        plt.savefig(fig_dir / f"sample1_{i+3}.pdf", transparent=True)


if __name__ == "__main__":
    main()
