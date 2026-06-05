from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.random import default_rng
from numpy.typing import NDArray
from scipy.linalg import pinvh

from src.constants import PATHS
from src.distance import bw_dist_decomposed, vec_distance
from src.interpolation import interpolate_bureswasserstein
from src.sampling.graphs import sample_graphs
from src.sampling.structures import permuted_sbm_structure
from src.utils import lapl_to_weights, square_to_vec, vec_to_square, weights_to_lapl

plt.rcParams.update(
    {
        "font.family": "serif",
        "text.usetex": True,
    }
)


def main(rcond=1e-8):
    rng = default_rng(1852)

    block_size = 20
    s0, _ = permuted_sbm_structure(
        [2 * block_size, block_size],
        [
            [0.4, 0.1],
            [0.1, 0.4],
        ],
        seed=rng,
    )
    s1, _ = permuted_sbm_structure(
        [block_size, block_size, block_size],
        [
            [0.3, 0.1, 0.1],
            [0.1, 0.3, 0.1],
            [0.1, 0.1, 0.3],
        ],
        seed=rng,
    )

    t = np.linspace(0, 1, num=25)
    graphs = sample_graphs(np.stack([s0, s1]), sampling_strategy="threshold-iid", seed=rng)
    n_nodes = graphs.shape[-1]
    interpolations = {
        "BW": lapl_to_weights(
            interpolate_bureswasserstein(
                weights_to_lapl(graphs[0]),
                weights_to_lapl(graphs[1]),
                t,
                rcond=rcond,
            )
        ),  # type: ignore
        "Linear": vec_to_square(
            t[:, None] * square_to_vec(graphs[[1]]) - (t - 1)[:, None] * square_to_vec(graphs[[0]])
        ),
    }
    inv_interpolations = {
        "BW": pinvh(weights_to_lapl(interpolations["BW"]), atol=rcond),
        "Linear": pinvh(weights_to_lapl(interpolations["Linear"]), atol=rcond),
    }

    distances: dict[tuple[str, str], dict[str, NDArray]] = {
        (dist, interp): {} for dist, interp in product(("BW", "L2"), interpolations.keys())
    }
    for (interp_name, interp), sampling_strategy in product(
        interpolations.items(),
        [
            "additive-normal-iid",
            "threshold-fixed",
            "threshold-edge-uniform",
            "threshold-iid",
        ],
    ):
        samples = np.stack(
            [
                sample_graphs(interp, sampling_strategy=sampling_strategy, seed=rng)
                for _ in range(20)
            ]
        )

        distances[("BW", interp_name)][sampling_strategy] = bw_dist_decomposed(
            inv_interpolations[interp_name][None, ...],
            pinvh(weights_to_lapl(samples), atol=rcond),
            rcond=rcond,
        )
        distances[("L2", interp_name)][sampling_strategy] = np.linalg.norm(
            square_to_vec(interp[None, ...]) - square_to_vec(samples), ord=2, axis=-1
        )

    for (dist_name, inter_name), dists in distances.items():
        handles = []
        labels = []
        _fig, ax = plt.subplots(figsize=(1.5, 1.5), dpi=200)

        palette = iter(sns.color_palette("colorblind"))
        for sampling_strategy, dist in dists.items():
            mean = dist.mean(0)
            std = dist.std(0)
            color = next(palette)
            handles.append(ax.plot(t, mean, label=sampling_strategy, color=color)[0])
            labels.append(sampling_strategy)
            ax.fill_between(t, mean - 2 * std, mean + 2 * std, color=color, alpha=0.1)

        ax.set(xlabel=r"$\tau$", ylabel=f"{dist_name} dist", xlim=(0, 1))
        plt.savefig(
            PATHS.figures / f"distance-geodesic-samples-{dist_name}-{inter_name}.pdf",
            bbox_inches="tight",
        )

        _fig, ax = plt.subplots(figsize=(3, 0.6), dpi=200)
        ax.legend(handles, labels, loc="center", ncol=len(handles) // 2)
        ax.axis("off")
        ax.autoscale(enable=True)
        plt.savefig(
            PATHS.figures / "distance-geodesic-samples-legend-2lines.pdf", bbox_inches="tight"
        )


if __name__ == "__main__":
    main()
