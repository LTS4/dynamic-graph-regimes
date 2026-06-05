import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from scipy.special import expit, logit
from scipy.stats import truncnorm

from src.utils import square_to_vec, vec_to_square


def _change_ego_length(node_list: NDArray, center: int, n_subnodes: int, rng: Generator):
    if n_subnodes < 1:
        return node_list

    n_sampled = node_list.shape[0]
    if n_sampled == n_subnodes:
        return node_list
    elif n_sampled > n_subnodes:
        return np.concatenate(
            [
                rng.choice(node_list[node_list != center], size=n_subnodes - 1, replace=False),
                [center],
            ]
        )
    else:
        # return np.concatenate([node_list, (n_subnodes - n_sampled) * [center]])
        raise ValueError("Not enough nodes")


def random_ego_subgraphs(
    adj: NDArray,
    n_hops: int,
    n_graphs: int = 1,
    n_subnodes=0,
    seed: int | Generator | None = None,
) -> NDArray | list[NDArray]:
    """Sample random ego subgraphs from a graph

    Args:
        adj (NDArray): Adjacency matrix of the graph.
        n_hops (int): Number of hops in ego subgraph.
        n_graphs (int, optional): Number of ego subgraphs to sample. Defaults to 1.
        seed (int | Generator | None, optional): Random seed. Defaults to None.

    Returns:
        NDArray | list[NDArray]: Indices of the nodes in the sampled ego subgraphs
    """
    rng = default_rng(seed)

    n_nodes = adj.shape[0]

    e_i = np.zeros((n_nodes, n_graphs))
    centers = rng.choice(n_nodes, size=n_graphs, replace=False)
    e_i[centers, np.arange(n_graphs)] = 1.0

    for _ in range(n_hops):
        e_i += adj @ e_i

    out = [
        _change_ego_length(np.nonzero(col)[0], center, n_subnodes, rng=rng)
        for col, center in zip(e_i.T, centers)
    ]

    if n_graphs == 1:
        return out[0]
    return out


def sample_graphs(
    structures: NDArray[np.float64],
    sampling_strategy: str,
    mask_low=0.0,
    mask_high=1.0,
    seed: int | Generator = None,
) -> NDArray[np.float64]:
    """Sample a sequence of graphs from given structures.

    Args:
        Structure (NDArray[np.float64]): Edge probabilities or weight means, with shape (n_samples,
            n_nodes, n_nodes).
        sampling_strategy (str): Strategy for sampling the sequence.  Options are:
            - 'iid': use edge weights as probabilities and sample independently
            - 'threshold-fixed': sample a fixed threshold for trajectory weights
            - 'threshold-edge': sample a threshold for each edge
        seed (int | Generator, optional): Random generator seed. Defaults to None.

    Raises:
        ValueError: If the sampling strategy is not defined

    Returns:
        NDArray[np.float64]: Sequence of discrete graphs
    """
    rng = default_rng(seed)

    n_samples, n_nodes, _ = structures.shape
    n_edges = (n_nodes**2 - n_nodes) // 2

    match sampling_strategy.split("-", maxsplit=1):
        case "no", _:
            return structures

        case "threshold", choice:
            match choice:
                case "iid":
                    masks = rng.uniform(mask_low, mask_high, size=(n_samples, n_edges))

                case "fixed":
                    masks = np.array([[0.5]])
                case "fixed-uniform":
                    masks = rng.uniform(mask_low, mask_high, size=(1, 1))
                case "fixed-mean":
                    # NOTE: We use a fixed threshold corresponding to the average edge probability
                    masks = square_to_vec(structures).mean().reshape(-1, 1)

                case "edge-normal":
                    # Bounded normal
                    masks = truncnorm.rvs(-3, 3, loc=0.5, scale=0.5 / 3, size=(1, n_edges))
                case "edge-normal-75":
                    # Bounded normal
                    masks = truncnorm.rvs(-6, 6, loc=0.5, scale=0.5 / 6, size=(1, n_edges))
                case "edge-uniform":
                    masks = rng.uniform(mask_low, mask_high, size=(1, n_edges))

                case "node-uniform-explogit":
                    masks = logit(rng.uniform(mask_low, mask_high, size=(1, n_nodes, 1)))
                    masks = square_to_vec(expit(masks + masks.swapaxes(-2, -1)))
                case "node-uniform-prod":
                    masks = rng.uniform(mask_low, mask_high, size=(1, n_nodes, 1))
                    masks = np.sqrt(square_to_vec(masks @ masks.swapaxes(-2, -1)))
                case "node-uniform-harmonic":
                    masks = rng.uniform(mask_low, mask_high, size=(1, n_nodes, 1))
                    masks = square_to_vec(2 / (1 / masks + 1 / masks.swapaxes(-2, -1)))
                case "node-uniform-mean":
                    masks = rng.uniform(mask_low, mask_high, size=(1, n_nodes, 1))
                    masks = square_to_vec((masks + masks.swapaxes(-2, -1)) / 2)
                case _:
                    raise ValueError(f"Unknown thresholding choice: {choice}.")

            return vec_to_square((masks <= square_to_vec(structures)).astype(float))

        case "additive", choice:
            match choice:
                case "normal-iid":
                    noise = rng.normal(scale=1 / n_nodes, size=(n_samples, n_edges))
                case "normal-iid-degree":
                    noise = rng.normal(
                        scale=structures.mean(),
                        size=(n_samples, n_edges),
                    )
                case _:
                    raise ValueError(f"Unknown additive noise choice: {choice}.")

            return vec_to_square(noise + square_to_vec(structures))

    raise ValueError(f"Unknown sampling strategy: {sampling_strategy}.")


if __name__ == "__main__":
    adj_ = np.diag(np.ones(9, dtype=float), k=1) + np.diag(np.ones(9, dtype=float), k=-1)
    adj_[0, -1] = adj_[-1, 0] = 1.0

    print(random_ego_subgraphs(adj_, 2, seed=10))
    print(random_ego_subgraphs(adj_, 2, seed=10, n_subnodes=3))
    print(random_ego_subgraphs(adj_, 2, seed=10, n_subnodes=8))
    print()
    print(random_ego_subgraphs(adj_, 2, n_graphs=3, seed=11))
    print(random_ego_subgraphs(adj_, 2, n_graphs=3, seed=11, n_subnodes=3))

    sg_idx = np.stack(random_ego_subgraphs(adj_, 2, n_graphs=3, seed=11, n_subnodes=8))
    print(adj_[sg_idx[:, :, np.newaxis], sg_idx[:, np.newaxis, :]])
