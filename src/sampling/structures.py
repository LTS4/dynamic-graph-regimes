"""Functions to generate graphs an dinterpolate between them"""

from itertools import product
from logging import warning
from typing import Iterable

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import ArrayLike, NDArray

################################################################################
# SBM structure sampling


def sbm_structure(sizes: list[int], probs: list[list[float]]) -> NDArray[np.float64]:
    """Create a matrix with edge probabilities between nodes for
    a Stochastic Block Model (SBM)

    Args:
        sizes (list[int]): Number of nodes in each block
        probs (list[list[float]]): Probabilities of edges between blocks

    Returns:
        NDArray[np.float64]: Structure matrix
    """
    n_nodes = np.sum(sizes)
    out = np.empty((n_nodes, n_nodes), float)
    starts = np.concatenate([[0], np.cumsum(sizes, dtype=int)])
    indices = [np.arange(start, start + size)[:, np.newaxis] for start, size in zip(starts, sizes)]

    for i, j in product(range(len(sizes)), range(len(sizes))):
        out[indices[i], indices[j].T] = probs[i][j]

    np.fill_diagonal(out, 0)
    return out


def permuted_sbm_structure(
    block_sizes: ArrayLike,
    probs: ArrayLike,
    seed: int | Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Sample a structure matrix for a Stochastic Block Model (SBM)

    Args:
        n_blocks (int): Number of blocks
        probs (ArrayLike): Probabilities of edges between blocks
        block_sizes (ArrayLike | None, optional): Block sizes.
        seed (int | None | np.random.Generator, optional): Random seed. Defaults to None.

    Raises:
        ValueError: If block weights do not sum to 1

    Returns:
        tuple[NDArray, NDArray]: Structure matrix and node partition between blocks
    """
    probs = sbm_structure(block_sizes, probs)

    rng = default_rng(seed)
    node_perm = rng.permutation(np.sum(block_sizes))

    sbm = probs[node_perm[:, None], node_perm[None, :]]
    np.fill_diagonal(sbm, 0)

    return sbm, node_perm


def sample_sbm_probs(
    avg_prob: float,
    block_sizes: ArrayLike,
    within_mean: float = 0.5,
    within_scale: float = 0.025,
    seed: int | Generator | None = None,
) -> NDArray[np.float64]:
    """Sample probabilities for a Stochastic Block Model (SBM) with a given average
    degree and block sizes

    Args:
        avg_prob (float): Average edge probability (`avg_edge_prob / n_nodes`)
        block_sizes (list[int]): Number of nodes in each block
        seed (Generator, optional): Random seed. Defaults to None.

    Returns:
        NDArray[np.float64]: Probabilities
    """
    if len(block_sizes) < 2:
        return np.array([[avg_prob]])

    rng = default_rng(seed)
    block_sizes = np.array(block_sizes)

    n_nodes = block_sizes.sum()

    # within_prob = max(min(rng.normal(within_mean, within_scale), 1), 0)
    within_prob = min(within_mean + rng.exponential(within_scale), 1)

    between_prob = (
        n_nodes**2 * avg_prob + within_prob * (n_nodes - np.sum(block_sizes**2))
    ) / (n_nodes**2 - np.sum(block_sizes**2))

    if not between_prob > 0:
        warning(
            f"Between-block probability dropped to zero with within_prob={within_prob},"
            " resampling. Consider reducing `within_scale` if stuck."
        )

        return sample_sbm_probs(avg_prob, block_sizes, within_mean, within_scale, seed=rng)

    n_blocks = len(block_sizes)
    probs = between_prob * np.ones((n_blocks, n_blocks), float)
    np.fill_diagonal(probs, within_prob)

    return probs


if __name__ == "__main__":
    # print("sbm_structure(size, probs)")
    # print(
    #     sbm_structure(
    #         sizes=[5, 5, 3],
    #         probs=[[0.3, 0.1, 0.1], [0.1, 0.3, 0.1], [0.1, 0.1, 0.3]],
    #     )
    # )
    # print()
    BLOCK_SIZES = [5, 5, 3]
    SEED = 13

    print(f"sample_sbm_probs(0.2, {BLOCK_SIZES}, 0.2, 0.4)")
    probs_ = sample_sbm_probs(0.2, BLOCK_SIZES, 0.2, 0.4, seed=SEED)
    print(np.round(probs_, 2))
    print(
        f"E[deg] = {0.2 * np.sum(BLOCK_SIZES)} ->",
        f"{np.sum(sbm_structure(BLOCK_SIZES, probs_)) / np.sum(BLOCK_SIZES):.3f}",
    )


def sample_sbm_structures(
    n_targets: int,
    n_nodes: int,
    n_blocks: Iterable[int],
    avg_edge_prob: float,
    within_mean: float = None,
    within_scale: float = None,
    seed: int | Generator = None,
    return_block_size=False,
    permuted_sbm=True,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], list[NDArray]]:
    """Sample multiple independent SBMs.

    Args:
        n_targets (int): Number of targets to sample.
        n_nodes (int): Number of nodes in each target.
        n_blocks (dict[int, float]): Probabilities of different block sizes.
        avg_edge_prob (float): Average edge probability.
        within_mean (float, optional): Mean of the within-block edge probabilities.
            Defaults to None, which means `avg_edge_prob / (n_nodes - 1)`.
        within_scale (float, optional): Scale of the within-block edge probabilities.
            Defaults to None, which means `within_mean / within_scale_factor`.
        seed (int | Generator, optional): Random seed or generator. Defaults to None.
        return_block_size (bool, optional): Whether to return the block sizes. Defaults to False.
        permuted_sbm (bool, optional): Whether to permute the SBM structure. Defaults to True.

    Returns:
        NDArray[np.float64]: Tensor with SBMs probabilities of size `(n_targets, n_nodes, n_nodes)`
    """
    # avg_edge_prob = avg_degree / (n_nodes - 1)
    within_mean = within_mean or avg_edge_prob

    if within_mean < 0:
        raise ValueError("within_mean must be greater than 0")

    # if (within_mean + 3 * within_scale) > 1:
    #     raise ValueError("within_scale must be smaller than 1")
    # if within_mean > within_scale:
    #     raise ValueError("within_mean must be smaller than within_scale")

    rng = default_rng(seed)

    targets = np.empty((n_targets, n_nodes, n_nodes))

    block_sizes = []
    for i, nb in enumerate(n_blocks):
        block_sizes.append(1 + rng.multinomial(n_nodes - nb, np.ones(nb, dtype=float) / nb))

        probs = sample_sbm_probs(
            avg_edge_prob,
            block_sizes=block_sizes[-1],
            within_mean=within_mean,
            within_scale=(within_mean / nb) if within_scale is None else within_scale,
            seed=rng,
        )
        if permuted_sbm:
            targets[i] = permuted_sbm_structure(
                block_sizes=block_sizes[-1],
                probs=probs,
                seed=rng,
            )[0]
        else:
            targets[i] = sbm_structure(
                sizes=block_sizes[-1],
                probs=probs,
            )

    if return_block_size:
        return targets, block_sizes
    else:
        return targets


def sample_num_blocks(n_targets: int, block_probs: dict[int, float] = None, seed=None):
    """Sample number of blocs for SBM,
    either as geometric, or as choice w/ given probabilities
    """
    rng = default_rng(seed)
    if block_probs is None:
        return rng.geometric(p=0.3, size=n_targets)
    else:
        return rng.choice(list(block_probs.keys()), size=n_targets, p=list(block_probs.values()))
    # print(sample_sequence(100, 10, change_prob=0.05, seed=10))
