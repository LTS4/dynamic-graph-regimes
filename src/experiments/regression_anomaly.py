from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from src.distance import bw_dist
from src.regression import ot_network_regression
from src.utils import lapl_to_weights, vec_to_square, weights_to_lapl


def get_sequence(name: str, sequence_pars: dict) -> NDArray:
    raise NotImplementedError


def noisy_realization(x: NDArray, p_add: float, p_drop: float, rng: Generator):
    n_samples, n_nodes, _ = x.shape

    obs_noise = vec_to_square(rng.uniform(size=(n_samples, (n_nodes**2 - n_nodes) // 2))) < x
    drop_noise = vec_to_square(rng.uniform(size=(n_samples, (n_nodes**2 - n_nodes) // 2)) < p_drop)
    add_noise = vec_to_square(rng.uniform(size=(n_samples, (n_nodes**2 - n_nodes) // 2)) < p_add)
    return (obs_noise & ~drop_noise | add_noise).astype(float)


@dataclass
class Config:
    n_trials: int

    p_add: float
    p_drop: float

    timestamps: NDArray[np.int64]
    regr_slice_d: dict[str, list[int]]

    rbf_gamma: float
    rcond: float

    seed: int

    return_samples_trial: Optional[int] = None
    return_lapls_trial: Optional[int] = None
    return_kernel_trial: Optional[int] = None


def regression_main(sequence: NDArray, cfg: Config):
    rng = np.random.default_rng(cfg.seed)

    # sequence: NDArray = get_sequence(...)

    results = {
        key: {
            "dist_samples": np.empty((cfg.n_trials, len(cfg.timestamps))),
            "dist_sequence": np.empty((cfg.n_trials, len(cfg.timestamps))),
        }
        for key in cfg.regr_slice_d.keys()
    }
    out = {"results": results}

    for trial in range(cfg.n_trials):
        noisy_sequence: NDArray = noisy_realization(sequence, cfg.p_add, cfg.p_drop, rng)

        disconnected_set = set(np.where(np.isclose(noisy_sequence.sum(1), 0).any(1))[0].tolist())

        regr_slice_d = {
            key: sorted(list(set(slice_list) - disconnected_set))
            for key, slice_list in cfg.regr_slice_d.items()
        }

        # From now on we consider Laplacians
        noisy_sequence = weights_to_lapl(noisy_sequence)

        if cfg.return_samples_trial == trial:
            out["samples"] = noisy_sequence

        if cfg.return_lapls_trial == trial:
            out["lapls"] = {}
        if cfg.return_kernel_trial == trial:
            out["kernel"] = {}

        for key, regr_slice in regr_slice_d.items():
            lapls, kernel = ot_network_regression(
                x_in=cfg.timestamps[regr_slice],
                laplacians=noisy_sequence[regr_slice],
                x_out=cfg.timestamps,
                retun_kernel=True,
                kernel="rbf",
                kernel_pars={"gamma": cfg.rbf_gamma},
                rcond=cfg.rcond,
            )

            dist_samples = np.abs(bw_dist(lapls, noisy_sequence, rcond=cfg.rcond))
            dist_sequence = np.abs(bw_dist(lapls, weights_to_lapl(sequence), rcond=cfg.rcond))

            results[key]["dist_samples"][trial] = dist_samples
            results[key]["dist_sequence"][trial] = dist_sequence

            if cfg.return_lapls_trial == trial:
                out["lapls"][key] = kernel
            if cfg.return_kernel_trial == trial:
                out["kernel"][key] = kernel

    return out
