from logging import error, info
from pathlib import Path

import hydra
import numpy as np
from joblib import Parallel, delayed
from numpy.random import Generator, default_rng
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sqlalchemy.orm import Session

from src.experiments.io import (
    Changepoints,
    Datasets,
    Sequences,
    SynthDataOptions,
    get_db_engine,
    hash_pickle,
    load_or_call,
    post_to_db,
    save_to_file,
)
from src.experiments.synthetic.config.data import DataConf
from src.sampling.graphs import sample_graphs
from src.sampling.sequences import (
    sample_changepoints,
    sample_geodesic_sequences,
    sample_geodesic_varying_speed,
    sample_static_regimes,
)
from src.sampling.structures import sample_num_blocks, sample_sbm_structures


def make_changepoints(cfg: DataConf, rng: Generator):
    changepoints = sample_changepoints(
        cfg.dataset.change_prob,
        cfg.dataset.n_samples,
        min_change_distance=cfg.dataset.min_change_distance,
        seed=rng.integers(1e8),
    )

    match cfg.sequences.choice:
        case "speed":
            n_targets = 2
        case "full":
            n_targets = changepoints.shape[0]
        case _:
            raise ValueError(f"Unknown sequence choice {cfg.sequences.choice}")

    # Sample numer of blocks
    n_blocks = sample_num_blocks(n_targets, cfg.dataset.block_probs, seed=rng.integers(1e8))

    # Sample SBM endpoints
    sbm_endpoints: NDArray = sample_sbm_structures(
        n_targets=n_targets,
        n_nodes=cfg.dataset.n_nodes,
        n_blocks=n_blocks,
        avg_edge_prob=cfg.dataset.avg_edge_prob,
        within_mean=cfg.dataset.within_mean,
        within_scale=cfg.dataset.within_scale,
        seed=rng.integers(1e8),
        permuted_sbm=cfg.dataset.permuted_sbm,
    )  # type:ignore

    assert np.all((sbm_endpoints >= 0) & (sbm_endpoints <= 1))

    graph_endpoints = sample_graphs(
        sbm_endpoints, sampling_strategy="threshold-iid", seed=rng.integers(1e8)
    )
    assert np.all((graph_endpoints >= 0) & (graph_endpoints <= 1))

    match cfg.sequences.choice:
        case "speed":
            speeds = np.concatenate(
                [[0], rng.uniform(low=0.3, high=1, size=changepoints.shape[0] - 1)]
            )
            speeds /= speeds.sum()
        case _:
            speeds = rng.uniform(low=0.3, high=1, size=n_targets)

    return changepoints, n_blocks, speeds, sbm_endpoints, graph_endpoints


def make_sequences(cfg: DataConf, rng: Generator, seq_d: Path, dataset_id: str):
    """Generate and save sequences according to config `cfg`"""
    # Sample change times based on change probability
    seq_d.mkdir(exist_ok=True)

    data_root = Path(cfg.exp.data_root)
    db_engine = get_db_engine(Path(cfg.exp.results_db))

    rng, rng_child = rng.spawn(2)

    cpts_id = str(seq_d.relative_to(data_root))
    with Session(db_engine) as sess:
        cpts = sess.get(Changepoints, cpts_id)
    if cpts:
        changepoints = np.asarray(cpts.cpts)
        n_blocks = np.asarray(cpts.n_blocks)
        speeds = np.asarray(cpts.partial_speeds)
        sbm_endpoints: NDArray = cpts.sbm_endpoints  # type:ignore
        graph_endpoints: NDArray = cpts.graph_endpoints  # type:ignore
    else:
        changepoints, n_blocks, speeds, sbm_endpoints, graph_endpoints = make_changepoints(
            cfg, rng=rng_child
        )
        post_to_db(
            db_engine,
            Changepoints(
                id=cpts_id,
                cpts=changepoints.tolist(),
                n_blocks=n_blocks.tolist(),
                partial_speeds=speeds.tolist(),
                sbm_endpoints=sbm_endpoints,
                graph_endpoints=graph_endpoints,
            ),
            instance_id=(Changepoints, cpts_id),
        )

    common_pars = {
        "changepoints": changepoints,
        "return_sparse": True,
        "seed": rng.integers(1e8),
    }

    match cfg.sequences.choice:
        case "full":
            sampling_func = sample_geodesic_sequences

            # Sample iid snapshots with regime changes
            static_path = seq_d / "static.npz"
            load_or_call(
                static_path,
                sample_static_regimes,
                changepoints,
                sbm_endpoints,
                return_sparse=True,
                seed=rng.integers(1e8),
            )

            seq_id = str(static_path.relative_to(data_root))

            post_to_db(
                db_engine,
                Sequences(
                    path=seq_id,
                    dataset_id=dataset_id,
                    change_type_id="iid",
                    changepoints_id=cpts_id,
                ),
                instance_id=(Sequences, seq_id),
            )

            sequence_pars_iter = {
                "_".join((level, interpolation, sampling, speed)): {
                    "graph_endpoints": graph_endpoints if level == "graph" else sbm_endpoints,
                    "interpolation_strategy": interpolation,
                    "sampling_strategy": sampling,
                    "speeds": None if speed == "full" else speeds,
                }
                for level in cfg.sequences.interpolation_levels
                for interpolation in cfg.sequences.interpolation_startegies
                for sampling in cfg.sequences.sampling_strategies
                for speed in cfg.sequences.speeds
            }

        case "speed":
            sampling_func = sample_geodesic_varying_speed
            common_pars["speeds"] = speeds
            sequence_pars_iter = {
                "_".join((level, interpolation, sampling)): {
                    "graph_endpoints": graph_endpoints if level == "graph" else sbm_endpoints,
                    "interpolation_strategy": interpolation,
                    "sampling_strategy": sampling,
                }
                for level in cfg.sequences.interpolation_levels
                for interpolation in cfg.sequences.interpolation_startegies
                for sampling in cfg.sequences.sampling_strategies
            }

    for change_type_id, change_type_pars in sequence_pars_iter.items():
        try:
            snapshots_path = seq_d / f"{change_type_id}.npz"
            snapshots = load_or_call(
                snapshots_path, sampling_func, **common_pars, **change_type_pars
            )
            if np.any(np.isnan(snapshots)):
                raise ValueError("NaN values in sampled snapshots")

            seq_id = str(snapshots_path.relative_to(data_root))
            post_to_db(
                db_engine,
                Sequences(
                    path=seq_id,
                    dataset_id=dataset_id,
                    change_type_id=change_type_id,
                    changepoints_id=cpts_id,
                ),
                instance_id=(Sequences, seq_id),
            )

        except:
            error("Error with changepoints %s on %s", cpts_id, change_type_id)
            raise


def make_sequences_ignore_err(cfg: DataConf, rng: Generator, seq_d: Path, dataset_id: str):
    try:
        make_sequences(cfg, rng, seq_d, dataset_id)
    except ValueError as err:
        error(err)


@hydra.main(
    version_base=None,
    config_name="data",
)
def main(cfg: DataConf):
    """Generate synthetic datasets"""
    data_path = Path(cfg.exp.data_root)
    data_id = hash_pickle(OmegaConf.to_object(cfg.dataset))
    info("Creating dataset %s", data_id)

    data_d = data_path / data_id
    data_d.mkdir(parents=True, exist_ok=True)

    save_to_file(OmegaConf.to_container(cfg.dataset), data_d / "config.yaml")

    post_to_db(
        get_db_engine(Path(cfg.exp.results_db)),
        SynthDataOptions(
            id=data_id,
            **OmegaConf.to_container(cfg.dataset, resolve=True),
        ),
        Datasets(
            id=data_id,
            path=str(data_d.relative_to(data_path)),
            options_id=data_id,
        ),
        instance_id=(Datasets, data_id),
    )

    rng = default_rng(cfg.dataset.seed)

    Parallel(n_jobs=-1, prefer="threads")(
        delayed(make_sequences_ignore_err)(cfg, rng_child, data_d / f"{i:0>2d}", data_id)
        for i, rng_child in enumerate(rng.spawn(cfg.sequences.n))
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
