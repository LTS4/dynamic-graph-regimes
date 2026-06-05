"""Feature extraction"""

from pathlib import Path

import hydra
import numpy as np
import sparse
from joblib import Parallel, delayed
from numpy.random import default_rng
from numpy.typing import NDArray
from omegaconf import OmegaConf
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from src import features
from src.experiments.io import (
    Datasets,
    Features,
    get_db_engine,
    hash_pickle,
    load_file,
    load_or_call,
    post_to_db,
)
from src.experiments.synthetic.config.features import (
    FeaturesConf,
    FeaturesFuncConf,
    SubgraphsConf,
)
from src.sampling.graphs import random_ego_subgraphs


def get_subgraphs(
    snapshots: sparse.COO, choice: str | list[str], options: dict | list[dict | None] | None = None
) -> sparse.COO | list[sparse.COO]:
    """Get subgraphs from snapshots based on the specified choice and options.

    Args:
        snapshots (sparse.COO): Sparse COO tensor representing the snapshots.
        choice (str): Choice of subgraph sampling method. Options are:
            - "global": Use the entire snapshot.
            - "ego": Sample ego subgraphs.
            - "iid": Sample independent and identically distributed nodes.
        options (dict): Options for subgraph sampling. The expected keys depend on the choice.

    Returns:
        sparse.COO | list[sparse.COO]: Subgraphs sampled from the snapshots.
    """
    if isinstance(choice, list):
        out = []
        if len(choice) != len(options):
            raise ValueError(
                "If multiple choices are provided, options must be a list of the same length"
            )
        for ch, opts in zip(choice, options):
            subgraphs = get_subgraphs(snapshots, ch, opts)
            out += subgraphs if isinstance(subgraphs, list) else [subgraphs]

        return out

    if options is None:
        options = {}
    elif isinstance(options, list):
        raise ValueError("If choice is a string, options must be a dict or None")

    n_nodes = snapshots.shape[-1]
    rng = default_rng(options.get("seed"))

    match choice:
        case "global":
            return snapshots

        case "ego":
            samples = options["samples"]
            if isinstance(samples, list):
                pass
            elif samples == "random":
                samples = rng.choice(snapshots.shape[0], replace=False, size=options["n_subgraphs"])
            else:
                raise ValueError(f"Invalid sample choice, got: {samples}")

            return [
                sparse.COO.from_numpy(
                    snapshots.todense()[:, sg_idx[:, np.newaxis], sg_idx[np.newaxis, :]]
                )
                for sg_idx in [
                    random_ego_subgraphs(
                        snapshots[sample],
                        n_hops=options["n_hops"],
                        n_subnodes=options["n_subnodes"],
                        seed=rng,
                    )
                    for sample in samples
                ]
            ]

        case "iid":
            # shape: n_subgraphs, n_subnodes
            # NOTE: choice without replacement affect all axes.
            sg_idx = np.stack(
                [
                    rng.choice(
                        n_nodes,
                        size=(options["n_subnodes"]),
                        replace=False,
                    )
                    for _ in range(options["n_subgraphs"])
                ]
            )

            # shape: (n_samples, n_sub, n_subnodes, n_subnodes)
            return sparse.COO.from_numpy(
                snapshots.todense()[:, sg_idx[:, :, np.newaxis], sg_idx[:, np.newaxis, :]]
            )
        case _:
            raise ValueError("Invalid subgraph choice")


def uniform_shape(x: NDArray, subgraph_choice: str):
    match subgraph_choice:
        case "ego":
            return x.transpose(1, 0, 2)
        case _:
            return x


def process_snapshots_features(
    snapshots_p: str,
    cfg_features: FeaturesFuncConf,
    cfg_subgraphs: SubgraphsConf,
    data_path: Path,
    db_engine: Engine,
):
    """Process snapshots to extract features and save them in the database."""
    data_path = Path(data_path)
    snapshots = load_file(data_path / snapshots_p)

    if np.any(np.isnan(snapshots)):
        raise ValueError(f"Invald graph encountered for {snapshots_p}")

    if cfg_features.func == "vec_weights":
        file_p = data_path / snapshots_p
    else:
        try:
            choice = OmegaConf.to_object(cfg_subgraphs.choice)
            choice_str = "+".join(choice)
        except ValueError:
            choice = choice_str = cfg_subgraphs.choice

        cfg_hash = hash_pickle(
            {
                "feats": OmegaConf.to_container(cfg_features),
                "subgraphs": OmegaConf.to_container(cfg_subgraphs),
            }
        )
        file_p: Path = (data_path / "features" / snapshots_p).with_suffix(
            ""
        ) / f"{cfg_hash}_{cfg_features.func}_{choice_str}.npy"

        if not file_p.exists():
            subs = get_subgraphs(
                snapshots,
                choice=choice,
                options=OmegaConf.to_container(cfg_subgraphs).get("options", {}),  # type:ignore
            )

            file_p.parent.mkdir(parents=True, exist_ok=True)

            feats = load_or_call(
                file_p,
                getattr(features, cfg_features.func),
                subs,
                **(cfg_features.options or {}),
            )

            if np.any(np.isnan(feats)):
                raise ValueError(f"Invald features encountered for {file_p}")

    rel_p = str(file_p.relative_to(data_path))
    post_to_db(
        db_engine,
        Features(
            name=cfg_features.func,
            cfg_features=OmegaConf.to_container(cfg_features),
            cfg_subgraphs=OmegaConf.to_container(cfg_subgraphs),
            path=rel_p,
            sequence_id=snapshots_p,
        ),
        instance_id=(Features, rel_p),
    )


@hydra.main(
    version_base=None,
    config_name="features",
)
def main(cfg: FeaturesConf):
    """Main function to extract features from snapshots."""
    data_root = Path(cfg.exp.data_root)
    db_engine = get_db_engine(Path(cfg.exp.results_db))

    with Session(db_engine) as sess:
        dataset = sess.get(
            Datasets,
            cfg.data,
        )
        if dataset is None:
            raise ValueError(f"Dataset {cfg.data} not in database")

        snapshots_paths = [sequence.path for sequence in dataset.sequences]

    if cfg.parallel:
        Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
            delayed(process_snapshots_features)(
                snapshots_p,
                cfg.features,
                cfg.subgraphs,
                data_root,
                db_engine,
            )
            for snapshots_p in snapshots_paths
        )
    else:
        for snapshots_p in snapshots_paths:
            process_snapshots_features(
                snapshots_p, cfg.features, cfg.subgraphs, data_root, db_engine
            )


if __name__ == "__main__":
    # Example configuration
    # pylint: disable=no-value-for-parameter
    main()
