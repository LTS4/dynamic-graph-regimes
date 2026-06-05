"""Changepoint detection on synthetic data."""

from itertools import product
from pathlib import Path

import hydra
import numpy as np
import ruptures as rpt
import sparse
from joblib import Parallel, delayed
from numpy.typing import NDArray
from omegaconf import OmegaConf
from ruptures import metrics as rptm
from ruptures.base import BaseEstimator
from ruptures.metrics.sanity_check import sanity_check
from sqlalchemy import Engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from tqdm import tqdm

from src.changepoints import (
    LAD,
    BaseCurvatureModel,
    BWBarycenterCost,
    BWCurvaturePeaks,
    BWRegressionCost,
    GraphVariationCost,
    LinearBarycenterCost,
    LinearCurvaturePeaks,
)
from src.experiments.io import (
    Algorithms,
    Changepoints,
    Features,
    Results,
    Sequences,
    get_db_engine,
    hash_pickle,
    load_file,
    post_to_db,
    result_exist,
)
from src.experiments.synthetic.config.detection import AlgoConf, DetectionConf
from src.utils import dictionary_inclusion, square_to_vec


def true_positives(true_bkps, my_bkps, margin=10) -> int:
    """Calculate the number of true positives of an estimated segmentation compared
    with the true segmentation. Adapted from :func:`ruptures.metrics.precision_recall`

    Args:
        true_bkps (list): list of the last index of each regime (true
            partition).
        my_bkps (list): list of the last index of each regime (computed
            partition).
        margin (int, optional): allowed error (in points).

    Returns:
        tuple: (precision, recall)
    """
    sanity_check(true_bkps, my_bkps)
    assert margin > 0, f"Margin of error must be positive (margin = {margin})"

    if len(my_bkps) == 1:
        return 0

    used = set()
    return len(
        set(
            true_b
            for true_b, my_b in product(true_bkps[:-1], my_bkps[:-1])
            if my_b - margin < true_b < my_b + margin and not (my_b in used or used.add(my_b))
        )
    )


def get_algo(cfg: AlgoConf) -> BaseEstimator | BaseCurvatureModel:
    """Instanciate algorithm according to cfg"""
    options: dict = OmegaConf.to_container(cfg.options)  # type:ignore
    match cfg.source:
        case "ruptures":
            match options.pop("custom_cost", None):
                case "bw-regression":
                    return getattr(rpt, cfg.name)(custom_cost=BWRegressionCost(**options))
                case "bw-barycenter":
                    return getattr(rpt, cfg.name)(custom_cost=BWBarycenterCost(**options))
                case "linear-bary":
                    return getattr(rpt, cfg.name)(custom_cost=LinearBarycenterCost(**options))
                case "graph-variation":
                    return getattr(rpt, cfg.name)(custom_cost=GraphVariationCost(**options))
                case _:
                    return getattr(rpt, cfg.name)(**cfg.options)

        case "src":
            match cfg.name:
                case "BWCurvaturePeaks":
                    return BWCurvaturePeaks(**options)
                case "LinearCurvaturePeaks":
                    return LinearCurvaturePeaks(**options)
                case "LAD":
                    return LAD(**options)
        case _:
            raise ValueError(f"Unknown source {cfg.source}")

    raise ValueError(f"Unknown algorithm {cfg.name} for source {cfg.source}")


def prepare_features(feat: Features, data_path: Path) -> NDArray:
    """Load and prepare features for detection"""
    x = load_file(data_path / feat.path)  # type: ignore
    if isinstance(x, sparse.COO):
        x = x.todense()
        if len(x.shape) > 2:
            assert x.shape[-2] == x.shape[-1]
            x = square_to_vec(x)
    elif isinstance(x, dict):
        raise ValueError("Invalide feature type, expected sparse.COO or NDArray")

    return x.reshape(x.shape[0], -1)  # type: ignore


def detection(
    feat: Features, cpts: NDArray, cfg: DetectionConf, data_path: Path, engine_db: Engine
):
    """Run detection algorithm and add results to DB"""
    cpts = cpts[1:]

    x = prepare_features(feat, data_path)

    if cfg.features.normalize:
        x = x - x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        x /= np.where(std > 0, std, 1)

    algo_id = hash_pickle(OmegaConf.to_object(cfg.algorithm))
    run_id = "_".join([hash_pickle(x), algo_id])
    algo: BaseEstimator = get_algo(cfg.algorithm).fit(x)  # type:ignore

    del x

    posted = False
    while not posted:
        try:
            with Session(engine_db) as sess:
                if not sess.get(Algorithms, algo_id):
                    sess.add(
                        Algorithms(
                            id=algo_id,
                            source=cfg.algorithm.source,
                            name=cfg.algorithm.name,
                            options=OmegaConf.to_object(cfg.algorithm.options),
                        )
                    )
                    sess.commit()

            posted = True
        except OperationalError:
            pass

    results: list[Results] = []
    try:
        # for penalty in tqdm(cfg.penalties, desc="Penalty loop", leave=False, ncols=80):
        for penalty in cfg.penalties:
            res_id = f"{run_id}_{penalty}"

            if not result_exist(res_id, engine_db):

                if cfg.penalty_kw:
                    result: list[int] = algo.predict(**{cfg.penalty_kw: penalty})  # type:ignore
                else:
                    result: list[int] = algo.predict(penalty)  # type:ignore

                tp = true_positives(cpts, result)
                fp = len(result) - tp - 1
                fn = len(cpts) - tp - 1

                try:
                    hauss = rptm.hausdorff(cpts, result)
                except ValueError:
                    hauss = cpts[-1]

                rand = rptm.randindex(cpts.tolist(), result)

                results.append(
                    Results(
                        id=res_id,
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        hausdorff=hauss,
                        rand_index=rand,
                        features=feat,
                        features_normalized=cfg.features.normalize,
                        algorithm_id=algo_id,
                        penalty=penalty,
                        pred=result,
                    )
                )
    except:
        if results:
            post_to_db(engine_db, *results)
        raise

    if results:
        post_to_db(engine_db, *results)

    # info("Completed %s", run_id)


def get_data(cfg: DetectionConf, db_engine: Engine) -> tuple[list[Features], list[NDArray]]:
    """Load features and checkpoints according to cfg"""
    stmt = select(Features, Sequences, Changepoints.cpts).where(
        Features.sequence_id == Sequences.path,
        Features.name == cfg.features.name,
        Sequences.dataset_id == cfg.dataset,
        Sequences.changepoints_id == Changepoints.id,
    )

    with Session(db_engine) as sess:
        return list(
            zip(
                *[
                    (row[0], np.asarray(row[2]))
                    for row in sess.execute(stmt)
                    if not cfg.features.options
                    or dictionary_inclusion(
                        row[0].options, OmegaConf.to_object(cfg.features.options)
                    )
                ]
            )
        )  # type:ignore


@hydra.main(
    version_base=None,
    config_name="detection",
)
def main(cfg: DetectionConf):
    """Run changepoint detection"""
    # Create/check for experiment database, w/ schema:
    #     data_code, change_type, feat_choice,   method, cost, penalty, metric1, ...
    # eg: 5d2ade15,  bw_full,     local_invs, Pelt,   rbf,  1.5,     0.6,
    data_root = Path(cfg.exp.data_root)
    db_engine = get_db_engine(Path(cfg.exp.results_db))

    feats, changepoints = get_data(cfg, db_engine)

    if cfg.parallel:
        # Launch detection using multiprocessing
        for _ in tqdm(
            Parallel(n_jobs=16, prefer="threads", return_as="generator_unordered")(
                delayed(detection)(feat, cpts, cfg, data_root, db_engine)
                for feat, cpts in zip(feats, changepoints)
            ),
            desc="Detection",
            leave=False,
            ncols=80,
            total=len(feats),
        ):
            pass
    else:
        for feat, cpts in tqdm(
            zip(feats, changepoints), desc="Detection", leave=False, ncols=80, total=len(feats)
        ):
            detection(feat, cpts, cfg, data_root, db_engine)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
