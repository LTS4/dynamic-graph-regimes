from dataclasses import dataclass, field
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from sqlalchemy import Engine, create_engine, func, select

from src.constants import PATHS
from src.experiments.io import (
    Algorithms,
    Datasets,
    Features,
    Results,
    Sequences,
    SynthDataOptions,
    load_or_call,
)

RESULTS_HPS = [
    "algo_name",
    "penalty",
    "feat_name",
    "features_normalized",
]
ALGO_HPS = [
    "model",
    "short_window",
    "long_window",
    "custom_cost",
    "ma_size",
    "interpolation",
    "distance",
]
FEAT_HPS = [
    "feat_dist_choice",
    "feat_n_prototypes",
    "subgraphs_choice",
]
METRICS = ["f1", "tp", "fp", "fn", "rand_index", "hausdorff"]

DATA_PARAMS = ["n_nodes", "avg_edge_prob", "permuted_sbm"]
EXP_COLUMNS = ["dataset_id", "change_type_id", "analysis_key"] + RESULTS_HPS + ALGO_HPS + FEAT_HPS

OUTPUT_COLUMNS = [
    "graph_bw_additive-normal-iid",
    "graph_bw_threshold-edge-uniform",
    "graph_bw_threshold-fixed",
    "graph_bw_threshold-iid",
    "graph_linear_additive-normal-iid",
    "graph_linear_threshold-edge-uniform",
    "graph_linear_threshold-iid",
]


def _json_extract_with_default(json_col, path: str, default=None):
    """Extract JSON value with default for missing keys.

    Args:
        json_col: SQLAlchemy column containing JSON
        path (str): JSON path (e.g., "$.key" or "$.nested.key")
        default: Default value if key missing (defaults to None)

    Returns:
        SQLAlchemy expression for COALESCE(json_extract(...), default)
    """
    return func.coalesce(func.json_extract(json_col, path), default)


def get_synthesis(engine: Engine) -> pd.DataFrame:
    """Fetch and synthesize experimental results from the database.

    Queries the results database and extracts nested algorithm options and feature
    configurations directly at the database level using SQLite JSON functions.
    Returns one result per unique combination of dataset, change type, and hyperparameters.

    Args:
        engine (Engine): SQLAlchemy database engine for querying results.

    Returns:
        pd.DataFrame: Synthesized results with columns:
            - dataset_id, seq_path, change_type_id: Experiment identifiers
            - ALGO_HPS: Algorithm hyperparameter columns
            - FEAT_HPS: Feature hyperparameter columns
            - Aggregated to first result per unique parameter combination
    """
    # Build select statement with JSON extraction at database level
    select_cols = [
        Results,
        Algorithms.name.label("algo_name"),
        Features.name.label("feat_name"),
        Sequences.path.label("seq_path"),
        Sequences.dataset_id,
        Sequences.change_type_id,
    ] + [
        # Extract algorithm options from ALGO_HPS (skip "algo_name" - already extracted)
        _json_extract_with_default(Algorithms.options, f"$.{key}").label(key)
        for key in ALGO_HPS
    ]

    # Extract feature configuration options from FEAT_HPS (skip "feat_name" - already extracted)
    for key in FEAT_HPS:
        if key.startswith("feat_"):
            select_cols.append(
                _json_extract_with_default(
                    Features.cfg_features, f"$.options.{key.replace("feat_","")}"
                ).label(key)
            )
        elif key.startswith("subgraphs_"):
            # Extract subgraphs_choice separately (comes from cfg_subgraphs, not cfg_features)
            select_cols.append(
                _json_extract_with_default(
                    Features.cfg_subgraphs, f"$.{key.replace("subgraphs_","")}"
                ).label("subgraphs_choice")
            )

    stmt = select(*select_cols).where(
        Results.features_id == Features.path,
        Results.algorithm_id == Algorithms.id,
        Features.sequence_id == Sequences.path,
        Sequences.dataset_id == Datasets.id,
    )

    results = pd.read_sql_query(stmt, engine.connect())
    print("N results:", len(results))

    # Post-processing: construct model field from other columns
    results["model"] = (
        results["model"]
        .fillna(
            results["custom_cost"] + "-" + results["interpolation"]
        )  # Do not convert to preserve NaN
        .fillna(results["algo_name"])
    )

    # Fill remaining NULLs with appropriate defaults
    results = results.fillna(results.dtypes.replace({"float64": 0.0, "O": "NULL"})).infer_objects()
    results["subgraphs_choice"] = results["subgraphs_choice"].apply(
        lambda s: "+".join(s) if isinstance(s, list) else s
    )

    return (
        results.groupby(
            [
                "dataset_id",
                "seq_path",
                "change_type_id",
                *RESULTS_HPS,
                *ALGO_HPS,
                *FEAT_HPS,
            ]
        )
        .first()
        .reset_index()
    )


def show_latex(df: pd.DataFrame, metric_choice: str):
    grouped = df.groupby(["analysis_key", "change_type_id"])[metric_choice].median().unstack()
    try:
        grouped = grouped[OUTPUT_COLUMNS]
    except KeyError:
        grouped = grouped[
            [col + "_full" for col in OUTPUT_COLUMNS] + [col + "_partial" for col in OUTPUT_COLUMNS]
        ]

    print(
        grouped.to_latex(float_format="{:.2f}".format)
        .replace("graph_", "")
        .replace("_full", "")
        .replace("Binseg-", "")
        .replace("_", " ")
    )


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:

    results["f1"] = 2 * results["tp"] / (2 * results["tp"] + results["fp"] + results["fn"])

    # This groupby is supposed to aggregate over the different sequences of the same dataframe
    gby = results.groupby(EXP_COLUMNS)

    df = gby[["tp", "fp", "fn"]].sum()
    df["f1"] = 2 * df["tp"] / (2 * df["tp"] + df["fp"] + df["fn"])
    df["rand_index"] = gby["rand_index"].mean()
    df["hausdorff"] = gby["hausdorff"].mean()
    df["algorithm_id"] = gby["algorithm_id"].unique()

    return df


def get_best(group: pd.DataFrame, key: str) -> pd.Series:
    am = np.argmax(group.sort_values(RESULTS_HPS + ALGO_HPS)[key])
    return group.iloc[am]


def extract_best_hps(datasets: pd.DataFrame, results: pd.DataFrame, seed_tr: int, seed_te: int):
    data_tr = datasets.loc[datasets["seed"] == seed_tr].copy()
    data_tr["test_id"] = (
        datasets.loc[datasets["seed"] == seed_te]
        .reset_index()
        .set_index(DATA_PARAMS)
        .loc[pd.MultiIndex.from_frame(data_tr[DATA_PARAMS]), "id"]
        .values
    )

    synthesis_tr: pd.DataFrame = (
        results.loc[datasets.index[datasets["seed"] == seed_tr]]
        .reset_index()
        .groupby(EXP_COLUMNS[:3])
        .apply(get_best, "f1", include_groups=False)
        .sort_index()
    )

    index_map = (
        synthesis_tr.reset_index().replace(data_tr["test_id"]).set_index(results.index.names).index
    )
    synthesis_te = (
        results.loc[pd.MultiIndex.from_tuples(set(index_map) & set(results.index))]
        .reset_index()
        .set_index(synthesis_tr.index.names)
        .sort_index()
    )

    return synthesis_tr, synthesis_te


@dataclass
class AnalysisConf:
    results_db_p: Path
    seed_tr: int
    seed_te: int


@hydra.main(
    version_base=None,
    config_name="analysis",
)
def analysis(cfg: AnalysisConf):
    engine = create_engine(f"sqlite:///{cfg.results_db_p}")

    datasets = pd.read_sql(
        select(
            Datasets,
            SynthDataOptions.n_nodes,
            SynthDataOptions.avg_edge_prob,
            SynthDataOptions.permuted_sbm,
            SynthDataOptions.seed,
        )
        .join(SynthDataOptions)
        .where((SynthDataOptions.seed == cfg.seed_te) | (SynthDataOptions.seed == cfg.seed_tr)),
        engine.connect(),
    ).set_index("id")

    synthesis_p = cfg.results_db_p.with_suffix(".parquet")
    if synthesis_p.exists() and (synthesis_p.stat().st_mtime < cfg.results_db_p.stat().st_mtime):
        synthesis_p.unlink()

    results = load_or_call(synthesis_p, get_synthesis, engine)

    results["analysis_key"] = results[
        ["algo_name", "feat_name", "distance", "feat_dist_choice"]
    ].apply(lambda row: "-".join(map(str, row)).replace("-NULL", ""), axis=1)
    results.sort_values("analysis_key", inplace=True)

    _synthesis_tr, synthesis_te = extract_best_hps(
        datasets, aggregate_results(results), cfg.seed_tr, cfg.seed_te
    )

    print("F1 SCORE")
    show_latex(100 * synthesis_te, "f1")

    print("\nRAND INDEX")
    show_latex(100 * synthesis_te, "rand_index")

    print("\nHAUSDORFF")
    show_latex(synthesis_te, "hausdorff")


cs = ConfigStore.instance()
cs.store(name="analysis", node=AnalysisConf)
cs.store(
    name="synth",
    node=AnalysisConf(
        results_db_p=PATHS.results / "experiments-synth.db",
        seed_tr=1814,
        seed_te=1861,
    ),
)
cs.store(
    name="speed",
    node=AnalysisConf(
        results_db_p=PATHS.results / "experiments-speed.db",
        seed_tr=1626,
        seed_te=1727,
    ),
)


if __name__ == "__main__":
    analysis()
    # print("Endpoint change experiment")
    # analysis(
    #
    # )

    # print(80 * "#", end="\n\n")
    # print("Speed change experiment")
