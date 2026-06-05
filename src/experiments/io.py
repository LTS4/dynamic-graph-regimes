import pickle
from hashlib import sha256
from logging import warning
from pathlib import Path
from time import sleep
from typing import Any, Callable

import numpy as np
import pandas as pd
import sparse

# import torch
import yaml
from sqlalchemy import JSON, Engine, ForeignKey, PickleType, create_engine, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

_rng = np.random.default_rng()


def hash_pickle(obj, size=8) -> str:
    h = sha256()
    h.update(pickle.dumps(obj))
    return h.hexdigest()[:size]


def load_directory(data_p: Path) -> dict[str, Any]:
    """Load all file in a directory and return dictionary of `{stem: content}`"""
    data = {}
    for file in data_p.iterdir():
        try:
            data[file.stem] = load_file(file)
        except IOError:
            data[file.stem] = f"Failed to load {file}"

    return data


def load_file(path: Path) -> Any:
    """Load file of pkl, npy or yaml format."""
    match path.suffix:
        case ".pkl":
            with path.open("rb") as f:
                out = pickle.load(f)
        case ".npy":
            with path.open("rb") as f:
                out = np.load(f)
        case ".npz":
            with path.open("rb") as f:
                out = sparse.load_npz(f)
        case ".yaml":
            with path.open("r", encoding="utf-8") as f:
                out = yaml.safe_load(f)
        # case ".pt":
        #     out = torch.load(path)
        case ".parquet":
            out = pd.read_parquet(path)
        case _:
            raise IOError("Cannot read file")

    return out  # type:ignore


def save_to_file(out, path: Path):
    """Save file in pkl, npy or yaml format."""
    match path.suffix:
        case ".pkl":
            with path.open("bw") as f:
                pickle.dump(out, f)
        case ".npy":
            with path.open("bw") as f:
                np.save(f, out)
        case ".npz":
            with path.open("bw") as f:
                sparse.save_npz(f, out)
        case ".yaml":
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(out, f)
        case ".parquet":
            out.to_parquet(path)
        case _:
            raise ValueError(f"Cannot save file of type {path.suffix}")


def load_or_call(path: Path, func: Callable, *args, **kwargs):
    """Check if file exists at `path` otherwise call function
    with given args and kwargs and return output, which is saved to path
    """
    if path.exists():
        out = load_file(path)
    else:
        out = func(*args, **kwargs)
        save_to_file(out, path)
    return out


################################################################################
# DATABASE
################################################################################


# Store in PATHS.results_db
class DBBase(DeclarativeBase):
    # pylint: disable=too-few-public-methods
    pass


class Algorithms(DBBase):
    """Database for sored algorithms"""

    __tablename__ = "algorithms"

    id: Mapped[str] = mapped_column(primary_key=True)

    source: Mapped[str]
    name: Mapped[str]

    options = mapped_column(JSON, nullable=True)

    results: Mapped[list["Results"]] = relationship(back_populates="algorithm")

    def __repr__(self) -> str:
        return (
            f"Algorithms(id={self.id}, source={self.source}, name={self.name}"
            + f" options={self.options})"
        )


class Results(DBBase):
    """Database table for storing experiments results."""

    __tablename__ = "results"

    id: Mapped[str] = mapped_column(primary_key=True)

    # Predicted

    pred = mapped_column(JSON)

    # Metrics
    tp: Mapped[int]
    fp: Mapped[int]
    fn: Mapped[int]
    hausdorff: Mapped[float]
    rand_index: Mapped[float]

    # Setting

    features: Mapped["Features"] = relationship(back_populates="results")
    features_id = mapped_column(ForeignKey("features.path"))
    features_normalized: Mapped[bool]

    algorithm_id: Mapped[str] = mapped_column(ForeignKey("algorithms.id"))
    algorithm: Mapped["Algorithms"] = relationship(back_populates="results")

    penalty: Mapped[float]

    def __repr__(self) -> str:
        return f"Results(id={self.id}, features_id={self.features_id}, algorithm={self.algorithm})"


class ChangeType(DBBase):
    """Database table for storing types of changes in datasets."""

    __tablename__ = "change_type"
    id: Mapped[str] = mapped_column(primary_key=True)

    sequences: Mapped[list["Sequences"]] = relationship(back_populates="change_type")

    # results: Mapped[list[Results]] = relationship(back_populates="change_type")

    def __repr__(self) -> str:
        return f"ChangeType(id={self.id})"


class Sequences(DBBase):
    """Database table for storing sequences of snapshots."""

    __tablename__ = "sequences"

    path: Mapped[str] = mapped_column(primary_key=True)

    dataset_id = mapped_column(ForeignKey("datasets.id"))
    dataset: Mapped["Datasets"] = relationship(back_populates="sequences")

    change_type_id = mapped_column(ForeignKey("change_type.id"))
    change_type: Mapped[ChangeType | None] = relationship(back_populates="sequences")

    features: Mapped[list["Features"]] = relationship(back_populates="sequence")

    changepoints: Mapped[list["Changepoints"]] = relationship(back_populates="sequences")
    changepoints_id = mapped_column(ForeignKey("changepoints.id"), nullable=True)

    def __repr__(self) -> str:
        return f"Sequences(id={self.path}, path={self.path})"


class Changepoints(DBBase):
    """Table for registering changepoints"""

    __tablename__ = "changepoints"

    id: Mapped[str] = mapped_column(primary_key=True)

    cpts = mapped_column(JSON)
    n_blocks = mapped_column(JSON)
    partial_speeds = mapped_column(JSON)
    sbm_endpoints = mapped_column(PickleType)
    graph_endpoints = mapped_column(PickleType)

    sequences: Mapped[list[Sequences]] = relationship(back_populates="changepoints")


class SynthDataOptions(DBBase):
    """Database table for storing options used to generate synthetic datasets."""

    __tablename__ = "synth_data_options"
    id: Mapped[str] = mapped_column(primary_key=True)

    n_samples: Mapped[int]
    n_nodes: Mapped[int]
    avg_edge_prob: Mapped[float | None]
    block_probs = mapped_column(JSON, nullable=True)  # dict[str, float]
    change_prob: Mapped[float | None]
    min_change_distance: Mapped[int | None]
    permuted_sbm: Mapped[bool | None]
    within_scale: Mapped[float | None]
    within_mean: Mapped[float | None]
    seed: Mapped[int | None]

    dataset: Mapped["Datasets"] = relationship(back_populates="options")

    def __repr__(self) -> str:
        return f"""SynthDataOptions(
    id={self.id},
    n_samples={self.n_samples},
    n_nodes={self.n_nodes},
    avg_edge_prob={self.avg_edge_prob},
    block_probs={self.block_probs},
    change_prob={self.change_prob},
    permuted_sbm={self.permuted_sbm},
    within_scale={self.within_scale},
    within_mean={self.within_mean},
    seed={self.seed}
)"""


class Datasets(DBBase):
    """Database table for storing dataset pointers."""

    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(primary_key=True)

    path: Mapped[str]
    sequences: Mapped[list[Sequences]] = relationship(back_populates="dataset")

    options_id: Mapped[str | None] = mapped_column(
        ForeignKey("synth_data_options.id"), nullable=True
    )
    options: Mapped[SynthDataOptions | None] = relationship(back_populates="dataset")

    # results: Mapped[list[Results]] = relationship(back_populates="dataset")

    def __repr__(self) -> str:
        return f"Datasets(id={self.id}, path={self.path})"


class Features(DBBase):
    """Database table for storing feature extraction settings."""

    __tablename__ = "features"

    path: Mapped[str] = mapped_column(primary_key=True)

    # Settings
    name: Mapped[str]
    cfg_features = mapped_column(JSON, nullable=True)
    cfg_subgraphs = mapped_column(JSON, nullable=True)

    # subgraph_choice: Mapped[str]

    # n_subgraphs: Mapped[int | None]
    # n_subnodes: Mapped[int | None]
    # seed: Mapped[int | None]
    # n_hops: Mapped[int | None]

    # Backtrack
    sequence: Mapped[Sequences] = relationship(back_populates="features")
    sequence_id = mapped_column(ForeignKey("sequences.path"))

    results: Mapped[list[Results]] = relationship(back_populates="features")

    def __repr__(self) -> str:
        return (
            f"Features(path={self.path}, name={self.name}, cfg_features={self.cfg_features},"
            f" cfg_subgraphs={self.cfg_subgraphs})"
        )


def get_db_engine(db_path: Path) -> Engine:
    """Create a database engine for the given path, and create tables if they do not exist."""
    # SQLite needs thread checks disabled for multi-threaded access (e.g., joblib).
    engine = create_engine(
        f"sqlite:///{str(db_path)}",
        echo=False,
        connect_args={"check_same_thread": False, "timeout": 30},
        pool_pre_ping=True,
        pool_recycle=3600,
    )

    if not db_path.is_file():
        DBBase.metadata.create_all(engine)

    return engine


def post_to_db(
    db_engine: Engine,
    *instances,
    instance_id: tuple = None,
    max_sleep=900,
    sleep_factor=1.1,
):
    # Use a short-lived session per operation to avoid cross-thread state.
    local_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def _op():
        with local_session() as sess:
            if instance_id is None or not sess.get(*instance_id):
                sess.add_all(instances)
                sess.commit()

    _with_retry(_op, db_engine, max_sleep=max_sleep, sleep_factor=sleep_factor)


def list_datasets_ids(
    db_engine: Engine, shuffle=True, max_sleep=900, sleep_factor=1.1
) -> list[str]:
    """Retrieve list of dataset ids from database.

    Args:
        db_path (Path): Path to database
        shuffle (bool, optional): Whether to shuffle data, to avoid problems on parallel execution.
            Defaults to True.
        max_sleep (int, optional): Maximum wait time, raise if sleep_time gets higher.
        sleep_factor (_type_, optional): Sleep time between connection trials. Defaults to 1.1

    Raises:
        OperationalError: _description_

    Returns:
        list[str]: _description_
    """
    local_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def _op():
        with local_session() as sess:
            datasets = list(sess.scalars(select(Datasets.id)))
        return _rng.permutation(datasets).tolist() if shuffle else datasets

    return _with_retry(_op, db_engine, max_sleep=max_sleep, sleep_factor=sleep_factor)


def result_exist(res_id, db_engine: Path, max_sleep=900, sleep_factor=1.1) -> bool:
    """Connect to db to check if resukt exists"""
    local_session = sessionmaker(bind=db_engine, expire_on_commit=False)

    def _op():
        with local_session() as sess:
            return bool(sess.get(Results, res_id))

    return _with_retry(_op, db_engine, max_sleep=max_sleep, sleep_factor=sleep_factor)


def _with_retry(fn: Callable[[], Any], db_engine: Engine, max_sleep=900, sleep_factor=1.1):
    """Retry helper for DB ops under concurrent access."""
    sleep_t = _rng.uniform(0.1, 1)
    i = 0
    waited = 0
    while waited < max_sleep:
        try:
            return fn()
        except OperationalError as err:
            if i % 10 == 0:
                warning("DB contention on %s after %d trials: %s", db_engine, i, err)
            sleep(sleep_t)
            waited += sleep_t
            sleep_t *= sleep_factor
            i += 1
    raise IOError(f"Impossible to connect to {db_engine} after {waited}s")
