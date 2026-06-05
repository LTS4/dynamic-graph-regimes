from dataclasses import dataclass
from pathlib import Path

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from src.constants import PATHS
from src.experiments.io import get_db_engine, list_datasets_ids


@dataclass
class ExpConf:

    data_root: str
    results_db: str


cs = ConfigStore.instance()
cs.store(name="exp", node=ExpConf)

cs.store(
    group="exp",
    name="synth",
    node=ExpConf(
        data_root=str(PATHS.data / "synthetic"),
        results_db=str(PATHS.results / "experiments-synth.db"),
    ),
)
cs.store(
    group="exp",
    name="speed",
    node=ExpConf(
        data_root=str(PATHS.data / "synthetic-speed"),
        results_db=str(PATHS.results / "experiments-speed.db"),
    ),
)


# Register resolver so dataset ids follow the configured DB path (can be overridden via CLI)
OmegaConf.register_new_resolver(
    "datasets_ids",
    lambda db_path: ",".join(map("'{}'".format, list_datasets_ids(get_db_engine(Path(db_path))))),
)
