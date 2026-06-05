from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .base import ExpConf


@dataclass
class DatasetConf:
    """General settings for synthetic datasets"""

    n_nodes: int = MISSING
    avg_edge_prob: float = MISSING
    permuted_sbm: bool = MISSING
    seed: int = MISSING

    n_samples: int = 50
    change_prob: float = 0.05
    min_change_distance: int = 10
    block_probs: dict[int, float] = field(
        default_factory=lambda: {
            2: 0.6,
            3: 0.4,
        }
    )
    within_mean: float | None = None
    within_scale: float | None = None


@dataclass
class SequencesConf:
    """Configuration for generated sequences"""

    n: int = MISSING
    choice: str = MISSING

    # Level options: graph, latent
    interpolation_levels: list[str] = field(default_factory=lambda: ["graph"])
    interpolation_startegies: list[str] = field(default_factory=lambda: ["bw", "linear"])
    sampling_strategies: list[str] = field(
        default_factory=lambda: [
            "threshold-fixed",
            "threshold-edge-uniform",
            "threshold-iid",
            "additive-normal-iid",
        ]
    )
    speeds: list[str] = field(default_factory=lambda: ["full", "partial"])


@dataclass
class DataConf:
    """Data generation config"""

    exp: ExpConf

    dataset: DatasetConf = field(default_factory=DatasetConf)
    sequences: SequencesConf = field(default_factory=SequencesConf)

    hydra: Any = field(
        default_factory=lambda: {
            "sweeper": {
                "params": {
                    "+dataset.n_nodes": "100, 200",
                    "+dataset.avg_edge_prob": "0.05, 0.1, 0.2, 0.3",
                    "+dataset.permuted_sbm": "false, true",
                }
            }
        }
    )


cs = ConfigStore.instance()
cs.store(name="data", node=DataConf)
