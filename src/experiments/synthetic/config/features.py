from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore

from .base import ExpConf


@dataclass
class FeaturesFuncConf:
    func: str
    options: Any = None  # dict[str, Any]


@dataclass
class IIDOpts:
    n_subgraphs: int
    n_subnodes: int
    seed: int


@dataclass
class EgoOpts:
    """Options for ego graphs"""

    samples: Any  # str | list[int]
    n_hops: int
    n_subgraphs: int | None  # ignored if samples are provided
    n_subnodes: int
    seed: int


@dataclass
class SubgraphsConf:
    choice: Any  # str | list[str]
    options: Any  # None | IIDOpts | EgoOpts


subgraphs_d = {
    "global": SubgraphsConf(choice="global", options=None),
    "iid_10": SubgraphsConf(choice="iid", options=IIDOpts(n_subgraphs=3, n_subnodes=10, seed=54)),
    "iid_20": SubgraphsConf(choice="iid", options=IIDOpts(n_subgraphs=3, n_subnodes=20, seed=54)),
    "ego_1": SubgraphsConf(
        choice="ego",
        options=EgoOpts(samples=[0, 100, 199], n_hops=1, n_subgraphs=None, n_subnodes=-1, seed=54),
    ),
    "ego_2": SubgraphsConf(
        choice="ego",
        options=EgoOpts(samples=[0, 100, 199], n_hops=2, n_subgraphs=None, n_subnodes=-1, seed=54),
    ),
    "ego_1_rand": SubgraphsConf(
        choice="ego",
        options=EgoOpts(samples="random", n_hops=1, n_subgraphs=3, n_subnodes=-1, seed=54),
    ),
    "ego_2_rand": SubgraphsConf(
        choice="ego",
        options=EgoOpts(samples="random", n_hops=2, n_subgraphs=3, n_subnodes=-1, seed=54),
    ),
}

func_d = {
    "graph_invs": FeaturesFuncConf(func="graph_invariants"),
    "vec_weights": FeaturesFuncConf(func="vec_weights"),
    "proto_dis_l1_4": FeaturesFuncConf(
        func="prototype_discrepancy", options={"n_prototypes": 4, "dist_choice": "l1"}
    ),
    "proto_dis_bw_4": FeaturesFuncConf(
        func="prototype_discrepancy", options={"n_prototypes": 4, "dist_choice": "bw"}
    ),
    "proto_dis_l1_8": FeaturesFuncConf(
        func="prototype_discrepancy", options={"n_prototypes": 8, "dist_choice": "l1"}
    ),
    "proto_dis_bw_8": FeaturesFuncConf(
        func="prototype_discrepancy", options={"n_prototypes": 8, "dist_choice": "bw"}
    ),
    "isomirror_16_8": FeaturesFuncConf(
        func="isomirror",
        options={
            "embedding_dim": 16,
            "mds_dim": 8,
            "use_procrustes": False,
            "diagaug": False,
        },
    ),
    "isomirror_1_3": FeaturesFuncConf(
        func="isomirror",
        options={
            "embedding_dim": 1,
            "mds_dim": 3,
            "use_procrustes": False,
            "diagaug": False,
        },
    ),
}


@dataclass
class FeaturesConf:
    """Configuration for features extraction"""

    exp: ExpConf

    data: str
    features: FeaturesFuncConf
    subgraphs: SubgraphsConf = field(
        default_factory=lambda: SubgraphsConf(
            choice=["global", "ego", "ego"],
            options=[
                None,
                EgoOpts(samples="random", n_hops=1, n_subgraphs=3, n_subnodes=-1, seed=54),
                EgoOpts(samples="random", n_hops=2, n_subgraphs=3, n_subnodes=-1, seed=55),
            ],
        )
    )

    parallel: bool = True
    n_jobs: int = -1

    hydra: Any = field(
        default_factory=lambda: {
            "sweeper": {
                "params": {
                    "+data": "${datasets_ids:${exp.results_db}}",
                }
            }
        }
    )


cs = ConfigStore.instance()
cs.store(name="features", node=FeaturesConf)

for name, node in subgraphs_d.items():
    cs.store(group="subgraphs", name=name, node=node)

for name, node in func_d.items():
    cs.store(group="features", name=name, node=node)
