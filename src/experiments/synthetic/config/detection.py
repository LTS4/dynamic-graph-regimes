from dataclasses import dataclass, field
from typing import Any, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from .base import ExpConf


@dataclass
class FeatConf:
    """Features config"""

    name: str = MISSING
    options: Any = None
    normalize: bool = False


@dataclass
class AlgoConf:
    """Base algorithm config"""

    source: str = MISSING
    name: str = MISSING
    options: dict = MISSING


@dataclass
class DetectionConf:
    """Config for changepoints detection experiments"""

    exp: ExpConf

    dataset: str = MISSING

    algorithm: AlgoConf = field(default_factory=AlgoConf)

    penalty_kw: Optional[str] = None
    penalties: list[float] = MISSING

    features: FeatConf = field(default_factory=FeatConf)

    parallel: bool = False

    hydra: Any = field(
        default_factory=lambda: {
            "sweeper": {
                "params": {
                    "+dataset": "${datasets_ids:${exp.results_db}}",
                }
            }
        }
    )


cs = ConfigStore.instance()
cs.store(name="detection", node=DetectionConf)

# It seems that only clinear is worth exploring (compared to l2, rbf, linear, ar)
# Also, binseg always outperforms Pelt
for model in ["clinear", "l2", "ar"]:
    cs.store(
        group="algorithm",
        name=f"binseg-{model}",
        node=AlgoConf(source="ruptures", name="Binseg", options={"model": model}),
    )
    cs.store(
        group="algorithm",
        name=f"pelt-{model}",
        node=AlgoConf(source="ruptures", name="Pelt", options={"model": model}),
    )

for distance in ["l2", "bw"]:
    for stat in ["mean", "sum"]:
        cs.store(
            group="algorithm",
            name=f"binseg-{stat}var-{distance}",
            node=AlgoConf(
                source="ruptures",
                name="Binseg",
                options={
                    "custom_cost": "graph-variation",
                    "ma_size": MISSING,
                    "interpolation": distance,
                    "distance": distance,
                    "stat": stat,
                },
            ),
        )

cs.store(
    group="algorithm",
    name="binseg-bw-barycenter",
    node=AlgoConf(
        source="ruptures",
        name="Binseg",
        options={
            "custom_cost": "bw-barycenter",
            "context_size": MISSING,
            "n_center": None,
            "center_choice": MISSING,
            "distance_choice": MISSING,
        },
    ),
)

cs.store(
    group="algorithm",
    name="bw-curvature",
    node=AlgoConf(
        source="src",
        name="BWCurvaturePeaks",
        options={
            "ma_size": MISSING,
            "window_size": MISSING,
            "curvature_smoothing": MISSING,
        },
    ),
)

cs.store(
    group="algorithm",
    name="linear-curvature",
    node=AlgoConf(
        source="src",
        name="LinearCurvaturePeaks",
        options={
            "ma_size": MISSING,
            "window": MISSING,
            "window_size": MISSING,
            "curvature_smoothing": MISSING,
            "distance": MISSING,
        },
    ),
)

cs.store(
    group="algorithm",
    name="lad",
    node=AlgoConf(
        source="src",
        name="LAD",
        options={
            "short_window": MISSING,
            "long_window": MISSING,
            "num_k": None,
            "typical_choice": "principal",
        },
    ),
)
