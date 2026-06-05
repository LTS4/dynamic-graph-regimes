"""Covid mobility experiement"""

import datetime

import numpy as np
import pandas as pd
import sparse
import torch
from ruptures import Dynp
from torch_geometric.utils import to_dense_adj
from torch_geometric_temporal.dataset import EnglandCovidDatasetLoader

from src.changepoints import LAD, GraphVariationCost
from src.constants import PATHS
from src.features import isomirror, prototype_discrepancy
from src.utils import square_to_vec


def main(model="rbf", min_size=3):
    """Covid mobility experiment"""
    results_p = PATHS.results / "covid.csv"
    if results_p.exists():
        df = pd.read_csv("results/covid.csv", index_col=0, parse_dates=[1, 2, 3, 4])
    else:
        data = EnglandCovidDatasetLoader().get_dataset(1)
        adjs = np.stack(
            [
                to_dense_adj(torch.tensor(ei), edge_attr=torch.tensor(ew))
                for ei, ew in zip(data.edge_indices, data.edge_weights)
            ]
        ).squeeze()

        # Make symmetric
        adjs += adjs.swapaxes(-1, -2)

        day0 = np.array(datetime.date(2020, 3, 3), dtype=np.datetime64)
        dates = day0 + np.arange(len(adjs), dtype=np.timedelta64)

        train_slice = slice(None, None, None)

        x_tr = adjs[train_slice]
        x_sparse = sparse.COO.from_numpy(x_tr)
        feats_l1 = 2 * prototype_discrepancy(
            square_to_vec(x_sparse)[..., np.newaxis], n_prototypes=4, dist_choice="l1"
        )
        feats_l1 -= feats_l1.mean()
        feats_l1 /= feats_l1.std()

        feats_bw = prototype_discrepancy(x_tr, n_prototypes=4, dist_choice="bw")
        feats_bw -= feats_bw.mean()
        feats_bw /= feats_bw.std()

        model_dict = {
            "LAD": LAD(short_window=3, long_window=12).fit(square_to_vec(x_tr)),
            "Iso-mirror": Dynp(model, jump=1, min_size=min_size).fit(
                isomirror(
                    x_sparse,
                    embedding_dim=1,
                    mds_dim=4,
                    use_procrustes=False,
                    diagaug=False,
                )
            ),
            "Protype diss L1": Dynp(model, jump=1, min_size=min_size).fit(feats_l1),
            "Protype diss BW": Dynp(model, jump=1, min_size=min_size).fit(feats_bw),
            "Graph-RSS L2": Dynp(
                custom_cost=GraphVariationCost(
                    ma_size=0, interpolation="l2", distance="l2", stat="sum", min_size=min_size
                ),
                jump=1,
            ).fit(square_to_vec(x_tr)),
            "Graph-RSS BW": Dynp(
                custom_cost=GraphVariationCost(
                    ma_size=0, interpolation="bw", distance="bw", stat="sum", min_size=min_size
                ),
                jump=1,
            ).fit(square_to_vec(x_tr)),
        }
        results = {}
        for name, model in model_dict.items():
            preds = np.array([0] + model.predict(n_bkps=4))
            print(name, preds)
            preds[-1] -= 1
            results[name] = dates[train_slice][preds]

        df = pd.DataFrame(results)
        df.to_csv(results_p)

    print(df.iloc[1:-1].apply(lambda col: col.dt.strftime("%d %b")).to_latex())


if __name__ == "__main__":
    main()
