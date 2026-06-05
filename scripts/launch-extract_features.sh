#!/usr/bin/bash

opt_list=(
    "+subgraphs=global +features=vec_weights"
    "+features=graph_invs"
    "+subgraphs=global +features=proto_dis_bw_4"
    "+subgraphs=global +features=proto_dis_bw_8"
    "+subgraphs=global +features=proto_dis_l1_4 ++n_jobs=16"
    "+subgraphs=global +features=proto_dis_l1_8 ++n_jobs=16"
    "+subgraphs=global +features=isomirror_16_8 ++n_jobs=16"
    "+subgraphs=global +features=isomirror_1_3 ++n_jobs=16"
)

for exp in speed synth; do
    for opts in "${opt_list[@]}"; do
        python -m src.experiments.synthetic.extract_features -m +exp=$exp $opts
    done
done