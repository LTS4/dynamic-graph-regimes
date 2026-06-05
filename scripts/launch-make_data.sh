#!/usr/bin/bash

opt_list=(
    "+exp=speed ++dataset.seed=1626 ++sequences.n=10 +sequences.choice=speed"
    "+exp=speed ++dataset.seed=1727 ++sequences.n=50 +sequences.choice=speed"
    "+exp=synth ++dataset.seed=1814 ++sequences.n=10 +sequences.choice=full"
    "+exp=synth ++dataset.seed=1861 ++sequences.n=50 +sequences.choice=full"
)

for opts in "${opt_list[@]}"; do
    python -m src.experiments.synthetic.make_data $opts -m
done