# Penalty according to AIC is 2*n_params
for exp in speed synth; do
    for feats in graph_invariants prototype_discrepancy; do
        penalties=$(seq -s, 0 5 100 | sed 's/,$//')
        python -m src.experiments.synthetic.detection +exp=$exp \
            +algorithm=binseg-clinear,binseg-l2,binseg-ar \
            +features.name=$feats \
            ++features.normalize=true \
            ++penalty_kw=pen ++penalties="[$penalties]" \
            -m
    done

    feats=isomirror
    penalties=$(seq -s, 0 0.05 5 | sed 's/,$//')
    python -m src.experiments.synthetic.detection +exp=$exp \
        +algorithm=binseg-clinear,binseg-l2,binseg-ar \
        +features.name=$feats \
        ++features.normalize=true \
        ++penalty_kw=pen ++penalties="[$penalties]" \
        -m

    dist=l2
    penalties=$(seq -s, 0 2 100 | sed 's/,$//')
    for ma_size in 0 3; do
        python -m src.experiments.synthetic.detection +exp=$exp \
            +algorithm=binseg-sumvar-$dist \
            +features.name=vec_weights \
            ++features.normalize=false \
            ++penalty_kw=pen ++penalties="[$penalties]" \
            +algorithm.options.ma_size=$ma_size -m
    done

    dist=bw
    penalties=$(seq -s, 0 0.001 0.2 | sed 's/,$//')
    for ma_size in 0 3; do
        python -m src.experiments.synthetic.detection +exp=$exp \
            +algorithm=binseg-sumvar-$dist \
            +features.name=vec_weights \
            ++features.normalize=false \
            ++penalty_kw=pen ++penalties="[$penalties]" \
            +algorithm.options.ma_size=$ma_size \
            ++parallel=false -m
    done


    # LAD benchmark from Huan et al. (2020)
    penalties=$(seq -s, 0.1 0.1 5 | sed 's/,$//')
    python -m src.experiments.synthetic.detection \
        +features.name=vec_weights \
        ++features.normalize=false \
        ++penalty_kw=z_thr ++penalties="[$penalties]" \
        +algorithm=lad \
        +algorithm.options.short_window=5 \
        +algorithm.options.long_window=10,20 \
        +exp=$exp -m
done
