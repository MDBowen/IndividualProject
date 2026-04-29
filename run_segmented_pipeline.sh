#!/bin/bash
python run_experiment.py \
    --timesteps 200_000 \
    --n_trials 8 \
    --only_sp100 True \
    --max_no_improvement_evals 40 \
    --eval_num 100