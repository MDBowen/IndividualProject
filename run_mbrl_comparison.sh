#!/bin/bash
python run_experiment.py \
    --compare_mbrl \
    --timesteps 100_000 \
    --number_of_runs 3\
    --max_no_improvement_evals 40
    --n_trials 3 \
    --only_sp100 True \
    --dynamics_rl_start_episode 7
