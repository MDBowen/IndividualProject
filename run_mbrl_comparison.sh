#!/bin/bash
python run_experiment.py \
    --compare_mbrl \
    --timesteps 50_000 \
    --n_trials 3 \
    --only_sp100 True \
    --dynamics_rl_start_episode 1
