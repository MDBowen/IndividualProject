
# Individual Project — Model-Based RL for Stock Trading

## Setup

```bash
pip install -r requirements.txt
```

---

## Running experiments

All experiments are run through `run_experiment.py`.

```bash
python run_experiment.py [OPTIONS]
```

### Quick-start examples

```bash
# Default run — all standard agents, SP100 + NASDAQ + CSI100, 3 trials
python run_experiment.py

# Fast test — SP100 only, shorter training
python run_experiment.py --only_sp100 True --timesteps 50_000 --n_trials 1

# Segmented pipeline — compares standard vs segmented model-based agents
python run_experiment.py --n_segments 4 --only_sp100 True --timesteps 200_000 --n_trials 5

# MBRL mode comparison — frozen / predictive / rl_transfer dynamics
python run_experiment.py --compare_mbrl --only_sp100 True --timesteps 100_000 --n_trials 3
```

Pre-configured versions of the last two are in:

```bash
bash run_segmented_pipeline.sh
bash run_mbrl_comparison.sh
```

---

## Arguments

### Core

| Argument | Default | Description |
|---|---|---|
| `--timesteps` | `100_000` | Total RL environment timesteps per training run |
| `--n_trials` | `3` | Independent trials per agent/dataset (different random ticker subsets) |
| `--number_of_runs` | `1` | Training runs per trial with the same ticker subset |
| `--assets_per_ep` | `10` | Number of assets randomly sampled from the market universe each trial |
| `--only_sp100` | `False` | Restrict to SP100 only (useful for faster iteration) |

### Dynamics model

| Argument | Default | Description |
|---|---|---|
| `--model_epochs` | `10` | Training epochs for the Autoformer / Dense dynamics model |
| `--dynamics_train_mode` | `frozen` | How the dynamics model updates during RL training: `frozen` (locked after pre-training), `predictive` (supervised forecast loss continues), `rl_transfer` (actor-loss gradients flow back into dynamics, TD3 only) |
| `--dynamics_rl_lr` | `1e-5` | Learning rate for dynamics model during RL training |
| `--dynamics_rl_start_episode` | `10` | RL episodes to complete before dynamics training begins |

### Segmented training

| Argument | Default | Description |
|---|---|---|
| `--n_segments` | `0` | Number of temporal segments for segmented training. When `> 0` the agent set switches to include `autoformer_td3_segmented` / `autoformer_ppo_segmented`. Total timesteps are split equally across segments; each segment's dynamics model is trained only on prior segments' data. |

### MBRL comparison mode

| Argument | Default | Description |
|---|---|---|
| `--compare_mbrl` | `False` | Run a structured comparison of all dynamics training modes. Overrides the default agent set with: `td3`, `ppo`, `autoformer_td3_frozen`, `autoformer_ppo_frozen`, `autoformer_td3_predictive`, `autoformer_ppo_predictive`, `autoformer_td3_transfer`, `autoformer_ppo_transfer` |

### Early stopping / evaluation

| Argument | Default | Description |
|---|---|---|
| `--eval_num` | `100` | Total number of evaluations during training (`eval_freq = timesteps / eval_num`) |
| `--n_eval_episodes` | `3` | Episodes per evaluation callback |
| `--max_no_improvement_evals` | `20` | Stop training after this many evaluations with no new best model |
| `--min_evals` | `10` | Minimum evaluations before early stopping can trigger |
| `--verbose` | `1` | Verbosity level for evaluation callbacks |

---

## Default agent set

When run without `--compare_mbrl` or `--n_segments`:

| Agent | Type |
|---|---|
| `buy_and_hold` | Baseline |
| `td3`, `ppo` | Model-free RL |
| `dense_td3`, `dense_ppo` | Model-based RL (MLP dynamics) |
| `dense_predictor` | Prediction-only strategy (MLP) |
| `autoformer_predictor` | Prediction-only strategy (Autoformer) |
| `autoformer_td3`, `autoformer_ppo` | Model-based RL (Autoformer dynamics) |

With `--n_segments > 0`, the agent set changes to `buy_and_hold`, `td3`, `ppo`, `autoformer_td3`, `autoformer_ppo`, `autoformer_td3_segmented`, `autoformer_ppo_segmented`.

With `--compare_mbrl`, the agent set is fixed to the frozen / predictive / rl_transfer variants listed above.
