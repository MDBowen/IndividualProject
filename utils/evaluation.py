import numpy as np
import torch as th
import pandas as pd
import gymnasium as gym
from collections.abc import Callable
from typing import Any
import warnings


from stable_baselines3.common import type_aliases
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped

from enviroments.test_env_stocktrading import StockTradingEnv

def evaluate_model(
    model: "type_aliases.PolicyPredictor",
    env: StockTradingEnv | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    reward_threshold: float | None = None,
    uses_predictor: bool = False,
) -> tuple[list, list, list, list, list]:
    """
    Evaluate a trained model on the environment and collect trajectories for plotting.

    :param model: The RL agent to evaluate.
    :param env: The StockTradingEnv or VecEnv to evaluate on.
    :param n_eval_episodes: Number of episodes to run.
    :param deterministic: Whether to use deterministic actions.
    :param render: Whether to render the environment.
    :param reward_threshold: If set, asserts mean episode reward exceeds this value.
    :param uses_predictor: If True, also collect dynamics model predictions and true next prices.
    :return: (rewards, actions, observations, predictions, trues)
        - rewards: list of per-episode reward arrays, shape (T,)
        - actions: list of per-episode action arrays, shape (T, action_dim)
        - observations: list of per-episode observation arrays, shape (T, obs_dim)
        - predictions: list of per-episode prediction arrays, shape (T, pred_len, stock_dim); empty if uses_predictor=False
        - trues: list of per-episode true next-price arrays, shape (T, stock_dim); empty if uses_predictor=False
    """
    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    # Patch _get_prediction to intercept predictions without a second buffer update
    last_prediction = [None]
    if uses_predictor:
        _original_get_prediction = model._get_prediction

        def _capturing_get_prediction(x, date):
            pred = _original_get_prediction(x, date)
            last_prediction[0] = pred.detach().cpu().numpy() if hasattr(pred, "cpu") else np.array(pred)
            return pred

        model._get_prediction = _capturing_get_prediction

    all_rewards = []
    all_actions = []
    all_observations = []
    all_predictions = []
    all_trues = []

    episode_rewards = []

    try:
        for _ in range(n_eval_episodes):
            obs = env.reset()
            # DummyVecEnv reset returns (obs, info) in newer gym; handle both
            if isinstance(obs, tuple):
                obs = obs[0]

            done = False
            states = None
            episode_starts = np.ones((env.num_envs,), dtype=bool)

            ep_rewards = []
            ep_actions = []
            ep_observations = []
            ep_predictions = []
            ep_trues = []

            while not done:
                ep_observations.append(obs.copy())

                if uses_predictor:
                    if env.num_envs == 1:
                        date = env.envs[0]._get_date()
                    else:
                        date = [env.envs[i]._get_date() for i in range(env.num_envs)]
                    action, states = model.predict(
                        obs,
                        state=states,
                        date=date,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                    )
                    # last_prediction is populated inside the patched _get_prediction
                    if last_prediction[0] is not None:
                        ep_predictions.append(last_prediction[0].copy())
                else:
                    action, states = model.predict(
                        obs,
                        state=states,
                        episode_start=episode_starts,
                        deterministic=deterministic,
                    )

                ep_actions.append(action.copy())
                new_obs, rewards, dones, infos = env.step(action)
                ep_rewards.append(float(rewards[0]))
                episode_starts = dones

                if uses_predictor:
                    # True values: actual prices in the next observation
                    stock_dim = model.dynamics_model.args.enc_in
                    ep_trues.append(new_obs[:, 1 : 1 + stock_dim].copy())

                if render:
                    env.render()

                done = bool(dones[0])
                obs = new_obs

            all_rewards.append(np.array(ep_rewards))
            all_actions.append(np.squeeze(np.array(ep_actions), axis=1))
            all_observations.append(np.squeeze(np.array(ep_observations), axis=1))
            if uses_predictor:
                all_predictions.append(np.array(ep_predictions))
                all_trues.append(np.squeeze(np.array(ep_trues), axis=1))

            episode_rewards.append(np.sum(ep_rewards))

    finally:
        # Always restore the original method
        if uses_predictor:
            model._get_prediction = _original_get_prediction

    mean_reward = float(np.mean(episode_rewards))
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            f"Mean reward below threshold: {mean_reward:.2f} < {reward_threshold:.2f}"
        )

    return all_rewards, all_actions, all_observations, all_predictions, all_trues




def autoformer_evaluate_policy(
    model: "type_aliases.PolicyPredictor",
    env: gym.Env | VecEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    reward_threshold: float | None = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> tuple[float, float] | tuple[list[float], list[int]]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to perform additional checks,
        called ``n_envs`` times after each step.
        Gets locals() and globals() passed as parameters.
        See https://github.com/DLR-RM/stable-baselines3/issues/1912 for more details.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])  # type: ignore[list-item, return-value]

    is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        
        if env.num_envs == 1:
            new_dates = env.envs[0]._get_date()
        else:
            new_dates = [env.envs[i]._get_date()  for i in range(env.num_envs)]

        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            date = new_dates,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:
                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0

        observations = new_observations

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward