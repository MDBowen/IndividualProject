from curses import window

import pandas as pd
from typing import Any, ClassVar, TypeVar
import numpy as np
from collections import deque
from sympy import sequence
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from copy import deepcopy

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from stable_baselines3.td3.policies import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
from stable_baselines3.common.buffers import DictReplayBuffer, NStepReplayBuffer, ReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, NStepReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv

from utils.timefeatures import time_features

SelfTD3 = TypeVar("SelfTD3", bound="ModelBasedTD3")

from utils.custom_buffers import Autoformer_Buffer, CustomReplayBuffer
from agents.modelbased_TD3.policies import ModelBasedTD3Policy

class TimestepBuffer:
    def __init__(self, seq_len, n_features):
        self.seq_len = seq_len
        self.n_features = n_features
        self.buffer = deque(maxlen=seq_len)
        self.reset()

    def reset(self):
        self.buffer.clear()
        for _ in range(self.seq_len):
            self.buffer.append(np.zeros(self.n_features, dtype=np.float32))

    def add(self, obs):
        self.buffer.append(np.array(obs, dtype=np.float32).flatten()[:self.n_features])

    def get(self):
        return np.stack(list(self.buffer))  # (seq_len, n_features)
    
    def get_as_batch(self):
        return np.stack(list(self.buffer))[np.newaxis, :]  # (1, seq_len, n_features)


    
    
class ModelBasedTD3(OffPolicyAlgorithm):
    """
    This is a modification of stable baselines3's TD3 which takes some additional params, including a dynamics prediciton modue
      and a FinRL enviroment

    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    TODO - modify below for the updates

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`td3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = { # TODO implement a policy for transformers and autoformers
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
        "ModelBasedMLP": ModelBasedTD3Policy
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: str | type[TD3Policy],
        env: StockTradingEnv,
        dynamics_model: th.nn.Module,
        learning_rate: float | Schedule = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[CustomReplayBuffer] = CustomReplayBuffer,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 1,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        dynamics_kwargs: dict[str, Any] | None = None
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_kwargs=policy_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            
        )

        dynamics_pred_len = dynamics_model.args.pred_len 
        dynamics_seq_len = dynamics_model.args.seq_len
        feature_dim = dynamics_model.args.enc_in

        self.policy_observation_space = spaces.Box(
            low = -1,
            high = 1,
            shape = (self.observation_space.shape[0] + dynamics_pred_len * feature_dim ,)
        )   # We need to redifine this as the policy now takes the concatenation of state, prediciton
        self.actor_observation_space = self.policy_observation_space
        self.critic_observation_space = self.observation_space
        self.dynamics_model = dynamics_model
        self.train_dynamics_model = False
        self.window_buffer = Autoformer_Buffer(max_size=100_000, 
                                               feature_size=feature_dim, 
                                               seq_len=dynamics_seq_len, 
                                               pred_len=dynamics_pred_len, 
                                               label_len=dynamics_model.args.label_len,
                                               freq = self.dynamics_model.args.freq)
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise
        self._last_date = None
        self._last_original_obs = None 

        self.dynamics_model_optim = self.dynamics_model._select_optimizer()
        self.dynamics_model_loss = self.dynamics_model._select_criterion()
    
        if _init_setup_model:
            self._setup_model()
        else: 
            self._setup_model()

        self._create_aliases()

    def get_prices(self, obs):
            stock_dims = self.dynamics_model.args.enc_in
            prices = obs[:, 1:1+stock_dims]
            assert prices.shape == (1,self.dynamics_model.args.enc_in,), f'Prices didnt come out the right shape, stock dim: {self.dynamics_model.args.enc_in}, obs: {obs.shape}, prices: {prices.shape}'
            return prices

    def train_dynamics_model(self, set_train):
        assert isinstance(set_train, bool), 'set_train true or false'
        self.train_dynamics_model = set_train

    # def _setup_model(self) -> None:
    #     super()._setup_model()
    #     self.setup
    #     self._create_aliases()
    #     # Running mean and running var
    #     self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
    #     self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
    #     self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
    #     self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray | dict[str, np.ndarray],
        reward: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        as dates are required, this stores the date from the env, 
        therefore ensure _store_transition needs to is called after step and before resets

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when dones is True)
        :param reward: reward for the current transition
        :param dones: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version

        if self.env.num_envs == 1:
            new_dates = self.env.envs[0]._get_date()
        else:
            new_dates = [self.env.envs[i]._get_date()  for i in range(self.env.num_envs)]

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, self._last_org_date, new_obs_, reward_ = self._last_obs,  self._last_date, new_obs, reward

        # Avoid modification by reference
        next_obs = deepcopy(new_obs_)
        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        for i, done in enumerate(dones):
            if done and infos[i].get("terminal_observation") is not None:
                if isinstance(next_obs, dict):
                    next_obs_ = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs_ = self._vec_normalize_env.unnormalize_obs(next_obs_)
                    # Replace next obs for the correct envs
                    for key in next_obs.keys():
                        next_obs[key][i] = next_obs_[key]
                else:
                    next_obs[i] = infos[i]["terminal_observation"]
                    # VecNormalize normalizes the terminal observation
                    if self._vec_normalize_env is not None:
                        next_obs[i] = self._vec_normalize_env.unnormalize_obs(next_obs[i, :])  # type: ignore[assignment]

        replay_buffer.add(
            self._last_original_obs,  # type: ignore[arg-type
            next_obs,  # type: ignore[arg-type]
            buffer_action,
            reward_,
            dones,
            infos,
            self._last_date
        )

        self._last_obs = new_obs
        self._last_date = new_dates
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses = [], []
        for _ in range(gradient_steps):
            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            # For n-step replay, discount factor is gamma**n_steps (when no early termination)

            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)

                next_obs = replay_data.next_observations

                x, y, x_mark, y_mark = replay_data.next_pred_observations
                pred, _ = self._dynamics_model_predict(x, y, x_mark, y_mark)

                x = th.cat((next_obs, pred.flatten(1)), 1).squeeze()
                assert x.shape[0] == next_obs.shape[0], f'Shapes wrong next_obs: {next_obs.shape}, pred: {pred.flatten(1).shape}, x: {x.shape}' 
                next_actions = (self.actor_target(x) + noise).clamp(-1, 1)
               
                # bellman equation : Q = r + Q'(s,a)

                # Compute the next Q-values: min over all critics targets
                next_obs = replay_data.next_observations
                next_q_values = th.cat(self.critic_target(next_obs, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            # Get current Q-values estimates for each critic network
            obs = replay_data.observations
            current_q_values = self.critic(obs, replay_data.actions)

            # Compute critic loss
            critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            critic_loss = th.Tensor(critic_loss)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                with th.no_grad():
                    x, y, x_mark, y_mark = replay_data.pred_observations
                    pred, _ = self._dynamics_model_predict(x, y, x_mark, y_mark)

                    x = th.cat((next_obs, pred.flatten(1)), 1).squeeze()

                actor_loss = -self.critic.q1_forward(replay_data.observations, self.actor(x)).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                self.actor.optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.critic_batch_norm_stats, self.critic_batch_norm_stats_target, 1.0)
                polyak_update(self.actor_batch_norm_stats, self.actor_batch_norm_stats_target, 1.0)

            # Optimize dynamics model if configured so
            if self.train_dynamics_model:
                x, y, x_mark, y_mark = replay_data.pred_observations
                pred, y = self._dynamics_model_predict(x, y, x_mark, y_mark, scale=False)
                
                pred = pred
                true = y
                loss = self.dynamics_model_loss(pred, true)
                loss.backward()
                self.dynamics_model_optim.step()

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "mb_TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["actor", "critic", "actor_target", "critic_target"]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []
    
    def _dynamics_model_predict(self, x, y, x_mark, y_mark, scale = True):
        reshape_and_scale = lambda transform, _x, shape: th.Tensor(transform(_x.reshape(shape[0] * shape[1], shape[2]).detach().numpy())).reshape(shape[0], shape[1], shape[2]).to(_x.device)

        if self.dynamics_model.args.scale and scale:

            batches = x.shape[0]
            window_x = x.shape[1]
            window_y = y.shape[1]
            feature_dim_x = x.shape[2]
            feature_dim_y = y.shape[2]

            x = reshape_and_scale(self.dynamics_model.scaler.transform, x, (batches, window_x, feature_dim_x))
            y = reshape_and_scale(self.dynamics_model.scaler.transform, y, (batches, window_y, feature_dim_y))

        pred, y = self.dynamics_model._predict(x, y, x_mark, y_mark)

        if self.dynamics_model.args.scale and scale:
            pred = reshape_and_scale(self.dynamics_model.scaler.inverse_transform, pred, pred.shape)

        return pred, y

    
    def _get_prediction(self, x, date):
        ''''x needs to be the prices in obs
            date needs to be the date from the env
        '''
        assert date is not None, 'date needs to be passed in for the time features'
        x= x.squeeze()
        assert len(x.shape) == 1, f'x needs to be 1d array of prices, got {x.shape}'
        self.window_buffer.add(x, date) 
        x, y, x_mark, y_mark = self.window_buffer.get_last(device = self.device)
        pred, _ = self._dynamics_model_predict(x, y, x_mark, y_mark)
        return pred.detach()

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        date: object | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        """
        This overwrites the predict implementation for stable_baslines3 algorithms in BaseAlgorithms

        :param observation: the input observation must be as (batch, features)
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """
        if isinstance(observation, dict):
            print('obs is dict',observation.keys())

        prices = self.get_prices(observation)
        prediction = self._get_prediction(prices, date = date).flatten(1)

        assert not isinstance(observation, dict), 'breakpoint for when observation in predict is dict, i dont think it should be a dict for finrl stockenvs'
        # pred = pred.flatten(1)
        x = np.concatenate((th.tensor(observation), prediction), axis=1)
        # x = th.cat((observation, pred.flatten(1)), 1).squeeze()
        a, b = self.policy.predict(x, None, None, deterministic)

        if len(a.shape) == 1:
            a = np.reshape(a, (1, a.shape[0]))
        return a, b

    def _setup_model(self) -> None:
        '''This is overwritten to redefine the policy's input space'''
        
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
                assert self.n_steps == 1, "N-step returns are not supported for Dict observation spaces yet."
            elif self.n_steps > 1:
                self.replay_buffer_class = NStepReplayBuffer
                # Add required arguments for computing n-step returns
                self.replay_buffer_kwargs.update({"n_steps": self.n_steps, "gamma": self.gamma})
            else:
                self.replay_buffer_class = ReplayBuffer

        if self.replay_buffer is None:
            # Make a local copy as we should not pickle
            # the environment when using HerReplayBuffer
            replay_buffer_kwargs = self.replay_buffer_kwargs.copy()
            if issubclass(self.replay_buffer_class, HerReplayBuffer):
                assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"
                replay_buffer_kwargs["env"] = self.env
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.dynamics_model.args.enc_in,
                self.action_space,
                device=self.device,
                n_envs=self.n_envs,
                optimize_memory_usage=self.optimize_memory_usage,
                **replay_buffer_kwargs,
            )
        
        self.learning_starts = max(self.learning_starts, self.replay_buffer.window)

        self.policy = self.policy_class(
            self.actor_observation_space,
            self.critic_observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])


    def _sample_action(
        self,
        learning_starts: int,
        action_noise: ActionNoise | None = None,
        n_envs: int = 1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample() for _ in range(n_envs)])
        else:
            # Note: when using continuous actions,
            # the policy internally uses tanh to bound the action but predict() returns
            # actions unscaled to the original action space [low, high]
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            assert self._last_obs is not None, "self._last_obs was not set"
            unscaled_action, _ = self.predict(self._last_obs, date = self._last_date, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action


