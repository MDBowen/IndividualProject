import numpy as np
import torch
import pandas as pd
from collections import deque
from collections.abc import Generator

import warnings
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)

from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import BaseBuffer
from utils.timefeatures import time_features, time_features_from_frequency_str

from stable_baselines3.common.preprocessing import get_obs_shape


class ModelBasedSample(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    pred_observations: tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
    next_pred_observations: tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]
    dones: th.Tensor
    rewards: th.Tensor
    # For n-step replay buffer
    discounts: th.Tensor | None = None

class MBRolloutSample(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor
    model_based_obs: th.Tensor
    pred_sequences: tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor] | None


class Autoformer_Buffer:
    def __init__(self,  max_size, feature_size, seq_len, pred_len, label_len, freq = 'D'):
        self.prices = deque(maxlen=max_size)
        self.dates = deque(maxlen=max_size)
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.pred_len = pred_len
        self.max_size = max_size
        self.label_len =  label_len
        self.freq = freq
        self.reset()
        
    def reset(self):
        self.prices.clear()
        self.dates.clear()

        for _ in range(self.max_size):
            self.prices.append(np.zeros(self.feature_size, dtype=np.float32))
            self.dates.append('1900-01-01')

    def add(self, x, date):

        assert isinstance(x, np.ndarray) and x.shape[0] == self.feature_size, f'input x needs to be a numpy array of shape (feature_size,) is instead {x}'

        self.prices.append(x)
        self.dates.append(date)

    def get_all(self):
        return list(self.prices), list(self.dates)
    
    def get_size(self):
        return len(list(self.prices))

    def get_last(self, device = 'cpu'):

        x = list(self.prices)[-self.seq_len:]
        y_label = list(self.prices)[-self.label_len:]

        x_dates = list(self.dates)[-self.seq_len:]
        y_dates = list(self.dates)[-self.label_len:]

        pred_dates = pd.date_range(y_dates[-1], periods=self.pred_len + 1, freq=self.freq)

        y_stamp = y_dates + list(pred_dates)[1:]
        
        x_mark = time_features(pd.to_datetime(x_dates), freq=self.freq).transpose(1, 0)
        y_mark = time_features(pd.to_datetime(y_stamp), freq=self.freq).transpose(1, 0)

        x = th.Tensor(x)
        y_label = th.Tensor(y_label)
        x_mark = th.Tensor(x_mark)
        y_mark = th.Tensor(y_mark)

        dec_inp = th.zeros(( self.pred_len, self.feature_size)).float()
        dec_inp = th.cat([y_label, dec_inp], dim = 0).float().to(device)

        x = x.reshape(1, self.seq_len, self.feature_size).float().to(device)
        dec_inp = dec_inp.reshape(1, self.label_len + self.pred_len, self.feature_size).float()
        x_mark = x_mark.reshape(1, self.seq_len, x_mark.shape[-1]).float().to(device)
        y_mark = y_mark.reshape(1, self.label_len + self.pred_len, y_mark.shape[-1]).float().to(device)
        
        return x, dec_inp, x_mark, y_mark


class SequenceReplayBuffer:
    def __init__(self, buffer_size, seq_len, pred_len, n_features, n_actions, device="cpu"):
        self.buffer_size = buffer_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.window = seq_len + pred_len  # total steps needed per sample
        self.n_features = n_features
        self.device = device
        self.pos = 0
        self.full = False

        self.observations = np.zeros((buffer_size, n_features), dtype=np.float32)
        self.next_observations = np.zeros((buffer_size, n_features), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_actions), dtype=np.float32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = np.zeros_like(obs)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(np.array(done).flatten()[0])
        self.pos = (self.pos + 1) % self.buffer_size
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        max_idx = self.buffer_size if self.full else max(self.pos, self.window + batch_size )

        indices = np.random.randint(self.window, max_idx, size=batch_size)

        # Input sequence: (batch_size, seq_len, n_features)
        obs_seq = np.stack([
            self.observations[i - self.window : i - self.pred_len] for i in indices
        ])

        # Target sequence: (batch_size, pred_len, n_features)
        target_seq = np.stack([
            self.observations[i - self.pred_len : i] for i in indices
        ])

        # Next obs sequence for TD learning: (batch_size, seq_len, n_features)
        next_obs_seq = np.stack([
            self.next_observations[i - self.window : i - self.pred_len] for i in indices
        ])

        return dict(
            observations=torch.tensor(obs_seq, dtype=torch.float32).to(self.device),
            next_observations=torch.tensor(next_obs_seq, dtype=torch.float32).to(self.device),
            targets=torch.tensor(target_seq, dtype=torch.float32).to(self.device),
            actions=torch.tensor(self.actions[indices], dtype=torch.float32).to(self.device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device),
            dones=torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device),
        )

    @property
    def size(self):
        return self.buffer_size if self.full else self.pos



class CustomReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    However this is modified for a transformer-model-based algorithms

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    pred_observations: np.ndarray
    next_observations: np.ndarray
    next_pred_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        price_dim: int,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        batch_size: int = 100,
        prediction_window: int = 96,
        label_len: int = 48,
        prediction_horizon: int = 24,
        dynamic_model_freq: str = 'D'
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
        self.pred_observations = np.zeros((self.buffer_size, self.n_envs, price_dim), dtype = observation_space.dtype)
        self.dates = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.batch_size = batch_size
        self.seq_len = prediction_window
        self.label_len = label_len
        self.pred_len = prediction_horizon
        self.window = self.seq_len + self.pred_len
        self.freq = dynamic_model_freq

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
            self.next_pred_observations = np.zeros((self.buffer_size, self.n_envs, price_dim), dtype=observation_space.dtype)
            self.next_dates = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

        self.price_dims = price_dim

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
        dates: list[float] | None = None,
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        # assert dates is not None, 'dates need to be passed in for the add function of model based td3'

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)
        self.dates[self.pos] = dates

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: VecNormalize | None = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            # return super().sample(batch_size=batch_size, env=env)
            upper_bound = self.buffer_size if self.full else self.pos
            batch_inds = np.random.randint(self.window, upper_bound, size=batch_size)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(self.window, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(self.window, self.pos - self.pred_len - 1 , size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ModelBasedSample:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        x_batch_dates =[
            self.dates[i - self.seq_len : i, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]
        y_batch_dates = [
            self.dates[i - self.label_len : i + self.pred_len, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]
        
        next_x_dates = [ 
            self.dates[i - self.seq_len + 1 : i + 1, env_indices[j]]  for j, i in enumerate(batch_inds)
            ] 

        next_y_dates = [
            self.dates[i - self.label_len + 1: i + self.pred_len + 1 , env_indices[j]]  for j, i in enumerate(batch_inds)
            ]

        x_mark_batch = [time_features(pd.to_datetime(x_dates), freq=self.freq).transpose(1, 0) for x_dates in x_batch_dates]
        y_mark_batch = [time_features(pd.to_datetime(y_dates), freq=self.freq).transpose(1, 0) for y_dates in y_batch_dates]
        next_x_mark_batch = [time_features(pd.to_datetime(x_dates), freq=self.freq).transpose(1, 0) for x_dates in next_x_dates]
        next_y_mark_batch = [time_features(pd.to_datetime(x_dates), freq=self.freq).transpose(1, 0) for x_dates in next_y_dates]

        
        x = self._normalize_obs( np.stack([
            self.observations[i - self.seq_len : i, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)
        y = self._normalize_obs( np.stack([
            self.observations[i - self.label_len : i + self.pred_len, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)
        next_x = self._normalize_obs( np.stack([
            self.next_observations[i - self.seq_len : i, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)

        next_y = self._normalize_obs( np.stack([
            self.next_observations[i - self.label_len : i + self.pred_len, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)

        reshape = lambda x: x[:, :, 1:1+self.price_dims]
        x, y, next_x, next_y = tuple(map(reshape, [x, y, next_x, next_y]))

        pred_x = th.tensor(x, dtype=th.float32, device=self.device)
        pred_y = th.tensor(y, dtype=th.float32, device=self.device)
        next_pred_x = th.tensor(next_x, dtype=th.float32, device=self.device)
        next_pred_y = th.tensor(next_y, dtype=th.float32, device=self.device)
        x_mark_batch = np.array(x_mark_batch)
        y_mark_batch = np.array(y_mark_batch)
        next_x_mark_batch = np.array(next_x_mark_batch)
        next_y_mark_batch = np.array(next_y_mark_batch)
        pred_x_mark = th.tensor(x_mark_batch, dtype=th.float32, device=self.device)
        pred_y_mark = th.tensor(y_mark_batch, dtype=th.float32, device=self.device)
        next_pred_x_mark = th.tensor(next_x_mark_batch, dtype=th.float32, device=self.device)
        next_pred_y_mark = th.tensor(next_y_mark_batch, dtype=th.float32, device=self.device)

        pred_x      = pred_x.reshape(len(batch_inds), self.seq_len, self.price_dims)
        pred_y      = pred_y.reshape(len(batch_inds), self.label_len + self.pred_len, self.price_dims)
        pred_x_mark = pred_x_mark.reshape(len(batch_inds), self.seq_len, x_mark_batch.shape[-1])
        pred_y_mark = pred_y_mark.reshape(len(batch_inds), self.label_len + self.pred_len, y_mark_batch.shape[-1])
        next_pred_x = next_pred_x.reshape(len(batch_inds), self.seq_len, self.price_dims)
        next_pred_y = next_pred_y.reshape(len(batch_inds), self.label_len + self.pred_len, self.price_dims)
        next_pred_x_mark = next_pred_x_mark.reshape(len(batch_inds), self.seq_len, x_mark_batch.shape[-1])
        next_pred_y_mark = next_pred_y_mark.reshape(len(batch_inds), self.label_len + self.pred_len, y_mark_batch.shape[-1])

        data = (
            th.tensor(self._normalize_obs(self.observations[batch_inds, env_indices, :], env), dtype=th.float32, device=self.device),
            th.tensor(self.actions[batch_inds, env_indices, :], dtype=th.float32, device=self.device),
            th.tensor(next_obs, dtype=th.float32, device=self.device),
            (pred_x, pred_y, pred_x_mark, pred_y_mark),
            (next_pred_x, next_pred_y, next_pred_x_mark, next_pred_y_mark),
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            th.tensor((self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1), dtype=th.float32, device=self.device),
            th.tensor(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env), dtype=th.float32, device=self.device),
        )
        return ModelBasedSample(*data)

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike | None) -> np.typing.DTypeLike | None:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype
    

class CustomRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        policy_obs_space: spaces.Space,
        price_dim: int,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        prediction_window: int = 96,
        label_len: int = 48,
        prediction_horizon: int = 24,
        dynamics_model_freq: str = 'D'
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False

        self.price_dim = price_dim
        self.pred_obs_dim = get_obs_shape(policy_obs_space)
        self.seq_len = prediction_window
        self.seq_label_len = label_len
        self.seq_pred_len = prediction_horizon
        self.time_feat_dim = len(time_features_from_frequency_str(dynamics_model_freq))

        self.reset()

    def reset(self) -> None:
        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=self.observation_space.dtype)
        self.pred_observations = np.zeros((self.buffer_size, self.n_envs, self.price_dim), dtype = self.observation_space.dtype)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dates  = np.zeros((self.buffer_size, self.n_envs), dtype=object)
        self.obs_pred_inputs = np.zeros((self.buffer_size, self.n_envs, *self.pred_obs_dim), dtype=self.observation_space.dtype)
        dec_len = self.seq_label_len + self.seq_pred_len
        self.seq_x      = np.zeros((self.buffer_size, self.n_envs, self.seq_len,  self.price_dim),    dtype=np.float32)
        self.seq_y      = np.zeros((self.buffer_size, self.n_envs, dec_len,       self.price_dim),    dtype=np.float32)
        self.seq_x_mark = np.zeros((self.buffer_size, self.n_envs, self.seq_len,  self.time_feat_dim), dtype=np.float32)
        self.seq_y_mark = np.zeros((self.buffer_size, self.n_envs, dec_len,       self.time_feat_dim), dtype=np.float32)
        self.generator_ready = False
        super().reset()

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
        dates: list[float] | None = None,
        obs_pred_input: object | None = None,
        pred_seq: tuple | None = None,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.obs_pred_inputs[self.pos] = obs_pred_input.cpu().numpy()
        if pred_seq is not None:
            x, y, x_mark, y_mark = pred_seq
            self.seq_x[self.pos]      = x.reshape(self.n_envs, self.seq_len, self.price_dim)
            self.seq_y[self.pos]      = y.reshape(self.n_envs, self.seq_label_len + self.seq_pred_len, self.price_dim)
            self.seq_x_mark[self.pos] = x_mark.reshape(self.n_envs, self.seq_len, self.time_feat_dim)
            self.seq_y_mark[self.pos] = y_mark.reshape(self.n_envs, self.seq_label_len + self.seq_pred_len, self.time_feat_dim)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[MBRolloutSample, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
                "seq_x",
                "seq_y",
                "seq_x_mark",
                "seq_y_mark",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: VecNormalize | None = None,
    ) -> MBRolloutSample:
        
        data = (
            self.observations[batch_inds],
            # Cast to float32 (backward compatible), this would lead to RuntimeError for MultiBinary space
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
            self.obs_pred_inputs[batch_inds],
        )
        pred_sequences = (
            self.to_torch(self.seq_x[batch_inds]),
            self.to_torch(self.seq_y[batch_inds]),
            self.to_torch(self.seq_x_mark[batch_inds]),
            self.to_torch(self.seq_y_mark[batch_inds]),
        )
        return MBRolloutSample(*tuple(map(self.to_torch, data)), pred_sequences=pred_sequences)
    