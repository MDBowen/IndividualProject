import numpy as np
import torch

import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None

from stable_baselines3.common.buffers import BaseBuffer


class ModelBasedSample(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    pred_obs: th.Tensor
    next_pred_obs: th.Tensor
    target_pred_obs: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    # For n-step replay buffer
    discounts: th.Tensor | None = None

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
        prediction_horizon: int = 24,
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
        self.batch_size = batch_size
        self.seq_len = prediction_window
        self.pred_len = prediction_horizon
        self.window = self.seq_len + self.pred_len

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)
            self.next_pred_observations = np.zeros((self.buffer_size, self.n_envs, price_dim), dtype=observation_space.dtype)

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

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

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
            batch_inds = np.random.randint(self.window, self.pos, size=batch_size)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: VecNormalize | None = None) -> ModelBasedSample:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        obs_seq = self._normalize_obs( np.stack([
            self.observations[i - self.window : i - self.pred_len, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)
        target_seq = self._normalize_obs( np.stack([
            self.next_observations[i - self.pred_len : i, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)
        next_obs_seq = self._normalize_obs( np.stack([
            self.next_observations[i - self.window : i - self.pred_len, env_indices[j]]  for j, i in enumerate(batch_inds)
            ]), env)

        reshape = lambda x: x[:, :, 1:1+self.price_dims]
        for arr in [obs_seq, target_seq, next_obs_seq]:
            arr = reshape(arr)

        obs_seq, target_seq, next_obs_seq = tuple(map(reshape, [obs_seq, target_seq, next_obs_seq]))

        assert obs_seq.shape == (len(batch_inds), self.seq_len, self.price_dims), f'obs_seq wrong shape is {obs_seq.shape} should be {(len(batch_inds), self.seq_len, self.price_dims)}'
        assert target_seq.shape == (len(batch_inds), self.pred_len, self.price_dims),  f'target_seq wrong shape is {target_seq.shape} should be {(len(batch_inds), self.pred_len, self.price_dims)}'
        assert next_obs_seq.shape == (len(batch_inds), self.seq_len, self.price_dims), f'next_obs_seq wrong shape is {next_obs_seq.shape} should be {(len(batch_inds), self.seq_len, self.price_dims)}'

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            obs_seq,
            next_obs_seq,
            target_seq,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ModelBasedSample(*tuple(map(self.to_torch, data)))

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