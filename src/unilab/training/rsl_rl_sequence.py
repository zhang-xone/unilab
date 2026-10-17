"""RSL-RL Sequence Encoder Wrapper — adapts UniLab env to VecEnv contract."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from unilab.algos.torch.ppo.rsl_rl.env import VecEnv
from unilab.utils.tensor import to_numpy, to_torch


class UniLabSequenceVecEnv(VecEnv):
    """Wrapper that adapts a UniLab ABEnv to the VecEnv interface expected by
    the reference OnPolicyRunner (with ActorCriticSequence encoder).

    Maintains an observation history buffer (sliding window) and exposes
    ``get_observations()`` returning ``(obs, obs_history)``.
    """

    def __init__(
        self,
        env: Any,
        device: str = "cpu",
        obs_history_length: int = 5,
    ) -> None:
        self._env = env
        self.device = device
        self.obs_history_length = obs_history_length
        self.num_envs = env.num_envs

        self.num_obs = int(env.obs_groups_spec.get("obs", 0))
        self.num_privileged_obs = int(env.obs_groups_spec.get("critic", self.num_obs))
        action_shape = env.action_space.shape
        if action_shape is None:
            raise ValueError("env.action_space.shape must be defined")
        self.num_actions = int(action_shape[0])

        self.max_episode_length = int(np.ceil(env.cfg.max_episode_seconds / env.cfg.ctrl_dt))

        self._obs_history = torch.zeros(
            self.num_envs, self.num_obs * self.obs_history_length, device=device
        )

        self._init_reset_pending = False
        if self._env.state is None:
            self._env.init_state()
            self._init_reset_pending = True

    def _get_obs(self) -> torch.Tensor:
        assert self._env.state is not None
        return to_torch(self._env.state.obs["obs"], self.device)

    def _get_priv(self) -> torch.Tensor:
        assert self._env.state is not None
        return to_torch(self._env.state.obs.get("critic", self._env.state.obs["obs"]), self.device)

    def _update_history(self, obs: torch.Tensor, dones: torch.Tensor) -> torch.Tensor:
        self._obs_history = torch.cat(
            (self._obs_history[:, self.num_obs:], obs), dim=-1
        )
        done_ids = torch.nonzero(dones).flatten()
        if len(done_ids) > 0:
            obs_done = obs[done_ids]
            new_history = self._obs_history.clone()
            new_history[done_ids] = obs_done.repeat(1, self.obs_history_length).clone()
            self._obs_history = new_history
        return self._obs_history

    def get_observations(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._get_obs(), self._obs_history

    def get_privileged_observations(self) -> torch.Tensor:
        return self._get_priv()

    def set_terrain_iteration(self, iteration: int) -> None:
        if hasattr(self._env, "set_terrain_iteration"):
            self._env.set_terrain_iteration(iteration)

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        actions_np = to_numpy(actions)
        state = self._env.step(actions_np)

        obs = self._get_obs()
        priv = self._get_priv()
        rew = to_torch(state.reward, self.device)
        terminated = to_torch(state.terminated, self.device).bool()
        truncated = to_torch(state.truncated, self.device).bool()
        dones = (terminated | truncated).bool()

        obs_history = self._update_history(obs, dones)

        infos: dict[str, torch.Tensor] = {}
        infos["time_outs"] = truncated

        if "log" in state.info:
            infos["log"] = state.info["log"]

        return obs, priv, rew, dones, infos, obs_history

    def reset(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._env.state is None:
            self._env.init_state()
            self._init_reset_pending = True

        if self._init_reset_pending:
            self._init_reset_pending = False
            obs = self._get_obs()
            self._obs_history = obs.repeat(1, self.obs_history_length).clone()
            return obs, self._obs_history

        env_indices = np.arange(self.num_envs, dtype=np.int32)
        new_obs, info = self._env.reset(env_indices)

        for key in self._env.state.obs:
            self._env.state.obs[key][env_indices] = new_obs[key]

        obs = self._get_obs()
        self._obs_history = obs.repeat(1, self.obs_history_length).clone()
        return obs, self._obs_history

    def close(self) -> None:
        self._env.close()