from __future__ import annotations

import logging
import os
from collections import namedtuple

import numpy as np
import torch
import torch.multiprocessing as mp

from env.snake_env import SnakeEnv
from ppo.memory import Memory

log = logging.getLogger(__name__)

MsgRewardInfo = namedtuple("MsgRewardInfo", ["agent", "episode", "reward"])
MsgUpdateRequest = namedtuple("MsgUpdateRequest", ["agent", "update"])
MsgMaxReached = namedtuple("MsgMaxReached", ["agent", "reached"])


class Agent(mp.Process):
    """Parallel agent process that collects rollouts in its own SnakeEnv."""

    def __init__(
        self,
        name: str,
        memory: Memory,
        pipe: mp.connection.Connection,
        env_kwargs: dict,
        max_episode: int,
        max_timestep: int,
        update_timestep: int,
        log_interval: int,
        gamma: float,
        seed: int | None = None,
    ):
        super().__init__(name=name)
        self.proc_id = name
        self.memory = memory
        self.pipe = pipe
        self.env_kwargs = env_kwargs
        self.max_episode = max_episode
        self.max_timestep = max_timestep
        self.update_timestep = update_timestep
        self.log_interval = log_interval
        self.gamma = gamma
        self.seed = seed

    def run(self) -> None:
        log.info("Agent %s started, PID %d", self.name, os.getpid())
        env = SnakeEnv(**self.env_kwargs)

        actions: list = []
        rewards: list = []
        states: list = []
        logprobs: list = []
        is_terminal: list = []
        timestep = 0
        running_reward = 0.0

        for i_episode in range(1, self.max_episode + 2):
            obs, _info = env.reset(
                seed=self.seed + i_episode if self.seed is not None else None
            )

            if i_episode == self.max_episode + 1:
                log.info("Agent %s: max episodes reached", self.name)
                self.pipe.send(MsgMaxReached(self.proc_id, True))
                break

            for _step in range(self.max_timestep):
                timestep += 1
                states.append(obs)

                with torch.no_grad():
                    action, logprob = self.memory.agent_policy.act(obs, False)

                obs, reward, terminated, truncated, _info = env.step(action)
                done = terminated or truncated

                actions.append(action)
                logprobs.append(logprob)
                rewards.append(reward)
                is_terminal.append(done)
                running_reward += reward

                if timestep % self.update_timestep == 0:
                    state_t, action_t, logprob_t, dis_return = (
                        self._experience_to_tensor(
                            states, actions, rewards, logprobs, is_terminal
                        )
                    )
                    self._add_experience_to_pool(
                        state_t, action_t, logprob_t, dis_return
                    )

                    self.pipe.send(MsgUpdateRequest(int(self.proc_id), True))
                    self.pipe.recv()

                    timestep = 0
                    actions = []
                    rewards = []
                    states = []
                    logprobs = []
                    is_terminal = []

                if done:
                    break

            if i_episode % self.log_interval == 0:
                avg_reward = running_reward / self.log_interval
                self.pipe.send(MsgRewardInfo(self.proc_id, i_episode, avg_reward))
                running_reward = 0.0

        env.close()

    def _experience_to_tensor(
        self,
        states: list,
        actions: list,
        rewards: list,
        logprobs: list,
        is_terminal: list,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state_t = torch.from_numpy(np.array(states, dtype=np.float32))
        action_t = torch.tensor(actions, dtype=torch.float32)
        logprob_t = torch.tensor(logprobs, dtype=torch.float32).detach()

        discounted_reward = 0.0
        dis_return: list[float] = []
        for reward, done in zip(reversed(rewards), reversed(is_terminal)):
            if done:
                discounted_reward = 0.0
            discounted_reward = reward + self.gamma * discounted_reward
            dis_return.insert(0, discounted_reward)

        dis_return_t = torch.tensor(dis_return, dtype=torch.float32)
        return state_t, action_t, logprob_t, dis_return_t

    def _add_experience_to_pool(
        self,
        state_t: torch.Tensor,
        action_t: torch.Tensor,
        logprob_t: torch.Tensor,
        dis_return_t: torch.Tensor,
    ) -> None:
        start = int(self.name) * self.update_timestep
        end = start + self.update_timestep
        self.memory.states[start:end] = state_t
        self.memory.actions[start:end] = action_t
        self.memory.logprobs[start:end] = logprob_t
        self.memory.disReturn[start:end] = dis_return_t
