import torch
from models.actor_critic import Agent
import numpy as np
import random
import time
from typing import Dict, List, Iterator, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class PPObuffer:

    def __init__(self, batch_size: int, gamma: float, gae_lambda: float = None) -> None:
        self.data: Dict[str, List] = {
            "states": [],
            "actions": [],
            "action_log_probs": [],
            "value_preds": [],
            "rewards": [],
            "advantages": [],
            "terminated": [],
            "truncated": [],
        }
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def add_step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        action_log_prob: torch.Tensor,
        value_pred: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """Add a single step from one environment."""

        # Convert tensors to lists for storage
        self.data["states"].append(state)
        self.data["actions"].append(action)
        self.data["action_log_probs"].append(action_log_prob)
        self.data["value_preds"].append(value_pred)
        self.data["rewards"].append(reward)
        self.data["terminated"].append(terminated)
        self.data["truncated"].append(truncated)

    def __len__(self) -> int:
        return len(self.data["states"])

    def compute_returns_and_advantages(self) -> None:
        """Compute returns and advantages using GAE."""

        steps_number = len(self.data["states"])
        new_episode_idx = 0

        self.data["returns"] = [0] * steps_number

        # compute cumulative discounted returns
        for i in range(steps_number):
            if (
                self.data["truncated"][i]
                or self.data["terminated"][i]
                or i == steps_number - 1
            ):
                for j in reversed(range(new_episode_idx, i + 1)):
                    if j == i:  # final episode
                        self.data["returns"][j] = self.data["rewards"][j]
                    else:
                        self.data["returns"][j] = (
                            self.data["rewards"][j]
                            + self.gamma * self.data["returns"][j + 1]
                        )
                new_episode_idx = i + 1

        # compute advantages

        if not self.gae_lambda:
            # compute advantages using TD error
            self.data["advantages"] = [
                self.data["returns"][i] - self.data["value_preds"][i]
                for i in range(steps_number)
            ]
        else:
            # compute advantages using GAE
            deltas = [0] * steps_number
            self.data["advantages"] = [0] * steps_number

            # compute deltas 
            for i in range(steps_number):
                if (
                    i == steps_number - 1
                    or self.data["terminated"][i]
                    or self.data["truncated"][i]
                ):
                    deltas[i] = self.data["rewards"][i] - self.data["value_preds"][i]
                else:        
                    deltas[i] = (
                        self.data["rewards"][i]
                        + self.gamma * self.data["value_preds"][i + 1]
                        - self.data["value_preds"][i]
                    )

            # compute advantages
            for i in reversed(range(steps_number)):
                if (
                    i == steps_number - 1
                    or self.data["terminated"][i]
                    or self.data["truncated"][i]
                ):
                    self.data["advantages"][i] = deltas[i]
                else:
                    self.data["advantages"][i] = (
                        deltas[i]
                        + self.gamma * self.gae_lambda * self.data["advantages"][i + 1]
                    )

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Creates an iterator that yields batches."""

        indices = list(range(len(self.data["states"])))
        random.shuffle(indices)

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            batch = {
                key: torch.stack([self.data[key][idx] for idx in batch_indices], dim=0)
                for key in self.data
            }
            yield batch

    def delete_data(self) -> None:
        self.data = {
            "states": [],
            "actions": [],
            "action_log_probs": [],
            "value_preds": [],
            "rewards": [],
            "advantages": [],
            "terminated": [],
            "truncated": [],
            "returns": [],
        }


class MultiEnvPPOBuffer:

    def __init__(
        self, num_env: int, batch_size: int, gamma: float, gae_lambda: float
    ) -> None:
        self.num_env = num_env
        self.batch_size = batch_size
        self.ppo_buffers = [
            PPObuffer(batch_size, gamma, gae_lambda) for _ in range(num_env)
        ]

    def add_steps(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        """
        Add steps from multiple environments.
        Each tensor should have shape (num_env, ...) where ... represents the shape for each item
        """
        # Distribute data to individual buffers
        for env_idx in range(self.num_env):
            self.ppo_buffers[env_idx].add_step(
                state=states[env_idx],
                action=actions[env_idx],
                action_log_prob=action_log_probs[env_idx],
                value_pred=value_preds[env_idx],
                reward=rewards[env_idx],
                terminated=terminated[env_idx],
                truncated=truncated[env_idx],
            )

    def compute_returns(self, gae=False):
        """Compute cumulative discounted returns"""
        for buffer in self.ppo_buffers:
            buffer.compute_returns_and_advantages()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Creates an iterator that yields batches combining data from all environments."""

        # Combine all data from all buffers
        combined_data = {key: [] for key in self.ppo_buffers[0].data.keys()}

        for buffer in self.ppo_buffers:
            for key in buffer.data:
                combined_data[key].extend(buffer.data[key])

        # Create indices and shuffle
        indices = list(range(len(combined_data["states"])))
        random.shuffle(indices)

        # Yield batches
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            try:
                batch = {
                    key: torch.stack(
                        [combined_data[key][idx] for idx in batch_indices], dim=0
                    )
                    for key in combined_data
                }
            except Exception as e:
                print(batch_indices)
                for key in combined_data.keys():
                    print(f"{key} len : {len(combined_data[key])}")
                exit()
            yield batch

    def get_all_data(self):
        """
        Returns all value predictions and returns across all buffers.
        This is used for computing accurate explained variance.
        """

        # Collect all value_preds and returns
        all_value_preds = []
        all_returns = []

        for buffer in self.ppo_buffers:
            all_value_preds.extend(buffer.data["value_preds"])
            all_returns.extend(buffer.data["returns"])
        return torch.stack(all_value_preds), torch.stack(all_returns)

    def delete_data(self) -> None:
        """Clear all buffers."""
        for buffer in self.ppo_buffers:
            buffer.delete_data()
