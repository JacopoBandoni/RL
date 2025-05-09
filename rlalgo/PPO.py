import torch
from models.actor_critic import Agent
import numpy as np
import random
import time
from typing import Dict, List, Iterator, Tuple
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn


class PPObuffer:

    def __init__(self, batch_size: int, gamma: float) -> None:
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

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float = None
    ) -> None:
        """Compute returns and advantages using GAE."""

        steps_number = len(self.data["states"])
        new_episode_idx = 0

        self.data["returns"] = [0] * steps_number

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

        self.data["advantages"] = [
            self.data["returns"][i] - self.data["value_preds"][i]
            for i in range(steps_number)
        ]

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

        # self.delete_data()

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

    def __init__(self, num_env: int, batch_size: int, gamma: float) -> None:
        self.num_env = num_env
        self.batch_size = batch_size
        self.gamma = gamma
        self.ppo_buffers = [PPObuffer(batch_size, gamma) for _ in range(num_env)]

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
            buffer.compute_returns_and_advantages(gamma=self.gamma)

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


class PPO:

    def __init__(
        self,
        obs_space_dims: int,
        action_space_dims: int,
        num_environments: int,
        learning_rate: float,
        gamma: float,
        gae_lambda: float,
        eps: float,
        epochs: int,
        batch_size: int,
        writer: SummaryWriter,
        value_clipping: bool,
        value_loss_coef: float,
        policy_loss_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        adam_eps: float,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps = eps
        self.epoches = epochs
        self.value_clipping = value_clipping
        self.num_envs = num_environments
        self.ppo_buffer = MultiEnvPPOBuffer(
            num_env=num_environments, batch_size=batch_size, gamma=self.gamma
        )
        self.model = Agent(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=adam_eps
        )
        self.writer = writer
        self.start_time = time.time()
        self.value_loss_coef = value_loss_coef
        self.policy_loss_coef = policy_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def process_step(
        self,
        states: np.ndarray,
        next_states: np.ndarray,
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
        last_step: bool = False,
    ) -> None:
        """Process steps from multiple environments."""

        # Convert numpy arrays to torch tensors if needed
        states_tensor = torch.tensor(states, dtype=torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        terminated_tensor = torch.tensor(terminated, dtype=torch.bool)
        truncated_tensor = torch.tensor(truncated, dtype=torch.bool)

        # Get next state values for non-terminated episodes
        if last_step:
            for i in range(self.num_envs):
                # HERE one could choose to bootstrap even for truncated episodes
                if terminated[i] == False and truncated[i] == False:
                    # bootstrap final
                    with torch.no_grad():
                        rewards_tensor[i] = rewards_tensor[
                            i
                        ] + self.gamma * self.model.get_state(next_states_tensor[i])

        self.ppo_buffer.add_steps(
            states=states_tensor,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            rewards=rewards_tensor,
            terminated=terminated_tensor,
            truncated=truncated_tensor,
        )

    def sample_action(
        self, state: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and return action, log probability, and value prediction."""
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            action, action_log_prob, _, state_value_prediction = self.model(
                state_tensor
            )
        return action, action_log_prob, state_value_prediction

    def update(self, global_step: int) -> None:

        clipfracs = []

        self.ppo_buffer.compute_returns()

        for epoch in range(self.epoches):
            for batch in self.ppo_buffer:

                # Save old policy log probs for KL calculation
                old_policy_log_prob = batch["action_log_probs"].detach()

                # Forward pass
                _, new_policy_log_prob, entropy, state_values = self.model(
                    batch["states"], batch["actions"]
                )

                # Calculate losses
                mse_value_function_loss = (
                    0.5 * ((state_values.squeeze() - batch["returns"]) ** 2).mean()
                )

                # Policy ratio for PPO update
                ratio = torch.exp(new_policy_log_prob - old_policy_log_prob)

                # Clipped policy loss
                clipped_ratio = torch.clamp(ratio, 1 - self.eps, 1 + self.eps)
                policy_loss = -torch.min(
                    ratio * batch["advantages"], clipped_ratio * batch["advantages"]
                ).mean()

                # Compute approximate KL divergence for early stopping
                with torch.no_grad():
                    # two distinct way to estimate the kl
                    old_approx_kl = (-new_policy_log_prob + old_policy_log_prob).mean()
                    approx_kl = (
                        (ratio - 1) - (new_policy_log_prob - old_policy_log_prob)
                    ).mean()
                    # Track clipping fraction
                    clipfracs.append(
                        ((ratio - 1.0).abs() > self.eps).float().mean().item()
                    )

                entropy_loss = entropy.mean()

                total_loss = (
                    self.value_loss_coef * mse_value_function_loss
                    + self.policy_loss_coef * policy_loss
                    - self.entropy_coef * entropy_loss
                )

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

        with torch.no_grad():
            all_value_preds, all_returns = self.ppo_buffer.get_all_data()
            y_pred = all_value_preds.detach().cpu().numpy()
            y_true = all_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = (
                np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            )

        # Log metrics
        self.writer.add_scalar(
            "charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step
        )
        self.writer.add_scalar(
            "losses/value_loss", mse_value_function_loss.item(), global_step
        )
        self.writer.add_scalar("losses/policy_loss", policy_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar(
            "losses/old_approx_kl", old_approx_kl.item(), global_step
        )
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)

        # Log SPS (steps per second)
        steps_per_sec = int(global_step / (time.time() - self.start_time))
        self.writer.add_scalar("charts/SPS", steps_per_sec, global_step)

        # Reset dataset
        self.ppo_buffer.delete_data()
