import torch
import numpy as np
import time
import gymnasium as gym
from typing import Dict, List, Iterator, Tuple, Any
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from models.actor_critic.factory import create_actor_critic
from rlalgo.ppo.buffer import MultiEnvPPOBuffer


class PPO:

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
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
        advantage_normalization: bool,
        annealed_lr: bool,
        target_kl: float,
        device: str,
    ) -> None:
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.eps = eps
        self.epoches = epochs
        self.value_clipping = value_clipping
        self.num_envs = num_environments
        self.ppo_buffer = MultiEnvPPOBuffer(
            num_env=num_environments,
            batch_size=batch_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            device=device,
        )
        # Create model using the factory
        self.model = create_actor_critic(observation_space, action_space, device=device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, eps=adam_eps
        )
        self.writer = writer
        self.start_time = time.time()
        self.value_loss_coef = value_loss_coef
        self.policy_loss_coef = policy_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.advantage_normalization = advantage_normalization
        self.annealed_lr = annealed_lr
        self.target_kl = target_kl
        self.device = device
        # Store spaces for reference
        self.observation_space = observation_space
        self.action_space = action_space

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
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminated_tensor = torch.tensor(
            terminated, dtype=torch.bool, device=self.device
        )
        truncated_tensor = torch.tensor(truncated, dtype=torch.bool, device=self.device)

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
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action, action_log_prob, _, state_value_prediction = self.model(
                state_tensor
            )
        return action, action_log_prob, state_value_prediction

    def update(self, global_step: int) -> None:
        """
        Update the PPO policy.
        """

        clipfracs = []

        self.ppo_buffer.compute_returns()

        for epoch in range(self.epoches):
            if self.annealed_lr:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.learning_rate * (1 - epoch / self.epoches)
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

                if self.advantage_normalization:
                    batch["advantages"] = (
                        batch["advantages"] - batch["advantages"].mean()
                    ) / (batch["advantages"].std() + 1e-8)

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
            if self.target_kl:
                if approx_kl > self.target_kl:
                    break

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
