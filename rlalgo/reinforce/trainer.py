import torch
import numpy as np
import time
import gymnasium as gym
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter

from models.actor.factory import create_actor

class REINFORCE:

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        learning_rate: float,
        gamma: float,
        eps: float,
        writer: SummaryWriter,
        device: str,
    ) -> None:
        self.log_probs = []
        self.rewards = []
        self.gamma = gamma

        self.model = create_actor(observation_space, action_space, device=device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, eps=eps
        )

        self.writer = writer
        self.start_time = time.time()
        self.device = device

    def register_reward(self, reward):
        # Store as float to avoid dtype issues later
        self.rewards.append(float(reward))

    def sample_action(
        self, state: np.ndarray
    ) -> Tuple[torch.Tensor]:
        """Sample an action and store its log-prob with gradients for REINFORCE."""
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        # No no_grad: we want grad through log_prob
        action, action_log_prob, _ = self.model(state_tensor)
        # Store log-prob for policy gradient
        self.log_probs.append(action_log_prob.squeeze())
        return action

    def update(self, global_step: int) -> None:

        # Compute discounted returns for the episode
        cumulative_rewards = []
        running_return = 0.0
        for reward in reversed(self.rewards):
            running_return = float(reward) + self.gamma * running_return
            cumulative_rewards.append(running_return)
        returns = torch.tensor(list(reversed(cumulative_rewards)), dtype=torch.float32, device=self.device)

        # Concatenate stored log-probs
        log_probs = torch.stack(self.log_probs)

        # Policy gradient loss (no baseline)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset storage for next episode
        self.log_probs = []
        self.rewards = []
