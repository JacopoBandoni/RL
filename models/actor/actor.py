import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

# Reuse shared enums and init util from the existing actor-critic module
from models.actor_critic.actor_critic import (
    ActionSpaceType,
    ObservationSpaceType,
    layer_init,
)


class Actor(nn.Module):
    """Policy-only network supporting discrete/continuous actions and observations.

    This mirrors the actor part of the existing `ActorCritic` while removing the critic.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        action_space_type: ActionSpaceType = ActionSpaceType.DISCRETE,
        obs_space_type: ObservationSpaceType = ObservationSpaceType.BOX,
        embedding_dim: int = 64,
        device: str = "cpu",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.action_space_type = action_space_type
        self.obs_space_type = obs_space_type
        self.device = device
        self.eps = eps

        # Observation processing
        if obs_space_type == ObservationSpaceType.DISCRETE:
            self.obs_embedding = nn.Embedding(in_features, embedding_dim)
            feature_input_dim = embedding_dim
        else:
            self.obs_embedding = None
            feature_input_dim = in_features

        # Policy head(s)
        if action_space_type == ActionSpaceType.DISCRETE:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, out_features), std=0.01),
            )
            # No additional params needed
        else:
            self.actor = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.actor_mean = layer_init(nn.Linear(64, out_features), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))

        self.to(device)

    def _process_obs(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if self.obs_space_type == ObservationSpaceType.DISCRETE:
            if not isinstance(x, torch.LongTensor) and not (x.dtype == torch.int64):
                x = x.long()
            x = self.obs_embedding(x)
        return x

    def _distribution(self, x: torch.Tensor):
        x = self._process_obs(x)
        if self.action_space_type == ActionSpaceType.DISCRETE:
            logits = self.actor(x)
            return Categorical(logits=logits)
        else:
            features = self.actor(x)
            mean = self.actor_mean(features)
            std = torch.exp(self.actor_logstd) + self.eps
            return Normal(mean, std)

    @torch.no_grad()
    def act(self, x: torch.Tensor):
        """Sample an action and return (action, log_prob, entropy)."""
        dist = self._distribution(x)
        action = dist.sample()
        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, x: torch.Tensor, action: torch.Tensor):
        """Evaluate given actions: return (log_prob, entropy)."""
        dist = self._distribution(x)
        if self.action_space_type == ActionSpaceType.CONTINUOUS:
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            # Ensure discrete actions are long for Categorical
            if not isinstance(action, torch.LongTensor) and not (action.dtype == torch.int64):
                action = action.long()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return log_prob, entropy

    def forward(self, x: torch.Tensor, action: torch.Tensor = None):
        """If action is provided, evaluate it; otherwise sample one.

        Returns (action, log_prob, entropy).
        """
        if action is None:
            return self.act(x)
        log_prob, entropy = self.evaluate_actions(x, action)
        return action, log_prob, entropy


