import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from enum import Enum


class ActionSpaceType(Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ObservationSpaceType(Enum):
    BOX = "box"  # Continuous observation space
    DISCRETE = "discrete"  # Discrete observation space


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize layer weights using orthogonal initialization."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Actor-Critic network that can handle both discrete and continuous action and observation spaces."""

    def __init__(
        self,
        in_features,
        out_features,
        action_space_type=ActionSpaceType.DISCRETE,
        obs_space_type=ObservationSpaceType.BOX,
        embedding_dim=64,  # For discrete observations
        device="cpu"
    ):
        super(Agent, self).__init__()
        self.action_space_type = action_space_type
        self.obs_space_type = obs_space_type
        self.device = device

        # Input processing based on observation space type
        if obs_space_type == ObservationSpaceType.DISCRETE:
            # For discrete observations, use an embedding layer
            self.obs_embedding = nn.Embedding(in_features, embedding_dim)
            feature_input_dim = embedding_dim
        else:  # BOX (continuous observations)
            self.obs_embedding = None
            feature_input_dim = in_features

        # No shared feature extractor - use separate networks for actor and critic
        # This matches the original implementation
        if action_space_type == ActionSpaceType.DISCRETE:
            # Critic network
            self.critic = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            
            # Actor network for discrete actions
            self.actor = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, out_features), std=0.01),
            )
        else:  # CONTINUOUS
            # Critic network
            self.critic = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            
            # Actor mean and std for continuous actions
            self.actor = nn.Sequential(
                layer_init(nn.Linear(feature_input_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
            )
            self.actor_mean = layer_init(nn.Linear(64, out_features), std=0.01)
            self.actor_logstd = nn.Parameter(torch.zeros(1, out_features))
        
        # Move everything to the specified device
        self.to(device)

    def _process_obs(self, x):
        """Process observation based on space type."""
        # Always ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        if self.obs_space_type == ObservationSpaceType.DISCRETE:
            # Convert to long for embedding if needed
            if not isinstance(x, torch.LongTensor) and not (x.dtype == torch.int64):
                x = x.long()
            # Use embedding layer for discrete observations
            x = self.obs_embedding(x)
        return x

    def get_state(self, x):
        """Get state value prediction."""
        x = self._process_obs(x)
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        """Sample action and return action, log probability, entropy, and value."""
        x = self._process_obs(x)
        value = self.critic(x)

        if self.action_space_type == ActionSpaceType.DISCRETE:
            logits = self.actor(x)
            probs = Categorical(logits=logits)

            if action is None:
                action = probs.sample()

            return action, probs.log_prob(action), probs.entropy(), value
        else:  # CONTINUOUS
            features = self.actor(x)
            action_mean = self.actor_mean(features)
            action_std = torch.exp(self.actor_logstd)
            probs = Normal(action_mean, action_std)

            if action is None:
                action = probs.sample()

            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), value

    def forward(self, x, action: torch.Tensor = None):
        """Forward pass."""
        return self.get_action_and_value(x, action)
