"""Factory for creating appropriate neural network models."""

import gymnasium as gym
from models.actor_critic import Agent, ActionSpaceType, ObservationSpaceType


def create_actor_critic(observation_space: gym.Space, action_space: gym.Space) -> Agent:
    """
    Create appropriate actor-critic model based on observation and action spaces.

    Args:
        observation_space: Gym observation space
        action_space: Gym action space

    Returns:
        Appropriate actor-critic network with the correct configuration
    """
    # Determine observation dimension and type
    if isinstance(observation_space, gym.spaces.Box):
        obs_dim = observation_space.shape[0]
        obs_type = ObservationSpaceType.BOX
    elif isinstance(observation_space, gym.spaces.Discrete):
        obs_dim = observation_space.n  # Number of discrete states
        obs_type = ObservationSpaceType.DISCRETE
    else:
        raise NotImplementedError(f"Unsupported observation space: {observation_space}")

    # Determine action dimension and type
    if isinstance(action_space, gym.spaces.Discrete):
        act_dim = action_space.n
        action_type = ActionSpaceType.DISCRETE
    elif isinstance(action_space, gym.spaces.Box):
        act_dim = action_space.shape[0]
        action_type = ActionSpaceType.CONTINUOUS
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    # Create appropriate model
    return Agent(
        in_features=obs_dim,
        out_features=act_dim,
        action_space_type=action_type,
        obs_space_type=obs_type,
    )
