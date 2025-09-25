"""Factory for creating actor-only policies."""

import gymnasium as gym
from models.actor.actor import Actor
from models.actor_critic.actor_critic import ActionSpaceType, ObservationSpaceType


def create_actor(observation_space: gym.Space, action_space: gym.Space, device: str = "cpu") -> Actor:
    """Create an `Actor` configured from gym spaces.

    - Supports `gym.spaces.Box` and `gym.spaces.Discrete` for both observation and action.
    """
    # Observation
    if isinstance(observation_space, gym.spaces.Box):
        obs_dim = observation_space.shape[0]
        obs_type = ObservationSpaceType.BOX
    elif isinstance(observation_space, gym.spaces.Discrete):
        obs_dim = observation_space.n
        obs_type = ObservationSpaceType.DISCRETE
    else:
        raise NotImplementedError(f"Unsupported observation space: {observation_space}")

    # Action
    if isinstance(action_space, gym.spaces.Discrete):
        act_dim = action_space.n
        action_type = ActionSpaceType.DISCRETE
    elif isinstance(action_space, gym.spaces.Box):
        act_dim = action_space.shape[0]
        action_type = ActionSpaceType.CONTINUOUS
    else:
        raise NotImplementedError(f"Unsupported action space: {action_space}")

    return Actor(
        in_features=obs_dim,
        out_features=act_dim,
        action_space_type=action_type,
        obs_space_type=obs_type,
        device=device,
    )


