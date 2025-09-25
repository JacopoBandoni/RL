from typing import Callable, Optional

import gymnasium as gym
import numpy as np


def make_env(
    gym_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    capture_episodes: int = 0,
) -> Callable[[], gym.Env]:
    """Create a single environment thunk with wrappers applied.

    Mirrors the behavior used in the current training script, but keeps
    it reusable and decoupled from any particular algorithm.
    """

    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array" if capture_video else None)
        if capture_video and idx == 0:
            if capture_episodes > 0:
                def episode_trigger(episode_id: int) -> bool:
                    if capture_episodes == 0:
                        return True
                    return episode_id % capture_episodes == 0

                env_local = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=episode_trigger
                )
                env = env_local
            else:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.NormalizeReward(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_vector_env(
    gym_id: str,
    num_envs: int,
    base_seed: int,
    capture_video: bool,
    run_name: str,
    capture_episodes: int = 0,
) -> gym.vector.VectorEnv:
    """Create a SyncVectorEnv from multiple make_env thunks."""
    thunks = [
        make_env(
            gym_id=gym_id,
            seed=base_seed + i,
            idx=i,
            capture_video=capture_video,
            run_name=run_name,
            capture_episodes=capture_episodes,
        )
        for i in range(num_envs)
    ]
    return gym.vector.SyncVectorEnv(thunks)


