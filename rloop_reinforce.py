import gymnasium as gym
import numpy as np
from rlalgo.reinforce.trainer import REINFORCE
import random

import argparse
import time
import wandb
import torch
from utils.envs import make_env
from utils.logging import setup_logging

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO training for Gymnasium environments"
    )

    # fmt: off
    # Environment parameters
    parser.add_argument("--gym-id", type=str, default="Humanoid-v5", help="Gym environment ID")
    parser.add_argument("--num-episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="Random seeds to use")
    # capture video sync with wandb is bugged with current version of wandb
    parser.add_argument("--capture-video", action="store_true", default=True, help="Capture videos of agent performance")
    parser.add_argument("--capture-episodes", type=int, default=100, help="Episode recording frequency (0 = all episodes, N = record one episode every N episodes)")
    # GPU support
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cuda or cpu)")
    # Algorithm hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--adam-eps", type=float, default=1e-5, help="Adam epsilon")

    # Logging parameters
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default=f"REINFORCE", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team) name")
    # fmt: on

    args = parser.parse_args()
    return args


def train_single_seed(args, seed):
    """Train a REINFORCE agent with a single seed."""
    print(f"Running with seed {seed}")

    # Setup unique run name and logging
    current_run_name = f"{args.gym_id}_{seed}_{int(time.time())}"
    writer = setup_logging(args, seed, current_run_name)

    # Track total episodes across all environments
    total_episodes = 0

    # Create a single environment (call the thunk)
    env_thunk = make_env(
        args.gym_id,
        seed,
        0,
        args.capture_video,
        current_run_name,
        args.capture_episodes,
    )
    env = env_thunk()

    # Initialize REINFORCE agent
    agent = REINFORCE(
        observation_space=env.observation_space,
        action_space=env.action_space,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        eps=args.adam_eps,
        writer=writer,
        device=args.device,
    )

    # Track metrics
    global_step = 0

    # Main training loop
    for episode in range(args.num_episodes):
        # Reset environment at the start of each episode
        obs, info = env.reset(seed=seed if episode == 0 else None)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            global_step += 1

            # Sample actions from the agent
            actions = agent.sample_action(obs)

            # Take a step in the environment
            action_np = actions.detach().cpu().numpy()
            if isinstance(env.action_space, gym.spaces.Box):
                # Ensure shape matches action space, e.g., (1,) for MountainCarContinuous
                action_np = action_np.reshape(env.action_space.shape)
            elif isinstance(env.action_space, gym.spaces.Discrete):
                action_np = int(action_np.squeeze())
            obs, reward, terminated, truncated, info = env.step(action_np)

            # Store the reward
            agent.register_reward(reward)

            # Log episodic stats from RecordEpisodeStatistics when available
            if "episode" in info:
                total_episodes += 1
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/total_episodes", total_episodes, global_step)

            if terminated or truncated:
                break


        # Update policy after the episode finished
        agent.update(global_step)

    # Cleanup
    env.close()
    writer.close()

    # Close wandb run
    if args.use_wandb:
        wandb.finish()


def main():
    """Main training function."""
    args = parse_args()

    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        train_single_seed(args, seed)

    print("Training complete!")


if __name__ == "__main__":
    main()
