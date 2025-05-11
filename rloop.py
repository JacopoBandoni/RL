import gymnasium as gym
import numpy as np
from rlalgo import PPO
import random

import argparse
import time
import wandb
import torch
from torch.utils.tensorboard import SummaryWriter
import os


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO training for Gymnasium environments"
    )

    # fmt: off
    # Environment parameters
    parser.add_argument("--gym-id", type=str, default="Walker2d-v5", help="Gym environment ID")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--num-updates", type=int, default=488, help="Number of update iterations") # 1953 per 1M
    parser.add_argument("--num-steps", type=int, default=2048, help="Number of steps per update")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1,2,3,5], help="Random seeds to use")
    # capture video sync with wandb is bugged with current version of wandb
    parser.add_argument("--capture-video", action="store_true", default=True, help="Capture videos of agent performance")
    parser.add_argument("--capture-episodes", type=int, default=100, help="Episode recording frequency (0 = all episodes, N = record one episode every N episodes)")
    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--eps", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for policy update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for policy update")
    parser.add_argument("--value-clipping", action="store_true", default=False, help="Use value function clipping")
    parser.add_argument("--value-loss-coef", type=float, default=0.5, help="Coefficient for value loss")
    parser.add_argument("--policy-loss-coef", type=float, default=1.0, help="Coefficient for policy loss")
    parser.add_argument("--entropy-coef", type=float, default=0, help="Coefficient for entropy loss")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max norm for gradient clipping")
    parser.add_argument("--adam-eps", type=float, default=1e-5, help="Epsilon value for Adam optimizer")
    # Logging parameters
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for logging")
    parser.add_argument("--wandb-project", type=str, default="PPO", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity (team) name")
    # fmt: on

    args = parser.parse_args()
    return args


def make_env(gym_id, seed, idx, capture_video, run_name, capture_episodes=0):
    """Create a single environment factory function.

    Args:
        gym_id: The Gymnasium environment ID
        seed: Random seed for the environment
        idx: Index of the environment in the vectorized env
        capture_video: Whether to capture video
        run_name: Name of the current run (for video directory)
        capture_episodes: Episode recording frequency (0 = all, N = record one every N episodes)
    """

    def thunk():
        env = gym.make(gym_id, render_mode="rgb_array" if capture_video else None)
        # Only capture video for the first environment (idx == 0)
        if capture_video and idx == 0:
            # If capture_episodes is specified, set up episode trigger
            if capture_episodes > 0:
                # Create a function to determine which episodes to record based on frequency
                def episode_trigger(episode_id):
                    # Record based on frequency
                    if capture_episodes == 0:
                        # Record all episodes
                        return True
                    else:
                        # Record one episode every capture_episodes
                        return episode_id % capture_episodes == 0

                env = gym.wrappers.RecordVideo(
                    env, f"videos/{run_name}", episode_trigger=episode_trigger
                )
            else:
                # Record all episodes (default behavior)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def setup_logging(args, seed, current_run_name):
    """Set up W&B and TensorBoard logging."""
    # Create hyperparameters dict
    hyperparams = vars(args)
    hyperparams["seed"] = seed

    # Initialize W&B if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=hyperparams,
            name=current_run_name,
            monitor_gym=False,  # CHECK IF SOLVED
            save_code=True,
        )

    # Create TensorBoard writer
    log_dir = os.path.join("runs", current_run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    # Format hyperparameters for TensorBoard
    hyperparams_table = "|param|value|\n|-|-|\n"
    for key, value in hyperparams.items():
        hyperparams_table += f"|{key}|{value}|\n"
    writer.add_text("hyperparameters", hyperparams_table)

    return writer


def train_single_seed(args, seed):
    """Train a PPO agent with a single seed."""
    print(f"Running with seed {seed}")

    # Setup unique run name and logging
    current_run_name = f"my_new_{args.gym_id}_{seed}_{int(time.time())}"
    writer = setup_logging(args, seed, current_run_name)

    # Track total episodes across all environments
    total_episodes = 0

    # Create vectorized environment
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.gym_id,
                seed + i,
                i,
                args.capture_video,
                current_run_name,
                args.capture_episodes,
            )
            for i in range(args.num_envs)
        ]
    )

    # Initialize PPO agent
    agent = PPO(
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        num_environments=args.num_envs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        eps=args.eps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        value_clipping=args.value_clipping,
        writer=writer,
        value_loss_coef=args.value_loss_coef,
        policy_loss_coef=args.policy_loss_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        adam_eps=args.adam_eps,
    )

    # Track metrics
    global_step = 0

    # Reset environments
    obs, info = envs.reset(seed=seed)

    # Main training loop
    for update in range(args.num_updates):

        # Collect rollout data
        for step in range(args.num_steps):
            global_step += args.num_envs

            # Sample actions from the agent
            actions, action_log_probs, value_preds = agent.sample_action(obs)

            # Take a step in the environment
            next_obs, rewards, terminated, truncated, info = envs.step(
                actions.cpu().numpy()
            )

            # Store the transition in buffer
            agent.process_step(
                states=obs,
                next_states=next_obs,
                actions=actions,
                action_log_probs=action_log_probs,
                value_preds=value_preds.squeeze(-1),
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                # last_step is necessary so it knows
                # wheter to bootstrap in case the episode is not finished
                last_step=step == (args.num_steps - 1),
            )

            # Update observation
            obs = next_obs

            if "episode" in info:
                for i, done in enumerate(info["episode"]["_r"]):
                    if done:
                        # Log episode stats
                        episodic_return = info["episode"]["r"][i]
                        episodic_length = info["episode"]["l"][i]
                        total_episodes += 1

                        # Log to tensorboard
                        writer.add_scalar(
                            "charts/episodic_return", episodic_return, global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", episodic_length, global_step
                        )
                        writer.add_scalar(
                            "charts/total_episodes", total_episodes, global_step
                        )

        # Update policy after collecting enough data
        agent.update(global_step)

    # Cleanup
    envs.close()
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
