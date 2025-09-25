import os
import wandb
from torch.utils.tensorboard import SummaryWriter

def setup_logging(args, seed, current_run_name):
    """Set up W&B and TensorBoard logging."""
    # Create hyperparameters dict
    hyperparams = vars(args)
    hyperparams["seed"] = seed

    # Initialize W&B if enabled
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project + "_" + args.gym_id,
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
