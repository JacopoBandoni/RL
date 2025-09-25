## RL: Modular Reinforcement Learning

Minimal, extensible RL playground. Ships with REINFORCE and PPO algos and a model factory that adapts actor-critic networks to the environment’s observation/action spaces (discrete/continuous and sizes).

### Structure
```
models/
  actor_critic.py  # Generic actor-critic for discrete/continuous spaces
  factory.py       # Space-aware model factory
rlalgo/
  ppo/
    buffer.py      # Multi-env rollout buffer with GAE
    trainer.py     # PPO update logic
rloop.py           # CLI entrypoint (envs, logging, training loop)
```

### Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Run PPO
```bash
# Discrete example
python rloop.py --gym-id CartPole-v1 --num-envs 1 --num-updates 50 --num-steps 1024

# Continuous example
python rloop.py --gym-id Humanoid-v5 --num-envs 1 --num-updates 100 --num-steps 2048
```
Logs: TensorBoard under `runs/…` (`tensorboard --logdir runs`). Optional W&B via `--use-wandb True`.
