## RL: Modular Reinforcement Learning

Minimal, extensible RL playground. Ships with REINFORCE and PPO algos and a model factory that adapts actor-critic networks to the environment’s observation/action spaces (discrete/continuous and sizes).

### Structure
```
models/
  actor/
    actor.py          # Policy networks for REINFORCE (discrete/continuous)
    factory.py        # Space-aware actor builder
  actor_critic/
    actor_critic.py   # Shared body + actor/critic heads
    factory.py        # Space-aware actor-critic builder

rlalgo/
  ppo/
    buffer.py         # Multi-env rollout buffer with GAE(lambda)
    trainer.py        # PPO update logic and optimization
  reinforce/
    trainer.py        # Monte-Carlo policy gradient (REINFORCE)

utils/
  envs.py             # Env helpers: seeding, wrappers, vectorization, video
  logging.py          # TensorBoard/W&B setup and utilities

rloop.py              # PPO CLI (args, envs, training loop)
rloop_reinforce.py    # REINFORCE CLI

runs/                 # TensorBoard logs
videos/               # Saved episode videos
wandb/                # Weights & Biases run artifacts
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

### Run REINFORCE

```bash
RL % python rloop_reinforce.py --gym-id MountainCarContinuous-v0 --num-episodes 1000
```