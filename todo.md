
**TODO LIST**

- [x] FIX THE BUG -> broadcasting bug merda!

- [x] take notes on the different kind of strategies not mentioned in the paper but present in the implementation.
  - [x] ridai una letta GAE
  - [x] vedi se GAE comprende una sommatoria fino a n (con bootstrap), non fino a fine episodio.
  - [x] scrivi la roba di GAE.

- [x] make vectorized environment
  - [x] change to vectorized
  - [x] I should not reset the environment every time.
  
- [x] implement proper learning algo to adact to previous changes
  - [x] implement total discounted returns
  - [x] implement advantage computation
   I should track next state and compute the state value function prediction for each terminal (NEXT) state. Not on the state where I have the reward.
   But I hacked it together by assigning to non terminal states in final steps the reward as the total discounted returns using bootstrp. Maybe TODO to make it more elegant.

- [x] add observability with w&b
  - [x] make the PPO algo parametric so that is possible to register all variable used in the learning algo
  - [x] integrate wandb
  - [x] integrate rloop metrics
  - [x] integrate PPO metrics
  - [x] integrate the debugging metrics and figure out their uses

- [x] compare and benchmark to reference implementation

- [x] clean the repo 
  - [x] clean logs
  - [x] clean redundant code (tracking twice)
  - [x] models
  - [x] do not hardcocde parameters
  - [x] allow video registering -> W&B non support sync ad ora ma stica
  - [x] fix determinism with seeds
  - [x] reorganize folder structure
  - [x] rewrite the buffer to make it more efficient

- [x] make the PPO parametric on input and output types: continous or discrete

- [x] fix bug when 1 num_envs

- [ ] make PPO command line parametric to distinct strategies:
    - [x] GAE
    - [ ] normalization of advantage
    - [ ] annealed lr

- [ ] Make PPO on Humanoid-v5 as good as clean RL
      https://huggingface.co/cleanrl/Humanoid-v2-ppo_continuous_action-seed1

- [ ] add support for GPU experiments

MOVE to DREAMER v3
