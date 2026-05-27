# AnyMDP

English | [中文](README.zh.md)

AnyMDP is a family of procedurally generated decision-making environments built around randomized MDPs and POMDPs. Instead of evaluating agents on one fixed transition graph, it samples new transition, reward, and observation structures for each task, which makes it useful for benchmarking generalization, in-context adaptation, and reinforcement learning algorithms.

## What This Module Provides

- Randomly generated MDP tasks.
- Randomly generated POMDP tasks with sampled observation models.
- Multi-token POMDP tasks with `MultiDiscrete` observation and action spaces.
- Gymnasium-compatible environment execution through `AnyMDPEnv`.
- Built-in reference solvers for fully observed MDPs.
- A task visualizer for inspecting sampled dynamics.

## Installation

Install from PyPI:

```bash
pip install "xenoverse[anymdp]"
```

Install from source:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[anymdp]"
```

## Pre-generated Data

The original README references two downloadable resources:

- Training dataset: [Kaggle large training set](https://www.kaggle.com/datasets/anonymitynobody/omnirl-training-data-d-large)
- Evaluation tasks and validation data: [Kaggle evaluation set](https://www.kaggle.com/datasets/anonymitynobody/omnirl-evaluation)

These are optional. You can use the task samplers directly without downloading any external data.

## Main Components

The most relevant entry points are:

- `xenoverse.anymdp.AnyMDPEnv`: Gymnasium environment class.
- `xenoverse.anymdp.AnyMDPTaskSampler`: standard random MDP sampler.
- `xenoverse.anymdp.GarnetTaskSampler`: GARNET-style random MDP sampler.
- `xenoverse.anymdp.AnyPOMDPTaskSampler`: POMDP sampler.
- `xenoverse.anymdp.MultiTokensAnyPOMDPTaskSampler`: multi-token POMDP sampler.
- `xenoverse.anymdp.AnyMDPSolverOpt`: optimal solver using ground-truth transition and reward tables.
- `xenoverse.anymdp.AnyMDPSolverMBRL`: model-based RL baseline for MDPs.
- `xenoverse.anymdp.AnyMDPSolverQ`: tabular Q-learning baseline for MDPs.
- `xenoverse.anymdp.anymdp_task_visualizer`: visualization helper.

The environment ID registered by the package is:

- `anymdp-v0`

## Recommended Workflow

The intended workflow is:

1. Create the environment.
2. Sample a task.
3. Call `env.set_task(task)`.
4. Call `env.reset()`.
5. Interact with the environment using `step(...)`.

Important: `reset()` requires that a task has already been set.

## Quick Start

### 1. Standard MDP

```python
import gymnasium as gym
import xenoverse.anymdp
from xenoverse.anymdp import AnyMDPTaskSampler

env = gym.make("anymdp-v0", max_steps=5000)

task = AnyMDPTaskSampler(
    state_space=128,
    action_space=5,
    min_state_space=None,
)

env.set_task(task)
obs, info = env.reset()
```

### 2. Run with random actions

```python
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Task Types

AnyMDP supports three task families through the same `env.set_task(task)` entry point.

### MDP

Use `AnyMDPTaskSampler(...)` when you want a fully observed finite MDP.

Typical parameters:

- `state_space`: upper bound on the number of observable states.
- `action_space`: number of actions.
- `min_state_space`: optional lower bound when sampling the actual active state subset.
- `seed`: random seed.
- `verbose`: print sampling diagnostics.

In the MDP case:

- Observation space is `Discrete(ns)`.
- Action space is `Discrete(na)`.
- The returned observation is the current state ID.

### POMDP

Use `AnyPOMDPTaskSampler(...)` when you want hidden states plus sampled observation emissions.

Example:

```python
from xenoverse.anymdp import AnyPOMDPTaskSampler

task = AnyPOMDPTaskSampler(
    state_space=64,
    action_space=5,
    observation_space=32,
    density=0.1,
)

env.set_task(task)
obs, info = env.reset()
```

In the POMDP case:

- Observation space is `Discrete(no)`.
- Action space is still `Discrete(na)`.
- Observations are sampled from the task's observation model instead of exposing the true state directly.

### Multi-token POMDP

Use `MultiTokensAnyPOMDPTaskSampler(...)` when you want multi-token observations and multi-token actions.

Example:

```python
from xenoverse.anymdp import MultiTokensAnyPOMDPTaskSampler

task = MultiTokensAnyPOMDPTaskSampler(
    state_space=128,
    action_space=5,
    observation_space=32,
    observation_tokens=4,
    action_tokens=2,
    density=0.2,
)

env.set_task(task)
obs, info = env.reset()
```

In the multi-token case:

- Observation space is `MultiDiscrete([no] * observation_tokens)`.
- Action space is `MultiDiscrete([na] * action_tokens)`.
- Each `step(...)` applies a sequence of action tokens inside the same environment step.

## Understanding the Sampled Task

The task samplers return a dictionary that is consumed by `env.set_task(task)`. Depending on task type, it can include:

- `ns`: total observable state-space size.
- `na`: action-space size.
- `max_steps`: suggested episode length for the sampled task.
- `state_mapping`: sampled active-state subset embedded into the observable state space.
- `transition`: transition kernel on active states.
- `reward`: reward tensor.
- `reward_noise`: reward noise tensor.
- `s_0` and `s_0_prob`: start states and their sampling probabilities.
- `s_e`: terminal states.
- `observation_transition`: observation-emission model for POMDP variants.
- `task_type`: one of `MDP`, `POMDP`, or `MTPOMDP`.

One important implementation detail: the environment can embed a smaller active state set inside a larger observable state space through `state_mapping`.

## Observation and Action Semantics

After `env.set_task(task)`:

- MDP tasks expose state-like observations directly.
- POMDP tasks expose sampled observations.
- Multi-token POMDP tasks expose vectors of observation tokens.

The action space depends on task type:

- MDP and POMDP tasks use scalar discrete actions.
- Multi-token POMDP tasks use a `MultiDiscrete` action vector.

Because of this, code written for standard MDP tasks should not be copied unchanged into multi-token tasks.

## Built-in Solvers

### Optimal solver

`AnyMDPSolverOpt` uses the ground-truth transition and reward tables stored in the task.

```python
from xenoverse.anymdp import AnyMDPSolverOpt

solver = AnyMDPSolverOpt(env)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    action = solver.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

Use this solver as an oracle-style reference baseline.

### Learned MDP baselines

`AnyMDPSolverMBRL` and `AnyMDPSolverQ` are designed for standard MDP tasks, not for POMDP or multi-token POMDP tasks.

Example:

```python
from xenoverse.anymdp import AnyMDPSolverMBRL

solver = AnyMDPSolverMBRL(env)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    action = solver.policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    solver.learner(obs, action, next_obs, reward, terminated, truncated)
    obs = next_obs
```

Notes:

- `AnyMDPSolverQ` explicitly asserts that the task type is `MDP`.
- The solver examples above assume the observation is sufficient for control, which is true for standard MDP tasks.
- For POMDP variants, use your own memory-based or recurrent policy, or refer to `test_ppo.py` for a training example.

## Visualization

The module exposes `anymdp_task_visualizer` from `visualizer.py`.

It is intended for inspecting sampled tasks such as:

- transition structure
- Markov-chain connectivity
- value-related structure

If you need to understand whether a sampled task looks reasonable before training, this is the first tool to use.

## Common Pitfalls

- Call `env.set_task(task)` before `env.reset()`.
- Do not assume the sampled active state count always equals `state_space`; the sampler may embed a smaller active set via `state_mapping`.
- `AnyMDPSolverQ` and `AnyMDPSolverMBRL` are for standard MDP settings, not general POMDP settings.
- Multi-token tasks use vector actions, so code written for scalar actions will need adjustment.
- The older README mentioned `Resampler`, but that interface is not part of the current exported API in this repository.

## File Guide

- `anymdp_env.py`: environment execution logic.
- `task_sampler.py`: MDP and POMDP task samplers.
- `anymdp_solver_opt.py`: oracle solver with access to ground truth.
- `anymdp_solver_mbrl.py`: model-based RL baseline.
- `anymdp_solver_q.py`: tabular Q-learning baseline.
- `visualizer.py`: task visualization utilities.
- `test_ppo.py`: PPO-style example for partially observed settings.

## References

```bibtex
@inproceedings{wang2025towards,
  title={Towards Large-Scale In-Context Reinforcement Learning by Meta-Training in Randomized Worlds},
  author={Fan Wang and Pengtao Shao and Yiming Zhang and Bo Yu and Shaoshan Liu and Ning Ding and Yang Cao and Yu Kang and Haifeng Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=b6ASJBXtgP}
}
```
