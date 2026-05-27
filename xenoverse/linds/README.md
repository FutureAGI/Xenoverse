# LinDS

English | [中文](README.zh.md)

LinDS is a family of procedurally generated linear dynamical-system environments. Each task samples a different linear state transition, observation model, control matrix, bias terms, target signal, and reward shaping, making it useful for continuous control, tracking, system identification, and robustness experiments.

## What This Module Provides

- Randomly generated linear dynamical-system tasks.
- Fixed-dimension and random-dimension task samplers.
- A Gymnasium-compatible environment for running sampled tasks.
- Optional padded observation and action dimensions for training models with fixed input/output sizes.
- A visualizer environment that records trajectories.
- An MPC baseline solver in `solver.py`.

## Underlying Model

The sampled tasks follow a linear state-space structure. In continuous-time notation, the module conceptually works with:

- `dx / dt = A x + B u + X`
- `o(t) = C x + Y`

Internally, the environment discretizes the system for simulation. Each task samples:

- `A`: state transition dynamics
- `B`: control matrix
- `C`: observation matrix
- `X`: state bias
- `Y`: observation bias

The reward encourages the observation to track a target command while penalizing control effort.

## Installation

Install from PyPI:

```bash
pip install "xenoverse[linds]"
```

Install from source:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[linds]"
```

## Main Components

The main public entry points are:

- `xenoverse.linds.LinearDSEnv`: base environment.
- `xenoverse.linds.LinearDSSampler`: fixed-dimension task sampler.
- `xenoverse.linds.LinearDSSamplerRandomDim`: random-dimension task sampler.
- `xenoverse.linds.LinearDSVisualizer`: environment with trajectory recording and plotting.
- `xenoverse.linds.LTISystemMPC`: MPC baseline controller.
- `xenoverse.linds.dump_linds_task` / `load_linds_task`: task serialization helpers.

The registered environment IDs are:

- `linear-dynamics-v0`
- `linear-dynamics-v0-visualizer`

## Recommended Workflow

The intended usage pattern is:

1. Create an environment.
2. Sample or load a task.
3. Call `env.set_task(task)`.
4. Call `env.reset()`.
5. Step the environment with continuous actions.

Important: `reset()` requires that a task has already been attached with `set_task(...)`.

## Quick Start

### 1. Sample a fixed-dimension task

```python
import numpy as np
from xenoverse.linds import LinearDSEnv, LinearDSSampler

task = LinearDSSampler(
    state_dim=16,
    observation_dim=8,
    action_dim=8,
)

env = LinearDSEnv(
    pad_observation_dim=16,
    pad_action_dim=8,
    pad_command_dim=16,
)

env.set_task(task, use_pad_dim=True)
obs, info = env.reset()
```

### 2. Step with random actions

```python
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## Sampling Tasks

### Fixed dimensions

Use `LinearDSSampler(...)` when you want an exact state, observation, and action size.

Key parameters:

- `state_dim`: latent state dimension.
- `observation_dim`: observation dimension.
- `action_dim`: action dimension.
- `seed`: optional random seed.

### Random dimensions

Use `LinearDSSamplerRandomDim(...)` when you want the task dimensions to vary across samples.

Example:

```python
from xenoverse.linds import LinearDSSamplerRandomDim

task = LinearDSSamplerRandomDim(
    max_state_dim=16,
    max_observation_dim=16,
    max_action_dim=8,
)
```

This is useful when you want one environment wrapper but tasks of varying intrinsic size.

## Padding Behavior

One of the main design choices in LinDS is support for padded dimensions.

Why this exists:

- Many sequence models or policies require a fixed observation size.
- Sampled tasks may have different intrinsic dimensions.

If `use_pad_dim=True` in `env.set_task(...)`:

- Observations are zero-padded up to `pad_observation_dim`.
- Actions passed to `step(...)` must match `pad_action_dim`.
- Command vectors returned in `info["command"]` are padded to `pad_command_dim`.

If `use_pad_dim=False`:

- Observation space matches the task's true `observation_dim`.
- Action space matches the task's true `action_dim`.

Example:

- Actual observation dimension is `3`.
- `pad_observation_dim=5`.
- Returned observation looks like `[o1, o2, o3, 0, 0]`.

Important: padding affects only the exposed interface shape, not the underlying system dimension.

## Observation, Action, and Info

### Observation

The environment returns a continuous observation vector produced from the current hidden state through the sampled observation model.

Depending on `use_pad_dim`, this vector is either:

- the true observation, or
- the zero-padded observation.

### Action

The action space is always continuous and bounded in `[-1, 1]` per dimension.

If padding is enabled:

- the environment expects an action vector of shape `(pad_action_dim,)`
- only the first `action_dim` components affect the actual task

### Info dictionary

`reset()` and `step()` return useful metadata including:

- `steps`: current step count
- `command`: current target command
- `command_type`: target mode reported at reset
- `error`: tracking error

## Target Types and Rewards

Sampled tasks can use either:

- `static_target`: one fixed command vector
- `dynamic_target`: a time-varying command trajectory

The reward combines:

- a base reward
- a penalty proportional to tracking error
- a penalty on control magnitude
- a termination penalty when the system diverges badly

In practice, higher reward means the system is tracking the target well without using excessive control.

## Termination Behavior

Episodes terminate early when the system becomes unstable enough, for example when:

- tracking error grows too large
- observation magnitude grows too large

Episodes are truncated when `max_steps` is reached.

## Using the Visualizer

The visualizer environment extends `LinearDSEnv` and records observations, states, actions, and rewards over time.

Example:

```python
import gymnasium as gym
from xenoverse.linds import LinearDSSamplerRandomDim

task = LinearDSSamplerRandomDim()
env = gym.make("linear-dynamics-v0-visualizer")

env.set_task(task)
obs, info = env.reset()
```

After rollout, you can call `visualize_and_save(...)` on the visualizer instance to save reward plots or optional t-SNE views of trajectories.

## MPC Baseline

`LTISystemMPC` provides a model-predictive-control baseline that uses the environment's discretized system matrices.

Example:

```python
from xenoverse.linds import LTISystemMPC

solver = LTISystemMPC(env, K=20, gamma=0.99)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    future_cmds = env.get_future_inner_cmds(K=solver.K)
    action = solver.solve(env.state, future_cmds)
    obs, reward, terminated, truncated, info = env.step(action)
```

Notes:

- This baseline depends on `osqp`, `scipy`, and related numeric dependencies.
- It is most useful as a reference controller rather than a learning baseline.

## Saving and Loading Tasks

The module exports helpers for task persistence:

```python
from xenoverse.linds import dump_linds_task, load_linds_task
```

Use them when you want reproducible experiments on the same sampled system.

## Common Pitfalls

- Call `env.set_task(task)` before `env.reset()`.
- If `use_pad_dim=True`, your action passed to `step(...)` must match `pad_action_dim`, not just `action_dim`.
- If you use padded mode, remember that trailing zeros in observations and commands are interface padding, not part of the true task.
- `pad_observation_dim`, `pad_action_dim`, and `pad_command_dim` must be large enough for the sampled task.
- `LinearDSSamplerRandomDim(...)` can produce smaller intrinsic dimensions than your environment pads to.

## File Guide

- `linds_env.py`: environment execution logic.
- `task_sampler.py`: task samplers and task serialization helpers.
- `solver.py`: MPC baseline and evaluation helpers.
- `visualizer.py`: trajectory recording and plotting.
- `test_ppo.py`: PPO-style training example.
