# Introduction

Scalable procedurally generated Linear Time Invariant system by randomizing the transition and rewards.

# Implementation

Linear Time Invariant (LTI) systems are linear dynamical systems that have a constant transition matrix and reward function.

$x_{t+1} = A x_t + B u_t + X$

$o_t = C x_t + Y$

In this implementation, we randomly generate the matrices $A$, $B$, $C$ as well as the bias vectors $X$ and $Y$.

The reward function is also procedurally generated, with a base reward and a factor that scales the distance between the current state and the command.

The initial states are randomly generated and the command is also randomly chosen.

# Usage

```python
import numpy as np
from xenoverse.linds import LinearDSEnv, LinearDSSampler

# Create a task sampler with 10-dimensional state and action spaces
task = LinearDSSampler(observation_dim=16, action_dim=8)

# Create an LINDS environment from the sampled task
env = LinearDSEnv()
env.set_task(task)

# Reset the environment
observation, info = env.reset()

# Take a step in the environment
action = np.random.randn(env.action_dim)
next_observation, reward, done, truncated, info = env.step(action)
```