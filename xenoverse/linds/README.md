# Introduction

Scalable procedurally generated Linear Time Invariant system by randomizing the transition and rewards.

# Implementation

Linear Time Invariant (LTI) systems represent a class of systems where the transition dynamics and reward function are linear functions. The state-space representation for an LTI system is:

$dx/dt = A x + B u + X$

$o(t) = C x(t) + Y$

In this implementation, we randomly generate the matrices $A$, $B$, $C$ as well as the bias vectors $X$ and $Y$.

The reward function is also procedurally generated, with a base reward and a factor that scales the distance between the current observation and the command.

$r(t)=r_0 - \alpha ||c(t) - D \cdot o(t) - E||^2 - \beta||u(t)||^2$

The initial states are randomly generated and the command is also randomly chosen.

# Usage

```python
import numpy as np
from xenoverse.linds import LinearDSEnv, LinearDSSamplerRandomDim

# Create a task sampler with 10-dimensional state and action spaces
task = LinearDSSampler(observation_dim=16, action_dim=8)

# LinearDSSamplerRandomDim samples tasks with random dimensions between 1 and the specified max values
task = LinearDSSamplerRandomDim(max_state_dim=16, max_action_dim=8)

# Create an LINDS environment from the sampled task
env = LinearDSEnv(pad_observation_dim=16, pad_action_dim=8, pad_command_dim=16)
# pad_observation_dim, pad_action_dim and pad_command_dim are used to ensure the environment's observation space constant
# the observation and action space of the actual task might be smaller than these values, but the environment will pad them with zeros to match these dimensions
# e.g., if the actual observation space is 3-dimensional and pad_observation_dim=5,
# then the observation returned by env will be [o1, o2, o3, 0, 0]
env.set_task(task, use_pad_dim=True) # use_pad_dim=True ensures that the environment uses padded dimensions

# Reset the environment
observation, info = env.reset()

# Take a step in the environment
action = np.random.randn(env.action_dim)
next_observation, reward, done, truncated, info = env.step(action)
```