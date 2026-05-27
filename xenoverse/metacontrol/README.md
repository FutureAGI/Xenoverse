# MetaControl

English | [中文](README.zh.md)

`xenoverse.metacontrol` provides procedurally randomized control environments built on top of Gymnasium classic-control tasks and MuJoCo humanoid control. The goal is to evaluate robustness, adaptation, and transfer under changing physical parameters and body configurations.

## What This Module Provides

- Randomized CartPole dynamics.
- Randomized Acrobot dynamics.
- Randomized Humanoid morphology through generated MuJoCo XML files.
- Simple task samplers for each environment family.
- `set_task(...)` hooks for injecting sampled task parameters into an environment instance.

## Environment Families

This module currently exposes three registered Gymnasium environments:

- `random-cartpole-v0`
- `random-acrobot-v0`
- `random-humanoid-v0`

## Installation and Dependencies

Base installation:

```bash
pip install xenoverse
```

Local development install:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

Important dependency notes:

- CartPole and Acrobot rely on Gymnasium classic-control support.
- Humanoid relies on Gymnasium MuJoCo support and a working MuJoCo runtime.
- The repository setup does not fully provision MuJoCo for you, so humanoid experiments may require extra manual environment setup.

## Main Public APIs

The module exports:

- `sample_cartpole`
- `sample_acrobot`
- `sample_humanoid`
- `get_humanoid_tasks`
- `RandomCartPoleEnv`
- `RandomAcrobotEnv`
- `RandomHumanoidEnv`

## Recommended Workflow

The intended usage pattern is:

1. Create an environment.
2. Sample a task.
3. Call `env.set_task(task)`.
4. Call `env.reset()`.
5. Interact with the environment using `step(...)`.

For CartPole and Acrobot, a task is a parameter dictionary.  
For Humanoid, a task is the path to a generated MuJoCo XML file.

## Quick Start

### Random CartPole

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole

env = gym.make("random-cartpole-v0")
task = sample_cartpole()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Random Acrobot

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_acrobot

env = gym.make("random-acrobot-v0")
task = sample_acrobot()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Random Humanoid

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_humanoid

env = gym.make("random-humanoid-v0")
task = sample_humanoid()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## 1. Random CartPole

Source:

- [random_cartpole.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_cartpole.py)

Task sampler:

- `sample_cartpole(...)`

Sampled parameters:

- gravity
- cart mass
- pole mass
- pole length

Environment:

- `RandomCartPoleEnv`

Notable behavior:

- extends Gymnasium `CartPoleEnv`
- supports `set_task(task_config)`
- supports configurable `frameskip`
- supports configurable randomized reset scale through `reset_bounds_scale`

Practical note:

- `step(...)` internally repeats the chosen action for `frameskip` simulator steps and accumulates reward.

## 2. Random Acrobot

Source:

- [random_acrobot.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_acrobot.py)

Task sampler:

- `sample_acrobot(...)`

Sampled parameters:

- link lengths
- link masses
- centers of mass
- gravity

Environment:

- `RandomAcrobotEnv`

Notable behavior:

- extends Gymnasium `AcrobotEnv`
- rewrites the dynamics so sampled physical parameters actually change the motion equations
- supports `set_task(task_config)`
- supports configurable `frameskip`
- supports configurable randomized reset scale through `reset_bounds_scale`

Practical note:

- termination depends on the randomized geometry, so difficulty can change significantly across tasks.

## 3. Random Humanoid

Sources:

- [random_humanoid.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_humanoid.py)
- [humanoid_xml_sampler.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/humanoid_xml_sampler.py)

Task sampler:

- `sample_humanoid(root_path=None, noise_scale=1.0)`

What it returns:

- a filesystem path to a generated MuJoCo XML file

What is randomized:

- limb sizes and proportions
- torso and pelvis geometry
- joint armature, damping, stiffness, and ranges
- actuator gear values
- morphology-related contact properties

Environment:

- `RandomHumanoidEnv`

Notable behavior:

- extends Gymnasium `HumanoidEnv`
- `set_task(task)` reloads the model from the sampled XML file
- healthy height range is adjusted from the sampled torso geometry

Practical note:

- a humanoid task is not a Python parameter dictionary like CartPole and Acrobot; it is an XML file path.

## Reusing Pre-generated Humanoid Tasks

If you already have a directory of generated humanoid XML files, you can list them with:

```python
from xenoverse.metacontrol import get_humanoid_tasks

tasks = get_humanoid_tasks("path/to/xml_dir")
```

This is useful when you want:

- reproducible evaluation sets
- fixed train/test task splits
- repeated experiments on the same morphology pool

## Reset and Task Injection Behavior

All three environments support `set_task(...)`, but the task payload differs:

- CartPole: dictionary of scalar physical parameters
- Acrobot: dictionary of scalar physical parameters
- Humanoid: path to a generated XML file

Recommended order:

1. `env = gym.make(...)`
2. `task = sample_*()`
3. `env.set_task(task)`
4. `env.reset()`

## Frameskip and Reset Randomization

CartPole and Acrobot both expose two useful knobs through environment construction:

- `frameskip`
- `reset_bounds_scale`

`frameskip`:

- repeats the chosen action for multiple simulator steps
- changes the effective control frequency

`reset_bounds_scale`:

- changes how large the randomized initial state can be
- higher values generally make episodes harder

The registered defaults in this repository are:

- CartPole: `frameskip=1`
- Acrobot: `frameskip=1`

## Choosing the Right Family

Use:

- CartPole when you want a lightweight classic-control benchmark with randomized dynamics.
- Acrobot when you want underactuated swing-up behavior with randomized physical structure.
- Humanoid when you want high-dimensional continuous control with randomized body morphology.

## Common Pitfalls

- Call `env.set_task(...)` before `env.reset()`.
- Do not treat `sample_humanoid()` like the other samplers; it returns an XML file path, not a parameter dict.
- Humanoid requires working MuJoCo support in your Python environment.
- Different task families have very different observation and action spaces; policies are not interchangeable across them.
- If you increase `frameskip` or reset randomness, your previous baselines may no longer be comparable.

## Testing

There is a basic humanoid smoke test at:

- [tests/test.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/tests/test.py)

At the moment, the included test focuses on humanoid rollout rather than exhaustive validation of all three families.
