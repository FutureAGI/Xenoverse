# MetaControl

`xenoverse.metacontrol` provides randomized control environments built on top of Gymnasium classic-control and MuJoCo humanoid tasks. The goal of this module is to benchmark policy robustness and adaptation under procedurally varied physical parameters and body configurations.

## Scope

This module currently includes three environment families:

- `random-cartpole-v0`: randomized cartpole dynamics
- `random-acrobot-v0`: randomized acrobot dynamics
- `random-humanoid-v0`: randomized humanoid morphology generated from MuJoCo XML

## What Is Randomized

### 1. Random CartPole

Source files:

- [random_cartpole.py](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_cartpole.py)

The helper `sample_cartpole(...)` samples task parameters including:

- gravity
- cart mass
- pole mass
- pole length

The environment class `RandomCartPoleEnv` extends Gymnasium `CartPoleEnv` and adds:

- `set_task(task_config)` for injecting sampled task parameters
- configurable `frameskip`
- configurable randomized reset scale through `reset_bounds_scale`

Registered Gym ID:

- `random-cartpole-v0`

## 2. Random Acrobot

Source files:

- [random_acrobot.py](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_acrobot.py)

The helper `sample_acrobot(...)` samples task parameters including:

- link lengths
- link masses
- centers of mass
- gravity

The environment class `RandomAcrobotEnv` extends Gymnasium `AcrobotEnv` and rewrites the system dynamics so that sampled physical parameters directly affect motion. It also supports:

- `set_task(task_config)` for task injection
- configurable `frameskip`
- configurable reset distribution through `reset_bounds_scale`

Registered Gym ID:

- `random-acrobot-v0`

## 3. Random Humanoid

Source files:

- [random_humanoid.py](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_humanoid.py)
- [humanoid_xml_sampler.py](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/humanoid_xml_sampler.py)

The helper `sample_humanoid(...)` generates a MuJoCo XML file for a randomized humanoid body. The XML sampler perturbs:

- limb sizes and proportions
- torso and pelvis geometry
- joint armature, damping, stiffness, and ranges
- actuator gear values
- contact and morphology-related attributes

The environment class `RandomHumanoidEnv` extends Gymnasium `HumanoidEnv` and loads a sampled XML file via `set_task(task)`.

Registered Gym ID:

- `random-humanoid-v0`

## Installation Notes

Base installation:

```bash
pip install xenoverse
```

For local development:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

Important dependency note:

- `random-cartpole-v0` and `random-acrobot-v0` rely on Gymnasium classic-control environments.
- `random-humanoid-v0` relies on Gymnasium MuJoCo support and a working MuJoCo installation.
- The current repository `setup.py` does not declare MuJoCo as an install dependency, so humanoid experiments may require extra manual setup in your local environment.

## Quick Start

### Random CartPole

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole

env = gym.make("random-cartpole-v0")
task = sample_cartpole()
env.set_task(task)

observation, info = env.reset()
terminated, truncated = False, False
while not terminated and not truncated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

### Random Acrobot

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_acrobot

env = gym.make("random-acrobot-v0")
task = sample_acrobot()
env.set_task(task)

observation, info = env.reset()
terminated, truncated = False, False
while not terminated and not truncated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

### Random Humanoid

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_humanoid

env = gym.make("random-humanoid-v0")
task = sample_humanoid()
env.set_task(task)

observation, info = env.reset()
terminated, truncated = False, False
while not terminated and not truncated:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
```

## Public APIs

The module exports:

- `sample_cartpole`
- `sample_acrobot`
- `sample_humanoid`
- `get_humanoid_tasks`
- `RandomCartPoleEnv`
- `RandomAcrobotEnv`
- `RandomHumanoidEnv`

## Testing

There is a basic humanoid smoke test at:

- [tests/test.py](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/tests/test.py)

At the moment, the included test file focuses on humanoid rollout.
