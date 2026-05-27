# MazeWorld

English | [中文](README.zh.md)

MazeWorld is a procedurally generated 3D maze-navigation environment. Each task randomizes maze topology, textures, navigation targets, command sequences, and physical scale, making it useful for navigation, exploration, meta-RL, in-context adaptation, and agent-environment interaction research.

Unlike simple object-navigation benchmarks that can often be handled with strong zero-shot priors alone, MazeWorld is designed to require iterative interaction, memory, and environment-specific adaptation.

<div style="width: 960; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-1.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-2.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-3.jpg" alt="Keyboard Demo">
</div>

## What This Module Provides

- Procedurally generated 3D maze tasks.
- A Gymnasium environment with both discrete and continuous action modes.
- Randomized command-following navigation tasks.
- Task resampling utilities.
- A built-in SLAM-style baseline agent.
- Local-map and global-map accessors for analysis and visualization.
- Keyboard and scripted demos.

## Installation

Install from PyPI:

```bash
pip install "xenoverse[mazeworld]"
```

Install from source:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[mazeworld]"
```

## Main Components

The main public entry points are:

- `xenoverse.mazeworld.MazeWorldContinuous3D`: main 3D environment class.
- `xenoverse.mazeworld.MazeTaskSampler`: random task sampler.
- `xenoverse.mazeworld.Resampler`: task resampler.
- `xenoverse.mazeworld.agents.SmartSLAMAgent`: built-in navigation baseline.

The registered environment ID is:

- `mazeworld-v2`

## Recommended Workflow

The standard usage pattern is:

1. Create the environment.
2. Sample or load a task.
3. Call `env.set_task(task)`.
4. Call `env.reset()`.
5. Run the environment with `step(...)`.

Important: `reset()` requires that a task has already been attached.

## Quick Start

### 1. Create the environment

```python
import gymnasium as gym
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler

env = gym.make(
    "mazeworld-v2",
    enable_render=False,
    action_space_type="Discrete16",
)
```

If you want on-screen rendering, set `enable_render=True`. That requires GUI access.

### 2. Sample and set a task

```python
task = MazeTaskSampler()
env.set_task(task)
obs, info = env.reset()
```

### 3. Run with random actions

```python
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## What a Task Contains

`MazeTaskSampler(...)` produces a task dictionary that includes randomized components such as:

- maze wall topology
- start position
- landmark locations
- command sequence
- cell size
- wall height
- agent height
- textures
- rewards
- field of view

So two sampled tasks can differ in geometry, scale, difficulty, visual appearance, and target order.

## Task Sampling

The task sampler supports a range of controls over maze generation.

Example:

```python
from xenoverse.mazeworld import MazeTaskSampler

task = MazeTaskSampler(
    n_range=(9, 25),
    allow_loops=True,
    cell_size_range=(1.5, 4.5),
    wall_height_range=(2.0, 6.0),
    agent_height_range=(1.6, 2.0),
    landmarks_number_range=(5, 10),
    commands_sequence=200,
    wall_density_range=(0.2, 0.4),
)
```

Useful parameters:

- `n_range`: maze grid-size range. The sampled maze is forced to odd size internally.
- `allow_loops`: whether the maze may contain loops rather than a strict tree structure.
- `cell_size_range`: physical size of each grid cell.
- `wall_height_range`: wall height range.
- `agent_height_range`: camera or robot height range.
- `landmarks_number_range`: number of possible navigation targets.
- `commands_sequence`: number of target commands in one task episode.
- `wall_density_range`: density used during maze generation.
- `step_reward`, `collision_reward`, `goal_reward`: reward shaping terms.

Example of forcing a fixed 15x15 maze with 2-meter cells:

```python
task = MazeTaskSampler(
    n_range=(15, 15),
    cell_size_range=(2.0, 2.0),
)
```

## Task Resampling

If you want to preserve the current maze layout but change targets, command sequences, or start positions, use `Resampler(...)`.

```python
from xenoverse.mazeworld import Resampler

new_task = Resampler(task)
```

Typical uses:

- keep maze geometry but resample commands
- keep geometry but resample start position
- optionally resample landmarks or landmark colors

This is useful when you want multiple related navigation episodes within one scenario.

## Observation and Info

### Observation

MazeWorld returns an RGB image observation from the agent's first-person 3D view.

By default:

- observation space is an image tensor with shape `(H, W, 3)`
- values are `uint8`

The exact shape depends on the `resolution` argument passed to the environment.

### Info dictionary

`reset()` and `step()` return metadata including:

- `steps`: current step count
- `command`: current target command represented as RGB color

The command color tells the agent which landmark should be reached next.

## Command Representation

MazeWorld uses color-coded commands. The current target is represented by a landmark color rather than text.

If you want the command embedded directly into the image observation, create the environment with:

```python
env = gym.make(
    "mazeworld-v2",
    command_in_observation=True,
    enable_render=False,
)
```

You can also read the current command color from `info["command"]`.

<div style="width: 480; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/CommandDemo.jpg" alt="command_in_observation">
</div>

## Action Spaces

MazeWorld supports three action-space modes through `action_space_type`:

- `Discrete16`
- `Discrete32`
- `Continuous`

Example:

```python
env = gym.make(
    "mazeworld-v2",
    action_space_type="Discrete16",
    enable_render=False,
)
```

### Discrete modes

- `Discrete16` is the default.
- `Discrete32` exposes a larger discrete action set.

These are the supported modes for the built-in `SmartSLAMAgent`.

### Continuous mode

In `Continuous` mode, the action is a 2D vector:

- first component: turning command
- second component: walking speed

Both components are clipped into `[-1, 1]`.

<div style="width: 240; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Dynamics.jpg" alt="Robot Dynamics">
</div>

## Rendering

If `enable_render=True`, calling `env.render()` shows:

- the first-person observation
- a global map view
- a local map view

Example:

```python
env = gym.make("mazeworld-v2", enable_render=True)
```

During manual control, keyboard interaction is available through the demo script described below.

## Keyboard Demo

You can control the environment manually with:

```bash
python -m xenoverse.mazeworld.demo.keyboard_play_demo --help
```

Common arguments include:

- `--max_steps`
- `--visibility_3D`
- `--save_replay`
- `--verbose`

## Built-in Smart Agent

MazeWorld includes a SLAM-and-planning-style baseline:

```python
from xenoverse.mazeworld.agents import SmartSLAMAgent

agent = SmartSLAMAgent(
    maze_env=env,
    memory_keep_ratio=0.25,
    render=False,
)

terminated = False
truncated = False
reward = 0.0

while not (terminated or truncated):
    action = agent.step(obs, reward)
    obs, reward, terminated, truncated, info = env.step(action)
```

Notes:

- `memory_keep_ratio=1.0` means near-perfect long-term retention.
- Lower values simulate forgetting.
- The built-in agent requires a discrete action space, not `Continuous`.
- Avoid using `agent.render=True` together with `env.enable_render=True`; both want to own a rendering window.

![Demonstration-Agent-Control](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/AgentDemo.gif)

## Agent and Teacher Use Cases

The built-in smart agent is best treated as:

- a strong heuristic baseline
- a trajectory generator
- a teacher policy for imitation or dataset generation

It is not guaranteed to be globally optimal.

## Accessing Maps

You can directly query both local and global maps from the environment.

```python
local_map = env.get_local_map()
global_map = env.get_global_map()
```

Important detail:

- each function returns a tuple containing a pygame surface and a NumPy array
- if you only want the NumPy array, unpack the second element

Example:

```python
local_surface, local_array = env.get_local_map()
global_surface, global_array = env.get_global_map()
```

The local map is aligned to the agent viewpoint. The global map shows the full maze layout.

## Saving the Agent Trajectory

At the end of an episode, you can save a trajectory visualization:

```python
env.save_trajectory("trajectory.png")
```

This produces an image showing the path taken by the agent on the maze map.

<div style="width: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/TrajectoryDemo.png" alt="Robot Trajectory">
</div>

## Rewards

The default reward structure includes:

- per-step reward, usually `0`
- positive reward for reaching the current target
- collision penalty

You can customize these through task-sampler arguments such as:

- `step_reward`
- `goal_reward`
- `collision_reward`

## Common Pitfalls

- Call `env.set_task(task)` before `env.reset()`.
- `SmartSLAMAgent` works only with `Discrete16` or `Discrete32`.
- Do not enable both agent-side rendering and environment-side rendering at the same time unless you intentionally want competing windows.
- `get_local_map()` and `get_global_map()` return `(surface, numpy_array)`, not just the array.
- `command_in_observation=True` changes the image observation by overlaying the command color bar.

## File Guide

- `envs/task_sampler.py`: maze task generation and resampling.
- `envs/maze_env.py`: Gymnasium wrapper and environment-facing APIs.
- `envs/maze_continuous_3d.py`: 3D observation and movement core.
- `agents/smart_slam_agent.py`: built-in SLAM-style baseline.
- `demo/keyboard_play_demo.py`: manual keyboard demo.
- `demo/agent_play_demo.py`: scripted smart-agent demo.

## References

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}

@article{wang2025context,
  title={Context and Diversity Matter: The Emergence of In-Context Learning in World Models},
  author={Wang, Fan and Chen, Zhiyuan and Zhong, Yuxuan and Zheng, Sunjian and Shao, Pengtao and Yu, Bo and Liu, Shaoshan and Wang, Jianan and Ding, Ning and Cao, Yang and others},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  year={2026}
}
```
