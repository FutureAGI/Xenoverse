# AnyHVAC

English | [中文](README.zh.md)

AnyHVAC is a procedurally generated HVAC control environment family for data-center-like indoor spaces. Each sampled task builds a different room layout, sensor placement, heat-source pattern, and cooling configuration, making it useful for control, reinforcement learning, and robustness experiments.

![AnyHVAC Visualizer](https://github.com/FutureAGI/DataPack/blob/main/demo/anyhvac/hvac_video.gif)

## What This Module Provides

- Randomly generated HVAC tasks with different room sizes, walls, sensors, coolers, and heat-emitting equipment.
- Temperature-control environments with Gymnasium-compatible `reset()` / `step()` APIs.
- A visualizer environment for interactive inspection.
- Built-in PID-style baseline solvers in `anyhvac_solver.py`.

## Installation

Install from PyPI:

```bash
pip install "xenoverse[anyhvac]"
```

Install from source:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[anyhvac]"
```

## Main Components

The most useful entry points are:

- `xenoverse.anyhvac.anyhvac_env.HVACEnv`: base HVAC environment.
- `xenoverse.anyhvac.anyhvac_env_vis.HVACEnvVisible`: environment with rendering.
- `xenoverse.anyhvac.anyhvac_sampler.HVACTaskSampler`: random task generator.
- `xenoverse.anyhvac.anyhvac_solver.HVACSolverGTPID`: baseline PID-style controller.

Note: the package `xenoverse.anyhvac` exports the environment classes, but the task sampler and solvers should be imported from their submodules directly.

## Recommended Workflow

AnyHVAC is easiest to use in four steps:

1. Create an environment instance.
2. Sample or load a task.
3. Call `env.set_task(task)`.
4. Call `env.reset()` and then interact with the environment.

Important: `reset()` assumes a task has already been attached. In practice, call `set_task(...)` before the first `reset()`.

## Quick Start

### 1. Create an environment and sample a task

```python
from xenoverse.anyhvac.anyhvac_env import HVACEnv
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler

env = HVACEnv(
    max_steps=5040,
    iter_per_step=600,
    set_lower_bound=16,
    set_upper_bound=32,
    verbose=False,
)

task = HVACTaskSampler(
    control_type="Temperature",
    target_temperature=26.0,
)

env.set_task(task)
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

This is the safest minimal example because the action format depends on how the environment is configured.

## Using the Visualizer

```python
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler

env = HVACEnvVisible(verbose=True)
task = HVACTaskSampler(control_type="Temperature")

env.set_task(task)
obs, info = env.reset()
```

The visualizer is useful for debugging task layouts, temperature diffusion, and cooler behavior.

## Understanding the Task

`HVACTaskSampler(...)` returns a task dictionary consumed by `env.set_task(task)`. A sampled task includes values such as:

- Room width and length.
- Grid resolution and floor height.
- Ambient temperature.
- Sensor objects.
- Cooling-unit objects.
- Heat-emitting equipment objects.
- Convection and heat-capacity parameters.
- Target temperature.

Because tasks are procedurally generated, the number of sensors, coolers, and equipment units changes from one sample to another.

## Observation Structure

After `set_task(...)`, the environment builds a dictionary observation space. The exact keys depend on configuration flags, but commonly include:

- `sensor_readings`: temperatures reported by sensors.
- `heat_readings`: current heat output of equipment.
- `action_temp`: last commanded cooling temperatures.
- `timestep`: current step index.

The observation is therefore not a single flat tensor by default. Most users should inspect `env.observation_space` after `set_task(...)`.

## Action Structure

The action space is also created after `set_task(...)`, because it depends on the number of coolers in the sampled task.

Common cases:

- If `no_switch_action=True` (default), the action is a `Box` with one normalized value per cooler.
- If `no_switch_action=False`, the action is a `Box` containing switch values and control values for each cooler.
- If `action_space_format="dict"`, the action becomes a dictionary with separate `switch` and `value` fields.

For temperature control, normalized action values are mapped into the configured setpoint range `[set_lower_bound, set_upper_bound]`.

## Reward and Termination Notes

The environment mixes several effects into the reward, including:

- Staying close to the target temperature.
- Energy-related cost.
- Action or switching penalties.
- Failure penalties when the environment overheats badly.

The exact balance depends on the environment configuration and reward mode. Internally, the failure threshold is derived from the sampled target temperature, so older examples that pass parameters such as `upper_limit` or `tolerance` are not accurate for the current code.

## Built-in PID Baselines

Baseline controllers live in `xenoverse.anyhvac.anyhvac_solver`.

Example import:

```python
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
```

Use these solvers as reference implementations rather than drop-in defaults. Their expected action format may not match every environment configuration, especially if you change:

- `no_switch_action`
- `action_space_format`
- the specific environment subclass

If you use a built-in solver, verify that the returned action shape matches `env.action_space`.

## Common Pitfalls

- Call `env.set_task(task)` before `env.reset()`.
- Import `HVACTaskSampler` from `xenoverse.anyhvac.anyhvac_sampler`, not from the package root.
- Do not rely on older examples that use `anyhvac-v0`; the current registration in this repository is `anyhvac-v1`.
- Check `env.action_space` after sampling a task, because the action dimension depends on the number of coolers.
- Check `env.observation_space` after sampling a task, because observation keys depend on configuration flags.

## File Guide

- `anyhvac_env.py`: base environment logic.
- `anyhvac_env_vis.py`: rendering and visualizer environment.
- `anyhvac_sampler.py`: random task generation.
- `anyhvac_solver.py`: PID-style baseline solvers.
- `hvac_config.py`: sampling and physical-parameter ranges.
