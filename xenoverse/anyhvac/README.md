# Introduction

Scalable simulation environments for HVAC in IDC, with casual models of second law of thermodynamics

![AnyHVACVisualizer](https://github.com/FutureAGI/DataPack/blob/main/demo/anyhvac/hvac_video.gif) 

# Install

```bash
pip install xenoverse[anyhvac]
```

#### For local installation, execute following commands:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .[anyhvac]
```

# Quick Start

## Import

Import and create the AnyHVAC environment:

```python
import gymnasium as gym
import xenoverse.anyhvac

env = gym.make("anyhvac-visualizer-v0", # use `anyhvac-v0` for non-visualizer version
                max_steps=86400, # max time in seconds
                target_temperature=28, # target temperature in Celsius, highest reward at this position
                upper_limit=80, # upper limit of temperature in Celsius, failure at this position
                iter_per_step=600） # number of iterations per step, actual time elapsed=iter_per_step * 0.2


```

## Sampling an HVAC control task

An HVAC task include random number of **sensors**, **coolers**, and **equipments** randomly aranged in the building. There might also be random walls in it. The task can be sampled by:

```python
from xenoverse.anyhvac import HVACTaskSampler

task = AnyMDPTaskSampler()
env.set_task(task)
observation, info = env.reset()
```

## Running the built-in PID solver based on sensor - actuator correlation
```python
from xenoverse.anyhvac import HVACSolverGTPID

solver = HVACSolverGTPID(env)  # Use PID controller and chtc to solve
state, info = env.reset()
terminated, truncated = False, False
while (not terminated) and (not truncated):
    action = solver.policy()
    state, reward, terminated, truncated, info = env.step(action)
```