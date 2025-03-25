# Introduction

Scalable simulation environments for HVAC in IDC, with casual models of second law of thermodynamics

![AnyHVACVisualizer](https://github.com/FutureAGI/DataPack/blob/main/demo/anymdp/AnyHVACVisualizer.gif) 

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
import gym
import xenoverse.anyhvac

env = gym.make("anyhvac-visualizer-v0", # use `anyhvac-v0` for non-visualizer version
                max_steps=86400, # max time in seconds
                target_temperature=28, # target temperature in Celsius, highest reward at this position
                upper_limit=80）
```

## Sampling an HVAC control task

An HVAC task include random number of **sensors**, **coolers**, and **equipments** randomly aranged in the building. There might also be random walls in it. The task can be sampled by:

```python
from xenoverse.anyhvac import HVACTaskSampler

task = AnyMDPTaskSampler()
env.set_task(task)
env.reset()
```

## Running the built-in PID solver based on sensor - actuator correlation
```python
from xenoverse.anyhvac import HVACSolverGTPID

solver = HVACSolverGTPID(env)  # Use PID controller and chtc to solve
state, info = env.reset()
done = False
while not done:
    action = solver.policy()
    state, reward, done, info = env.step(action)
```

