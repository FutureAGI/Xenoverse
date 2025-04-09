# Introduction

Differences from Version 1.0:

1.Air Conditioning Control: The system now manages two parameters: the switch and the set temperature, rather than directly controlling power. The output power is calculated based on the air conditioner's set temperature and the return air temperature.​

2.Area Division: An additional layer of areas has been added on top of the existing cells. Each area has its own target temperature, fluctuating by ±1°C around a baseline target temperature.​

3.Reward Function: The temperature penalty in the reward function is now calculated based on the difference between the maximum sensor temperature in each area and its target temperature. Areas without sensors are excluded from this calculation. The coefficients have been adjusted in new reward function.


# Quick Start

## Import

Import and create the AnyHVAC environment:

```python
import gym
import xenoverse.anyhvac

env = gym.make("anyhvac-visualizer-v1", # use `anyhvac-v0` for non-visualizer version
                max_steps=86400, # max time in seconds
                target_temperature=28, # target temperature in Celsius, highest reward at this position
                upper_limit=80, # upper limit of temperature in Celsius, failure at this position
                iter_per_step=600, 
                set_lower_bound=16, # lower limit of ac set temperature
                set_upper_bound=32, # upper limit of ac set temperature
                tolerance=1, # temperature tolerance for reward calculation
               ) # number of iterations per step, actual time elapsed=iter_per_step * 0.2


```

## Sampling an HVAC control task

An HVAC task include random number of **sensors**, **coolers**, and **equipments** randomly aranged in the building. There might also be random walls in it. The task can be sampled by:

```python
from xenoverse.anyhvacv2 import HVACTaskSampler

task = HVACTaskSampler()
env.set_task(task)
env.reset()
```

## Running with random actions
```python

state, info = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

