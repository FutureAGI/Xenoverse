# Introduction

MazeWorld is a 3D environment with randomly generated mazes and randomly generated navigation targets. It has been implemented in Numpy and supports both discrete and continuous action spaces. MazeWorld can be regarded as one type of ObjectNav tasks. However, unlike other ObjectNav tasks which can be mainly solved by ***Zero-Shot*** capabilities, MazeWorld requires iterative interaction and ***self-adaption*** between the agent and the environment to solve the task. Moreover, due to domain randomization, the maze can not be solved by zero-shot capabilities. MazeWorld is mainly used for research on Meta Reinforcement Learning (*Meta-RL*), especially In-Context Reinforcement Learning (*ICRL*).
<div style="width: 960; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-1.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-2.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-3.jpg" alt="Keyboard Demo">
</div>

## Keyboard Demonstrations

You may try MazeWorld with your own keyboard with the following commands:
```bash
python -m xenoverse.mazeworld.demo.keyboard_play_demo --help
  --max_steps MAX_STEPS
  --visibility_3D VISIBILITY_3D     #3D vision range, Only valid in 3D mode
  --save_replay SAVE_REPLAY         #Save the replay trajectory in file
  --verbose VERBOSE
```

## Smart SLAM-based Agent

We implement a smart SLAM-based agent that can do SLAM & Planning automatically, you can try it with the following command:
```bash
python -m xenoverse.mazeworld.demo.agent_play_demo --help
  --max_steps MAX_STEPS
  --save_replay FILE_NAME #SAVE_REPLAY Save the replay trajectory in file
  --memory_keep_ratio FLOAT #MEMORY_KEEP_RATIO Keep ratio of memory when the agent switch from short to long term memory. 1.0 means perfect memory, 0.0 means no memory
  --verbose VERBOSE 
```

![Demonstration-Agent-Control](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/AgentDemo.gif) 

# Installation

#### Remote installation
```bash
pip install xenoverse[mazeworld]
```

#### Local installation
```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .[mazeworld]
```

# Quick Start with the APIs

Here is an example of creating and running MazeWorld environments

## Creating Maze Environments
```python
import gymnasium as gym
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler

# Make sure you have access to GUI if setting enable_render=True
maze_env = gym.make("mazeworld-v2", enable_render=True)

# In case you want to run the environment in the backend, set enable_render=False
# maze_env = gym.make("mazeworld-v2", enable_render=False)
```

## Sampling a maze task

```python
#Sample a random maze task
task = MazeTaskSampler()
```

It is important to note that sampling a task might result in a maze that is variant in topology, texture, navigation targets, size, height of the robot, height of the wall, commands etc.

In case you want to resample a task, while keeping the some of your environment settings unchanged, you can do the following:

```python
#Sample a new task from a existing task, keep the scenario unchanged, only change the commands and start point
from xenoverse.mazeworld import Resampler
new_task = Resampler(task)
```

## Running agent-environment interaction

Here is an simplest version of running the maze environment step by step

```python
#Set the task configuration to the meta environment
maze_env.set_task(task)
initial_observation, initial_information = maze_env.reset()

#Start the task
done = False
while not done:
    action = maze_env.action_space.sample() # Replace it with your own policy function
    observation, reward, terminated, truncated, info = maze_env.step(action)
    maze_env.render()
```

## Using the built-in agents

We implement a smart agent with simulated SLAM and planning abilities. It can effectively employ a Exploration-then-Exploitation strategy to explore the environment and navigate to the target. Notice that the agent can be used as high-level baseline and teacher for generating trajectories efficiently, but it is not guarranteed to be achieve global optimal performance.

```python
from xenoverse.mazeworld.agents import SmartSLAMAgent

agent = SmartSLAMAgent(maze_env=maze_env, memory_keep_ratio=0.25, render=True) # memory_keep_ratio=0.25 means the agent only keeps 25% of what it sees in the long term memory
action = agent.step(observation, reward)
```
It's important to be aware that the "render=True" option cannot be utilized concurrently with "enable_render=True" when configuring the maze environment.

# High-level APIs

## Configurating the task sampler

You may pass arguments to the task sampler to control the generation of maze tasks

```python
MazeTaskSampler(
  n_range=(9, 25),  # The range of the grids used in the maze
  allow_loops=True,  # Whether to allow loops in the maze
  cell_size_range=(1.5, 4.5), # The range of the size of each grid (cell)
  wall_height_range=(2.0, 6.0),  # The range of the height of the wall
  agent_height_range=(1.6, 2.0), # The range of the height of the robot
  landmarks_number_range=(5, 10), # The range of the number of landmarks (navigation targets)
  commands_sequence=200, # The number of commands in navigation
  wall_density_range=(0.2, 0.4)) # The range of the density that controls the fraction of the wall
```

For instance if you want to generate mazes of 15x15 grids with grid size being always 2.0 (which would give a 30mx30m (900m^2) maze, you can do the following:)
```python
task = MazeTaskSampler(n_range=(15, 15), cell_size_range=(2.0, 2.0))
```

## Robot Action Space

You may choose different action space for the robot by setting the "action_space_type" argument when configuring the maze environment. E.g., the default choice for action space is "Discrete16", which means the robot have 16 different actions to choose from.

```python
maze_env = gym.make("mazeworld-v2", action_space_type="Discrete16", enable_render=False)
```

It is also possible to use "Discrete32" and "Continuous". If using the built-in agent, you can not set the "action_space_type" argument to "Continuous", only "Discrete16" and "Discrete32" are supported.

The action space for the MazeWorld follows the dynamics of the two-wheel-steering robot. For "Continuous" action space, the action is a 2-dimensional vector, where the first element controls the steering angle and the second element controls the speed. As shown in the figure below:

<div style="width: 240; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Dynamics.jpg" alt="Robot Dynamics">
</div>

## Accessing the local and global map

You might want to directly access the local and global map of the maze environment. You can do so by calling "get_local_map()" and "get_global_map()", which return a numpy array of the local and global map respectively. The local map is represented in the local coordinate system, while the global map is represented in the global coordinate system.

```python
maze_env = gym.make("mazeworld-v2", enable_render=False)
local_map = maze_env.get_local_map()
global_map = maze_env.get_global_map()
```

## Retrieve the trajectory of the agent

To retrieve the trajectory of the agent, you can call "save_trajectory()" at the end of each episode. The function returns a image with the trajectory of the agent in the global map, as shown in the figure below:

<div style="width: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/TrajectoryDemo.png" alt="Robot Trajectory">
</div>

## Reward Setting

The MazeWorld defaultly uses a reward 0 for each step, a positive reward relating to the scale of the maze for reaching the target, and punishment for collision. You can also customize the reward function by setting "step_reward", "goal_reward", and "collision_punishment" in the MazeTaskSampler function.

## Reading the commands

The commands in MazeWorld are represented by specific color. You might choose to embed the command as a color bar in the 3D observtion by setting "command_in_sequence=True" when initializing the environment. You may also directly access the rgb color of the command by the returned information of "step()" function:

```python
...
  ...
    ...
    observation, reward, done, info = maze_env.step(action)
    rgb_command = info["command"]
```

<div style="width: 480; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/CommandDemo.jpg" alt="command_in_observation">
</div>