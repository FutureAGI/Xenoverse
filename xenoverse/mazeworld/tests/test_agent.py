#!/usr/bin/env python
# coding=utf8
# File: test.py
import gymnasium as gym
import sys
import xenoverse.mazeworld
import time
from xenoverse.mazeworld import MazeTaskSampler, Resampler
from xenoverse.mazeworld.agents import SmartSLAMAgent
from numpy import random

def test_agent_maze(max_steps=1000):
    maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps)
    task = MazeTaskSampler(verbose=True)
    maze_env.set_task(Resampler(task))

    # Must intialize agent after reset
    agent = SmartSLAMAgent(maze_env=maze_env, memory_keep_ratio=0.25, render=False)

    terminated, truncated=False, False
    observation = maze_env.reset()
    sum_reward = 0
    reward = 0
    while not terminated and not truncated:
        action = agent.step(observation, reward)
        observation, reward, terminated, truncated, _ = maze_env.step(action)
        loc_map = maze_env.get_local_map()
        global_map = maze_env.get_global_map()
        sum_reward += reward
    print("...Test Finishes. Get score %f, steps = %s\n\n---------\n\n"%(sum_reward, max_steps))

if __name__=="__main__":
    for _ in range(10):
        test_agent_maze(max_steps=100)
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
