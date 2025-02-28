#!/usr/bin/env python
# coding=utf8
# File: test.py
import gym
import sys
import l3c.mazeworld
import time
from l3c.mazeworld import MazeTaskSampler, Resampler
from l3c.mazeworld.agents import SmartSLAMAgent
from numpy import random

def test_agent_maze(max_steps=1000):
    maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps)
    task = MazeTaskSampler(verbose=True)
    maze_env.set_task(Resampler(task))

    # Must intialize agent after reset
    agent = SmartSLAMAgent(maze_env=maze_env, memory_keep_ratio=0.25, render=False)

    done=False
    observation = maze_env.reset()
    sum_reward = 0
    reward = 0
    while not done:
        action = agent.step(observation, reward)
        observation, reward, done, _ = maze_env.step(action)
        loc_map = maze_env.get_local_map()
        global_map = maze_env.get_global_map()
        sum_reward += reward
    print("...Test Finishes. Get score %f, steps = %s\n\n---------\n\n"%(sum_reward, max_steps))

if __name__=="__main__":
    for _ in range(10):
        test_agent_maze(max_steps=100)
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
