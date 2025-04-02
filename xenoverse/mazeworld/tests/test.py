#!/usr/bin/env python
# coding=utf8
# File: test.py
import gymnasium as gym
import sys
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler
from numpy import random

def test_maze(max_steps=1000):
    maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps)
    task = MazeTaskSampler(verbose=True)
    maze_env.set_task(task)

    maze_env.reset()
    terminated, truncated=False, False
    sum_reward = 0
    while not terminated and not truncated:
        state, reward, terminated, truncated, _ = maze_env.step(maze_env.action_space.sample())
        sum_reward += reward
    print("...Test Finishes. Get score %f, steps = %s\n\n---------\n\n"%(sum_reward, max_steps))

if __name__=="__main__":
    for _ in range(10):
        test_maze()
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
