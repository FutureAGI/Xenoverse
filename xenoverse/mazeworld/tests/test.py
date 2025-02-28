#!/usr/bin/env python
# coding=utf8
# File: test.py
import gym
import sys
import l3c.mazeworld
from l3c.mazeworld import MazeTaskSampler
from numpy import random

def test_maze(max_steps=1000):
    maze_env = gym.make("mazeworld-v2", enable_render=False, max_steps=max_steps)
    task = MazeTaskSampler(verbose=True)
    maze_env.set_task(task)

    maze_env.reset()
    done=False
    sum_reward = 0
    while not done:
        state, reward, done, _ = maze_env.step(maze_env.action_space.sample())
        sum_reward += reward
    print("...Test Finishes. Get score %f, for maze_type = %s task_type = %s, n = %d, steps = %s\n\n---------\n\n"%(sum_reward, maze_type, task_type, n, max_steps))

if __name__=="__main__":
    for n in [9, 15, 25]:
        for task_type in ["NAVIGATION", "SURVIVAL"]:
            for maze_type in ["Discrete2D", "Discrete3D", "Continuous3D"]:
                n_landmarks=random.randint(2,10)
                density=random.random() * 0.50
                test_maze(n=n, task_type=task_type, density=density, maze_type=maze_type, n_landmarks=n_landmarks)
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
