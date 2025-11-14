#!/usr/bin/env python
# coding=utf8
# File: test.py
import gymnasium as gym
import sys
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_humanoid, get_humanoid_tasks

def test_humanoid(max_steps=1000):
    env = gym.make("random-humanoid-v0")
    task = sample_humanoid()
    env.set_task(task)

    env.reset()
    terminated, truncated=False, False
    sum_reward = 0
    while not terminated and not truncated:
        state, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        sum_reward += reward
    print("...Test Finishes. Get score %f, steps = %s\n\n---------\n\n"%(sum_reward, max_steps))

if __name__=="__main__":
    for _ in range(10):
        test_humanoid()
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")