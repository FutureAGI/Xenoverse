#!/usr/bin/env python
# coding=utf8
# File: test.py
import os

import gymnasium as gym

import xenoverse.metacontrol
from xenoverse.metacontrol import sample_humanoid


def test_humanoid(max_steps=1000):
    env = gym.make("random-humanoid-v0")
    task = sample_humanoid()
    try:
        env.set_task(task)
        env.reset()
        terminated, truncated = False, False
        sum_reward = 0
        steps = 0
        while not terminated and not truncated and steps < max_steps:
            _, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            sum_reward += reward
            steps += 1
        print(
            "...Test Finishes. Get score %f, steps = %s\n\n---------\n\n"
            % (sum_reward, steps)
        )
    finally:
        env.close()
        os.unlink(task)


if __name__ == "__main__":
    for _ in range(10):
        test_humanoid()
    print("\n\nCongratulations!!!\n\nAll Tests Have Been Passed\n\n")
