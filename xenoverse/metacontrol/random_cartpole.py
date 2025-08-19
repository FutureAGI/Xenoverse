"""
Gym Environment For Any MDP
"""
import numpy
import gymnasium as gym
import pygame
from numpy import random
from numba import njit
from gymnasium import spaces
from xenoverse.utils import pseudo_random_seed, versatile_sample
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

def sample_cartpole(gravity_scope=True,
                    masscart_scope=True,
                    masspole_scope=True,
                    length_scope=True):
    # Sample a random cartpole task
    pseudo_random_seed(0)
    gravity = versatile_sample(gravity_scope, (1, 11), 9.8)
    masscart = versatile_sample(masscart_scope, (0.5, 2.0), 1.0)
    masspole = versatile_sample(masspole_scope, (0.05, 0.20), 0.1)
    length = versatile_sample(length_scope, (0.25, 1.0), 0.5)  # actually half the pole's length

    return {
        "gravity": gravity,
        "masscart": masscart,
        "masspole": masspole,
        "length": length
    }

class RandomCartPoleEnv(CartPoleEnv):

    def __init__(self, *args, **kwargs):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.frameskip = kwargs.get("frameskip", 5)
        kwargs.pop("frameskip", None)
        super().__init__(*args, **kwargs)

    def set_task(self, task_config):
        for key, value in task_config.items():
            setattr(self, key, value)
        self.polemass_length = self.masspole * self.length
        self.total_mass = self.masspole + self.masscart

    def step(self, action):
        total_reward = 0
        terminated = False
        truncated = False
        for _ in range(self.frameskip):
            obs, reward, terminated, truncated, info = super().step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info