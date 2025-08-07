"""
Gym Environment For Any MDP
"""
import numpy
import gymnasium as gym
import pygame
from numpy import random
from numba import njit
from gymnasium import spaces
from xenoverse.utils import pseudo_random_seed
from gymnasium.envs.classic_control.cartpole import CartPoleEnv

def sample_cartpole(gravity_scope=[9.8, 9.8],
                    masscart_scope=[1.0, 1.0],
                    masspole_scope=[0.1, 0.1],
                    length_scope=[0.5, 0.5]):
    # Sample a random cartpole task
    pseudo_random_seed(0)
    def sample(scope):
        return random.uniform(scope[0], scope[1])
    return {
        "gravity": sample(gravity_scope),
        "masscart": sample(masscart_scope),
        "masspole": sample(masspole_scope),
        "length": sample(length_scope)
    }

class RandomCartPoleEnv(CartPoleEnv):

    def __init__(self, *args, **kwargs):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        super().__init__(*args, **kwargs)

    def set_task(self, task_config):
        print("Setting task with config:", task_config)
        self.gravity = task_config.get("gravity", 9.8)
        self.masscart = task_config.get("masscart", 1.0)
        self.masspole = task_config.get("masspole", 0.1)
        self.length = task_config.get("length", 0.5)  # actually half the pole's length
        self.polemass_length = self.masspole * task_config.get("length", 0.5)
        self.total_mass = self.masspole + self.masscart