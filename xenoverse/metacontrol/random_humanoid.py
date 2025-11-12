import numpy
import gymnasium as gym
import pygame
from numpy import random
from numba import njit
from gymnasium import spaces
from xenoverse.utils import pseudo_random_seed, versatile_sample
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv


class RandomHumanoidEnv(HumanoidEnv):
    """
    Randomly sampled humanoid environment from mujoco-py
    """
    def __init__(self, seed=None, **kwargs):
        self.kwargs = kwargs
        super().__init__(**self.kwargs)
        self.seed(seed)

    def seed(self, seed=None):
        if(seed is None):
            pseudo_random_seed(0)
        else:
            pseudo_random_seed(seed)

    def set_task(self, task):
        super().__init__(xml_file=task, **self.kwargs)
