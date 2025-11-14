import numpy
import gymnasium as gym
import pygame
import xml.etree.ElementTree as ET
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
        tree = ET.parse(task)
        root = tree.getroot()

        for body in root.findall('.//body'):
            print(body.get('name'))
            if(body.get('name') == 'torso'):
                size = body.get('pos', '0 0 0').split()
                torso_height = float(size[2])
        max_height = torso_height * 2
        min_height = torso_height / 2

        super().__init__(xml_file=task, 
                        healthy_z_range = (min_height, max_height),
                         **self.kwargs)
