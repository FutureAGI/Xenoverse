import os
import numpy
import gymnasium as gym
import pygame
import xml.etree.ElementTree as ET
from numpy import random
from numba import njit
from gymnasium import spaces
from xenoverse.utils import pseudo_random_seed, versatile_sample, generate_secure_strings
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from humanoid_xml_sampler import humanoid_xml_sampler

def sample_humanoid(root_path=None, noise_scale=1.0):
    # Sample a random humanoid task
    if(root_path is None):
        root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(root_path, 'assets'))
    if(os.path.exists(root_path) is False):
        os.makedirs(root_path)
    file_id = generate_secure_strings(1, length=8)[0]
    file_path = os.path.join(root_path, f'random_humanoid_{file_id}.xml')
    humanoid_xml_sampler(file_path, noise_scale=noise_scale)
    return file_path

def get_humanoid_tasks(directory):
    # Acquire a list of tasks from the specified directory
    xml_files = [f for f in os.listdir(directory) if f.endswith('.xml')]
    xml_lists = []
    for xml_file in xml_files:
        if 'random_humanoid' in xml_file:
            xml_lists.append(os.path.join(directory, xml_file))
    if(len(xml_lists) == 0):
        raise ValueError(f"No random_humanoid XML files found in directory: {directory}")
    return xml_lists

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