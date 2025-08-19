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

def sample_acrobot(link_length_1=True,
                   link_length_2=True,
                   link_mass_1=True,
                   link_mass_2=True,
                   link_com_1=True,
                   link_com_2=True,
                   gravity=True):
    # Sample a random acrobot task
    pseudo_random_seed(0)
    link_length_1 = versatile_sample(link_length_1, (0.5, 3.0), 1.0)
    link_length_2 = versatile_sample(link_length_2, (0.5, 3.0), 1.0)
    link_mass_1 = versatile_sample(link_mass_1, (0.5, 3.0), 1.0)
    link_mass_2 = versatile_sample(link_mass_2, (0.5, 3.0), 1.0)
    link_com_1 = versatile_sample(link_com_1, (0.25, 0.75), 0.5) * link_length_1
    link_com_2 = versatile_sample(link_com_2, (0.25, 0.75), 0.5) * link_length_2
    gravity = versatile_sample(gravity, (1.0, 15.0), 9.8)

    return {
        "link_length_1": link_length_1,
        "link_length_2": link_length_2,
        "link_mass_1": link_mass_1,
        "link_mass_2": link_mass_2,
        "link_com_1": link_com_1,
        "link_com_2": link_com_2,
        "gravity": gravity
    }

class RandomAcrobotEnv(CartPoleEnv):

    def __init__(self, *args, **kwargs):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.frameskip = kwargs.get("frameskip", 5)
        kwargs.pop("frameskip", None)
        super().__init__(*args, **kwargs)

    # Rewrite the dynamics for acrobot
    def _dsdt(self, s_augmented):
        m1 = self.link_mass_1
        m2 = self.link_mass_2
        l1 = self.link_length_1
        lc1 = self.link_com_1
        lc2 = self.link_com_2
        I1 = self.link_mass_1 * (self.link_com_1**2 + (self.link_length_1 - self.link_com_1)**2) / 6.0
        I2 = self.link_mass_2 * (self.link_com_2**2 + (self.link_length_2 - self.link_com_2)**2) / 6.0
        g = self.gravity
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            # the following line is consistent with the description in the
            # paper
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            # the following line is consistent with the java implementation and the
            # book
            ddtheta2 = (
                a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2
            ) / (m2 * lc2**2 + I2 - d2**2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0

    def _terminal(self):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > self.link_length_1)

    def set_task(self, task_config):
        print("Setting task with config:", task_config)
        for key, value in task_config.items():
            setattr(self, key, value)

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