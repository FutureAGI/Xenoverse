"""
Gym Environment For Any MDP
"""
import numpy
import gymnasium as gym
import pygame
import random as rnd
from numpy import random

from gym import error, spaces, utils
from gym.utils import seeding
from xenoverse.utils import pseudo_random_seed
from copy import deepcopy

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.observation_space = spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False

    def set_task(self, task, verbose=False, reward_shaping=False):
        for key in task:
            setattr(self, key, task[key])
        # 定义无界的 observation_space
        self.observation_space = gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(self.state_dim,), dtype=float)
        # 定义 action_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)
        self.task_set = True
        self.need_reset = True
        self.reward_shaping = reward_shaping
        if(verbose):
            print('Task Mode:', self.mode)
            print('ndim:', self.ndim)

    def reset(self, *args, **kwargs):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())

        loc, noise = rnd.choice(self.born_loc)
        self._inner_state = loc + noise * random.normal(size=self.ndim)
        self._state = self.observation_map(self._inner_state)

        return self._state, {"steps": self.steps}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert numpy.shape(action) == (self.action_dim,)

        ### update inner state (dynamics)
        inner_deta = self.action_map(self._inner_state, action)
        next_inner_state = (self._inner_state + 
            inner_deta * self.action_weight + 
            self.transition_noise * random.normal(size=(self.ndim,)))

        ### Essential Rewards specified by different goals
        reward = self.average_cost
        terminated = False
        for goal in self.goals:
            r,d,info=goal(self._inner_state, next_inner_state, t=self.steps, need_reward_shaping=self.reward_shaping)
            if(self.reward_shaping):
                r = info["shaped_reward"]
            reward += r
            terminated = terminated or d

        ### Calculate Universal Random Reward
        if("random_reward_fields" in self.__dict__):
            reward += self.random_reward_fields(self._inner_state)

        ### Add Noise to Reward
        if(abs(reward) > 0.5):
            reward *= 1.0 + self.reward_noise * random.normal()

        ### update state (observation)
        self.steps += 1
        info = {"steps": self.steps}
        self._inner_state = next_inner_state
        self._state = self.observation_map(self._inner_state)
        oob = (numpy.abs(self._inner_state) > self.box_size)
        terminated = oob.any() or terminated
        truncated = (self.steps >= self.max_steps)
 
        if(terminated or truncated):
            self.need_reset = True

        return self._state, reward, terminated, truncated, info
    
    @property
    def state(self):
        return numpy.copy(self._state)
    
    @property
    def inner_state(self):
        return numpy.copy(self._inner_state)