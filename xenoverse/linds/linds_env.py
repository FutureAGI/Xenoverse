"""
Gym Environment For Any MDP
"""
import numpy
import gymnasium as gym
import pygame
import random as rnd
from numpy import random

from gymnasium import error, spaces, utils
from xenoverse.utils import pseudo_random_seed, weights_and_biases
from copy import deepcopy
from scipy.linalg import expm  # 矩阵指数

class LinearDSEnv(gym.Env):
    def __init__(self, dt=0.1, max_steps=1000):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.observation_space = spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(1,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False
        self.dt = dt

    def set_task(self, task, verbose=False, reward_shaping=False):
        for key in task:
            setattr(self, key, task[key])
        # 定义无界的 observation_space
        self.observation_space = gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(self.observation_dim,), dtype=float)
        # 定义 action_space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)
        self.task_set = True
        self.need_reset = True
        if(verbose):
            print('Task has been set:')
            print('state dim:', self.state_dim)
            print('observation dim:', self.observation_dim)
            print('action dim:', self.action_dim)
            print('reward dim:', self.reward_dim)
            print('max steps:', self.max_steps)
        self.build_dynamics_matrices()

    def get_vector_reward_space(self, observation):
        return self.reward_weight @ observation + self.reward_bias

    def build_dynamics_matrices(self):
        # ZOH discretization
        M = numpy.block([
            [self.ld_A, numpy.eye(self.state_dim)],  # 增广矩阵用于积分
            [numpy.zeros((self.state_dim, 2*self.state_dim))]
        ])
        M_exp = expm(M * self.dt)
        self.ld_phi = M_exp[:self.state_dim, :self.state_dim]           # e^(A*dt)
        self.ld_gamma = M_exp[:self.state_dim, self.state_dim:] @ self.ld_B    # ∫e^(A*τ)dτ * B
        self.ld_Xt = self.ld_X * self.dt

    def dynamics(self, action):
        noise = numpy.random.randn(self.state_dim) * self.noise_drift * self.dt
        return self.ld_phi @ self._state + self.ld_gamma @ numpy.array(action) + self.ld_Xt + noise
    
    #Get the current observations from the current state
    @property
    def get_observation(self):
        return self.ld_C @ self._state + self.ld_Y
    
    def reset(self, *args, **kwargs):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())

        self._state = numpy.copy(rnd.choice(self.initial_states))

        # if command is not fixed, sample a random command whenever reset
        if(self.target_type=="static_target"):
            self._cmd = numpy.copy(self.command)
        else:
            self._cmd = random.randn(self.reward_dim) * random.choice([0, 1])

        return self.get_observation, {"steps": self.steps, 
                                      "command": self._cmd, 
                                      "command_type": self.target_type}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert numpy.shape(action) == (self.action_dim,)
        act = numpy.clip(action, self.action_space.low, self.action_space.high)

        self._state = self.dynamics(act)
        obs = self.get_observation
        dist = numpy.linalg.norm(self.get_vector_reward_space(obs) - self._cmd)

        if(dist > 10.0):
            terminated = True
            reward = -self.terminate_punish
        else:
            terminated = False
            reward = 0.0

        observation = self.get_observation

        reward += (self.reward_base - self.reward_factor * dist \
            - self.action_cost * numpy.sum(numpy.square(action))) * self.dt
        self.steps += 1
        truncated = (self.steps >= self.max_steps - 1)

        return obs, reward, terminated, truncated, {"steps": self.steps, "command": self._cmd}
    
    @property
    def state(self):
        return numpy.copy(self._state)