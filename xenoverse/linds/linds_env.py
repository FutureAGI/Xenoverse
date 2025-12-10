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
    def __init__(self, dt=0.1, max_steps=1000, 
                 pad_observation_dim=16,
                 pad_command_dim=16,
                 pad_action_dim=8):
        """
        Pay Attention max_steps might be reseted by task settings
        pad_observation_dim: int
            The padded observation dimension when task is not set
            e.g., if actual observation dim is 3 but pad_observation_dim=5,
            then the observation returned by env will be [o1, o2, o3, 0, 0]
        pad_action_dim: int
            The padded action dimension when task is not set
            e.g., if actual action dim is 3 but pad_action_dim=5,
            then the action returned by env will be [a1, a2, a3, 0, 0]
        """
        self.pad_observation_dim = pad_observation_dim
        self.pad_action_dim = pad_action_dim
        self.pad_command_dim = pad_command_dim
        self.observation_space = spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(pad_observation_dim,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(pad_action_dim,), dtype=float)
        self.max_steps = max_steps
        self.task_set = False
        self.dt = dt

    def set_task(self, task, verbose=False, reward_shaping=False, use_pad_dim=True):
        for key in task:
            setattr(self, key, task[key])
        self.use_pad_dim = use_pad_dim
        if(not use_pad_dim):
            # 定义无界的 observation_space
            self.observation_space = gym.spaces.Box(low=-numpy.inf, high=numpy.inf, shape=(self.observation_dim,), dtype=float)
            # 定义 action_space
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_dim,), dtype=float)
        else:
            # The actual observation and action space might be smaller than the padded one
            assert self.observation_dim <= self.pad_observation_dim, \
                f"Padded observation dimension {self.pad_observation_dim} is smaller than actual observation dimension {self.observation_dim}"
            assert self.action_dim <= self.pad_action_dim, \
                f"Padded action dimension {self.pad_action_dim} is smaller than actual action dimension {self.actiion_dim}" 

        self.task_set = True
        self.need_reset = True

        if(verbose):
            print('Task has been set:')
            print('state dim:', self.state_dim)
            print('observation dim:', self.observation_dim)
            print('action dim:', self.action_dim)
            print('max steps:', self.max_steps)
        self.build_dynamics_matrices()

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
        actual_obs = self.ld_C @ self._state + self.ld_Y
        if(self.use_pad_dim):
            padded_obs = numpy.zeros((self.pad_observation_dim,))
            padded_obs[:self.observation_dim] = actual_obs
            return padded_obs.tolist()
        else:
            return actual_obs
        
    def get_inner_cmd(self, step=0):
        # if command is not fixed, sample a random command whenever reset
        if(self.target_type=="static_target"):
            return numpy.copy(self.command) * self.target_valid
        else:
            return self.command(step) * self.target_valid
        
    def get_final_cmd(self, cmd):
        if(self.use_pad_dim):
            padded_cmd = numpy.zeros((self.pad_command_dim,))
            padded_cmd[:self.observation_dim] = cmd
            return padded_cmd
        else:
            return cmd

    def reset(self, *args, **kwargs):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())
        self._cmd_list = []

        self._state = numpy.copy(rnd.choice(self.initial_states))

        # pre-fill the command list up to current step
        if(self.target_type=="static_target"):
            self._cmd_list.append(self.get_inner_cmd())
        else:
            for t in range(self.target_delay, -1, -1):
                self._cmd_list.append(self.get_inner_cmd(-t))
        obs = self.get_observation
        cmd = self._cmd_list[-1]
        error = numpy.linalg.norm((numpy.array(obs[:self.observation_dim]) - cmd) * self.target_valid)
        return self.get_observation, {"steps": self.steps, 
                                      "command": self.get_final_cmd(self._cmd_list[-1]), 
                                      "command_type": self.target_type,
                                      "error": error}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        if(self.use_pad_dim):
            assert numpy.shape(action) == (self.pad_action_dim,), f"Action shape mismatch: expected {(self.pad_action_dim,)}, got {numpy.shape(action)}"
            act = numpy.clip(action, self.action_space.low, self.action_space.high)
            act = act[:self.action_dim]
        else:
            assert numpy.shape(action) == (self.action_dim,), f"Action shape mismatch: expected {(self.action_dim,)}, got {numpy.shape(action)}"
            act = numpy.clip(action, self.action_space.low, self.action_space.high)

        self._state = self.dynamics(act)
        obs = self.get_observation

        self.steps += 1
        # get the current command
        # when there is a delay, the command is from several steps ago
        self._cmd_list.append(self.get_inner_cmd(self.steps))
        cmd = self._cmd_list.pop(0)

        error = numpy.linalg.norm((numpy.array(obs[:self.observation_dim]) - cmd) * self.target_valid)
        obs_scale = numpy.linalg.norm(obs)

        if(error > 10.0 or obs_scale > 20.0):
            terminated = True
            reward = -self.terminate_punish
        else:
            terminated = False
            reward = 0.0

        reward += (self.reward_base - self.reward_factor * error \
            - self.action_cost * numpy.sum(numpy.square(action))) * self.dt
        truncated = (self.steps >= self.max_steps - 1)

        return obs, reward, terminated, truncated, {"steps": self.steps, 
                                                    "command": self.get_final_cmd(self._cmd_list[-1]),
                                                    "error": error}
    
    def get_future_inner_cmds(self, K=None):
        if(K is None or K <= 0):
            return deepcopy(self._cmd_list)
        else:
            if(self.target_type=="static_target"):
                return [deepcopy(self.command) for i in range(K)]
            elif(self.target_type=="dynamic_target"):
                cmd_list = deepcopy(self._cmd_list)
                k = 0
                while(len(cmd_list) < K):
                    k += 1
                    cmd_list.append(self.get_inner_cmd(self.steps + k))
                return cmd_list

    @property
    def state(self):
        return numpy.copy(self._state)
