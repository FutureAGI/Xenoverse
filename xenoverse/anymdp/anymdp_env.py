"""
Gym Environment For Any MDP
"""
import numpy
import gymnasium as gym
import pygame
from numpy import random

from gymnasium import spaces
from xenoverse.utils import pseudo_random_seed

class AnyMDPEnv(gym.Env):
    def __init__(self, max_steps):
        """
        Pay Attention max_steps might be reseted by task settings
        """
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.max_steps = max_steps
        self.task_set = False

    def set_task(self, task_config):
        self.transition_matrix = task_config["transition"]
        self.reward_matrix = task_config["reward"]
        self.reward_noise = task_config["reward_noise"]
        self.reward_noise_type = task_config["reward_noise_type"]
        self.state_mapping = task_config["state_mapping"]
        self.reset_triggers = task_config["reset_triggers"]
        self.reset_states = task_config["reset_states"]
        self.n_states = task_config["state_space"]
        self.n_actions = task_config["action_space"]

        if("max_steps" in task_config):
            self.max_steps = task_config["max_steps"]

        ns1, na1, ns2 = self.transition_matrix.shape
        ns3, na2, ns4 = self.reward_matrix.shape

        assert ns1 == ns2 and ns2 == ns3 and ns3==ns4, \
            "Transition matrix and reward matrix must have the same number of states"
        assert na1 == na2, \
            "Transition matrix and reward matrix must have the same number of actions"
        assert ns1 > 0, "State space must be at least 1"
        assert na1 > 1, "Action space must be at least 2"

        self.observation_space = spaces.Discrete(self.n_states)
        self.action_space = spaces.Discrete(self.n_actions)

        self.task_set = True
        self.need_reset = True

    def reset(self, *args, **kwargs):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())

        self._state = numpy.random.choice(len(self.reset_states),
                                          replace=True,
                                          p=self.reset_states)
        return self.state_mapping[self._state], {"steps": self.steps}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        assert action < self.n_actions, "Action must be less than the number of actions"
        transition_gt = self.transition_matrix[self._state, action]

        next_state = random.choice(len(self.state_mapping), p=transition_gt)

        reward_gt = self.reward_matrix[self._state, action, next_state]
        reward_gt_noise = self.reward_noise[self._state, action, next_state]

        if(self.reward_noise_type == 'normal'):
            reward = random.normal(reward_gt, reward_gt_noise)
        elif(self.reward_noise_type == 'binomial'):
            if(reward_gt > 0):
                reward = float(random.binomial(1, reward_gt))
            else:
                reward = - float(random.binomial(1, abs(reward_gt)))

        info = {"steps": self.steps, "reward_gt": reward_gt}

        if(self.state_mapping.ndim == 1):
            transition_ext_gt = numpy.zeros((self.n_states,))
            for i,s in enumerate(self.state_mapping):
                transition_ext_gt[s] = transition_gt[i]
            info["transition_gt"] = transition_ext_gt

        self.steps += 1
        self._state = next_state
        #print("inner", next_state, "outer", self.state_mapping[next_state], self.max_steps,
        #      "triggers", self.reset_triggers, "starts", self.reset_states)
        terminated = self.reset_triggers[self._state]
        truncated = self.steps >= self.max_steps
        if(terminated or truncated):
            self.need_reset = True
        return self.state_mapping[next_state], reward, terminated, truncated, info
    
    @property
    def state(self):
        return self.state_mapping[self._state]
    
    @property
    def inner_state(self):
        return self._state

class AnyMDPEnvD2C(AnyMDPEnv):
    """
    Transfer a AnyMDPEnv to Continuous State Space without resampling a task
    """
    def __init__(self, max_steps, state_dim):
        super().__init__(max_steps)
        self.observation_space = spaces.Box(0., 1., shape=(state_dim,))
        self.state_dim = state_dim

    def set_task(self, task_config):
        super().set_task(task_config)
        n_inner_states = self.state_mapping.shape[0]
        self.state_mapping = numpy.random.normal(size=(n_inner_states, self.state_dim))
        self.observation_space = spaces.Box(0., 1., shape=(self.state_dim,))