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

@njit(cache=True)
def map_transition_reward(t, r, t_obs, r_obs, s_mapping):
    ns, na, _ = t.shape
    for i,si in enumerate(s_mapping):
        for a in range(na):
            for j,sj in enumerate(s_mapping):
                t_obs[si][a][sj] = t[i][a][j]
                r_obs[si][a][sj] = r[i][a][j]
    return t_obs, r_obs

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
        for k,v in task_config.items():
            setattr(self, k, v)

        assert self.transition.shape == self.reward.shape
        assert self.transition.shape[0] == len(self.state_mapping) and self.transition.shape[1] == self.na

        assert self.ns > 0, "State space must be at least 1"
        assert self.na > 1, "Action space must be at least 2"

        self.observation_space = spaces.Discrete(self.ns)
        self.action_space = spaces.Discrete(self.na)

        # inverse mapping from observation to inner state
        self.obs2inner = list(range(self.ns))
        for i,s in enumerate(self.state_mapping):
            self.obs2inner[s] = i # do permutation to avoid empty mapping
            self.obs2inner[i] = s
        # Get observation transition and reward
        self.transition_obs = numpy.zeros((self.ns, self.na, self.ns))
        self.reward_obs = numpy.zeros((self.ns, self.na, self.ns))
        self.transition_obs, self.reward_obs = map_transition_reward(self.transition, self.reward, self.transition_obs, self.reward_obs, self.state_mapping)

        # check transition matrix is valid
        t_mat_sum = numpy.sum(self.transition, axis=-1)
        error = (t_mat_sum - 1.0)**2
        if(len(self.s_e) > 0):
            error[self.s_e] = 0.0
        if((error >= 1.0e-6).any()):
            raise Exception(f'Transition Matrix Sum != 1 at {numpy.where(error>=1.0e-6)}')
        
        # check if there is any state that is both start and end
        intersection = numpy.intersect1d(self.s_0, self.s_e)
        if(len(intersection) > 0):
            raise Exception(f'State {intersection} is {self.s_0} and {self.s_e}')

        self.task_set = True
        self.need_reset = True

    def reset(self, *args, **kwargs):
        if(not self.task_set):
            raise Exception("Must call \"set_task\" first")
        
        self.steps = 0
        self.need_reset = False
        random.seed(pseudo_random_seed())

        self._state = numpy.random.choice(self.s_0, p=self.s_0_prob)
        return int(self.state_mapping[self._state]), {"steps": self.steps}

    def step(self, action):
        if(self.need_reset or not self.task_set):
            raise Exception("Must \"set_task\" and \"reset\" before doing any actions")
        if(self._state in self.s_e and self.ns > 1):
            raise Exception(f"Unexpected Error: Given an ended state: {self.inner_state()}")
        assert action < self.na, "Action must be less than the number of actions"
        transition_gt = numpy.copy(self.transition[self._state, action])
        next_state = random.choice(len(self.state_mapping), p=transition_gt)

        # sample the reward
        reward_gt = self.reward[self._state, action, next_state]
        reward_gt_noise = self.reward_noise[self._state, action, next_state]
        reward = random.normal(reward_gt, reward_gt_noise)

        info = {"steps": self.steps, "reward_gt": reward_gt}
        info["transition_gt"] = self.transition_obs[self.state, action]

        self.steps += 1
        self._state = next_state
        terminated = (self._state in self.s_e) or (self.ns < 2)
        truncated = self.steps >= self.max_steps

        if(terminated or truncated):
            self.need_reset = True
        return int(self.state_mapping[next_state]), reward, terminated, truncated, info
    
    @property
    def state(self):
        return int(self.state_mapping[self._state])
    
    @property
    def inner_state(self, query_state=None):
        if(query_state is None):
            return int(self._state)
        else:
            return int(self.obs2inner[query_state])
    
    def get_gt_transition(self):
        return self.transition_obs
    
    def get_gt_reward(self):
        return self.reward_obs
