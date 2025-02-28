import gym
import numpy
from numpy import random
from numba import njit
from xenoverse.anymdp.solver import update_value_matrix, get_final_transition, get_final_reward


class AnyMDPSolverOpt(object):
    """
    Solver for AnyMDPEnv with Bellman Equation and Value Iteration
    Suppose to know the ground truth of the environment
    """
    def __init__(self, env, gamma=0.99):
        if(not env.task_set):
            raise Exception("AnyMDPEnv is not initialized by 'set_task', must call set_task first")
        self.n_actions = env.action_space.n
        self.n_states = env.observation_space.n
        self.transition_matrix = get_final_transition(
                transition=env.transition_matrix,
                reset_states=env.reset_states,
                reset_triggers=env.reset_triggers)
        self.reward_matrix = get_final_reward(
                reward=env.reward_matrix,
                reset_triggers=env.reset_triggers,
        )
        self.state_mapping = env.state_mapping
        self.value_matrix = numpy.zeros((self.n_states, self.n_actions))
        self.gamma = gamma
        self.inverse_state_mapping = dict()
        for i,state in enumerate(self.state_mapping):
            self.inverse_state_mapping[state] = i
        self.q_solver()

    def q_solver(self, gamma=0.99):
        self.value_matrix = update_value_matrix(self.transition_matrix, self.reward_matrix, gamma, self.value_matrix)

    def policy(self, state):
        return numpy.argmax(self.value_matrix[self.inverse_state_mapping[state]])