import numpy
from numpy import random
from numba import njit
from xenoverse.anymdp.solver import update_value_matrix


class AnyMDPSolverOpt(object):
    """
    Solver for AnyMDPEnv with Bellman Equation and Value Iteration
    Suppose to know the ground truth of the environment
    """
    def __init__(self, env, gamma=0.99):
        self.na = env.action_space.n
        self.ns = env.observation_space.n
        self.transition_matrix = env.transition
        self.reward_matrix = env.reward
        self.state_mapping = env.state_mapping
        self.value_matrix = numpy.zeros((len(env.state_mapping), self.na))
        self.gamma = gamma
        self.inverse_state_mapping = dict()
        for i,state in enumerate(self.state_mapping):
            self.inverse_state_mapping[state] = i
        self.q_solver(gamma=gamma)

    def q_solver(self, gamma=0.99):
        self.value_matrix = update_value_matrix(self.transition_matrix, self.reward_matrix, gamma, self.value_matrix)
    
    def learner(self, *args, **kwargs):
        pass

    def policy(self, state, **kwargs):
        return numpy.argmax(self.value_matrix[self.inverse_state_mapping[state]])