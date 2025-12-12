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
        self.na = env.na
        self.ns = env.ns
        self.env=  env
        self.task_type = env.task_type
        if(self.task_type != 'MTMDP'):
            self.da = 1
        else:
            self.da = env.da

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

    def policy(self, *args, **kwargs):
        # optimal solver directly utilize the inner states
        state_dist = numpy.zeros((self.ns,))
        state_dist[self.env.inner_state] = 1.0
        toks = []
        for i in range(self.da):
            value_dist = self.value_matrix.T @ state_dist
            toks.append(numpy.argmax(value_dist))
            state_dist = self.transition_matrix[:, toks[-1], :] @ state_dist
        if(len(toks) == 1):
            toks = toks[0]
        else:
            toks = numpy.array(toks, dtype=int)
        return toks