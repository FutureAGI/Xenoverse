import numpy
from numpy import random

class AnyMDPSolverQ(object):
    """
    Solver for AnyMDPEnv with Q-Learning
    """
    def __init__(self, env, gamma=0.99, alpha=0.50, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.env = env
        self.na = env.action_space.n
        self.ns = env.observation_space.n
        self.value_matrix = numpy.zeros((self.ns, self.na)) + 1.0/(1.0 - gamma)
        self.sa_visitied = numpy.ones((self.ns, self.na))
        self.s_visitied = numpy.ones((self.ns,))
        self.gamma = gamma
        self.alpha = alpha
        self.max_steps = max_steps
        self.avg_r = 0.0
        self.avg_r2 = 0.0
        self.r_std = 0.01
        self.r_cnt = 0
        self.lr = numpy.ones((self.ns, self.na))

    def learner(self, s, a, ns, r, terminated, truncated):
        
        self.avg_r = (self.avg_r * self.r_cnt + r) / (self.r_cnt + 1)
        self.avg_r2 = (self.avg_r2 * self.r_cnt + r**2) / (self.r_cnt + 1)
        self.r_cnt = min(self.r_cnt + 1, 10000)
        self.r_std = numpy.sqrt(max(self.avg_r2 - self.avg_r**2, 1.0e-4))

        # Learning rate decay
        self.lr[s,a] = numpy.sqrt(max((self.max_steps + 1) / (self.max_steps + self.sa_visitied[s,a]), 1.0e-3))

        if(terminated):
            target = r
            self.value_matrix[ns] = 0.0
        else:
            target = r + self.gamma * max(self.value_matrix[ns])

        error = target - self.value_matrix[s][a]
        self.value_matrix[s][a] += self.alpha * self.lr[s,a] * error
        self.sa_visitied[s][a] += 1
        self.s_visitied[s] += 1

    def policy(self, state, is_test=False):
        if(is_test):
            return numpy.argmax(self.value_matrix[state])

        value = self.value_matrix[state] - numpy.max(self.value_matrix[state])
        stiffness = min((self.max_steps + self.s_visitied[state]) / (self.max_steps + 1), 10.0)
        value = value / max(numpy.std(value), 1.0e-2) * stiffness
        value = numpy.exp(value) / numpy.sum(numpy.exp(value))
        return random.choice(range(len(value)), p=value)