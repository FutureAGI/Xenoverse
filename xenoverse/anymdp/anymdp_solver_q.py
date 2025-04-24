import numpy
from numpy import random

class AnyMDPSolverQ(object):
    """
    Solver for AnyMDPEnv with Q-Learning
    """
    def __init__(self, env, gamma=0.99, c=0.05, alpha=0.01, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.na = env.action_space.n
        self.ns = env.observation_space.n
        self.value_matrix = numpy.zeros((self.ns, self.na))
        self.sa_visitied = numpy.ones((self.ns, self.na))
        self.gamma = gamma
        self.alpha = alpha
        self.max_steps = max_steps
        self._c = c / (1.0 - self.gamma)
        self.value_std = numpy.zeros((self.ns,))
        self.avg_r = 0.0
        self.avg_r2 = 0.0
        self.r_std = 0.10
        self.r_cnt = 0


    def learner(self, s, a, ns, r, done):
        if(done):
            target = r
        else:
            target = r + self.gamma * max(self.value_matrix[ns])
        
        self.avg_r = (self.avg_r * self.r_cnt + r) / (self.r_cnt + 1)
        self.avg_r2 = (self.avg_r2 * self.r_cnt + r**2) / (self.r_cnt + 1)
        self.r_cnt = min(self.r_cnt + 1, 10000)
        self.r_std = numpy.sqrt(self.avg_r2 - self.avg_r**2)
        error = target - self.value_matrix[s][a]

        self.value_matrix[s][a] += self.alpha * error
        self.sa_visitied[s][a] += 1

        self.value_std = numpy.clip(numpy.std(self.value_matrix, axis=-1), 0.10, None)


    def policy(self, state):
        # Apply UCB with dynamic noise (Thompson Sampling)
        values = self._c * self.value_std[state] * numpy.sqrt(numpy.log(self.max_steps + 1) / self.sa_visitied[state]) * \
                numpy.maximum(numpy.random.randn(self.na), 0) + \
                self.value_matrix[state]
        return numpy.argmax(values)