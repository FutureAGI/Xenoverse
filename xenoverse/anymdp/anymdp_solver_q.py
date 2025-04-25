import numpy
from numpy import random

class AnyMDPSolverQ(object):
    """
    Solver for AnyMDPEnv with Q-Learning
    """
    def __init__(self, env, gamma=0.99, c=1.0, alpha=0.01, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.env = env
        self.na = env.action_space.n
        self.ns = env.observation_space.n
        self.value_matrix = numpy.zeros((self.ns, self.na)) + 0.1/(1.0 - gamma)
        self.sa_visitied = numpy.ones((self.ns, self.na))
        self.gamma = gamma
        self.alpha = alpha
        self.max_steps = max_steps
        self._c = c
        self.avg_r = 0.0
        self.avg_r2 = 0.0
        self.r_std = 0.01
        self.r_cnt = 0

    def learner(self, s, a, ns, r, terminated, truncated):
        
        self.avg_r = (self.avg_r * self.r_cnt + r) / (self.r_cnt + 1)
        self.avg_r2 = (self.avg_r2 * self.r_cnt + r**2) / (self.r_cnt + 1)
        self.r_cnt = min(self.r_cnt + 1, 10000)
        self.r_std = numpy.sqrt(max(self.avg_r2 - self.avg_r**2, 1.0e-4))
        b_t = self._c * self.r_std * numpy.sqrt(numpy.log(self.max_steps + 1) / self.sa_visitied[s,a])
        lr = max((self.max_steps + 1) / (self.max_steps + self.sa_visitied[s,a]), 2.0e-3)

        rnd_vec = random.uniform(0.0, 1.0, size=b_t.shape)
        if(terminated):
            target = r + 1.0 / (1.0 - self.gamma) * b_t * rnd_vec
        else:
            target = r + b_t * rnd_vec + self.gamma * max(self.value_matrix[ns])

        error = target - self.value_matrix[s][a]
        self.value_matrix[s][a] += self.alpha * lr * error
        self.sa_visitied[s][a] += 1

    def policy(self, state):
        return numpy.argmax(self.value_matrix[state])