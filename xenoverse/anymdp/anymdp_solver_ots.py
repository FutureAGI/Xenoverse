import numpy
from numpy import random


class AnyMDPSolverOTS(object):
    """
    Implementing the RL Solver  of the paper
    Hu, Bingshan, et al. "Optimistic Thompson sampling-based algorithms for 
        episodic reinforcement learning." 
        Uncertainty in Artificial Intelligence. PMLR, 2023.
    """
    def __init__(self, env, gamma=0.99, c=0.05, alpha=0.01, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.na = env.action_space.n
        self.ns = env.observation_space.n
       
        self.est_r = numpy.zeros((self.ns, self.na))
        self.est_r_cnt = numpy.zeros((self.ns, self.na))
        self.est_t = 0.1 * numpy.ones((self.ns, self.na, self.ns))
        self.gamma = gamma
        self.lr = 1.0 - alpha
        self._c = c / (1.0 - self.gamma)
        self.max_steps = max_steps

        self.value_matrix = numpy.zeros((self.ns, self.na))
        self.value_std = numpy.zeros((self.ns,))
        self.est_r_global_avg = 0
        self.est_r_global_cnt = 0
        self.alpha = alpha

    def learner(self, s, a, ns, r, terminated, truncated):
        # Update the environment model estimation
        if(self.est_r_cnt[s,a] > 0):
            self.est_r[s,a] += (r - self.est_r[s,a])/self.est_r_cnt[s,a]
        else:
            self.est_r[s,a] = r

        if(self.est_r_global_cnt > 0):
            self.est_r_global_avg += (r - self.est_r_global_avg) * self.alpha
        else:
            self.est_r_global_avg = r

        self.est_r_cnt[s,a]+= 1
        self.est_t[s,a,ns] += 1
        self.est_r_global_cnt += 1

        est_t = self.est_t / numpy.sum(self.est_t, axis=-1, keepdims=True)

        if(terminated):
            est_value = self.est_r[s,a]
        else:
            est_target = numpy.sum(est_t[s, a] * numpy.max(self.value_matrix, axis=-1))
            est_value = self.est_r[s,a] + self.gamma * est_target
        
        self.value_matrix[s,a] += self.alpha * (est_value - self.value_matrix[s,a])
        self.value_std = numpy.clip(numpy.std(self.value_matrix, axis=-1), 0.10, None)

    def policy(self, state):
        # Apply UCB with dynamic noise (Thompson Sampling)
        values = self._c * numpy.sqrt(numpy.log(self.max_steps + 1) / numpy.clip(self.est_r_cnt[state], 1.0, None)) * \
                numpy.maximum(numpy.random.randn(self.na), 0) * self.value_std[state] + \
                self.value_matrix[state]
        return numpy.argmax(values)