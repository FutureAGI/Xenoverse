import numpy
from numpy import random
from xenoverse.anymdp.solver import update_value_matrix


class AnyMDPSolverMBRL(object):
    """
    Implementing the RL Solver  of the paper
    Hu, Bingshan, et al. "Optimistic Thompson sampling-based algorithms for 
        episodic reinforcement learning." 
        Uncertainty in Artificial Intelligence. PMLR, 2023.
    """
    def __init__(self, env, gamma=0.99, c=1.0, max_steps=4000):
        """
        The constructor for the class AnyMDPSolverQ
        The exploration strategy is controlled by UCB-H with c as its hyperparameter. Increasing c will encourage exploration
        Simulation of the ideal policy when the ground truth is not known
        """
        self.ns = env.observation_space.n
        self.na = env.action_space.n

        self.est_r = numpy.zeros((self.ns, self.na, self.ns))
        self.vis_cnt = 0.01 * numpy.ones((self.ns, self.na, self.ns))
        self.vis_cnt_sa = numpy.ones((self.ns, self.na))

        self.gamma = gamma
        self._c = c / (1.0 - self.gamma)
        self.max_steps = max_steps

        self.value_matrix = numpy.zeros((self.ns, self.na))
        self.est_r_global_avg = 0
        self.est_r_global_cnt = 0

        self.est_r_std = 0.01

        self.s_0 = []
        self.s_0_cnt = []
        self.s_e = []

        self.update_estimator()

    def update_estimator(self):
        t_mat = numpy.copy(self.vis_cnt)
        # use 0.01 to make sure those with all transition = 0 will stay 0
        t_mat_valid = numpy.clip(numpy.sum(t_mat, axis=-1, keepdims=True), 0.01, None)
        self.t_mat = t_mat / t_mat_valid
        self.r_mat = numpy.copy(self.est_r)

        self.est_r_std = max(numpy.std(self.est_r), 0.01)
        self.b_mat = self._c * self.est_r_std / numpy.sqrt(self.vis_cnt_sa)

        self.value_matrix = update_value_matrix(self.t_mat, self.r_mat, self.gamma, self.value_matrix, max_iteration=1)

        if(len(self.s_0) > 0):
            self.s_0_prob = numpy.array(self.s_0_cnt) / numpy.sum(self.s_0_cnt)

    def learner(self, s, a, ns, r, terminated, truncated):
        # Update the environment model estimation
        cnt = self.vis_cnt[s,a,ns]
        self.est_r[s,a,ns] = (self.est_r[s,a,ns] * cnt + r) / (cnt + 1)
        self.vis_cnt[s,a,ns] += 1
        self.vis_cnt_sa[s,a] += 1

        if(terminated):
            self.vis_cnt[ns] = 0
            self.est_r[ns] = 0
            if(ns not in self.s_e):
                self.s_e.append(ns)

        if(terminated or truncated):
            self.update_estimator()

    def policy(self, state, is_test=False):
        # UCB Exploration
        if(is_test):
            return numpy.argmax(self.value_matrix[state])
        rnd_vec = random.uniform(0.0, 1.0, size=(self.na))
        return numpy.argmax(self.value_matrix[state] + self.b_mat[state] * rnd_vec)
    
    def set_reset_states(self, s):
        if(s not in self.s_0):
            self.s_0.append(s)
            self.s_0_cnt.append(1)
        else:
            idx = self.s_0.index(s)
            self.s_0_cnt[idx] += 1