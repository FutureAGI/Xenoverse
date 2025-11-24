import numpy
import numpy as np
from numpy import random
from numba import njit
import scipy.stats as stats
from scipy.stats import spearmanr
from copy import deepcopy

import numpy as np
import scipy.linalg as la
from scipy import sparse
import osqp
import matplotlib.pyplot as plt

class LTISystemMPC(object):
    """
    Solving Linear Time-Invariant System Model Predictive Control Problem
    x_{k+1} = A x_k + B u_k + X
    y_k = C x_k + Y
    """
    def __init__(self, env,
                 K=20, # forward looking steps
                 gamma=0.99): # cost for each action relative to observation
        self.Na = env.action_space.shape[0]
        self.Nx = env.state_dim  # state dimension
        self.Nu = env.action_dim  # action dimension
        self.Ny = env.reward_dim  # optimize in reward dimension
        self.A = env.ld_phi
        self.B = env.ld_gamma
        self.C = env.reward_weight @ env.ld_C   # directly map observation to reward space
        self.X = env.ld_Xt.reshape(-1, 1)
        self.Y = env.reward_weight @ env.ld_Y.reshape(-1, 1) + env.reward_bias.reshape(-1, 1)
        # MPC parameters
        self.K = K  # forward looking steps
        Q = np.diag(gamma ** np.arange(K))  # discount factor weights
        P = np.eye(K) * env.action_cost / env.reward_factor  # action cost weights
        self.W_Q = np.kron(Q, np.eye(self.Ny))
        self.W_P = np.kron(P, np.eye(self.Nu))
        
        # constraints
        self.u_min = env.action_space.low[:self.Nu]
        self.u_max = env.action_space.high[:self.Nu]

        self.u_lb = np.kron(np.ones(self.K), self.u_min)
        self.u_ub = np.kron(np.ones(self.K), self.u_max)
        
        # construct constraints
        self._pre_build_matrices()
        
    def _pre_build_matrices(self):
        
        """
        forward looking prediction model
        OUTPUT = [x1, x2, ..., x_K]
        OUTPUT = H @ U + F1 @ x0 + F2
        Output: shape (K * Ny, 1)
        H: shape (K * Ny, K * Nu)
        U: shape (K * Nu, 1)
        x0: shape (Nx, 1)
        F1: shape (K * Ny, Nx)
        F2: shape (K * Ny, 1)
        """

        self.H = np.zeros((self.K * self.Ny, self.K * self.Nu))
        self.F1 = np.zeros((self.K * self.Ny, self.Nx))
        self.F2 = np.zeros((self.K * self.Ny, 1))

        A_power = list()
        tmp = np.eye(self.Nx)
        A_power.append(tmp)
        for i in range(self.K + 1):
            tmp = tmp @ self.A
            A_power.append(tmp)
        A_power = A_power[::-1]  # 逆序存储A的幂次
        for i in range(self.K):
            for j in range(self.K-i):
                k = self.K - i - j - 1
                self.H[i*self.Ny:(i+1)*self.Ny, k*self.Nu:(k+1)*self.Nu] = self.C @ A_power[i] @ self.B
            self.F1[i*self.Ny:(i+1)*self.Ny, :] = self.C @ A_power[i+1]
            self.F2[i*self.Ny:(i+1)*self.Ny, :] = self.C @ A_power[i] @ self.X + self.Y

        # Pre-compute Sigma matrices
        self.W_H = self.H.T @ self.W_Q @ self.H + self.W_P

        # Pre-compute Bias effect
        self.G_11 = self.F1.T @ self.W_Q @ self.F1
        self.G_12 = self.F1.T @ self.W_Q 
        self.G_21 = self.G_12.T

        self.A_cons = np.kron(np.eye(self.K), np.eye(self.Nu))

    def solve(self, x_current, ref_trajectory):
        """
        Reference trajectory can be shorter than K steps
        """
        if(ref_trajectory.ndim == 1):
            Y_ref = np.kron(np.ones((self.K)), ref_trajectory[:self.Ny]).reshape(-1, 1)
        else:
            Y_ref = np.zeros((self.K * self.Ny, 1))
            for i in range(self.K):
                if(ref_trajectory.shape[0] > i):
                    Y_ref[self.Ny * i:self.Ny * (i + 1), 0] = ref_trajectory[i, :self.Ny].flatten()
                else:
                    Y_ref[self.Ny * i:self.Ny * (i + 1), 0] = ref_trajectory[-1, :self.Ny].flatten()
        x = x_current.reshape(-1, 1)

        f = self.F1 @ x + self.F2 - Y_ref
        f = f.T @ self.W_Q @ self.H
        f = f.flatten()
        prob = osqp.OSQP()
        prob.setup(sparse.csc_matrix(self.W_H), f, sparse.csc_matrix(self.A_cons), 
                   self.u_lb, self.u_ub, verbose=False)
        res = prob.solve()
        
        if res.info.status != 'solved':
            print(f"Fail to solve QP: {res.info.status}")
            return None

        # return the optimal control sequence
        u_opt = numpy.zeros((self.Na, ))
        u_opt[:self.Nu] = res.x[:self.Nu]
        return u_opt

def test_mpc(env, use_mpc=True, plot=False):
    mpc = LTISystemMPC(env, K=50, gamma=0.99)
    
    T_sim = 400
    obs, info = env.reset()
    x_current = env._state
    cmd = info["command"]

    error_history = []
    reward_history = []
    
    for t in range(T_sim):
        #action = env.action_space.sample()
        if(use_mpc is False):
            action = env.action_space.sample()
        else:
            action = mpc.solve(x_current, cmd)

        obs, reward, terminated, truncated, info = env.step(action)
        cmd = info["command"]
        #print("action:", action, "observation", obs, "command", cmd, "reward", reward)

        error = np.linalg.norm(env.get_vector_reward_space(obs) - cmd)
        error_history.append(error)
        reward_history.append(reward)
        x_current = env._state

        if terminated or truncated:
            obs, info = env.reset()
            x_current = env._state
            u_current = np.zeros((mpc.Nu, 1))
            cmd = info["command"]
        
    if(plot):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(error_history, 'b-', label='errors')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(reward_history, 'g-', label='rewards')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    tracking_error = np.mean(error_history)
    rewards = np.mean(reward_history)
    if(tracking_error < 0.01):
        print(cmd, task["reward_weight"], task["reward_dim"])
        print(task["ld_B"])

    name = "MPC" if use_mpc else "Random"
    
    print(f"--{name}-- Tracking Errors: {tracking_error:.4f} Rewards: {rewards:.4f}")


if __name__ == "__main__":
    import gymnasium as gym
    import numpy
    from xenoverse.linds import LinearDSSamplerRandomDim

    task = LinearDSSamplerRandomDim()
    env = gym.make("linear-dynamics-v0-visualizer")
    env.set_task(task)

    print("Start MPC solver demonstration...")
    test_mpc(env, use_mpc=False)
    test_mpc(env)
    print("...Test Passed")