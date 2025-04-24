import numpy
import numpy as np
from numpy import random
from numba import njit
import networkx as nx
import scipy.stats as stats
from scipy.stats import spearmanr
from copy import deepcopy

def normalized_mrr(scores1, scores2, k=None):
    assert numpy.shape(scores1) == numpy.shape(scores2)
    n = numpy.shape(scores1)[0]

    if k is None:
        k = n
    else:
        k = min(k, n)

    indices1 = np.argsort(-scores1)
    indices2 = np.argsort(-scores2)
    indices1_rev = indices1[::-1]

    ranks = np.zeros(n)
    for i, idx in enumerate(indices2):
        ranks[idx] = i + 1

    invranks = np.zeros(n)
    for i, idx in enumerate(indices1_rev):
        invranks[idx] = i + 1

    mrrmax = 0.0
    mrrmin = 0.0
    mrr = 0.0

    for i in range(k):
        idx = indices1[i]
        mrrmax += 1.0 / (i + 1) ** 2
        mrrmin += 1.0 / ((i + 1) * invranks[idx])
        mrr += 1.0 / ((i + 1) * ranks[idx])

    return (mrr - mrrmin) / (mrrmax - mrrmin)
    
def mean_mrr(X, Y, k=None):
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if(X.ndim == 1):
        return normalized_mrr(X, Y, k)
    nmrrs = []

    for i in range(X.shape[0]):
        x_col = X[i]
        y_col = Y[i]
        nmrr = normalized_mrr(x_col, y_col)
        nmrrs.append(nmrr)
    return numpy.mean(nmrrs)

@njit(cache=True)
def update_value_matrix(t_mat, r_mat, gamma, vm, max_iteration=-1, is_greedy=True):
    diff = 1.0
    cur_vm = numpy.copy(vm)
    ns, na, _ = r_mat.shape
    iteration = 0
    while diff > 1.0e-4 and (
            (max_iteration < 0) or 
            (max_iteration > iteration and max_iteration > 1) or
            (iteration < 1 and random.random() < max_iteration)):
        iteration += 1
        old_vm = numpy.copy(cur_vm)
        for s in range(ns):
            for a in range(na):
                exp_q = 0.0
                for sn in range(ns):
                    if(is_greedy):
                        exp_q += t_mat[s,a,sn] * numpy.max(cur_vm[sn])
                    else:
                        exp_q += t_mat[s,a,sn] * numpy.mean(cur_vm[sn])
                cur_vm[s,a] = numpy.dot(r_mat[s,a], t_mat[s,a]) + gamma * exp_q
        diff = numpy.sqrt(numpy.mean((old_vm - cur_vm)**2))
    return cur_vm

def get_opt_trajectory_dist(s0, s0_prob, se, ns, na, transition, vm, K=8):
    a_max = numpy.argmax(vm, axis=1)
    i_indices = np.arange(ns)[:, None]
    j_indices = np.arange(ns)
    max_trans = transition[i_indices, a_max[:, None], j_indices]
    for s in se:
        max_trans[s, s0] = s0_prob # s_e directly lead to s0

    for _ in range(K):
        max_trans = numpy.matmul(max_trans, max_trans)
    
    gini_impurity = []
    normal_entropy = []

    for s in s0:
        stable_prob = max_trans[s] + 1.0e-12 # calculation safety
        gini_impurity.append(1.0 - numpy.sum(stable_prob * stable_prob))
        normal_entropy.append(-numpy.sum(stable_prob * numpy.log(stable_prob)) / numpy.log(ns))

    # Check gini impurity
    return numpy.min(gini_impurity), numpy.min(normal_entropy)

def check_valuefunction(task, verbose=False):
    t_mat = numpy.copy(task["transition"])

    r_mat = numpy.copy(task["reward"])
    ns, na, _ = t_mat.shape
    gamma = numpy.power(2, -1.0 / ns)
    vm_opt = update_value_matrix(t_mat, r_mat, gamma, numpy.zeros((ns, na), dtype=float), is_greedy=True)
    vm_rnd = update_value_matrix(t_mat, r_mat, gamma, numpy.zeros((ns, na), dtype=float), is_greedy=False)

    vm_wht = numpy.ones((ns, ), dtype=int)
    vm_wht[task["s_e"]] = 0
    vm_rnd_std = numpy.std(vm_rnd[numpy.where(vm_wht>0)])

    # Get Average Reward
    avg_vm_opt = vm_opt * (1.0 - gamma) * task["max_steps"]
    avg_vm_rnd = vm_rnd * (1.0 - gamma) * task["max_steps"]
    vm_diffs = []

    for s in task["s_0"]:
        vm_diff = numpy.max(avg_vm_opt[s]) - numpy.max(avg_vm_rnd[s])
        if(vm_diff < (0.10 * vm_rnd_std + 0.01)):
            return False
        vm_diffs.append(vm_diff)
        
    # check the stationary distribution of the optimal value function
    K = int(numpy.log2(task["max_steps"])) + 1
    gini, ent = get_opt_trajectory_dist(
                            deepcopy(task["s_0"]), 
                            numpy.copy(task["s_0_prob"]),
                            deepcopy(task["s_e"]), 
                            ns, na, 
                            numpy.copy(t_mat), 
                            vm_opt, 
                            K=K)
    
    t_mat_sum = numpy.sum(t_mat, axis=-1)
    error = (t_mat_sum - 1.0)**2
    error[task["s_e"]] = 0.0
    if((error >= 1.0e-6).any()):
        if(verbose):
            print("Transition Matrix Error: ", numpy.where(error>=1.0e-6))
        return False
    
    vm_diffs = numpy.mean(vm_diffs)
    if(verbose):
        print("Value Diff: {:.4f}, Gini Impurity: {:.4f}, Normalized Entropy: {:.4f}, final_goal_terminate: {}".format(vm_diffs, gini,ent, task["final_goal_terminate"]))
    return gini > 0.7 and ent > 0.5