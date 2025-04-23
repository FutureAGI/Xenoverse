import numpy
import numpy as np
from numpy import random
from numba import njit
import networkx as nx
import scipy.stats as stats
from scipy.stats import spearmanr

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
    
def check_valuefunction(task):
    t_mat = task["transition"]
    r_mat = task["reward"]
    ns, na, _ = t_mat.shape
    gamma = 0.994
    vm_opt = update_value_matrix(t_mat, r_mat, gamma, numpy.zeros((ns, na), dtype=float), is_greedy=True)
    vm_rnd = update_value_matrix(t_mat, r_mat, gamma, numpy.zeros((ns, na), dtype=float), is_greedy=False)
    print(vm_opt, vm_rnd)
    for s in task["s_0"]:
        if(numpy.max(vm_opt[s]) - numpy.max(vm_rnd[s]) < 0.01):
            return False
    return True