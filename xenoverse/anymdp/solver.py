import numpy
from numpy import random
from numba import njit
import networkx as nx
import scipy.stats as stats
from scipy.stats import spearmanr

def graph_diameter(t_mat, threshold=1.0e-4):
    G = nx.DiGraph()
    ss_trans = numpy.sum(t_mat, axis=1)
    ss_trans = ss_trans / numpy.sum(ss_trans, axis=1, keepdims=True)
    
    for i in range(len(ss_trans)):
        for j in range(len(ss_trans)):
            if(ss_trans[i][j] > threshold):
                G.add_edge(i, j, weight=ss_trans[i][j])
    if(not nx.is_strongly_connected(G)):
        diameter = -1
    else:
        diameter = nx.diameter(G)
    return diameter

import numpy as np
from scipy.stats import spearmanr

def mean_spearmanr(X, Y):
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape")
    if(X.ndim == 1):
        coeff = spearmanr(X, Y)[0]
        if(numpy.isnan(coeff)):
            return 1.0
        else:
            return coeff
    spearman_coeffs = []

    for i in range(X.shape[0]):
        x_col = X[i]
        y_col = Y[i]
        coeff, _ = spearmanr(x_col, y_col)
        if(not numpy.isnan(coeff)):
            spearman_coeffs.append(coeff)
    mean_coeff = numpy.mean(spearman_coeffs)
    if(len(spearman_coeffs) == 0):
        return 1.0
    return mean_coeff

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

def task_diameter(task):
    return graph_diameter(task['transition'])

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

def get_final_transition(**task):
    t_mat = numpy.copy(task["transition"])
    if(t_mat.shape[0] < 2):
        return t_mat
    
    reset_dist = task["reset_states"]
    reset_trigger = numpy.where(task["reset_triggers"] > 0)

    for s in reset_trigger:
        t_mat[s, :] = reset_dist

    return t_mat

def get_final_reward(**task):
    r_mat = numpy.copy(task["reward"])
    if(r_mat.shape[0] < 2):
        return r_mat
    reset_trigger = numpy.where(task["reset_triggers"] > 0)

    r_mat[reset_trigger, :, :] = 0.0

    return r_mat

def check_transition(t_mat):
    ns = t_mat.shape[0]
    if(t_mat is None):
        return quality
    # acquire state - to - state distribution
    log_ns = int(numpy.floor(numpy.log2(ns)))
    ss_trans = numpy.sum(t_mat, axis=1)
    ss_trans = ss_trans / numpy.sum(ss_trans, axis=1, keepdims=True)
    quality = 0
    for i in range(log_ns):
        ss_trans = numpy.matmul(ss_trans, ss_trans)
        ss_unreach = numpy.sum(ss_trans < 1.0e-6)
        if(ss_unreach > 0):
            quality = max(quality, i / log_ns + ss_unreach / ns / ns)
    ss_unreach = numpy.sum(ss_trans < 1.0e-6, axis=1)
    if(numpy.any(ss_unreach > 0)): # not connected
        return False
    if(ns < 6):
        return True # where states below 6, transition is all ok as long as strongly connected
    return (quality > 0.50)

def check_transition_2(t_mat):
    quality = -1
    if(t_mat is None):
        return quality
    d = graph_diameter(t_mat)
    # Not connected
    if(d < 0):
        return False
    
    ns = t_mat.shape[0]
    d_H = 2.0 * numpy.sqrt(ns)
    return d > d_H
    
def check_valuefunction(t_mat, r_mat):
    # returns 0 (invalid) and 1 (valid)
    if(t_mat is None or r_mat is None):
        return 0
    ns, na, _ = r_mat.shape
    if(ns < 2): # For bandit problem, only check rewards
        if(numpy.std(r_mat) > 0.01):
            return True
        else:
            return False

    vm_l = update_value_matrix(t_mat, r_mat, 0.994, numpy.zeros((ns, na), dtype=float), max_iteration=5)
    vm_s = update_value_matrix(t_mat, r_mat, 0.50, numpy.zeros((ns, na), dtype=float), max_iteration=5)
    vm_r = update_value_matrix(t_mat, r_mat, 0.994, numpy.zeros((ns, na), dtype=float), max_iteration=5, is_greedy=False)

    corr_ls_pi = mean_mrr(vm_l, vm_s, k=3) # ndcg@3
    corr_lr_pi = mean_mrr(vm_l, vm_r, k=3) # ndcg@3

    inter_s_std = numpy.std(vm_l)
    intra_s_std = numpy.mean(numpy.std(vm_l, axis=1))

    vm_l_max = numpy.max(vm_l, axis=1)
    vm_s_max = numpy.max(vm_s, axis=1)

    corr_ls = mean_spearmanr(vm_l_max, vm_s_max)

    corr_thres=0.85
    corr_ls_pi_thres=0.75
    corr_lr_pi_thres=0.75

    if(inter_s_std < 0.1 or intra_s_std < 0.1): # value function too flat
        return False
    elif(corr_ls > corr_thres):
        return False
    elif(corr_ls_pi > corr_ls_pi_thres):
        return False
    elif(corr_lr_pi > corr_lr_pi_thres):
        return False
    
    return True

def check_task_trans(task, transition_check_type=1):
    """
    Check the quality of the task
    Requiring: Q value is diverse
               State is connected
               Route is complex
    Returns:
        float: transition quality
        float: value function quality
    """
    if(task is None or 
       not "transition" in task or 
       not "reset_states" in task or
       not "reset_triggers" in task):
        return -1
    if(task["transition"].shape[0] < 2):
        return 1
    t_mat = get_final_transition(**task)
    if(transition_check_type == 1):
        return check_transition(t_mat)
    elif(transition_check_type == 2):
        return check_transition_2(t_mat)
    else:
        raise ValueError(f"Unknown transition check type: {transition_check_type}")


def check_task_rewards(task):
    """
    Check the quality of the task
    Requiring: Q value is diverse
               State is connected
               Route is complex
    Returns:
        float: transition quality
        float: value function quality
    """
    if(task is None or 
       not "transition" in task or 
       not "reset_states" in task or
       not "reset_triggers" in task or 
       not "reward" in task):
        return False

    t_mat = get_final_transition(**task)
    r_mat = get_final_reward(**task)
    return check_valuefunction(t_mat, r_mat)