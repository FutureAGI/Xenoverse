import numpy
from numpy import random
from numba import njit
import networkx as nx
import scipy.stats as stats

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
    quality = -1
    if(t_mat is None):
        return quality
    # acquire state - to - state distribution
    ns = t_mat.shape[0]
    log_ns = int(numpy.floor(numpy.log2(ns)))
    ss_trans = numpy.sum(t_mat, axis=1)
    ss_trans = ss_trans / numpy.sum(ss_trans, axis=1, keepdims=True)

    for i in range(log_ns):
        ss_trans = numpy.matmul(ss_trans, ss_trans)
        ss_unreach = numpy.sum(ss_trans < 1.0e-6)
        if(ss_unreach > 0):
            quality = max(quality, i / log_ns + ss_unreach / ns / ns)
    ss_unreach = numpy.sum(ss_trans < 1.0e-6, axis=1)
    if(numpy.any(ss_unreach > 0)):
        return 0
    if(ns < 4):
        return 1 # where states below 4, transition is all ok as long as strongly connected
    return quality

def check_valuefunction(t_mat, r_mat):
    if(t_mat is None or r_mat is None):
        return -100
    ns, na, _ = r_mat.shape
    if(ns < 2): # For bandit problem, only check rewards
        if(numpy.std(r_mat) > 1.0e-3):
            return 1
        else:
            return -100

    vm_l = update_value_matrix(t_mat, r_mat, 0.99, numpy.zeros((ns, na), dtype=float), max_iteration=5)
    vm_s = update_value_matrix(t_mat, r_mat, 0.70, numpy.zeros((ns, na), dtype=float), max_iteration=5)
    vm_r = update_value_matrix(t_mat, r_mat, 0.99, numpy.zeros((ns, na), dtype=float), max_iteration=5, is_greedy=False)

    vm_l = numpy.max(vm_l, axis=1)
    vm_s = numpy.max(vm_s, axis=1)
    vm_r = numpy.max(vm_r, axis=1)

    vbase = numpy.sqrt(numpy.mean(vm_r ** 2))

    wht = 1.5

    corr_ls, _ = stats.spearmanr(vm_l, vm_s)
    corr_lr, _ = stats.spearmanr(vm_l, vm_r)
    qdelta = wht * numpy.tanh(numpy.mean((vm_l - vm_r)) / vbase / wht)
    qstd = wht * numpy.tanh(numpy.std(vm_l) / vbase / wht)

    if(qstd < 0.1): # value function too flat
        return -100

    if(numpy.isnan(corr_lr)):
        corr_lr = 1
    if(numpy.isnan(corr_ls)):
        corr_ls = 1

    q_random = numpy.log((1 + 1.0e-10 - corr_lr))
    q_longshort = numpy.log((1 + 1.0e-10 - corr_ls))

    quality = q_random + q_longshort + qdelta + qstd

    return quality


def check_task_trans(task):
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
    return check_transition(t_mat)


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
        return -100

    t_mat = get_final_transition(**task)
    r_mat = get_final_reward(**task)
    return check_valuefunction(t_mat, r_mat)