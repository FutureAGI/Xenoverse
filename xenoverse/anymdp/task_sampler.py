"""
Any MDP Task Sampler
"""
import numpy
import gym
import pygame
import time
from numpy import random
from copy import deepcopy
from xenoverse.utils import pseudo_random_seed
from xenoverse.anymdp.solver import check_task_trans, check_task_rewards


def reward_sampler_sas(state_space:int, 
                   action_space:int, 
                   reward_sparsity:float=0.25,
                   exp_positive_ratio:float=0.10,
                   reward_noise_max:float=0.30):
    reward_mask = numpy.zeros((state_space,))
    reward_sparsity = min(reward_sparsity * random.exponential(1.0), 1.0)
    reward_mask = random.binomial(1, reward_sparsity, size=(state_space,))
    reward_noise_mask = random.binomial(1, reward_sparsity, size=(state_space,))
    reward_noise = reward_noise_mask * reward_noise_max  * random.random() * random.rand(state_space)

    positive_ratio = min(exp_positive_ratio * random.exponential(1.0), 1.0)
    sign_mask = 2.0 * random.binomial(1, positive_ratio, size=(state_space,)) - 1
    reward_matrix = numpy.abs(random.normal(loc=0, scale=1.0, size=(state_space,))) * reward_mask * sign_mask
    return reward_matrix, reward_noise

def comb_sampler(n):
    # 生成范围[0, 2^n -1]，确保均匀性
    num = random.randint(0, 2 ** n - 1)
    # 使用 [2:] 仅移除 '0b' 前缀，保留所有有效位
    comb_type = bin(num)[2:]
    # 补零或截断至 n 位
    if len(comb_type) < n:
        comb_type = '0' * (n - len(comb_type)) + comb_type
    else:
        comb_type = comb_type[-n:]
    return comb_type

def reward_sampler(task:dict,
                   state_space:int, 
                   action_space:int,
                   reward_noise_choice:list):

    if("reset_triggers_positive" not in task):
        return {}

    reward_scale = random.uniform(0.01, 0.10) # Scale the reward noise
    reward_noise_type =  random.choice(reward_noise_choice)

    if(state_space < 2):
        reward_matrix = numpy.random.random((1, action_space, 1)) * reward_scale
        reward_noise = 0.3 * numpy.random.random((1, action_space, 1)) * reward_scale
        return {"reward": reward_matrix,
                "reward_noise": reward_noise,
                "reward_noise_type": reward_noise_type}

    # Sample Pitfalls
    v_scale = random.uniform(10, 100) * reward_scale
    rewards_neg = - numpy.random.random(size=[state_space]) * v_scale \
        * task["reset_triggers_negative"]
    # Sample Sucesses
    rewards_pos = task["reset_triggers_positive"] * \
        numpy.clip(
            numpy.random.exponential(scale=v_scale,
            size=state_space), v_scale, None)

    # Crucial Rewards
    rewards = rewards_neg + rewards_pos

    # Reshape the reward to ns, na, ns
    reward_matrix = numpy.zeros((state_space, action_space, state_space))
    reward_noise = numpy.zeros((state_space, action_space, state_space))

    reward_matrix += rewards[None, None, :]
    reward_is_crucial = (numpy.abs(rewards) > 1.0e-6)
    reward_noise_factor = v_scale * random.random()
    reward_noise += reward_noise_factor * (numpy.random.random(size=[state_space]) * reward_is_crucial.astype(float))[None, None, :]
    crucial_index = numpy.where(reward_is_crucial)

    reward_type=comb_sampler(3)
    if(reward_type[0] == '1'): # Add random -s reward
        rm, rn = reward_sampler_sas(state_space, action_space)
        rm[crucial_index] = 0.0 # Make sure the reset triggers are not rewarded
        rn[crucial_index] = 0.0 # Make sure the reset triggers are not rewarded
        reward_matrix += random.exponential(0.10) * reward_scale * rm[None, None, :]
        reward_noise  += random.exponential(0.10) * reward_scale * rn[None, None, :]
    if(reward_type[1] == '1'): # Add random a reward
        reward_matrix += random.exponential(0.01) * reward_scale * (numpy.random.random((1, action_space, 1)) - 0.5)
    if(reward_type[2] == '1'): # Add a random s-s reward
        ndim = task["state_embedding"].shape[1]
        r_direction = numpy.random.random((1, ndim)) # Sample a random reward direction
        r_direction /= numpy.linalg.norm(r_direction)
        r_direction = r_direction * task["state_embedding"]
        r_direction *= random.random() * reward_scale
        r_direction = numpy.sum(r_direction, axis=1)
        reward_matrix += r_direction[None, None, :] - r_direction[:, None, None]

    if(numpy.max(reward_matrix) > 1.01 or numpy.min(reward_matrix) < -1.01):
        reward_noise_type = 'normal'
    else:
        reward_matrix = numpy.clip(reward_matrix, -1.0, 1.0)

    return {"reward": reward_matrix,
            "reward_noise": reward_noise,
            "reward_noise_type": reward_noise_type}

def transition_sampler(state_space:int, 
                    action_space:int, 
                    min_state_space:int, 
                    transition_diversity:int):
        
        # Sample a subset of states
        if(min_state_space is None):
            min_state_space = state_space
        else:
            min_state_space = min(min_state_space, state_space)
        sample_state_space = random.randint(min_state_space, state_space + 1)        
        state_mapping = numpy.random.permutation(state_space)[:sample_state_space]

        if(state_space < 1):
            raise ValueError("State space must be at least 1")

        if(state_space < 2):
            return {"state_mapping": state_mapping,
                    "reset_triggers": numpy.zeros((1,)),
                    "reset_states": numpy.ones((1,)),
                    "transition": numpy.ones((1, action_space, 1)),
                    "reset_triggers_negative": None,
                    "reset_triggers_positive": None,
                    "state_embedding": None}
    
        # Sample the reset states
        eps = 1e-6

        # Now sample dimension of the tasks
        ndim = random.randint(2, 8)

        # Sample the location of states
        loc_s = numpy.random.random((sample_state_space, ndim))

        # Sample the main axis length
        max_axis_length = random.randint(2, sample_state_space)
        loc_s[:, 0] *= max_axis_length

        # Sample positive reset trigger and reset states
        iter = 0
        max_iter = 10000
        need_reset_trigger = (random.random() > 0.70) # 30% probability of not requiring reset triggers
        reset_trigger_positive = numpy.zeros(sample_state_space)
        reset_dist = numpy.zeros(sample_state_space)
        while(((numpy.sum(reset_trigger_positive) < 1 and not need_reset_trigger)
              or numpy.sum(reset_dist) < 1)
              and iter < max_iter):

            proc_s = loc_s[:, 0] / max_axis_length
            max_proc_s = numpy.max(proc_s) - eps
            min_proc_s = numpy.min(proc_s) + eps
            max_proc_s_start = min(0.60, max_proc_s - 0.20)
            min_proc_s_end  = max(0.40, min_proc_s + 0.20)

            if(not need_reset_trigger):
                positive_prob = numpy.clip(proc_s - random.uniform(max_proc_s_start, max_proc_s), 0.0, 1.0)
                positive_prob *= max(1, 0.05 * random.random() * sample_state_space) / (numpy.sum(positive_prob) + eps) 
                positive_prob = numpy.clip(positive_prob, 0.0, 1.0)
                reset_trigger_positive = numpy.random.binomial(1, positive_prob)

            # Sample the reset state distribution
            reset_prob = numpy.clip(random.uniform(min_proc_s, min_proc_s_end) - proc_s, 0.0, 1.0)
            reset_prob *= max(1, 0.05 * random.random() * sample_state_space) / (numpy.sum(reset_prob) + eps) 
            reset_prob = numpy.clip(reset_prob, 0.0, 1.0)
            if(numpy.sum(reset_prob) < 1):
                reset_prob += eps
            reset_dist = numpy.random.binomial(1, reset_prob).astype(float) * (1.0 - reset_trigger_positive) # Avoid resetting to triggers

            # Sample negative reset trigger - Uniform distribution
            # A upperbound of 20% of the state triggers reset
            reset_trigger_negative = numpy.random.binomial(1, 
                                            0.2 * random.random(), 
                                            size=(sample_state_space,)) * \
                                    (1.0 - reset_trigger_positive) * (1.0 - reset_dist)

            reset_trigger = reset_trigger_positive + reset_trigger_negative

            iter += 1

        if(iter >= max_iter):
            raise RuntimeError("Failed to sample a valid task")

        reset_dist = reset_dist / numpy.sum(reset_dist)

        # Sample a upper bound of the transition
        adj_ub = random.randint(2, min(transition_diversity + 1, sample_state_space + 1))

        # Calculate the distance between states
        dist_s = numpy.linalg.norm(loc_s[None, :, :] - loc_s[:, None, :], axis=2)

        # Calculate the minimum k distance between states
        avg_dist = numpy.mean(numpy.sort(dist_s, axis=1)[:, 1:adj_ub])

        # Now sample action of transitions with shape [ns, na, ndim]
        dir_s_a = numpy.random.randn(sample_state_space, action_space, ndim)
        dir_s_a = dir_s_a / numpy.linalg.norm(dir_s_a, axis=2, keepdims=True)
        dist_s_a = (1.0 + 2.0 * numpy.random.random(size=(sample_state_space, action_space))) * avg_dist
        deta_s_a = dir_s_a * dist_s_a[:, :, None]

        # Now calculate the target loc by projecting a with shape[ns, na, ndim]
        target_s_a = deta_s_a + loc_s[:, None, :]

        # Now caculate the state-action-state distance after taking action [ns, na, ns]
        dist_s_a_s = numpy.linalg.norm(target_s_a[:, :, None, :] - loc_s[None, None, :, :], axis=3)
        sigma = avg_dist * random.exponential(scale=1.0)

        prob_s_a_s = numpy.exp(- (dist_s_a_s / sigma) ** 2)

        # Now sample the transition by masking the state-action-state distance
        dist_s_a_s_index = numpy.argsort(dist_s_a_s, axis=2)[:, :, :adj_ub]

        extra_mask = numpy.random.choice([0, 1], size=dist_s_a_s_index.shape)
        extra_mask_sum = extra_mask.sum(axis=2)
        extra_mask_zero = numpy.where(extra_mask_sum == 0)

        for idx in zip(*extra_mask_zero):
            extra_mask[idx[0], idx[1], numpy.random.randint(adj_ub)] = 1
        dist_s_a_s_index = dist_s_a_s_index * extra_mask - (1 - extra_mask)

        # Set the distance beyond n_s_a to inf
        bool_arr = numpy.zeros_like(dist_s_a_s, dtype=bool)
        for i in range(adj_ub):
            bool_arr |= (dist_s_a_s_index[:, :, i][:, :, None] == numpy.arange(sample_state_space)[None, None, :])
        
        prob_s_a_s[bool_arr==False] = 0.0
        transition_matrix = prob_s_a_s / numpy.sum(prob_s_a_s, axis=2, keepdims=True)

        return {"state_mapping": state_mapping,
                "reset_triggers": reset_trigger,
                "reset_states": reset_dist,
                "transition": transition_matrix,
                "reset_triggers_negative": reset_trigger_negative,
                "reset_triggers_positive": reset_trigger_positive,
                "state_embedding": loc_s}

def AnyMDPTaskSampler(state_space:int=128,
                 action_space:int=5,
                 min_state_space:int=None,
                 epoch_state_visit:int=4,
                 max_transition_diversity:int=8,
                 max_iteration:int=50,
                 quality_threshold_transition:float=0.55,
                 quality_threshold_valuefunction:float=0.0,
                 reward_noise_choice:list=['normal'],
                 seed=None,
                 keep_metainfo=False,
                 verbose=False):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(pseudo_random_seed())
    
    max_steps = max(int(numpy.random.exponential(epoch_state_visit * state_space)), epoch_state_visit * state_space)

    # Generate Transition Matrix While Check its Quality
    task = {"state_space": state_space,
            "action_space": action_space,
            "max_steps": max_steps}

    qtrans = -1
    qvf = -1

    trans_step = 0
    vf_step = 0

    qtrans_max = -100
    qvf_max = -100
    best_task = None

    while (qtrans < quality_threshold_transition) and trans_step < max_iteration:
        task.update(transition_sampler(state_space, 
                                       action_space,
                                       min_state_space,
                                       max_transition_diversity))
        qtrans = check_task_trans(task)
        if(qtrans > qtrans_max):
            qtrans_max = qtrans
            best_task = deepcopy(task)
        trans_step += 1
    task = best_task

    while (qvf < quality_threshold_valuefunction) and vf_step < max_iteration:
        substate_space = task["state_mapping"].shape[0]
        task.update(reward_sampler(task,
                                   substate_space, 
                                   action_space,
                                   reward_noise_choice))
        qvf = check_task_rewards(task)
        if(qvf > qvf_max):
            qvf_max = qvf
            best_task = deepcopy(task)
        vf_step += 1

    if(verbose):
        print(f"Resample transitions {trans_step} times, quality: {qtrans_max:.2f}",
              f"Rsample rewards {vf_step} times, quality: {qvf_max:.2f}")

    if(not keep_metainfo):
        del task["state_embedding"]
        del task["reset_triggers_positive"]
        del task["reset_triggers_negative"]

    return best_task