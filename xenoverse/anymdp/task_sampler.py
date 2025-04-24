"""
Any MDP Task Sampler
"""
import numpy
from numpy import random
from copy import deepcopy
from xenoverse.utils import pseudo_random_seed
from xenoverse.anymdp.solver import check_valuefunction
from xenoverse.utils import RandomFourier

eps = 1e-10

def sample_mdp(state_number, na,
              s0_range=3, verbose=False):
    task = dict()
    assert state_number >=8, "state_number must be at least 8 for MDP"

    # sample S_0
    assert s0_range > 0
    if(s0_range < 2):
        s_0_prob = numpy.array([1.0], dtype=int)
        s_0 = [0]
    else:
        while (numpy.sum(s_0_prob) < eps):
            s_0_prob = numpy.clip(random.normal(loc=0, scale=1, size=(s0_range)), 0, None)
        s_0 = numpy.where(s_0_prob > eps)[0]
        s_0_prob = s_0_prob[s_0]
        s_0_prob = s_0_prob / numpy.sum(s_0_prob)
    task.update({"s_0": numpy.copy(s_0),
                 "s_0_prob": numpy.copy(s_0_prob)})
    
    # sample S_E
    p_s_e_base = numpy.clip(numpy.random.uniform(-0.20, 0.40), 0.0, None) # 40% pitfalls at maximum
    while True:
        s_e = numpy.random.choice([0, 1], size=state_number, p=[1 - p_s_e_base, p_s_e_base])
        if(numpy.sum(s_e) < state_number * p_s_e_base + 1):
            break
    s_e[s_0] = 0 # make sure S_0 do not reset
    if(random.random() < 0.3): # with 30% probability the last state is goal. Sampling is balanced by value function filtering.
        s_e[-1] = 1
        final_goal = True
    else:
        s_e[-1] = 0
        final_goal = False
    s_e = numpy.where(s_e == 1)[0]
    task.update({"s_e": s_e})

    # sample transition s-s'
    trans_ss = numpy.zeros((state_number, state_number), dtype=float)
    max_leap = random.randint(2, min(state_number // 4, 6) + 1) # from 2 to 6
    max_retreat = random.randint(2, min(state_number // 2, 8) + 1) # from 2 to 8

    ss_from = numpy.zeros(state_number, dtype=int)
    ss_to = numpy.zeros(state_number, dtype=int)
    for s in range(state_number):
        if(s in s_e): continue

        s_from_min = max(0, s - max_retreat)
        if(s > 2):
            s_from = random.randint(s_from_min, s-1)  # start of the transition
        else:
            s_from = 0
        s_to = random.randint(s + 1, s + max_leap) + 1 # end of the transition (exclusive)

        s_to = min(s_to, state_number)

        while(s_to < state_number):
            valid_leap = False
            for s_future in range(s + 1, s_to):
                if(s_future not in s_e):
                    valid_leap = True
                    break
            if(valid_leap):
                break
            s_to += 1

        ss_from[s] = s_from
        ss_to[s] = s_to

        trans_ss[s, s_from:s_to] = random.uniform(0.20, 1, size=(s_to - s_from))
        trans_ss[s] = trans_ss[s] / numpy.sum(trans_ss[s])

    # sample average rewards s-s'

    # sample potential reward
    if(random.random() < 0.5):
        potential_reward_base = 0
    else:
        potential_reward_base = random.exponential(1.0)
    potential_reward_generator = RandomFourier(ndim=1, 
                                        max_order=5, 
                                        max_item=3, 
                                        max_steps=state_number * 2,
                                        box_size=max(random.uniform(-potential_reward_base, potential_reward_base), 0.0))
    potential_reward = []
    for s in range(state_number):
        potential_reward.append(potential_reward_generator(s)[0])
    potential_reward = numpy.array(potential_reward)

    # calculate potential cost
    potential_cost = numpy.max(potential_reward[-1] - potential_reward[s_0])

    # add state-dependent reward
    position_reward = numpy.zeros(state_number)
    position_reward_noise = numpy.zeros(state_number)
    position_reward_base = random.exponential(1.0)
    position_reward_noise_base  = numpy.clip(random.uniform(-0.30, 0.30), 0.0, None)

    # award those at the last part
    for s in range(state_number):
        if(s in s_e or s in s_0): continue
        offset = (s + 1) / state_number
        # Add position reward at the final half of the states
        if(s > state_number // 2 and random.random() < offset - 0.5 and not final_goal):
            position_reward[s] = random.uniform(0.4, offset) * position_reward_base
            position_reward_noise[s] = position_reward_noise_base * position_reward[s]
            # avoid easy-to-get rewards at the same state
            trans_ss[s, s] /= 10.0 # at least reduce it to < 0.1
            trans_ss[s] = trans_ss[s] / numpy.sum(trans_ss[s])
        elif(random.random() < 0.15): # set random cost
            position_reward[s] = - random.uniform(0.2, 0.50) * position_reward_base
            position_reward_noise[s] = - position_reward_noise_base * position_reward[s]

    # punish the pitfalls
    if(random.random() < 0.2):
        pitfalls_reward_base = 0
    elif(random.random() < 0.60):
        pitfalls_reward_base = - random.exponential(0.20)
    else:
        pitfalls_reward_base = - random.exponential(2.0)

    for s in s_e:
        if(s < state_number - 1): # not the final goal
            position_reward[s] = pitfalls_reward_base
    
    # sample step cost / survive award
    if(random.random() < 0.50):
        step_reward = 0.0
    else:
        if(final_goal):
            step_reward = - random.exponential(0.02)
        else:
            step_reward = random.exponential(0.02)
    
    goal_cost = max(potential_cost, 0) + max(random.exponential(0.20), 0.05)
    
    if(final_goal):
        position_reward[-1] = random.uniform(4.0 * goal_cost, 10.0 * goal_cost)
    else:
        position_reward[-1] = random.uniform(4.0 * position_reward_base, 10.0 * position_reward_base)
        trans_ss[-1, -1] /= 10.0 # at least reduce it to < 0.1
        trans_ss[-1] = trans_ss[-1] / numpy.sum(trans_ss[-1])

    # now further decompose the transition
    transition = numpy.zeros((state_number, na, state_number), dtype=float)

    for s in range(state_number):
        if(s in s_e): continue
        a_center = random.uniform(ss_from[s], ss_to[s], size=na)

        # na x ns dimension, representing the distance of the action to the corresponding state
        a_dist = (a_center[:, None] - numpy.arange(ss_from[s], ss_to[s])[None, :]) ** 2
        sigma = max(random.exponential(1.0), 0.5)
        
        a_prob = numpy.exp(-a_dist / sigma**2)

        # now calculate the weight for each action
        s_sum_prob = numpy.sum(a_prob, axis=0)

        # in case some element of s_weight < eps, just find the nearest action
        for i in numpy.where(s_sum_prob < eps)[0]:
            a_prob[numpy.argmin(a_dist[:, i]), i] = 1.0

        # normalize probability according to dimension na
        a_prob = a_prob / numpy.sum(a=a_prob, axis=0)

        transition[s, :, ss_from[s]:ss_to[s]] = a_prob * trans_ss[s:s+1, ss_from[s]:ss_to[s]]

        transition[s] = transition[s] / numpy.sum(transition[s], axis=-1, keepdims=True)

    # normalize the transition

    # prepare the reward matrix
    reward = numpy.zeros((state_number, na, state_number), dtype=float)
    reward_noise = numpy.zeros((state_number, na, state_number), dtype=float)
    rnd_reward_base = numpy.clip(random.exponential(0.05), 0.0, 0.10)

    reward += potential_reward[:, None, None] - potential_reward[None, None, :]
    reward += position_reward[None, None, :]
    reward += step_reward

    sparsity = numpy.clip(numpy.random.uniform(-0.7, 0.3), 0, None)

    if(sparsity > eps):
        reward += (random.normal(size=(state_number, na)) * rnd_reward_base * (numpy.random.rand(state_number, na) < sparsity).astype(float))[:, :, None]
        reward_noise += random.uniform(0.0, 0.30, size=reward.shape) * rnd_reward_base

    reward_noise += position_reward_noise[None, None, :]

    task.update({"transition": numpy.copy(transition),
                 "reward": numpy.copy(reward),
                 "reward_noise": numpy.copy(reward_noise),
                 "final_goal_terminate": final_goal})

    return task

def sample_bandit(na):
    base = random.exponential(1.0)
    noise_base = numpy.clip(random.uniform(-0.30, 0.30), 0.0, None)
    transition = numpy.ones((1, na, 1), dtype=float)
    while True:
        reward = random.uniform(0.5 * base, base, size=(1, na, 1))
        if(numpy.std(reward) > 0.01):
            break
    reward_noise = noise_base * reward
    return {"transition": numpy.copy(transition),
           "reward": numpy.copy(reward),
           "reward_noise": numpy.copy(reward_noise),
           "s_0": numpy.array([0]),
           "s_e": numpy.array([]),
           "s_0_prob": numpy.array([1.0])}

def AnyMDPTaskSampler(state_space:int=128,
                 action_space:int=5,
                 min_state_space:int=None,
                 seed=None,
                 verbose=False):
    # Sampling Transition Matrix and Reward Matrix based on Irwin-Hall Distribution and Gaussian Distribution
    if(seed is not None):
        random.seed(seed)
    else:
        random.seed(pseudo_random_seed())

    assert(state_space >= 8 or state_space == 1),"State Space must be at least 8 or 1 (Multi-armed Bandit)!"
    
    if(state_space < 2):
        max_steps = 1
    else:
        lower_bound = max(4.0 * state_space, 100)
        upper_bound = min(15.0 * state_space, 1000)
        max_steps = random.uniform(lower_bound, upper_bound)
    
    # Sample a subset of states
    if(min_state_space is None):
        min_state_space = state_space
        real_state_space = state_space
    else:
        min_state_space = min(min_state_space, state_space)
        assert(min_state_space >= 8), "Minimum State Space must be at least 8!"
        real_state_space = random.randint(min_state_space, state_space + 1)
    state_mapping = numpy.random.permutation(state_space)[:real_state_space]

    # Generate Transition Matrix While Check its Quality
    task = {"ns": state_space,
            "na": action_space,
            "max_steps": max_steps,
            "state_mapping": state_mapping}
    
    while(True):
        if(real_state_space == 1):
            task.update(sample_bandit(action_space))
            break
        else:
            task.update(sample_mdp(real_state_space, action_space, verbose))
            if(check_valuefunction(task, verbose=verbose)):
                break

    return task