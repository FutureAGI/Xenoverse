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
    p_s_e_base = numpy.random.uniform(0, 0.5) # 50% pitfalls at maximum
    s_e = numpy.random.choice([0, 1], size=state_number, p=[1 - p_s_e_base, p_s_e_base])
    s_e[s_0] = 0 # make sure S_0 do not reset
    if(random.random() < 0.5): # with 50% probability the last state is goal
        s_e[-1] = 1
        final_goal = True
    else:
        s_e[-1] = 0
        final_goal = False
    task.update({"s_e": numpy.where(s_e == 1)[0]})

    # sample transition s-s'
    trans_ss = numpy.zeros((state_number, state_number), dtype=float)
    max_leap = random.randint(2, min(state_number // 4, 6)) # from 2 to 6
    ss_from = numpy.zeros(state_number, dtype=int)
    ss_to = numpy.zeros(state_number, dtype=int)
    for s in range(state_number):
        if(s in s_e): continue

        if(s > 2):
            s_from = random.randint(0, s-1)  # start of the transition
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

        trans_ss[s, s_from:s_to] = random.uniform(0, 1, size=(s_to - s_from))
        trans_ss[s] = trans_ss[s] / numpy.sum(trans_ss[s])

    # sample average rewards s-s'

    # sample potential reward
    potential_reward_generator = RandomFourier(ndim=1, 
                                        max_order=5, 
                                        max_item=3, 
                                        max_steps=state_number * 2,
                                        box_size=max(random.uniform(-5.0, 5.0), 0.0))
    potential_reward = []
    for s in range(state_number):
        potential_reward.append(potential_reward_generator(s)[0])
    potential_reward = numpy.array(potential_reward)

    # calculate potential cost
    potential_cost = numpy.max(potential_reward[-1] - potential_reward[s_0])

    # calculate point cost
    # probability to achieve the states
    achieve_probability = numpy.sum(trans_ss, axis=0)

    # award those hard to achieve
    position_reward = numpy.zeros(state_number)
    position_reward_noise = numpy.zeros(state_number)
    base = random.exponential(1.0)
    noise_base = numpy.clip(random.uniform(-0.30, 0.30), 0.0, None)

    # award those hard to achieve
    if(random.random() < 0.67):
        for s in range(state_number):
            if(s in s_e or s in s_0): continue
            award_bar = random.uniform(0.05, 0.40)
            punish_bar = random.uniform(0.80, 1.20)
            if(achieve_probability[s] < award_bar):
                position_reward[s] = random.uniform(0.5, 1.0) * base
                position_reward_noise[s] = noise_base * position_reward[s]
            elif(achieve_probability[s] > punish_bar): # those easy to achieve
                position_reward[s] = - random.uniform(0.5, 1.0) * base
                position_reward_noise[s] = - noise_base * position_reward[s]

    # award the pitfalls
    pitfalls_base = numpy.clip(random.uniform(-100.0, 10.0), None, 0.0)
    for s in s_e:
        if(s < state_number - 1): # not the final goal
            position_reward[s] = pitfalls_base
    
    # sample step cost / survive award
    if(random.random() < 0.33):
        step_reward = 0.0
    else:
        if(final_goal):
            step_reward = - random.exponential(0.10)
        else:
            step_reward = random.exponential(0.10)

    # dynamic programming to find the cost
    cur_cost = numpy.ones(state_number) * 1e10
    cur_cost[s_0] = 0
    active_queue = [s for s in s_0]
    while(len(active_queue) > 0):
        s = active_queue.pop(0)
        for s_next in numpy.where(trans_ss[s] != 0)[0]:
            new_cost = cur_cost[s] - min(position_reward[s_next], 0) - min(step_reward) # add the cost
            if(new_cost < cur_cost[s_next]):
                cur_cost[s_next] = new_cost
                active_queue.append(s_next)
    
    final_cost = cur_cost[-1]
    goal_cost = final_cost + potential_cost

    if(final_goal):
        position_reward[-1] = random.uniform(1.5 * goal_cost, 4.0 * goal_cost)
    else:
        position_reward[-1] = max(random.uniform(2.0 * base, 10.0 * base),
                                  2.0 * state_number * numpy.abs(step_reward))
        
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

    # prepare the reward matrix
    reward = numpy.zeros((state_number, na, state_number), dtype=float)
    reward_noise = numpy.zeros((state_number, na, state_number), dtype=float)

    reward += potential_reward[:, None, None] - potential_reward[None, None, :]
    reward += position_reward[None, None, :]
    reward += step_reward

    sparsity = numpy.clip(numpy.random.uniform(-0.7, 0.3), 0, None)
    if(sparsity > eps):
        reward += (random.normal(size=(state_number, na)) * base * (numpy.random.rand(state_number, na) < sparsity).astype(float))[:, :, None]

    reward_noise += position_reward_noise[None, None, :]

    task.update({"transition": numpy.copy(transition),
                 "reward": numpy.copy(reward),
                 "reward_noise": numpy.copy(reward_noise)})

    return task

def sample_bandit(na):
    base = random.exponential(1.0)
    noise_base = numpy.clip(random.uniform(-0.30, 0.30), 0.0)
    transition = numpy.zeros((1, na, 1), dtype=float)
    reward = random.uniform(0.5 * base, base, size=(1, na, 1))
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
        else:
            task.update(sample_mdp(real_state_space, action_space, verbose))
        if(check_valuefunction(task)):
            break

    return task