import numpy
from numpy import random
from copy import deepcopy
from xenoverse.utils import pseudo_random_seed
from xenoverse.anymdp.solver import check_valuefunction, update_value_matrix
from xenoverse.utils import RandomFourier, random_partition

eps = 1e-10

def sample_potential_reward(ns, avg=1.0, low=0.20, high=5.0): 
    """
    Potential reward serves as the a reward shaping to the task
    Returns: reward, noise of shape (ns, 1, ns)
    """
    if(random.random() < 0.5):
        potential_reward_base = 0
    else:
        potential_reward_base = numpy.clip(random.exponential(avg), low, high)
    potential_generator = RandomFourier(ndim=1, 
                                        max_order=5, 
                                        max_item=3, 
                                        max_steps=ns * 2,
                                        box_size=max(random.uniform(-potential_reward_base, potential_reward_base), 0.0))
    potential = []
    for s in range(ns):
        potential.append(potential_generator(s)[0])
    potential= numpy.array(potential)
    potential_reward = potential[:, None, None] - potential[None, None, :]

    return potential_reward

def sample_position_reward(ns, s_e, avg=0.2, low=0.04, high=1.0):
    """
    Position reward serves as continuous punishment or reward to the task
    Returns: reward, noise of shape (1, 1, ns)
    """
    position_reward_base = random.exponential(0.2)

    random_pdf = numpy.clip(random.normal(size=(ns,)), 0.0, None) 
    random_pdf[-1] += 0.20
    random_pdf *= position_reward_base
    random_cdf = numpy.cumsum(random_pdf)
    random_baseline = random.uniform(0.1 * random_cdf[-1], 0.9 * random_cdf[-1])
    
    position_reward = random_cdf - random_baseline
    position_reward_noise = numpy.clip(random.uniform(-0.30, 0.30, size=position_reward.shape), 0.0, None) * position_reward_base
    position_reward[s_e] = 0.0
    position_reward_noise[s_e] = 0.0

    return position_reward[None, None, :], position_reward_noise[None, None, :]

def sample_state_action_cost(ns, na, avg=0.05, low=0.0, high=0.10, sparsity=0.3):
    """
    A random state - action cost / reward at each step
    Returns: reward, noise of shape (ns, na, 1)
    """
    reward = numpy.zeros((ns, na), dtype=float)
    reward_noise = numpy.zeros((ns, na), dtype=float)
    rnd_reward_base = numpy.clip(random.exponential(0.05), 0.0, 0.10)
    sparsity = (numpy.random.uniform(-0.7, 0.3, size=(ns, na)) > 0).astype(float)
    reward = rnd_reward_base * random.normal(size=(ns, na)) * sparsity
    reward_noise = 0.30 * rnd_reward_base * numpy.clip(random.normal(size=(ns, na)), 0, None) * sparsity
    return reward[:, :, None], reward_noise[:, :, None]

def sample_transition(ns, na, s0_range=3):
    # sample S_0
    assert s0_range > 0
    if(s0_range < 2):
        s_0_prob = numpy.array([1.0], dtype=int)
        s_0 = [0]
    else:
        s_0_prob = 0.0
        while (numpy.sum(s_0_prob) < eps):
            s_0_prob = numpy.clip(random.normal(loc=0, scale=1, size=(s0_range)), 0, None)
        s_0 = numpy.where(s_0_prob > eps)[0]
        s_0_prob = s_0_prob[s_0]
        s_0_prob = s_0_prob / numpy.sum(s_0_prob)
    
    # sample S_E
    p_s_e_base = numpy.clip(numpy.random.uniform(-0.20, 0.40), 0.0, None) # 40% pitfalls at maximum
    while True:
        s_e = numpy.random.choice([0, 1], size=ns, p=[1 - p_s_e_base, p_s_e_base])
        if(numpy.sum(s_e) < ns * p_s_e_base + 1):
            break
    s_e[s_0] = 0 # make sure S_0 do not reset
    if(random.random() < 0.3): # with 30% probability the last state is goal. Sampling is balanced by value function filtering.
        s_e[-1] = 1
        final_terminate = True
    else:
        s_e[-1] = 0
        final_terminate = False
    s_e = numpy.where(s_e == 1)[0].tolist()

    # sample transition s-s'
    trans_ss = numpy.zeros((ns, ns), dtype=float)
    min_leap = 2
    max_leap = max(min_leap + 1, ns // 4 + 1) # max forward leap
    min_back = 4
    max_back = max(min_back + 1, ns // 2 + 1) # max backward leap

    ss_from = numpy.zeros(ns, dtype=int)
    ss_to = numpy.zeros(ns, dtype=int)
    for s in range(ns):
        if(s in s_e): continue

        s_from_min = max(0, s - max_back)
        s_from_max = max(0, s - min_back, s_from_min + 1)
        s_to_max = min(ns,     s + max_leap)
        s_to_min = min(ns - 1, s + min_leap, s_to_max - 1)

        s_from = random.randint(s_from_min, s_from_max)  # start of the transition
        s_to = random.randint(s_to_min, s_to_max) # end of the transition (exclusive)

        while(s_to < ns):
            valid_leap = []
            for s_future in range(s + 1, s_to):
                if(s_future not in s_e):
                    valid_leap.append(s_future)
            if(len(valid_leap) > 1):  # At least one leap forward is valid
                break
            s_to += 1

        ss_from[s] = s_from
        ss_to[s] = s_to
        if(final_terminate):
            valid_leap.append(ns - 1)

        if(len(valid_leap) > 1):
            while numpy.sum(trans_ss[s][valid_leap]) < 1.0e-3 or numpy.where(trans_ss[s] > 1.0e-3)[0].size < 2: # At least 2 transition
                trans_ss[s, s_from:s_to] = numpy.clip(random.normal(size=(s_to - s_from)), 0.10, 1.0)
        else:
            while numpy.sum(trans_ss[s]) < 1.0e-3 or numpy.where(trans_ss[s] > 1.0e-3)[0].size < 2:
                trans_ss[s, s_from:s_to] = numpy.clip(random.normal(size=(s_to - s_from)), 0.10, 1.0)

        # avoid self-loop
        trans_ss[s, s] /= 2.0
        if(s == ns - 1): # last state stop self loop
            trans_ss[s, s] = 0

        # normalize the transition probability
        trans_ss[s] = trans_ss[s] / numpy.sum(trans_ss[s])


    # now further decompose the transition into states
    transition = numpy.zeros((ns, na, ns), dtype=float)

    for s in range(ns):
        if(s in s_e): continue
        a_center = random.uniform(ss_from[s] - 1, ss_to[s], size=na)

        # na x ns dimension, representing the distance of the action to the corresponding state
        a_dist = (a_center[:, None] - numpy.arange(ss_from[s], ss_to[s])[None, :]) ** 2
        sigma = numpy.clip(random.exponential(1.0), 0.20, 1.6)
        
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
        
    return s_0, s_0_prob, s_e, final_terminate, transition

def sample_mdp(ns, na, max_steps, s0_range=3, verbose=False, max_try=5):
    task = dict()
    assert ns >=8, "ns must be at least 8 for MDP"

    s_0, s_0_prob, s_e, final_terminate, transition = sample_transition(ns, na, s0_range=s0_range)
    task.update({"s_0": numpy.copy(s_0),
                 "s_0_prob": numpy.copy(s_0_prob),
                 "s_e": numpy.copy(s_e),
                 "transition": numpy.copy(transition),
                 "final_goal_terminate": final_terminate})

    # sample potential reward (V(s') - V(s))
    r_pot = sample_potential_reward(ns)

    # add state-dependent reward
    r_s, r_s_noise = sample_position_reward(ns, s_e)

    # add state-action dependent reward
    r_sa, r_sa_noise = sample_state_action_cost(ns, na)

    if(final_terminate): # we may add negative step_reward
        r_step = min(random.normal(), 0.0) * 0.01
    elif(len(s_e) > 0): # we may add survival step reward when there is a pitfall
        r_step = max(random.normal(), 0.0) * 0.01
    else: # otherwise, no need for step reward
        r_step = 0.0
    
    raw_reward = r_pot + r_s + r_sa + r_step
    reward_noise = r_s_noise + r_sa_noise

    # start with a reward for pitfalls
    term_reward = numpy.zeros(ns, dtype=float)
    term_reward[-1] = 1.0

    gamma = 0.99

    # now check the value function, reduce the pitfalls to make sure value of pitfalls falls below any other places
    pitfalls = deepcopy(s_e)
    if(final_terminate):
        last_valid_s = ns-2
        pitfalls.remove(ns-1)
    else:
        last_valid_s = ns-1

    non_pitfalls = [i for i in range(ns) if i not in s_e]

    t_mat = numpy.copy(transition)
    vm = numpy.zeros((ns, na), dtype=float)
    cur_try = 0

    while cur_try < max_try:
        r_mat = raw_reward + term_reward[None, None, :]
        vm = update_value_matrix(t_mat, r_mat, gamma, vm)
        vsm = numpy.max(vm, axis=-1)

        pitgain = numpy.min(term_reward) - numpy.min(vsm[non_pitfalls]) + 1.0
        goalfall = numpy.max(vsm[s_0]) - vsm[last_valid_s] + random.uniform(2.0, 5.0)

        if(pitgain <= 0 and goalfall <= 0):
            break

        if(pitgain > 0.0): # try to keep pit always below
            term_reward[pitfalls] -= pitgain + random.uniform(1.0, 10.0)
        if(goalfall > 0.0): # try keeping the goal always above
            deta_v = max(2.0 * goalfall, random.uniform(1.0, 10.0))
            if(final_terminate):
                term_reward[-1] += deta_v
            else:
                term_reward[-1] += (1.0 - gamma) * deta_v
        cur_try += 1
    if(cur_try >= max_try): # just can not fix the environment
        return None

    reward = raw_reward + term_reward[None, None, :]

    task.update({"transition": numpy.copy(transition),
                 "reward": numpy.copy(reward),
                 "reward_noise": numpy.copy(reward_noise)})

    return task

def sample_bandit(na):
    base = numpy.clip(random.exponential(1.0), 0.05, 2.0)
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

def sample_sparse_matrix(n, m, k, seed=None):
    if k <= 0 or k > n:
        raise ValueError("k must satisfy 0 < k <= n")
    
    if seed is not None:
        numpy.random.seed(seed)
    
    matrix = numpy.zeros((n, m, n))
    arr = numpy.arange(n)
        
    for i in range(n):
        for j in range(m):
            sample = random.choice(arr, size=k, replace=False)
            partition = random_partition(k)
            matrix[i, j, sample] = partition
    
    return matrix

def sample_garnet(ns, na, max_steps, b, sigma=0.2, r_mean=0.0, verbose=False):
    task = dict()
    assert ns >=8, "ns must be at least 8 for MDP"

    s_0_prob = numpy.array([1.0], dtype=int)
    s_0 = [0]
    transition = sample_sparse_matrix(ns, na, b)

    task.update({"s_0": numpy.copy(s_0),
                 "s_0_prob": numpy.copy(s_0_prob),
                 "s_e": numpy.array([]),
                 "transition": numpy.copy(transition),
                 "final_goal_terminate": False})

    reward = random.normal(size=(ns, na, ns)) * sigma + r_mean
    reward_noise = numpy.zeros((ns, na, ns), dtype=float)

    task.update({"transition": numpy.copy(transition),
                 "reward": numpy.copy(reward),
                 "reward_noise": numpy.copy(reward_noise)})

    return task

def sample_procon_stage_1(ns, na, l, cyc):
    """
    Sample a ProCon stage 1 task with given parameters
    """
    assert ns >= 8, "ns must be at least 8 for ProCon stage 1" 
    assert na >= 2, "na must be at least 2 for ProCon stage 1"
    assert l >= 2, "l must be at least 2 for ProCon stage 1"
    assert cyc >= 2 and cyc < ns, "cyc must be at least 2 and at most ns - 1 for ProCon stage 1"
    state_seq = []
    s_0 = random.randint(0, ns)
    s_0_prob = numpy.zeros(ns, dtype=float)
    s_0_prob[s_0] = 1.0
    state_seq.append(s_0)
    state_all = list(range(ns))
    state_left = list(range(ns))
    state_left.remove(s_0)
    opt_transition = numpy.zeros((ns, ns), dtype=float)
    edges = []
    s = prev_s

    while True:
        prev_s = state_seq[-1]
        while s in state_seq[-cyc:]:
            s = random.choice(state_left)
        opt_transition[prev_s, s] = random.uniform(0.01, 1.0)
        state_seq.append(s)
        state_left.remove(s)
        edges.append((prev_s, s))
        if(s in state_seq[:-(cyc + 1)]):
            break

    state_idx = deepcopy(state_seq)
    while len(state_left) > 0:
        s = random.choice(state_left)
        while True:
            next_s = random.choice(state_all)
            opt_transition[s, next_s] = random.uniform(0.01, 1.0)
            edges.append((prev_s, s))
            state_left.remove(s)
            state_idx.append(s)
            if next_s in state_seq:
                break

    return opt_transition, edges, s_0, s_0_prob, state_seq