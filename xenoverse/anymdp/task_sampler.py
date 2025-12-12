"""
Any MDP Task Sampler
"""
import numpy
import scipy.sparse as sp
from numpy import random
from copy import deepcopy
from xenoverse.utils import pseudo_random_seed
from xenoverse.anymdp.solver import check_valuefunction
from xenoverse.utils import RandomFourier
from xenoverse.anymdp.task_sampler_utils import sample_bandit, sample_mdp, sample_garnet

eps = 1e-10

def AnyMDPTaskSampler(state_space:int=64,
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
        upper_bound = max(min(8.0 * state_space, 500), lower_bound + 1)
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
            "state_mapping": state_mapping,
            "task_type": "MDP"}
    
    while(True):
        if(real_state_space == 1):
            task.update(sample_bandit(action_space))
            break
        else:
            res = sample_mdp(real_state_space, action_space, max_steps, verbose=verbose)
            if(res is not None):
                task.update(res)
                if(check_valuefunction(task, verbose=verbose)):
                    break
            elif(verbose):
                print("Failed to generate valid MDP, trying again...")

    return task

def AnyPOMDPTaskSampler(
                 state_space:int=64,
                 action_space:int=5,
                 min_state_space:int=None,
                 observation_space:int=64,
                 density=0.20,
                 maximum_distribution=4,
                 seed=None,
                 verbose=False):
    
    task = AnyMDPTaskSampler(state_space, action_space, min_state_space, seed, verbose)
    task["no"] = observation_space
    task["task_type"] = "POMDP"
    real_states = task["state_mapping"].shape[0]
    density = min(density, maximum_distribution / observation_space)
    obs_mat=sp.random(real_states, observation_space, density=density, format='csr').toarray()
    for i in range(real_states):
        if(numpy.sum(obs_mat[i]) == 0):
            obs_mat[i][random.randint(observation_space)] = 1
        obs_mat[i] /= numpy.sum(obs_mat[i])
    task["observation_transition"] = obs_mat
    return task

def MultiTokensAnyPOMDPTaskSampler(
                 state_space:int=256,
                 action_space:int=5,
                 min_state_space:int=None,
                 observation_space:int=64,
                 observation_tokens:int=4,
                 action_tokens:int=2,
                 density=0.20,
                 maximum_distribution=4,
                 seed=None,
                 verbose=False):
    
    task = AnyMDPTaskSampler(state_space, action_space, min_state_space, seed, verbose)
    task["no"] = observation_space
    task["do"] = observation_tokens
    task["da"] = action_tokens
    task["task_type"] = "MTPOMDP"
    real_states = task["state_mapping"].shape[0]
    task["observation_transition"] = []

    for obs_tok in range(observation_tokens):
        density = min(density, maximum_distribution / observation_space)
        obs_mat=sp.random(real_states, observation_space, density=density, format='csr').toarray()
        for i in range(real_states):
            if(numpy.sum(obs_mat[i]) == 0):
                obs_mat[i][random.randint(observation_space)] = 1
            obs_mat[i] /= numpy.sum(obs_mat[i])
        task["observation_transition"].append(obs_mat)
    return task

def GarnetTaskSampler(state_space:int=128,
                 action_space:int=5,
                 min_state_space:int=None,
                 b:int=2,
                 sigma:float=0.1,
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
        upper_bound = max(min(8.0 * state_space, 500), lower_bound + 1)
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
    
    task.update(sample_garnet(real_state_space, action_space, max_steps, b, sigma, verbose=verbose))

    return task