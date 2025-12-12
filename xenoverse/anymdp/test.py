if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import xenoverse.anymdp
    from xenoverse.anymdp import AnyMDPTaskSampler, GarnetTaskSampler, AnyPOMDPTaskSampler, MultiTokensAnyPOMDPTaskSampler
    from xenoverse.anymdp.test_utils import train
    from xenoverse.utils import dump_task, load_task

    """
    # Test MDP Task Sampler
    task = AnyMDPTaskSampler(state_space=16, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    # Test Garnet Task Sampler
    task = GarnetTaskSampler(state_space=16, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    # Test POMDP Task Sampler
    task = AnyPOMDPTaskSampler(state_space=16, 
                             action_space=5,
                             min_state_space=None,
                             observation_space=16,
                             density = 0.1,
                             verbose=True)

    task = MultiTokensAnyPOMDPTaskSampler(state_space=128, 
                             action_space=5,
                             min_state_space=None,
                             observation_space=32,
                             observation_tokens=4,
                             action_tokens=1,
                             density = 0.1,
                             verbose=True)
    """

    task = MultiTokensAnyPOMDPTaskSampler(state_space=128, 
                             action_space=5,
                             min_state_space=None,
                             observation_space=32,
                             observation_tokens=4,
                             action_tokens=1,
                             density = 0.1,
                             verbose=True)
    dump_task("./task.pkl", task)
    
    env = gym.make("anymdp-v0")
    env.set_task(task)
    
    train_rewards, test_rewards, train_steps = train(env, max_epochs=100, gamma=0.99, solver_type='random', lr=0.20)
    train_rewards, test_rewards, train_steps = train(env, max_epochs=100, gamma=0.99, solver_type='opt', lr=0.20)
    train_rewards, test_rewards, train_steps = train(env, max_epochs=10000, gamma=0.99, solver_type='q', lr=0.20)

    print("Test Passed")