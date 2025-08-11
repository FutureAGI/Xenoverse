if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import xenoverse.anymdp
    from xenoverse.anymdp import AnyMDPTaskSampler, GarnetTaskSampler
    from xenoverse.anymdp.test_utils import train

    task = AnyMDPTaskSampler(state_space=16, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    # Test Garnet Task Sampler
    """
    task = GarnetTaskSampler(state_space=16, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    """
    
    env = gym.make("anymdp-v0")
    env.set_task(task)
    
    train_rewards, test_rewards, train_steps = train(env, max_epochs=100, gamma=0.99, solver_type='random', lr=0.20)
    train_rewards, test_rewards, train_steps = train(env, max_epochs=100, gamma=0.99, solver_type='opt', lr=0.20)
    train_rewards, test_rewards, train_steps = train(env, max_epochs=10000, gamma=0.99, solver_type='q', lr=0.20)

    print("Test Passed")