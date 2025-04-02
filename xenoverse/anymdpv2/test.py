if __name__=="__main__":
    import gymnasium as gym
    import numpy
    from xenoverse.anymdpv2 import AnyMDPv2TaskSampler

    task = AnyMDPv2TaskSampler(state_dim=128, 
                             action_dim=16)
    max_steps = 5000
    prt_freq = 100

    # Test Random Policy
    env = gym.make("anymdp-v2-visualizer")
    env.set_task(task)
    state = env.reset()
    acc_reward = 0
    epoch_reward = 0
    done = False
    obs_arr = []
    act_arr = []
    state_arr = []
    step_lst = []

    steps = 0
    episode_steps = 0
    while steps < max_steps:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        episode_steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(terminated or truncated):
            step_lst.append(episode_steps)
            episode_steps = 0
            state, info = env.reset()
    print(f"Random Policy Summary: {acc_reward}, Average Episode Length:{numpy.mean(step_lst)}")
    env.visualize_and_save()

    print("Test Passed")