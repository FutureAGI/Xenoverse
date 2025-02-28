if __name__=="__main__":
    import gym
    import xenoverse.anymdp
    from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ, AnyMDPTaskSampler

    task = AnyMDPTaskSampler(state_space=64, 
                             action_space=5,
                             min_state_space=32,
                             verbose=True)
    max_steps = 32000
    prt_freq = 1000

    # Test D2C Random Policy
    env = gym.make("anymdp-d2c-v0")
    env.set_task(task)

    # Test Random Policy
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    done = False

    steps = 0
    while steps < max_steps:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
    print("D2C Random Policy Summary: {}".format(acc_reward))

    # Test Random Policy
    env = gym.make("anymdp-v0")
    env.set_task(task)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    done = False

    steps = 0
    while steps < max_steps:
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
    print("Random Policy Summary: {}".format(acc_reward))

    # Test AnyMDPSolverOpt
    solver = AnyMDPSolverOpt(env)
    state, info = env.reset()
    done = False
    acc_reward = 0
    epoch_reward = 0
    steps = 0

    while steps < max_steps:
        action = solver.policy(state)
        state, reward, done, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
            state_list = []
    print("Optimal Solver Summary:  {}".format(acc_reward))

    # Test AnyMDPSolverQ
    solver = AnyMDPSolverQ(env)
    state, info = env.reset()
    done = False
    acc_reward = 0
    epoch_reward = 0
    steps = 0

    while steps < max_steps:
        action = solver.policy(state)
        next_state, reward, done, info = env.step(action)
        solver.learner(state, action, next_state, reward, done)
        acc_reward += reward
        epoch_reward += reward
        state = next_state
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
    print("Q Solver Summary: {}".format(acc_reward))

    # Test AnyMDPSolverOTS
    solver = AnyMDPSolverOTS(env)
    state, info = env.reset()
    done = False
    acc_reward = 0
    epoch_reward = 0
    steps = 0

    while steps < max_steps:
        action = solver.policy(state)
        next_state, reward, done, info = env.step(action)
        solver.learner(state, action, next_state, reward, done)
        acc_reward += reward
        epoch_reward += reward
        state = next_state
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(done):
            state, info = env.reset()
    print("OTS Solver Summary: {}".format(acc_reward))

    print("Test Passed")