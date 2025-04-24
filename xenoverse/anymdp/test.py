if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import xenoverse.anymdp
    from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ, AnyMDPTaskSampler

    task = AnyMDPTaskSampler(state_space=32, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    max_steps = 200000
    max_steps_rnd = 10000
    prt_freq = 1000
    gamma = 0.98
    c = 0.01
    lr = 0.10

    # Test Random Policy
    env = gym.make("anymdp-v0")
    env.set_task(task)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0

    steps = 0
    while steps < max_steps_rnd:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(terminated or truncated):
            state, info = env.reset()
    print("Random Policy Summary: {}".format(acc_reward))

    # Test AnyMDPSolverOpt
    solver = AnyMDPSolverOpt(env)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    steps = 0
    epoch_step = 0
    epoch_steps = []
    epoch_trajectory = [int(state)]
    while steps < max_steps_rnd:
        action = solver.policy(int(state))
        state, reward, terminated, truncated, info = env.step(action)
        epoch_trajectory.append(int(state))
        acc_reward += reward
        epoch_reward += reward
        steps += 1
        epoch_step += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(terminated or truncated):
            state, info = env.reset()
            epoch_trajectory.append(int(state))
            epoch_steps.append(epoch_step)
            epoch_step = 0
            state_list = []
    print("Optimal Solver Summary:  {}, Averge Length: {}, Epoch Trajectory: {}".format(acc_reward, numpy.mean(epoch_steps), epoch_trajectory[:100]))

    # Test AnyMDPSolverQ
    solver = AnyMDPSolverQ(env, gamma=gamma, c=c, alpha=lr)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    steps = 0

    while steps < max_steps:
        action = solver.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        solver.learner(state, action, next_state, reward, terminated, truncated)
        acc_reward += reward
        epoch_reward += reward
        state = next_state
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(terminated or truncated):
            state, info = env.reset()
    print("Q Solver Summary: {}".format(acc_reward))

    # Test AnyMDPSolverOTS
    solver = AnyMDPSolverOTS(env, gamma=gamma, c=c, alpha=lr)
    state, info = env.reset()
    acc_reward = 0
    epoch_reward = 0
    steps = 0

    while steps < max_steps:
        action = solver.policy(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        solver.learner(state, action, next_state, reward, terminated, truncated)
        acc_reward += reward
        epoch_reward += reward
        state = next_state
        steps += 1
        if(steps % prt_freq == 0 and steps > 0):
            print("Step:{}\tEpoch Reward: {}".format(steps, epoch_reward))
            epoch_reward = 0
        if(terminated or truncated):
            state, info = env.reset()
    print("OTS Solver Summary: {}".format(acc_reward))

    print("Test Passed")