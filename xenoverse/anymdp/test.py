if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import xenoverse.anymdp
    from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPSolverOTS, AnyMDPSolverQ, AnyMDPTaskSampler
    from xenoverse.anymdp.debug import run_rnd_opt

    task = AnyMDPTaskSampler(state_space=128, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    run_rnd_opt(task, max_steps=10000)

    max_steps = 1000000
    prt_freq = 1000
    gamma = 0.98
    c = 0.01
    lr = 0.10

    env = gym.make("anymdp-v0")
    env.set_task(task)

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