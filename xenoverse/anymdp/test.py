if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import xenoverse.anymdp
    from xenoverse.anymdp import  AnyMDPSolverOpt, AnyMDPSolverMBRL, AnyMDPSolverQ, AnyMDPTaskSampler
    from xenoverse.anymdp.debug import run_rnd_opt

    def test_q(env, max_steps, gamma, c, lr):
        # Test AnyMDPSolverQ
        solver = AnyMDPSolverQ(env, gamma=gamma, c=c, alpha=lr)
        state, info = env.reset()
        acc_reward = 0
        epoch_reward = 0
        steps = 0
        prt_freq = 1000

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

    def test_mbrl(env, max_steps, gamma, c):
        # Test AnyMDPSolverMBRL
        solver = AnyMDPSolverMBRL(env, gamma=gamma, c=c)
        state, info = env.reset()
        solver.set_reset_states(state)
        acc_reward = 0
        epoch_reward = 0
        steps = 0
        prt_freq = 1000

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
                solver.set_reset_states(state)

        print("MBRL Solver Summary: {}".format(acc_reward))

    task = AnyMDPTaskSampler(state_space=8, 
                             action_space=5,
                             min_state_space=None,
                             verbose=True)
    
    run_rnd_opt(task, max_steps=10000)
    
    env = gym.make("anymdp-v0")
    env.set_task(task)
    #test_q(env, max_steps=1000000, gamma=0.99, c=0.50, lr=0.10)
    test_mbrl(env, max_steps=1000000, gamma=0.99, c=0.05)
    print("Test Passed")