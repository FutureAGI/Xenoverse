if __name__ == "__main__":
    import numpy
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID

    env = HVACEnvVisible(verbose=True)
    print("Sampling hvac tasks...")
    task = HVACTaskSampler(control_type='Temperature')
    print("... Finished Sampling")
    env.set_task(task)
    terminated, truncated = False,False
    obs, info = env.reset()
    agent = HVACSolverGTPID(env)
    while (not terminated) and (not truncated):
        # Random Policy
        #action = env.action_space.sample()
        # Correlation PID
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)