if __name__ == "__main__":
    from xenoverse.anyhvac.anyhvac_env import HVACEnv
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
    env = HVACEnvVisible()
    print("Sampling hvac tasks...")
    task = HVACTaskSampler()
    print("... Finished Sampling")
    env.set_task(task)
    done = False
    obs = env.reset()
    agent = HVACSolverGTPID(env)
    while not done:
        action = agent.policy()
        obs, reward, done, info = env.step(action)
        print("sensors - ", obs, "actions - ", action, "rewards - ", reward, "ambient temperature - ", env.ambient_temp)