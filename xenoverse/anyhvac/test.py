if __name__ == "__main__":
    import gymnasium as gym
    from xenoverse.anyhvac.anyhvac_env import HVACEnv
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
    env = gym.make("anyhvac-visualizer-v0")
    print("Sampling hvac tasks...")
    task = HVACTaskSampler()
    print("... Finished Sampling")
    env.set_task(task)
    terminated, truncated = False, False
    obs, info = env.reset()
    agent = HVACSolverGTPID(env)
    while (not terminated) and (not truncated):
        action = agent.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        print("sensors - ", obs, "actions - ", action, "rewards - ", reward, "ambient temperature - ", env.ambient_temp)