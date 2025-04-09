if __name__ == "__main__":

    from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler

    env = HVACEnvVisible()
    print("Sampling hvac tasks...")
    task = HVACTaskSampler()
    print("... Finished Sampling")
    env.set_task(task)
    terminated, truncated = False,False
    obs = env.reset()
    while (not terminated) and (not truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print("sensors - ", obs, "\nactions - ", action, "\nrewards - ", reward, "ambient temperature - ", env.ambient_temp)