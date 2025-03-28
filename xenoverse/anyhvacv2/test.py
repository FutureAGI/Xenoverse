if __name__ == "__main__":

    from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler

    env = HVACEnvVisible()
    print("Sampling hvac tasks...")
    task = HVACTaskSampler()
    print("... Finished Sampling")
    env.set_task(task)
    done = False
    obs = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print("sensors - ", obs, "\nactions - ", action, "\nrewards - ", reward, "ambient temperature - ", env.ambient_temp)