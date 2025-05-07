if __name__ == "__main__":
    import numpy 
    import numbers
    from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvacv2.anyhvac_solver import HVACSolverGTPID
    import pickle 

    env = HVACEnv()
    TASK_CONFIG_PATH = "./hvac_task_config.pkl"
    try:
        with open(TASK_CONFIG_PATH, "rb") as f:
            task = pickle.load(f)
        print(f"Loaded existing task config from {TASK_CONFIG_PATH}")
    
    except FileNotFoundError:
        print("Sampling new HVAC tasks...")
        task = HVACTaskSampler(control_type='Temperature')
        with open(TASK_CONFIG_PATH, "wb") as f:
            pickle.dump(task, f)
        print(f"... Saved new task config to {TASK_CONFIG_PATH}")
    env.set_task(task)
    terminated, truncated = False,False
    obs = env.reset()
    max_steps = 10000
    current_stage = []
    steps = 0
    while steps < max_steps:
        action = env.sample_action(mode="pid")
        obs, reward, terminated, truncated, info = env.step(action)
        current_stage.append(reward)
        if steps < 1:
            info_sums = {key: 0.0 for key in info.keys()}
            info_counts = {key: 0 for key in info.keys()}
        for key, value in info.items():
            if isinstance(value, (int, float)):
                info_sums[key] += value
                info_counts[key] += 1
        
        steps += 1
        # print("sensors - ", obs, "\nactions - ", action, "\nrewards - ", reward, "ambient temperature - ", env.ambient_temp)
        if steps % 100 == 0:
            mean_reward = numpy.mean(current_stage)
            cool_power = round(numpy.mean(info.get("cool_power", 0)),4)
            heat_power = round(numpy.mean(info.get("heat_power", 0)),4)

            fail_step_percrentage = info["fail_step_percrentage"] if isinstance(info["fail_step_percrentage"], numbers.Real) else 0
            info_total = f"energy_cost: {round(info.get('energy_cost', 0),4)}, target_cost: {round(info.get('target_cost', 0),4)}, switch_cost: {round(info.get('switch_cost', 0),4)},cool_power: {cool_power}, heat_power: {heat_power}"
            

            
            # 格式化输出


            print(f"Step {steps} | fail_step_percrentage:{fail_step_percrentage} | Reward: {mean_reward:.2f} | {info_total}| cool_power: {cool_power:.2f} | heat_power:{heat_power:.2f} ", flush=True)
            
            # 重置统计量
            current_stage = []
            info_sums = {k:0.0 for k in info_sums}
            info_counts = {k:0 for k in info_counts}