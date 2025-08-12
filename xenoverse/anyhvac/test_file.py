if __name__ == "__main__":
    import numpy
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
    import pickle 

    env = HVACEnv()
    TASK_CONFIG_PATH = "./hvac_task_config.pkl"
    try:
        with open(TASK_CONFIG_PATH, "rb") as f:
            task = pickle.load(f)
        print(f"Loaded existing task config from {TASK_CONFIG_PATH}")
    
    except FileNotFoundError:
        print("Sampling new HVAC tasks...")
        task = HVACTaskSampler()
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
            
            # 计算各信息字段均值
            info_means = {
                key: info_sums[key] / info_counts[key] 
                for key in info_sums
            }
            
            # 格式化输出
            info_str = " | ".join([f"{k}:{v:.4f}" for k,v in info_means.items()])
            print(f"Step {steps} | Reward: {mean_reward:.2f} | {info_str}", flush=True)
            
            # 重置统计量
            current_stage = []
            info_sums = {k:0.0 for k in info_sums}
            info_counts = {k:0 for k in info_counts}