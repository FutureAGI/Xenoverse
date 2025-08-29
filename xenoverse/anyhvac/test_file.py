if __name__ == "__main__":
    import numpy
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
    import pickle 
    from rl_trainer import HVACRLTester

    env = HVACEnv(verbose=True)
    TASK_CONFIG_PATH = "./task_file/hvac_task_config_0828_1.pkl"
    RL_MODEL_PATH = "./models/0828/temp/1/sac_random_start_no_normal/sac_stage2.zip"
    model = HVACRLTester(RL_MODEL_PATH, "sac", "cpu")
    try:
        with open(TASK_CONFIG_PATH, "rb") as f:
            task = pickle.load(f)
        print(f"Loaded existing task config from {TASK_CONFIG_PATH}")
        print("t_ambient", task['ambient_temp'])
    
    except FileNotFoundError:
        print("Sampling new HVAC tasks...")
        task = HVACTaskSampler()
        with open(TASK_CONFIG_PATH, "wb") as f:
            pickle.dump(task, f)
        print(f"... Saved new task config to {TASK_CONFIG_PATH}")
        print("t_ambient", task['ambient_temp'])
    env.set_task(task)
    print("target_temperature: ", task['target_temperature'])
    # env.set_control_type("power")
    # env.set_return_normilized_obs(True)
    
    terminated, truncated = False,False
    obs = env.reset()[0]
    pid = HVACSolverGTPID(env)
    max_steps = 3000
    current_stage = []
    steps = 0
    n_coolers = len(env.coolers)
    n_sensors = len(env.sensors)
    values = []
    lstm_states = None
    cool_power_sum = 0.0
    cool_power_count = 0
    while steps < max_steps:
        # action = env.sample_action(mode="max")
        action = model.predict(obs)
        
        # pid
        # action = pid.policy(obs[:n_sensors])

        switch = action[:n_coolers]
        value = action[n_coolers:]
        for i in range(len(switch)):
            if switch[i]<0.5:
                value[i] = -1.0

        # print(value)
        values.append(value)
        
        obs, reward, terminated, truncated, info = env.step(action)
        # print("t: ",env.t)
        # print(obs)
        if 'cool_power' in info:
            cool_power_sum += numpy.sum(info['cool_power'])
            cool_power_count += 1
        # if terminated or truncated:
        #     break
        current_stage.append(reward)
        if steps < 1:
            print("info.keys(): ", info.keys())
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
                key: info_sums[key] / count if (count := info_counts.get(key, 0)) != 0 else 0
                for key in info_sums
            }
            
            # 格式化输出
            info_str = " | ".join([f"{k}:{v:.4f}" for k,v in info_means.items()])
            print(f"Step {steps} | Reward: {mean_reward:.2f} | {info_str}", flush=True)
            
            # 重置统计量
            current_stage = []
            info_sums = {k:0.0 for k in info_sums}
            info_counts = {k:0 for k in info_counts}
    
    if cool_power_count > 0:
        avg_cool_power = cool_power_sum / cool_power_count
        print(f"\n平均冷却功率: {avg_cool_power:.2f} kW")
    else:
        print("\n未找到冷却功率数据")
    
    # 保存数据到CSV文件
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 将values转换为二维数组 (时间步数 × cooler数量)
    values_array = numpy.array(values)

    # 创建DataFrame并保存为CSV
    df = pd.DataFrame(values_array, columns=[f"cooler_{i}" for i in range(n_coolers)])
    csv_path = os.path.join(output_dir, "cooler_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"保存数据到: {csv_path}")

    # 可视化每个cooler的值
    plt.figure(figsize=(12, 8))

    # 计算子图的行列数
    n_cols = min(4, n_coolers)  # 每行最多4个子图
    n_rows = (n_coolers + n_cols - 1) // n_cols

    for i in range(n_coolers):
        plt.subplot(n_rows, n_cols, i+1)
        
        # 提取当前cooler的所有时间步的值
        cooler_values = values_array[:, i]
        
        # 绘制折线图
        plt.plot(cooler_values, 'b-', linewidth=1.5)
        
        # 设置标题和坐标轴
        plt.title(f"Cooler {i}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.ylim(-1.1, 1.1)  # 固定y轴范围
        plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局并保存图像
    plt.tight_layout()
    plot_path = os.path.join(output_dir, "cooler_values_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"保存可视化结果到: {plot_path}")

    # 显示图像（可选）
    plt.show()