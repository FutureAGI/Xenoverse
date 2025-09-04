if __name__ == "__main__":
    import numpy
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
    from xenoverse.anyhvac.anyhvac_env import HVACEnvDiscreteAction, HVACEnvDiffAction
    import pickle 
    from rl_trainer import HVACRLTester

    env = HVACEnvDiffAction(verbose=True)
    TASK_CONFIG_PATH = "./task_file/hvac_task_config_0904_diff_action.pkl"
    RL_MODEL_PATH = "./models/0828/temp/1/sac_random_start_no_normal/sac2_stage1.zip"
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
    max_steps = 500
    current_stage = []
    steps = 0
    n_coolers = len(env.coolers)
    n_sensors = len(env.sensors)
    values = []
    lstm_states = None
    cool_power_sum = 0.0
    cool_power_count = 0
    while steps < max_steps:
        # Max coolers power
        # action = env.sample_action(mode="max")

        # RL
        # action = model.predict(obs)
        
        # pid
        action = pid.policy(obs[:n_sensors])
        
        obs, reward, terminated, truncated, info = env.step(action)
        env_action = env.last_action
        switch = env_action["switch"]
        value = env_action["value"]
        for i in range(len(switch)):
            if switch[i]<0.5:
                value[i] = -1.0

        # print(value)
        values.append(value)

        # print("t: ",env.t)
        # print(obs)
        if 'cool_power' in info:
            cool_power_sum += numpy.sum(info['cool_power'])
            cool_power_count += 1
        if terminated or truncated:
            break
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
            
            info_means = {
                key: info_sums[key] / count if (count := info_counts.get(key, 0)) != 0 else 0
                for key in info_sums
            }
            
            info_str = " | ".join([f"{k}:{v:.4f}" for k,v in info_means.items()])
            print(f"Step {steps} | Reward: {mean_reward:.2f} | {info_str}", flush=True)
            
            current_stage = []
            info_sums = {k:0.0 for k in info_sums}
            info_counts = {k:0 for k in info_counts}
    
    if cool_power_count > 0:
        avg_cool_power = cool_power_sum / cool_power_count
        print(f"\navg coolers power: {avg_cool_power:.2f} kW")
    else:
        print("\nno data")
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    values_array = numpy.array(values)

    df = pd.DataFrame(values_array, columns=[f"cooler_{i}" for i in range(n_coolers)])
    csv_path = os.path.join(output_dir, "cooler_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"save data to: {csv_path}")

    min_width_per_plot = 4
    min_height_per_plot = 2
    max_width = 24
    max_height = 36 

    n_cols = min(4, n_coolers)
    n_rows = (n_coolers + n_cols - 1) // n_cols

    fig_width = min(n_cols * min_width_per_plot, max_width)
    fig_height = min(n_rows * min_height_per_plot, max_height)

    plt.figure(figsize=(fig_width, fig_height))

    plt.subplots_adjust(
        left=0.05, 
        right=0.95, 
        bottom=0.05, 
        top=0.95,
        wspace=0.3,  
        hspace=0.4   
    )

    for i in range(n_coolers):
        ax = plt.subplot(n_rows, n_cols, i+1)
        
        cooler_values = values_array[:, i]
        
        ax.plot(cooler_values, 'b-', linewidth=1.5)
        
        ax.set_title(f"Cooler {i}", fontsize=10)
        ax.set_xlabel("Time Step", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if n_coolers > 12:
            ax.tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout(pad=2.0)  
    plot_path = os.path.join(output_dir, "cooler_values_plot.png")
    plt.savefig(plot_path, dpi=300)
    print(f"figure save to: {plot_path}")

    plt.show()