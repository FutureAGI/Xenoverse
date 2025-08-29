

if __name__ == "__main__":

    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvac.anyhvac_env import HVACEnv
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from rl_trainer import HVACRLTrainer
    import gymnasium as gym

    import pickle 
    TASK_CONFIG_PATH = "./task_file/hvac_task_config_0828_1.pkl"
    load_path = "./models/0828/temp/1/sac_random_start_no_normal/sac_stage2.zip"
    save_path = "./models/0828/temp/1/sac_random_start_no_normal/sac2_stage1.zip"
    visual = False


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

    # task = HVACTaskSampler(control_type='Temperature')  # power
    if visual:
        def make_env():
            env = HVACEnvVisible()
            env.set_task(task)
            return env
    else:
        def make_env():
            env = HVACEnv()
            env.set_task(task)
            # env.set_return_normilized_obs(True)
            env.set_random_start_t(True)
            # env.set_control_type("power")
            return env

    trainer = HVACRLTrainer(
        env_maker=make_env,
        n_envs=72,  # 并行环境数
        vec_env_type="subproc",  # 向量环境类型
        algorithm="sac",  # 可选 "ppo" 或 "sac"
        stage_steps=100,  # 每5000步统计一次平均奖励
        vec_env_args={
            "start_method": "spawn"
        },
        verbose = 0,
        device="cuda"  # 使用GPU加速
    )

    # 训练模型
    # trainer.load_model(load_path)
    trainer.train(total_steps=1000000)
    
    # 保存模型

    trainer.save_model(save_path)
    print(f"model saved in {save_path}")


