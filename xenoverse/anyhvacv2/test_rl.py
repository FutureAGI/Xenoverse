if __name__ == "__main__":

    from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvacv2.anyhvac_env import HVACEnv
    from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
    from rl_trainer import HVACRLTrainer
    import gymnasium as gym

    import pickle 
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

    visual = False
    if visual:
        def make_env():
            env = HVACEnvVisible()
            env.set_task(task)
            return env
    else:
        def make_env():
            env = HVACEnv()
            env.set_task(task)
            return env

    trainer = HVACRLTrainer(
        env_maker=make_env,
        n_envs=12,  # 并行环境数
        vec_env_type="subproc",  # 向量环境类型
        algorithm="ppo",  # 可选 "ppo" 或 "sac"
        stage_steps=100,  # 每5000步统计一次平均奖励
        vec_env_args={
            "start_method": "spawn"
        },
        verbose = 0,
        device="cpu"  # 使用GPU加速
    )

    # 训练模型
    trainer.train(total_steps=1e6)
    
    # 保存模型
    trainer.save_model("./models/hvac_sac_model.zip")
