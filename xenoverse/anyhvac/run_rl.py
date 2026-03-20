

if __name__ == "__main__":

    from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
    from xenoverse.anyhvac.anyhvac_env import HVACEnv, HVACEnvDiscreteAction, HVACEnvDiffAction
    from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
    from rl_trainer import HVACRLTrainer
    import gymnasium as gym
    from numpy import random as rnd

    import pickle 
    TASK_CONFIG_PATH = "./models/0309/hvac_task_1057/hvac_task_1057.pkl"
    load_path = "./models/0309/hvac_task_1057/rppo_reward_mode_1_stage2_generateF.zip"
    save_path = "./models/0309/hvac_task_1057/rppo_reward_mode_1_stage2_generateF.zip"
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
            env = HVACEnvDiffAction(reward_mode=1)
            # env = HVACEnvDiscreteAction(reward_mode=1)
            env.set_task(task,discretize_rl_action_space=True, add_action_cost=False, too_cold_limit=False)
            # env.set_return_normilized_obs(True)
            env.set_random_start_t(True)
            env.set_generate_record(False)
            overheat_no_reset = rnd.uniform(0.0, 1.0) > 0.5
            env.set_overheat_no_termiated_training_only(overheat_no_reset)
            # env.set_control_type("power")2
            return env

    trainer = HVACRLTrainer(
        env_maker=make_env,
        n_envs=16,  # 并行环境数
        vec_env_type="subproc",  # 向量环境类型
        algorithm="rppo",  # 可选 "ppo" 或 "sac"
        stage_steps=100,  # 每5000步统计一次平均奖励
        vec_env_args={
            "start_method": "spawn"
        },
        verbose = 0,
        device="cpu"  # 使用GPU加速
    )

    # 训练模型
    trainer.load_model(load_path)
    trainer.train(total_steps=500000)
    
    # 保存模型

    trainer.save_model(save_path)
    print(f"model saved in {save_path}")


