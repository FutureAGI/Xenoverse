if __name__=="__main__":
    import gymnasium as gym
    import numpy
    import argparse
    from xenoverse.linds import LinearDSSamplerRandomDim

    from stable_baselines3 import PPO, SAC
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.evaluation import evaluate_policy
    from xenoverse.linds.task_sampler import dump_linds_task,load_linds_task


    task = LinearDSSamplerRandomDim()

    env = gym.make("linear-dynamics-v0-visualizer") 
    env.set_task(task, verbose=True, reward_shaping=True)

    args = argparse.ArgumentParser()
    args.add_argument("--max_step", type=int, default=200000)
    args.add_argument("--lr", type=float, default=3e-4)
    args.add_argument("--run", choices=["mlp", "lstm", "both"], default="both")
    args.add_argument("--task", type=str, default=None)
    args = args.parse_args()
    if(args.task is not None):
        task = load_linds_task(args.task)
    else:
        task = LinearDSSamplerRandomDim()
        dump_linds_task("./task.pkl", task)

    max_step = args.max_step
    lr = args.lr

    model_mlp = PPO(
        "MlpPolicy",      # 使用 MLP 策略网络
        env,                  # 环境对象
        verbose=1,            # 打印训练日志
        learning_rate=lr,   # 学习率
        batch_size=64,        # 批量大小
        gamma=0.99,           # 折扣因子
        tensorboard_log="./ppo_tensorboard/"  # TensorBoard 日志目录
    )

    model_lstm = RecurrentPPO(
        "MlpLstmPolicy",      # 使用 MLP 策略网络
        env,                  # 环境对象
        verbose=1,            # 打印训练日志
        learning_rate=lr,   # 学习率
        n_steps=2048,         # 每个环境每次更新的步数
        batch_size=64,        # 批量大小
        n_epochs=10,          # 每次更新的迭代次数
        gamma=0.99,           # 折扣因子
        gae_lambda=0.95,      # GAE 参数
        policy_kwargs={
            "lstm_hidden_size": 32,    # LSTM 隐藏层大小
            "n_lstm_layers": 2,        # LSTM 层数
            "enable_critic_lstm": True # Critic 网络也使用 LSTM
        },
        clip_range=0.2,       # PPO 的 clip 范围
        tensorboard_log="./ppo_tensorboard/"  # TensorBoard 日志目录
    )


    if(args.run == "mlp" or args.run == "both"):

        print(f"Training MLP Policy for {max_step} steps")

        mean_reward, std_reward = evaluate_policy(model_mlp, env, n_eval_episodes=20)
        print(f"Before Training: Mean reward: {mean_reward}, Std reward: {std_reward}")

        model_mlp.learn(total_timesteps=max_step)

        mean_reward, std_reward = evaluate_policy(model_mlp, env, n_eval_episodes=20)
        print(f"After Training: Mean reward: {mean_reward}, Std reward: {std_reward}")

    if(args.run == "lstm" or args.run == "both"):

        print(f"Training LSTMLSTM Policy for {max_step} steps")

        mean_reward, std_reward = evaluate_policy(model_lstm, env, n_eval_episodes=20)
        print(f"Before Training: Mean reward: {mean_reward}, Std reward: {std_reward}")

        model_lstm.learn(total_timesteps=max_step)

        mean_reward, std_reward = evaluate_policy(model_lstm, env, n_eval_episodes=20)
        print(f"After Training: Mean reward: {mean_reward}, Std reward: {std_reward}")