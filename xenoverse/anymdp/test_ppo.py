import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from xenoverse.utils import dump_task, load_task
import gymnasium as gym
import xenoverse.anymdp
from xenoverse.anymdp import AnyMDPTaskSampler, GarnetTaskSampler, AnyPOMDPTaskSampler, MultiTokensAnyPOMDPTaskSampler
from xenoverse.anymdp.test_utils import train
import argparse



def create_env(task):
    """创建并包装环境"""
    env = gym.make("anymdp-v0")
    env.set_task(task)
    
    # 检查环境是否符合 Gym 规范（重要！）
    check_env(env, warn=True)
    
    # 使用 Monitor 包装以记录统计信息
    env = Monitor(env)
    return env

def train_multi_discrete_ppo(task):
    """训练 PPO 模型"""
    
    # 创建向量化环境（SB3 推荐）
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # 可以并行多个环境加速训练
    n_envs = 4
    env = DummyVecEnv([lambda:create_env(task) for _ in range(n_envs)])
    
    # 配置 PPO 超参数
    model = RecurrentPPO(
        policy="MlpLstmPolicy",           # 多层感知机策略
        env=env,
        learning_rate=3e-4,           # 学习率
        n_steps=2048,                 # 每个环境采集的步数
        batch_size=256,               # 小批量大小
        n_epochs=10,                  # 每次更新的训练轮数
        gamma=0.99,                   # 折扣因子
        gae_lambda=0.95,              # GAE参数
        clip_range=0.2,               # PPO裁剪范围
        verbose=1,                    # 打印训练信息
        tensorboard_log="./ppo_multi_discrete_tensorboard/",  # TensorBoard日志
        device="auto"                 # 自动选择设备（CPU/GPU）
    )
    
    # 开始训练
    total_timesteps = 500_000  # 总训练步数
    print(f"\n开始训练，总步数: {total_timesteps}")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # 保存模型
    model.save("ppo_multi_discrete_model")
    print("模型已保存: ppo_multi_discrete_model.zip")
    
    env.close()
    return model


def evaluate_model(model_path=None, n_eval_episodes=10):
    """评估训练好的模型"""
    
    # 加载模型或直接使用传入的模型
    if model_path:
        model = PPO.load(model_path)
    else:
        model = train_multi_discrete_ppo()
    
    # 创建评估环境
    eval_env = create_env()
    
    # 使用 SB3 内置评估函数
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,  # 使用确定性策略评估
        render=False
    )
    
    print("\n" + "="*50)
    print(f"评估结果（{n_eval_episodes} 回合）:")
    print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print("="*50)
    
    # 可视化一个回合
    print("\n可视化一个评估回合...")
    obs, _ = eval_env.reset()
    total_reward = 0
    for step in range(100):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        total_reward += reward
        print(f"Step {step}: Action {action}, Reward {reward:.3f}, Obs {obs[:2]}")
        if terminated or truncated:
            break
    
    print(f"回合总奖励: {total_reward:.2f}")
    eval_env.close()
    
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="task file")
    args = parser.parse_args()
    if(args.task != None):
        task = load_task(args.task)
    else:
        """
        # Test MDP Task Sampler
        task = AnyMDPTaskSampler(state_space=16, 
                                action_space=5,
                                min_state_space=None,
                                verbose=True)
        # Test Garnet Task Sampler
        task = GarnetTaskSampler(state_space=16, 
                                action_space=5,
                                min_state_space=None,
                                verbose=True)
        # Test POMDP Task Sampler
        task = AnyPOMDPTaskSampler(state_space=16, 
                                action_space=5,
                                min_state_space=None,
                                observation_space=16,
                                density = 0.1,
                                verbose=True)

        task = MultiTokensAnyPOMDPTaskSampler(state_space=128, 
                                action_space=5,
                                min_state_space=None,
                                observation_space=32,
                                observation_tokens=4,
                                action_tokens=1,
                                density = 0.1,
                                verbose=True)
        """

        task = MultiTokensAnyPOMDPTaskSampler(state_space=128, 
                                action_space=5,
                                min_state_space=None,
                                observation_space=32,
                                observation_tokens=4,
                                action_tokens=1,
                                density = 0.1,
                                verbose=True)

    
    # 选项1: 仅训练
    model = train_multi_discrete_ppo(task)
    
    # 选项2: 训练并评估
    model = evaluate_model(n_eval_episodes=5)
    
    # 选项3: 加载已有模型并评估
    # model = evaluate_model(model_path="ppo_multi_discrete_model.zip", n_eval_episodes=5)
    
    print("\n训练与评估完成！")