# rl_trainer.py
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import Union, Type

class HVACRLTrainer:
    def __init__(
        self,
        env_maker,  # env maker
        algorithm: str = "ppo",  # 算法类型
        policy_type: str = "MlpLstmPolicy",  # 策略网络类型
        stage_steps: int = 10000,  # 每阶段统计步数
        n_envs: int = 4,                   # 并行环境数
        vec_env_type: str = "subproc",     # 向量环境类型
        vec_env_args: dict = None,         # 向量环境参数
        verbose: int = 1,
        device: str = "auto"
    ):
        # 环境包装
        self.n_envs = n_envs
        self.env = self._make_vec_env(
            env_maker=env_maker,
            vec_type=vec_env_type.lower(),
            vec_args=vec_env_args or {}
        )
        self.algorithm = algorithm.lower()
        self.stage_steps = stage_steps
        self.stats = {"stage_rewards": []}
        
        # 初始化模型
        policy_map = {
            "ppo": PPO,
            "sac": SAC
        }
        if self.algorithm not in policy_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.model_class = policy_map[self.algorithm]
        self.model = self._init_model(policy_type, verbose, device)
        
        # 训练回调
        self.logger_callback = TrainingLoggerCallback(
            check_freq=stage_steps,
            verbose=verbose
        )

    def _make_vec_env(self, env_maker, vec_type: str, vec_args: dict):
        """ 创建兼容类型检查的向量环境 """
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        from stable_baselines3.common.monitor import Monitor
        
        def _make_env(rank):
            def _init():
                env = env_maker()  
                if hasattr(env, "seed"):
                    env.seed(42 + rank)
                return Monitor(env)
            return _init

        vec_env_classes = {
            "dummy": DummyVecEnv,
            "subproc": SubprocVecEnv
        }
        if vec_type not in vec_env_classes:
            raise ValueError(f"Unsupported vec env type: {vec_type}")
        

        return vec_env_classes[vec_type](
            [ _make_env(i) for i in range(self.n_envs) ],
            **vec_args
        )
    
    def _init_model(self, policy_type: str, verbose: int, device: str):
        """ 初始化强化学习模型 """
        common_params = {
            "policy": policy_type,
            "env": self.env,
            "verbose": verbose,
            "device": device
        }
        
        if self.algorithm == "ppo":
            return RecurrentPPO(
                batch_size=64 * self.n_envs,
                n_steps=1024 // self.n_envs,
                **common_params
            )
        elif self.algorithm == "sac":
            return SAC(
                batch_size=256,
                learning_starts=10000 // self.n_envs,
                **common_params
            )

    def train(self, total_steps: int = 100000):
        """ 执行训练流程 """
        self.model.learn(
            total_timesteps=total_steps,
            callback=self.logger_callback,
            reset_num_timesteps=False
        )
        self._update_stats()

    def evaluate(self, n_episodes: int = 10):
        """ 策略评估 """
        total_rewards = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

    def save_model(self, path: str):
        """ 保存模型和配置 """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """ 加载已有模型 """
        self.model = self.model_class.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def _update_stats(self):
        """ 更新训练统计数据 """
        if self.logger_callback.stage_rewards:
            self.stats["stage_rewards"].extend(
                self.logger_callback.stage_rewards
            )

class TrainingLoggerCallback(BaseCallback):
    """ 自定义训练日志回调 """
    def __init__(self, check_freq: int, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.stage_rewards = []
        self.current_stage = []
        self.current_info = []

    def _on_step(self) -> bool:
        """ 记录每个步骤的奖励和信息统计 """
        # 原始数据记录
        self.current_stage.append(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        self.current_info.append(info)  # 保存完整的info字典
        
        # 动态初始化统计字段
        if not hasattr(self, 'info_sums'):
            self.info_sums = {key: 0.0 for key in info.keys()}
            self.info_counts = {key: 0 for key in info.keys()}
        
        # 累积统计量
        for key, value in info.items():
            if isinstance(value, (int, float)):
                self.info_sums[key] += value
                self.info_counts[key] += 1
        
        if self.n_calls % self.check_freq == 0:
            # 计算主奖励均值
            mean_reward = np.mean(self.current_stage)
            
            # 计算各信息字段均值
            info_means = {
                key: self.info_sums[key] / self.info_counts[key] 
                for key in self.info_sums
            }
            
            # 格式化输出
            info_str = " | ".join([f"{k}:{v:.4f}" for k,v in info_means.items()])
            print(f"Step {self.model.num_timesteps} | Reward: {mean_reward:.2f} | {info_str}", flush=True)
            
            # 重置统计量
            self.current_stage = []
            self.info_sums = {k:0.0 for k in self.info_sums}
            self.info_counts = {k:0 for k in self.info_counts}
            
        return True
