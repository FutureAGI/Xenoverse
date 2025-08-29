# rl_trainer.py
import os
import numpy as np
import gymnasium as gym
import numbers
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
        policy_type: str = "MlpLstmPolicy",  # 策略网络类型 PPO: MlpPolicy 
        stage_steps: int = 10000,  # 每阶段统计步数
        n_envs: int = 4,                   # 并行环境数
        vec_env_type: str = "dummy",     # 向量环境类型  dummy  subproc
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
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        policy_type_map = {
            "ppo": "MlpPolicy",
            "rppo": "MlpLstmPolicy",
            "sac": "MlpPolicy"
        }
        
        if self.algorithm not in policy_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.model_class = policy_map[self.algorithm]
        self.model = self._init_model(policy_type_map[self.algorithm], verbose, device)
        
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
        
        if self.algorithm == "rppo":
            print("Use RecurrentPPO!")
            return RecurrentPPO(
                batch_size=64 * self.n_envs,
                n_steps=1024 // self.n_envs,
                **common_params
            )
        elif self.algorithm == "ppo":
            print("Use PPO!")
            return PPO(
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
        self.current_overheat = []
        self.current_over_tolerace = []

    def _on_step(self) -> bool:
        """ 记录每个步骤的奖励和信息统计 """

        # 原始数据记录
        self.current_stage.append(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        self.current_info.append(info)  # 保存完整的info字典
        self.current_overheat.append(info["over_heat"])
        self.current_over_tolerace.append(info["over_tolerace"])

        
        # 动态初始化统计字段
        if not hasattr(self, 'info_sums'):
            self.info_sums = {key: 0.0 for key in info.keys()}
            self.info_counts = {key: 0 for key in info.keys()}
            
        # 累积统计量
        for key, value in info.items():
            if isinstance(value, numbers.Real):
                self.info_sums[key] += value
                self.info_counts[key] += 1
        
        if self.n_calls % self.check_freq == 0:
            # 计算主奖励均值
            mean_reward = np.mean(self.current_stage)
            cool_power = round(np.sum(info.get("cool_power", 0)),4)
            heat_power = round(np.sum(info.get("heat_power", 0)),4)
            over_heat = np.mean(self.current_overheat)
            over_tolerace = np.mean(self.current_over_tolerace)
            self.current_overheat.clear()
            self.current_over_tolerace.clear()


            # print("self.current_stage",self.current_stage )
            info_total = f"energy_cost: {round(info.get('energy_cost', 0),4)}, target_cost: {round(info.get('target_cost', 0),4)}, switch_cost: {round(info.get('switch_cost', 0),4)},cool_power: {cool_power}, heat_power: {heat_power}"
            
            # 计算各信息字段均值
            # info_means = {
            #     key: self.info_sums[key] / self.info_counts[key] 
            #     for key in self.info_sums
            # }

            # 格式化输出
            print(f"Step {self.model.num_timesteps} | over_heat:{over_heat} | over_tolerace:{over_tolerace} | Reward: {mean_reward:.2f} | {info_total}", flush=True)
            
            # 重置统计量
            self.current_stage = []
            self.info_sums = {k:0.0 for k in self.info_sums}
            self.info_counts = {k:0 for k in self.info_counts}
            
        return True

class HVACRLTester:
    def __init__(
        self,
        model_path: str,
        algorithm: str = "ppo",
        device: str = "auto"
    ):
        """
        参数:
            model_path: 模型文件路径
            algorithm: 算法类型 (ppo, rppo, sac)
            device: 运行设备 (cpu, cuda, auto)
        """
        self.algorithm = algorithm.lower()
        self.model_path = model_path
        self.device = device
        
        # 直接加载模型，不需要环境
        self.model = self._load_model()
    
    def _load_model(self):
        """加载训练好的模型"""
        # 算法到模型类的映射
        model_classes = {
            "ppo": PPO,
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        
        if self.algorithm not in model_classes:
            raise ValueError(f"不支持的算法类型: {self.algorithm}")
        
        model_class = model_classes[self.algorithm]
        
        # 加载模型，不传入环境
        model = model_class.load(
            self.model_path,
            device=self.device
        )
        
        print(f"成功加载 {self.algorithm.upper()} 模型: {self.model_path}")
        return model
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        根据观测值预测动作
        
        参数:
            obs: 观测值数组 (一维)
            deterministic: 是否使用确定性策略
        
        返回:
            动作数组 (一维)
        """
        # 确保输入是正确形状的数组
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs, dtype=np.float32)
        
        # 添加批次维度 (1, *obs_shape)
        obs = obs[np.newaxis, :]
        
        # 模型预测
        action, _ = self.model.predict(obs, deterministic=deterministic)
        
        # 返回动作 (移除批次维度)
        return action[0]

    


class MySB3CompatibleEnv(gym.Env):
    def __init__(self, config=None):
        super().__init__()
        # 定义观测空间和动作空间 (根据你的实际环境修改)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self._state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._episode_length = 0
        self._max_episode_steps = 200 # 示例：最大步数

    def step(self, action):
        # 模拟环境 step 逻辑 (根据你的实际环境修改)
        # 简单地更新状态并计算奖励
        self._state += action.flatten() * 0.1
        reward = -np.sum(np.square(self._state)) # 示例：简单的惩罚
        self._episode_length += 1
        terminated = self._episode_length >= self._max_episode_steps
        truncated = False # 示例：不考虑 truncated
        info = {}

        return self._state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 模拟环境 reset 逻辑 (根据你的实际环境修改)
        self._state = self.observation_space.sample() * 0.1 # 示例：随机初始状态
        self._episode_length = 0
        info = {}
        return self._state, info

    # 如果需要渲染，实现 render 方法
    # def render(self):
    #     pass

    # 如果需要关闭环境，实现 close 方法
    # def close(self):
    #     pass

