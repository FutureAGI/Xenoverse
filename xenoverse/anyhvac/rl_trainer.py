# rl_trainer.py
import os
import torch
import numpy as np
import gymnasium as gym
import numbers
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from typing import Union, Type, Optional

class HVACRLTrainer:
    def __init__(
        self,
        env_maker,  
        algorithm: str = "ppo", 
        stage_steps: int = 10000,
        n_envs: int = 4,
        vec_env_type: str = "subproc",
        vec_env_args: dict = None,
        verbose: int = 1,
        device: str = "auto",
        log_path: str = None
    ):
        # env wrapper
        self.n_envs = n_envs
        self.env = self._make_vec_env(
            env_maker=env_maker,  
            vec_type=vec_env_type.lower(),
            vec_args=vec_env_args or {}
        )
        self.algorithm = algorithm.lower()
        self.stage_steps = stage_steps
        self.stats = {"stage_rewards": []}
        
        # init model
        policy_map = {
            "ppo": PPO,
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        policy_type_map = {
            "ppo": "MultiInputPolicy",
            "rppo": "MultiInputLstmPolicy",
            "sac": "MultiInputPolicy"
        }
        
        if self.algorithm not in policy_map:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        self.model_class = policy_map[self.algorithm]
        self.model = self._init_model(policy_type_map[self.algorithm], verbose, device)
        
        # training callback
        self.logger_callback = TrainingLoggerCallback(
            check_freq=stage_steps,
            verbose=verbose,
            log_path=log_path
        )

    def _make_vec_env(self, env_maker, vec_type: str, vec_args: dict):

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
        """ init RL model """
        common_params = {
            "policy": policy_type,
            "env": self.env,
            "verbose": verbose,
            "device": device
        }
        
        if self.algorithm == "rppo":
            n_steps = 128
            batch_size = int(n_steps * self.n_envs / 4)

            print("Use RecurrentPPO!")

            return RecurrentPPO(
                batch_size= batch_size,
                n_steps= n_steps,
                gamma=0.95,
                gae_lambda=0.90,
                # 降低学习率：原默认一般3e-4，减半到1.5e-4（若你原lr是其他值，按1/2~1/3调整）
                learning_rate=1.5e-4,
                clip_range=0.15,
                policy_kwargs = {
                    "net_arch": {
                        "pi": [512, 256],      # 两层 MLP
                        "vf": [512, 256],
                    },
                    "features_extractor_class": NormalizedCombinedExtractor,
                    "lstm_hidden_size": 512,      
                    "n_lstm_layers": 1,         
                    "shared_lstm": False,         # 独立 LSTM
                    "enable_critic_lstm": True,   # Critic 也有 LSTM
                    "lstm_kwargs": {
                        "dropout": 0.0,  # 可选：防止过拟合
                    },
                    "activation_fn": torch.nn.ReLU,
                },
                **common_params
            )
        
        elif self.algorithm == "ppo":
            print("Use PPO!")
            return PPO(
                batch_size= int(32 * self.n_envs / 4),
                n_steps= 32,
                **common_params
            )
        elif self.algorithm == "sac":
            return SAC(
                policy_kwargs={
                    "net_arch": {
                        "pi": [512, 256, 128],  # Actor（策略网络）的隐藏层结构
                        "qf": [512, 512, 256]   # Critic（价值网络）的隐藏层结构
                    },
                    "activation_fn": torch.nn.ReLU  # 高维数据常用ReLU增强非线性表达
                },
                buffer_size = 10000000, # 1e7
                batch_size = 1024,  # 增大批量（利用多核CPU并行计算，同时稳定梯度）
                learning_starts = 10000, # 先采集1万步初始经验（高维环境需要更多初始探索）
                train_freq=(1, "step"),  # 每步都训练（长周期环境需及时更新策略）
                gradient_steps=4,    # 每次训练更新4步（充分利用采样的batch数据）
                gamma=0.995, # 稍大于默认的0.99（长周期环境更重视远期奖励）
                use_sde=True,    # 开启SDE（高维动作空间需要更强的随机探索）
                sde_sample_freq=16,  # 每16步重新采样噪声（平衡探索随机性和稳定性）
                ent_coef="auto",
                **common_params
            )

    def train(self, total_steps: int = 100000):
        self.model.learn(
            total_timesteps=total_steps,
            callback=self.logger_callback,
            reset_num_timesteps=False
        )
        self._update_stats()

    def evaluate(self, n_episodes: int = 10):
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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model = self.model_class.load(path, env=self.env)
        print(f"Model loaded from {path}")

    def _update_stats(self):
        if self.logger_callback.stage_rewards:
            self.stats["stage_rewards"].extend(
                self.logger_callback.stage_rewards
            )

class TrainingLoggerCallback(BaseCallback):
    def __init__(self, check_freq: int, verbose=1, log_path=None):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.stage_rewards = []
        self.current_stage = []
        self.current_info = []
        self.current_overheat = []
        self.current_over_tolerace = []
        self.log_path = log_path

    def _on_step(self) -> bool:
        current_obs = self.model.env.get_attr("current_obs")[0]
        current_action = self.model.env.get_attr("current_action")[0]
        self.current_stage.append(self.locals["rewards"][0])
        info = self.locals["infos"][0]
        self.current_overheat.append(info["over_heat"])
        self.current_over_tolerace.append(info["over_tolerace"])

        if not hasattr(self, 'info_sums'):
            self.info_sums = {key: 0.0 for key in info.keys()}
            self.info_counts = {key: 0 for key in info.keys()}

        for key, value in info.items():
            if isinstance(value, numbers.Real):
                self.info_sums[key] += value
                self.info_counts[key] += 1
        
        if self.n_calls % self.check_freq == 0:
            mean_reward = np.mean(self.current_stage)
            cool_power = round(np.sum(info.get("cool_power", 0)),4)
            heat_power = round(np.sum(info.get("heat_power", 0)),4)
            over_heat = np.mean(self.current_overheat)
            over_tolerace = np.mean(self.current_over_tolerace)
            self.current_overheat.clear()
            self.current_over_tolerace.clear()

            info_total = f"energy_cost: {round(info.get('energy_cost', 0), 4)}, " \
                         f"target_cost: {round(info.get('target_cost', 0), 4)}, " \
                         f"switch_cost: {round(info.get('switch_cost', 0), 4)}, " \
                         f"action_cost: {round(info.get('action_cost', 0), 4)}, " \
                         f"cool_power: {cool_power}, heat_power: {heat_power}"

            if self.log_path is None:
                print(f"Step {self.model.num_timesteps} | over_heat:{over_heat} | over_tolerace:{over_tolerace} | Reward: {mean_reward:.2f} | {info_total}", flush=True)
                print("current_obs: ", current_obs)
            else:
                log_line1 = f"Step {self.model.num_timesteps} | over_heat:{over_heat} | over_tolerace:{over_tolerace} | Reward: {mean_reward:.2f} | {info_total}"
                log_line2 = f"current_obs: {current_obs}"
                with open(self.log_path, 'a') as f:
                    f.write(log_line1 + '\n')
                    f.write(log_line2 + '\n')
                    f.flush()


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
        self.algorithm = algorithm.lower()
        self.model_path = model_path
        self.device = device
        
        self.model = self._load_model()
    
    def _load_model(self):

        model_classes = {
            "ppo": PPO,
            "rppo": RecurrentPPO,
            "sac": SAC
        }
        
        if self.algorithm not in model_classes:
            raise ValueError(f"Unsupported model: {self.algorithm}")
        
        model_class = model_classes[self.algorithm]
        
        model = model_class.load(
            self.model_path,
            device=self.device
        )
        
        print(f"Load {self.algorithm.upper()} model: {self.model_path}")
        return model
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Predict actions based on observed values
        
        Param:
            obs: Observation value array (one-dimensional)
            deterministic: Whether to use a deterministic strategy
        
        Return:
            Action array (one-dimensional)
        """
        # if not isinstance(obs, np.ndarray):
        #     obs = np.array(obs, dtype=np.float32)
        
        # obs = obs[np.newaxis, :]
        
        if self.algorithm == "rppo":
            if self._episode_start:
                episode_start = np.array([True])
                self._episode_start = False
            else:
                episode_start = np.array([False])
            action, self._last_lstm_states = self.model.predict(obs, 
                                                                state=self._last_lstm_states, 
                                                                episode_start=episode_start, 
                                                                deterministic=deterministic)
        else:
            action, _ = self.model.predict(obs, deterministic=deterministic)
        
        return action
    
    def predict_with_distribution(
        self, 
        obs: Union[np.ndarray, dict[str, np.ndarray]], 
        state: Optional[tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> tuple[np.ndarray, dict, tuple]:
        """
        预测动作并返回动作分布信息（支持循环策略）
        """

        # 1. 设置模型为评估模式
        self.model.policy.set_training_mode(False)
        
        # 2. 转换 observation 为 tensor
        observation, vectorized_env = self.model.policy.obs_to_tensor(obs)
        
        # 3. 确定 batch size
        if isinstance(observation, dict):
            n_envs = observation[next(iter(observation.keys()))].shape[0]
        else:
            n_envs = observation.shape[0]
        
        # 4. 准备 LSTM 状态
        if state is None:
            state = np.concatenate([np.zeros(self.model.policy.lstm_hidden_state_shape) for _ in range(n_envs)], axis=1)
            state = (state, state)
        
        if episode_start is None:
            episode_start = np.array([False for _ in range(n_envs)])
        
        # 5. 转换为 torch tensor
        with torch.no_grad():
            states = (
                torch.tensor(state[0], dtype=torch.float32, device=self.model.device),
                torch.tensor(state[1], dtype=torch.float32, device=self.model.device)
            )
            episode_starts = torch.tensor(episode_start, dtype=torch.float32, device=self.model.device)
            
            # 6. 获取分布
            distribution, new_states = self.model.policy.get_distribution(
                observation, 
                lstm_states=states, 
                episode_starts=episode_starts
            )
            
            # 7. 获取动作
            actions_raw = distribution.get_actions(deterministic=deterministic)
            
            # 8. 提取分布信息
            distribution_info = self._extract_distribution_info(distribution)
            
            # 9. 转换为 numpy
            actions = actions_raw.cpu().numpy()
            new_states = (new_states[0].cpu().numpy(), new_states[1].cpu().numpy())
            
            # 10. 根据动作空间类型进行后处理
            if isinstance(self.model.action_space, gym.spaces.Box):
                # 处理连续动作空间
                if self.model.policy.squash_output:
                    actions = self.model.policy.unscale_action(actions)
                else:
                    actions = np.clip(actions, self.model.action_space.low, self.model.action_space.high)
                
            elif isinstance(self.model.action_space, gym.spaces.MultiDiscrete):
                # 处理 MultiDiscrete 动作空间
                # 确保动作是整数类型
                if actions.dtype != np.int32 and actions.dtype != np.int64:
                    actions = actions.astype(np.int32)
                
                # 验证动作在合法范围内（可选，用于调试）
                nvec = self.model.action_space.nvec
                if np.any(actions < 0) or np.any(actions >= nvec):
                    print(f"[WARNING] 动作超出范围!")
                    print(f"  - 合法范围: [0, {nvec})")
                    print(f"  - 实际 min: {np.min(actions)}, max: {np.max(actions)}")
                    # 裁剪到合法范围
                    actions = np.clip(actions, 0, nvec - 1)

                    
            elif isinstance(self.model.action_space, gym.spaces.Discrete):
                # 处理 Discrete 动作空间
                # 确保是整数
                if actions.ndim > 0:
                    actions = actions.flatten()
                actions = actions.astype(np.int32)
                
            else:
                print(f"[DEBUG] 不支持的 action_space 类型: {type(self.model.action_space)}，跳过处理")
            
            # 11. 处理非向量化环境
            if not vectorized_env:
                actions = actions.squeeze(axis=0)
                new_states = (new_states[0].squeeze(axis=1), new_states[1].squeeze(axis=1))
            
            return actions, distribution_info, new_states
        
    def _extract_distribution_info(self, distribution):
        """
        从分布对象中提取信息（支持连续、离散和 MultiDiscrete 动作空间）
        
        返回:
            dict: 包含分布参数和统计信息
        """
         
        info = {}
        
        # 检查是否有 distribution 属性
        if hasattr(distribution, 'distribution'):
            inner_distribution = distribution.distribution

            # 检查是否是列表/元组（MultiCategorical 的情况）
            if isinstance(inner_distribution, (list, tuple)):
                if len(inner_distribution) > 0:
                    print(f"    - 第一个分布类型: {type(inner_distribution[0])}")
                
                info['type'] = 'multi_categorical'
                info['probs'] = []
                info['logits'] = []
                info['entropy'] = []
                
                # 遍历每个维度的分布
                for i, dist in enumerate(inner_distribution):
                
                    if hasattr(dist, 'probs') and hasattr(dist, 'logits'):
                        probs = dist.probs.cpu().numpy()
                        logits = dist.logits.cpu().numpy()
                        entropy = dist.entropy().cpu().numpy()
                        
                        info['probs'].append(probs)
                        info['logits'].append(logits)
                        info['entropy'].append(entropy)
                    else:
                        print(f"      [WARNING] 第 {i} 个分布没有 probs 或 logits 属性")
                
                # 转换为数组
                info['probs'] = np.array(info['probs'], dtype=object)
                info['logits'] = np.array(info['logits'], dtype=object)
                info['entropy'] = np.array(info['entropy'])
                info['total_entropy'] = np.sum(info['entropy'])

                
            else:
                # 单个分布
                # 对于连续动作（高斯分布）
                if hasattr(inner_distribution, 'loc') and hasattr(inner_distribution, 'scale'):
                    print(f"  [INFO] 检测到 Gaussian 分布")
                    info['type'] = 'gaussian'
                    info['mean'] = inner_distribution.loc.cpu().numpy()
                    info['std'] = inner_distribution.scale.cpu().numpy()
                    info['log_std'] = torch.log(inner_distribution.scale).cpu().numpy()
                    
                    # 计算熵
                    info['entropy'] = distribution.entropy().cpu().numpy()
                    
                # 对于离散动作（Categorical分布）
                elif hasattr(inner_distribution, 'probs'):
                    info['type'] = 'categorical'
                    info['probs'] = inner_distribution.probs.cpu().numpy()
                    info['logits'] = inner_distribution.logits.cpu().numpy()
                    
                    # 计算熵
                    info['entropy'] = distribution.entropy().cpu().numpy()
        
        else:
            print(f"  [WARNING] distribution 对象没有 'distribution' 属性")
            print(f"    - 可用属性: {[attr for attr in dir(distribution) if not attr.startswith('_')]}")
        
        return info

    def reset(self):
        if self.algorithm == "rppo":
            self._episode_start = True
            self._last_lstm_states = None

    
class NormalizedCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, **kwargs):
        super().__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            dim = get_flattened_obs_dim(subspace)
            
            # 所有向量观测都加 LayerNorm
            extractors[key] = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.LayerNorm(dim)  # ✅ 解决尺度失衡
            )
            total_concat_size += dim
        
        self.extractors = torch.nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
    
    def forward(self, obs):
        encoded = []
        for key, extractor in self.extractors.items():
            encoded.append(extractor(obs[key]))
        return torch.cat(encoded, dim=1)

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

