import argparse
import os
import time
import copy
import numpy as np
import random
import torch
import multiprocessing
import pickle
import gc
from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from sb3_contrib import RecurrentPPO
from typing import Dict, Tuple, Optional, Any, List

from xenoverse.anyhvacv2.anyhvac_env import HVACEnv
from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible
from xenoverse.anyhvacv2.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvacv2.anyhvac_solver import HVACSolverGTPID


class MultiAgentDataGenerator:
    """
    Modified HVAC multi-agent data generator with corrected dimensions and policies
    """
    def __init__(self, 
                 task_config_path=None,
                 visual=False, 
                 use_time_in_observation=False,
                 use_heat_in_observation=True,
                 policies_to_use=None,
                 model_paths=None,
                 seed=None,
                 multiprocess_mode=False):
        
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        self.visual = visual
        self.use_time_in_observation = use_time_in_observation
        self.use_heat_in_observation = use_heat_in_observation
        self.multiprocess_mode = multiprocess_mode
        
        # Save config parameters
        self.task_config_path = task_config_path
        # 初始化LSTM状态管理变量
        self.behavior_lstm_states = None
        self.reference_lstm_states = None
        self.episode_start = True  # 新增：episode开始标志
        
        # Load or create task config
        if task_config_path and os.path.exists(task_config_path):
            with open(task_config_path, 'rb') as f:
                self.task = pickle.load(f)
            print(f"Loaded task config from {task_config_path}")
        else:
            print("Creating new HVAC task...")

            self.task = HVACTaskSampler(control_type='Temperature')
            if task_config_path:
                with open(task_config_path, 'wb') as f:
                    pickle.dump(self.task, f)
                print(f"Saved task config to {task_config_path}")
        
        # Store task info for processing
        self.task_info = {
            'ambient_temp': self.task.get('ambient_temp', 25.0),
            'target_temperature': self.task.get('target_temperature', 24.0),
            'n_width': self.task.get('n_width', 10),
            'n_length': self.task.get('n_length', 10),
            'cell_size': self.task.get('cell_size', 2.0),
            'floor_height': self.task.get('floor_height', 3.0)
        }
        
        # Create environment
        self._create_environment()
        
        # Initialize policies
        if policies_to_use is None:
            policies_to_use = ["random", "pid", "constant"]
        self.policies_to_use = policies_to_use
        self.model_paths = model_paths or {}
        
        # Initialize models
        self.pid_solver = None
        self.ppo_model = None
        self.sac_model = None

        
        # Initialize noise distillation parameters
        self.noise_level = 0.1  # Start with high noise
        self.noise_decay_rate = 0.01  # Will be set based on action space
        self.total_learning_steps = 0
        
        # PPO augmentation config
        self.augment_config = {
            'noise_probability': 0.3,  # Probability of applying noise
            'noise_scale': 0.1,
            'use_binary_noise': True,  # True for original behavior, False for Gaussian
        }
        
        # Policy weights - updated according to requirements
        self.policy_weights = {
            'random': 0,
            'constant': 0.2,
            'pid': 0.3,
            'ppo_aug': 0.3,
            'ppo': 0.2,
            'unknown': 0.0
        }
        
        # Policy switching
        self.per_step_random_policy = True

        self.policy_switch_prob = 0.008

        self.policy_keep_prob = 0.992
        
        self._initialize_policies()
        
        # Updated tag mapping
        self.tag_mapping = {
            'random': 0,
            'constant': 1,
            'pid': 2,
            'ppo_aug': 3,
            'ppo': 4,
            'unknown': 5
        }
        
        # Masking probabilities
        self.mask_all_tag_prob = 0
        self.mask_episode_tag_prob = 0.15
        
        # Temperature failure threshold
        self.temperature_failure_threshold = 40.0

    def _create_environment(self):
        """创建HVAC环境实例"""
        if self.visual:
            self.env = HVACEnvVisible()
        else:
            self.env = HVACEnv(
                include_time_in_observation=self.use_time_in_observation,
                include_heat_in_observation=self.use_heat_in_observation
            )
        self.env.set_task(self.task)
        
        # Reset environment
        self.env.reset()
        
        # Get environment parameters
        self.n_sensors = len(self.env.sensors)
        self.n_coolers = len(self.env.coolers)
        self.n_heaters = len(self.env.equipments)
        
        # Get topology matrix
        self.cooler_sensor_topology = self.env.cooler_sensor_topology
    
    def _initialize_policies(self):
        """Initialize all available policies"""
        self.policies = {}
        
        # Set noise decay rate based on action space size
        if hasattr(self.env.action_space, 'shape'):
            action_dim = np.prod(self.env.action_space.shape)

            state_dim = np.prod(self.env.observation_space.shape)
            self.noise_decay_rate = random.uniform(0.0, 1.0 / (state_dim * action_dim))
        else:
            self.noise_decay_rate = 0.001  # Default decay rate
        
        # Random policy
        if "random" in self.policies_to_use:
            self.policies["random"] = self._random_policy
        
        # Constant temperature policy
        if "constant" in self.policies_to_use:
            self.policies["constant"] = self._constant_policy
        
        # PID policy
        if "pid" in self.policies_to_use:
            self.policies["pid"] = self._pid_policy
        
        # PPO policy
        if "ppo" in self.policies_to_use:
            if "ppo" in self.model_paths and os.path.exists(self.model_paths["ppo"]):
                try:
                    self.ppo_model = RecurrentPPO.load(
                        self.model_paths["ppo"], 
                        env=self.env,
                        device="cuda" if torch.cuda.is_available() else "cpu"
                    )
                    self.policies["ppo"] = self._ppo_policy
                    print(f"Loaded PPO model from {self.model_paths['ppo']}")
                except Exception as e:
                    print(f"Failed to load PPO model: {e}")
                    self.policies["ppo"] = self._random_policy
            else:
                print("PPO model not found, using random policy")
                self.policies["ppo"] = self._random_policy
        
        # PPO augmented policy
        if "ppo_aug" in self.policies_to_use:
            if "ppo" in self.model_paths and os.path.exists(self.model_paths["ppo"]):
                # Choose which PPO augmentation to use
                if self.augment_config.get('use_binary_noise', True):
                    # Use original behavior (binary: random or optimal)
                    self.policies["ppo_aug"] = self._ppo_aug_policy
                    print("PPO_AUG strategy enabled (binary noise)")
                else:
                    # Use Gaussian noise approach
                    self.policies["ppo_aug"] = self._ppo_aug_policy_gaussian
                    print("PPO_AUG strategy enabled (Gaussian noise)")
            else:
                print("PPO_AUG requires PPO model, falling back to random")
                self.policies["ppo_aug"] = self._random_policy
    
    def _random_policy(self, obs):
        """Random policy"""
        action = self.env.action_space.sample()
        return action, None
    
    def _constant_policy(self, obs):
        """Constant temperature control policy"""
        action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        n_coolers = len(self.env.coolers)
        
        # Set switch part - all ON
        action[:n_coolers] = 1.0
        
        # Set constant temperature value (24°C)
        target_temp = 24.0
        lb = self.env.lower_bound
        ub = self.env.upper_bound
        
        # Normalize to 0-1 range
        normalized_value = (target_temp - lb) / (ub - lb)
        normalized_value = np.clip(normalized_value, 0.0, 1.0)
        
        action[n_coolers:] = normalized_value

        return action, None
    
    def _pid_policy(self, obs):
        """PID policy"""
        if self.pid_solver is None:
            self.pid_solver = HVACSolverGTPID(self.env)
        action = self.pid_solver.policy_mask(obs, self.pid_mask) # 加mask，pid只控mask
        # action = self.pid_solver.policy(obs)
        return action, None
    
    def _ppo_policy(self, obs):
        """PPO policy"""
        if self.ppo_model is not None:
            episode_starts = np.array([self.episode_start], dtype=bool)  # 单环境 
            action, self.behavior_lstm_states = self.ppo_model.predict(
                obs, 
                state=self.behavior_lstm_states,
                episode_start=episode_starts, 
                deterministic=True
            )
            # 重置episode_start标志
            self.episode_start = False
            return action, self.behavior_lstm_states
        return self._random_policy(obs)

    def _ppo_aug_policy(self, obs):
        """PPO augmented policy - matches AnyMDPOptNoiseDistiller behavior"""
        if self.ppo_model is not None:
            # Decide whether to use noise or optimal action
            if random.random() < self.noise_level:
                # 返回随机动作（但仍需要更新LSTM状态）
                episode_starts = np.array([self.episode_start], dtype=bool)
                # 调用predict来维护LSTM状态的一致性
                _, self.behavior_lstm_states = self.ppo_model.predict(
                    obs, 
                    state=self.behavior_lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                # Return random action (like original)
                action = self.env.action_space.sample()
                self.episode_start = False
                return action, self.behavior_lstm_states
            else:
                # 返回PPO动作
                episode_starts = np.array([self.episode_start], dtype=bool)
                
                action, self.behavior_lstm_states = self.ppo_model.predict(
                    obs, 
                    state=self.behavior_lstm_states,
                    episode_start=episode_starts,
                    deterministic=True
                )
                
                self.episode_start = False
                return action, self.behavior_lstm_states
        return self._random_policy(obs)
    
    def _ppo_aug_policy_gaussian(self, obs):
        """Alternative: PPO augmented with Gaussian noise"""
        if self.ppo_model is not None:
            # Get base PPO action
            action, self.behavior_lstm_states = self.ppo_model.predict(
                obs, 
                state=self.behavior_lstm_state,
                deterministic=True
            )
            
            # Apply action augmentation with probability
            if random.random() < self.augment_config['noise_probability']:
                augmented_action = self._apply_action_noise_gaussian(action)
                return augmented_action, self.behavior_lstm_state
            else:
                return action, self.behavior_lstm_state
        return self._random_policy(obs)
    
    def _apply_action_noise_gaussian(self, action):
        """Apply Gaussian noise to action (improved version)"""
        action = np.array(action).flatten()
        
        # Add Gaussian noise
        noise_scale = self.augment_config['noise_scale']
        noise = np.random.normal(0, noise_scale, action.shape)
        
        # Apply noise with current noise level (instead of random decay)
        effective_noise = noise * self.noise_level
        
        augmented_action = action + effective_noise
        
        # Clip to valid range
        augmented_action = np.clip(augmented_action, 
                                  self.env.action_space.low, 
                                  self.env.action_space.high)
        
        return augmented_action.astype(np.float32)
    
    def update_noise_level(self):
        """Update noise level - call this after each episode or batch"""
        self.noise_level = max(0.0, self.noise_level - self.noise_decay_rate)
        self.total_learning_steps += 1
    
    def _compute_action_temperature_differences_with_graph(self, behavior_temp_settings, reference_temp_settings, 
                                                          obs_graph, obs):
        """
        利用obs_graph计算每个cooler的温度设定值与其最近k个sensor目标温度的差值
        
        Args:
            behavior_temp_settings: 行为策略的温度设定值 (n_coolers,)
            reference_temp_settings: 参考策略的温度设定值 (n_coolers,)
            obs_graph: sensor-cooler关系图 (n_sensors, n_coolers)，转置后为 (n_coolers, n_sensors)
        
        Returns:
            diff_behavior: 行为策略温度差值 (n_coolers,)
            diff_reference: 参考策略温度差值 (n_coolers,)
        """
        n_coolers = self.n_coolers
        diff_behavior = np.zeros(n_coolers, dtype=np.float32)
        diff_reference = np.zeros(n_coolers, dtype=np.float32)
        
        # 获取目标温度
        target_temp = self.task_info['target_temperature']
        if isinstance(target_temp, (list, np.ndarray)):
            # 如果是数组，为每个sensor使用对应的目标温度
            target_temps = np.array(target_temp)
        else:
            # 如果是标量，所有sensor使用相同的目标温度
            target_temps = np.full(self.n_sensors, target_temp)
        
        # 转置obs_graph得到 (n_coolers, n_sensors) 格式
        cooler_sensor_graph = obs_graph.T
        
        for cooler_idx in range(n_coolers):
            # 从graph中获取该cooler关联的sensors（值为1的位置）
            connected_sensors = np.where(cooler_sensor_graph[cooler_idx] == 1.0)[0]
            
            if len(connected_sensors) > 0:
                # 计算关联sensors的平均目标温度
                avg_target_temp = np.mean(target_temps[connected_sensors])
                # avg_target_temp = np.mean(obs[connected_sensors])
            else:
                # 如果没有关联的sensor，使用全局平均目标温度
                avg_target_temp = np.mean(target_temps)
                # avg_target_temp = np.mean(obs)
            
            # 计算温度差值（设定值 - 平均目标温度）

            diff_behavior[cooler_idx] = behavior_temp_settings[cooler_idx] - avg_target_temp
            diff_reference[cooler_idx] = reference_temp_settings[cooler_idx] - avg_target_temp
            
        return diff_behavior, diff_reference
    
    
    
    def _select_policies_for_coolers(self, current_policies=None):
        """
        Select policy for each cooler independently
        Returns a list of policy names, one for each cooler
        """
        if not self.per_step_random_policy:
            return None
        new_policies = []
        
        for cooler_idx in range(self.n_coolers):
            # Check if we should keep current policy for this cooler
            if current_policies is not None:
                current_policy = current_policies[cooler_idx]
                if current_policy in self.policies and random.random() < self.policy_keep_prob:
                    new_policies.append(current_policy)
                    continue
            
            # Select new policy based on weights
            available_policies = []
            weights = []
            
            for policy_name in self.policies.keys():
                if policy_name in self.policy_weights:
                    available_policies.append(policy_name)
                    weights.append(self.policy_weights[policy_name])
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(available_policies)] * len(available_policies)
            
            # Select policy for this cooler
            selected_policy = random.choices(available_policies, weights=weights)[0]
            new_policies.append(selected_policy)
        
        return new_policies
    
    def _compute_all_policy_actions(self, obs):
        """
        Compute actions for all policies for the given observation
        Returns a dictionary: {policy_name: (action, lstm_state)}
        """
        all_actions = {}
        # 保存原始的LSTM状态
        original_lstm_states = self.behavior_lstm_states
        original_episode_start = self.episode_start

        
        for policy_name, policy_func in self.policies.items():
            if policy_name in ['ppo', 'ppo_aug']:
                # 为每个LSTM策略创建独立的状态副本
                if policy_name == 'ppo':
                    # 使用reference的LSTM状态
                    temp_lstm_states = copy.deepcopy(self.reference_lstm_states) if self.reference_lstm_states is not None else None
                    temp_episode_start = self.episode_start
                else:  # ppo_aug
                    # 使用behavior的LSTM状态
                    temp_lstm_states = copy.deepcopy(self.behavior_lstm_states) if self.behavior_lstm_states is not None else None
                    temp_episode_start = self.episode_start
                # 临时设置状态 - 但这里不修改实例变量
                original_behavior_states = self.behavior_lstm_states
                original_episode_start = self.episode_start

                try:
                    if policy_name == 'ppo':
                        # 对于PPO，我们需要特殊处理，因为它用于reference
                        # 这里不应该修改实例的状态，而是直接调用模型
                        if self.ppo_model is not None:
                            episode_starts = np.array([temp_episode_start], dtype=bool)
                            action, updated_lstm_states = self.ppo_model.predict(
                                obs, 
                                state=temp_lstm_states,
                                episode_start=episode_starts,
                                deterministic=True
                            )
                            all_actions[policy_name] = (action, updated_lstm_states)
                        else:
                            action, lstm_state = self._random_policy(obs)
                            all_actions[policy_name] = (action, lstm_state)
                    else:  # ppo_aug

                        self.behavior_lstm_states = temp_lstm_states
                        self.episode_start = temp_episode_start
                        
                        action, updated_lstm_states = policy_func(obs)
                        all_actions[policy_name] = (action, updated_lstm_states)
                finally:
                    # 恢复原始状态
                    self.behavior_lstm_states = original_behavior_states
                    self.episode_start = original_episode_start
                
            else:
                # 非LSTM策略
                action, lstm_state = policy_func(obs)
                all_actions[policy_name] = (action, lstm_state)
    
        # 恢复原始状态（避免污染）
        self.behavior_lstm_states = original_lstm_states
        self.episode_start = original_episode_start
        
        return all_actions
    
    def _reorganize_actions_by_cooler_policies(self, all_policy_actions, cooler_policies):
        """
        Reorganize actions based on selected policy for each cooler
        """
        n_coolers = self.n_coolers
        
        # Initialize combined action
        combined_action = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        
        # Process each cooler
        for cooler_idx, policy_name in enumerate(cooler_policies):
            if policy_name in all_policy_actions:
                action, _ = all_policy_actions[policy_name]
                action = np.array(action).flatten()
                
                # Extract this cooler's switch and value from the policy action
                combined_action[cooler_idx] = action[cooler_idx]  # Switch
                if len(action) > n_coolers:
                    combined_action[n_coolers + cooler_idx] = action[n_coolers + cooler_idx]  # Value
        
        return combined_action
    
    
    
    def _compute_temperature_deviation(self, obs):
        """计算温度偏差"""
        # 提取sensor读数
        if self.use_time_in_observation or self.use_heat_in_observation:
            sensor_readings = obs[:self.n_sensors]
        else:
            sensor_readings = obs
        
        # 获取目标温度
        target_temp = self.task_info['target_temperature']
        if isinstance(target_temp, (list, np.ndarray)):
            # 如果是数组，计算每个sensor与其对应目标温度的差值
            temperature_deviation = sensor_readings - np.array(target_temp)
        else:
            # 如果是标量，所有sensor使用相同的目标温度
            temperature_deviation = sensor_readings - target_temp
        
        return temperature_deviation
    
    def _check_temperature_failure(self, obs):
        """Check if any sensor exceeds temperature threshold"""
        if self.use_time_in_observation or self.use_heat_in_observation:
            sensor_readings = obs[:self.n_sensors]
        else:
            sensor_readings = obs


        # Check if any sensor exceeds threshold
        return np.any(sensor_readings > self.temperature_failure_threshold)
    
    
    def _create_sensor_cooler_graph(self, num_closest_sensors: int = 3):
        """
        Create sensor-cooler relationship graph.

        Args:
            num_closest_sensors (int): The number of closest sensors to consider for each cooler.
                                       Defaults to 3.

        Returns:
            np.ndarray: The sensor-cooler relationship graph.
        """
        # Create a new matrix with the same shape as the original (cooler, sensor) matrix
        obs_graph_orig = np.zeros_like(self.cooler_sensor_topology)
        n_coolers, n_sensors = self.cooler_sensor_topology.shape
        
        for cooler_id in range(n_coolers):
            sensor_weights = self.cooler_sensor_topology[cooler_id, :]
            if n_sensors >= num_closest_sensors:
                # Get the indices of the 'num_closest_sensors' smallest values (closest sensors)
                closest_sensor_idx = np.argpartition(sensor_weights, num_closest_sensors)[:num_closest_sensors]
            else:
                # If there are fewer than 'num_closest_sensors' sensors, use all sensors
                closest_sensor_idx = np.arange(len(sensor_weights))
            
            # Set the positions corresponding to the closest sensors to 1
            obs_graph_orig[cooler_id, closest_sensor_idx] = 1.0
        
        # Transpose the matrix to match the required shape (sensor, cooler)
        obs_graph = obs_graph_orig.T.astype(np.float32)
        return obs_graph

    def _create_cooler_cooler_graph(self, k_nearest_coolers: int = 3):
        """
        Create cooler-cooler relationship graph using KNN.

        Args:
            k_nearest_coolers (int): The number of nearest coolers to consider for each cooler.
                                     Defaults to 3.

        Returns:
            np.ndarray: The cooler-cooler relationship graph.
        """
        n_coolers = self.n_coolers
        agent_graph = np.zeros((n_coolers, n_coolers), dtype=np.float32)
        
        # Get cooler positions
        # Assuming self.env.coolers is an iterable of objects with a 'loc' attribute
        cooler_positions = np.array([cooler.loc for cooler in self.env.coolers])
        
        # Compute pairwise distances
        for i in range(n_coolers):
            for j in range(n_coolers):
                if i != j:
                    dist = np.linalg.norm(cooler_positions[i] - cooler_positions[j])
                    agent_graph[i, j] = dist
        
        # Convert to KNN graph (k='k_nearest_coolers' nearest neighbors)
        k = min(k_nearest_coolers, n_coolers - 1)
        for i in range(n_coolers):
            # Get k nearest neighbors
            distances = agent_graph[i, :]
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            # Set connections
            agent_graph[i, :] = 0
            agent_graph[i, nearest_indices] = 1
        
        # Make symmetric and add self-connections (if desired, currently commented out)
        # agent_graph = np.maximum(agent_graph, agent_graph.T)
        # np.fill_diagonal(agent_graph, 1.0) # Uncomment if self-connections are needed
        
        return agent_graph
        
    def generate_multiagent_data(self, epoch_id, max_steps=5040):
        """Generate multi-agent data with corrected dimensions and immediate reset on overheating"""
        print(f"\nGenerating data for epoch {epoch_id:06d}")
        
        # Initialize progress tracking
        start_time = time.time()
        
        # Initialize data storage - 按agent组织
        
        observations = []
        diff_observations = [] 
        actions_behavior = []
        actions_label = []
        diff_actions_behavior = []  
        diff_actions_label = []     
        tags = []
        tags_dict = {}
        rewards = []
        resets = []
        obs_graph = []
        agent_graph = []
        
        # Initialize environment
        obs, info = self.env.reset()
        self._reset_lstm_states()
        # 初始化参考策略的LSTM状态
        reference_episode_start = True
        # Masking decisions
        mask_all_tag = random.random() < self.mask_all_tag_prob
        mask_episode_tag = random.random() < self.mask_episode_tag_prob
        
        # Statistics
        steps = 0
        total_reward = 0
        episode_count = 0
        overheating_resets = 0  # Track overheating resets
        num_closest_sensors = random.randint(3,5)
        policy_usage_count = {name: 0 for name in self.policies.keys()}
        cooler_policy_usage = {i: {name: 0 for name in self.policies.keys()} for i in range(self.n_coolers)}
        
        # Initial policy selection
        # current_policy_name = self._select_policy_for_step(self.policies)
        current_cooler_policies = self._select_policies_for_coolers()
        # behavior_policy = self.policies[current_policy_name]
        # Reference policy (PPO for labels)
        reference_policy = self.policies.get("ppo", self._random_policy)

        print(f"  Initial cooler policies: {current_cooler_policies}")
        print(f"  Initial noise level: {self.noise_level:.4f}")
        self.pid_mask = np.array([policy == 'pid' for policy in current_cooler_policies], dtype=bool)  # 加mask，pid只控mask
        while steps < max_steps:
            # print(steps)
            # Progress update
            if steps % 1000 == 0 and steps > 0:
                print(f"  Progress: {steps}/{max_steps} steps")
            # Select policies for each cooler
            if self.per_step_random_policy:
                current_cooler_policies = self._select_policies_for_coolers(current_cooler_policies)
                self.pid_mask = np.array([policy == 'pid' for policy in current_cooler_policies], dtype=bool) # 加mask，pid只控mask
                # new_policy_name = self._select_policy_for_step(self.policies, current_policy_name)
                # Update usage statistics
                for cooler_idx, policy_name in enumerate(current_cooler_policies):
                    policy_usage_count[policy_name] += 1
                    cooler_policy_usage[cooler_idx][policy_name] += 1

            # Compute actions for all policies
            # 1. 计算behavior动作
            all_policy_actions = self._compute_all_policy_actions(obs)        
            behavior_action = self._reorganize_actions_by_cooler_policies(
                all_policy_actions, current_cooler_policies
            )
            # 2. 独立计算reference动作 - 不通过_compute_all_policy_actions

            if self.ppo_model is not None:
                reference_episode_starts = np.array([reference_episode_start], dtype=bool)
                # print("reference_episode_starts", reference_episode_starts)
                reference_action, self.reference_lstm_states = self.ppo_model.predict(
                    obs, 
                    state=self.reference_lstm_states,
                    episode_start=reference_episode_starts,
                    deterministic=True
                )
                reference_episode_start = False  # 重置参考策略的episode_start标志 只在第一步为True
            else:
                reference_action, _ = self._random_policy(obs)

            # 确保动作格式正确
            behavior_action = np.array(behavior_action).flatten()
            reference_action = np.array(reference_action).flatten()


            # Execute action
            obs, reward_behavior_action, done, truncated, info = self.env.step(behavior_action)
            # obs_reference_action, reward_reference_action, done, truncated, info = self.env.step(reference_action)


            
            # Check for overheating BEFORE processing the observation
            temperature_failed = self._check_temperature_failure(obs)
            # temperature_failed = self._check_temperature_failure(obs_reference_action)
            # Extract sensor readings for temperature deviation
            if self.use_time_in_observation or self.use_heat_in_observation:
                sensor_readings = obs[:self.n_sensors]
            else:
                sensor_readings = obs
            
            # 额外保存温差
            temperature_deviations = self._compute_temperature_deviation(obs)
            temperature_ori = sensor_readings
            # Extract switch and value from actions
            n_coolers = self.n_coolers
            
            # Behavior action
            behavior_switch = behavior_action[:n_coolers]
            behavior_switch = np.where(behavior_switch > 0.5, 1.0, 0.0)  # 添加二值化操作
            behavior_values = behavior_action[n_coolers:n_coolers*2] if len(behavior_action) > n_coolers else np.ones(n_coolers) * 0.5
            turn_on_mask = (behavior_switch > 0.5)

            # Reference action
            reference_switch = reference_action[:n_coolers]
            reference_switch = np.where(reference_switch > 0.5, 1.0, 0.0)  # 添加二值化操作
            reference_values = reference_action[n_coolers:n_coolers*2] if len(reference_action) > n_coolers else np.ones(n_coolers) * 0.5

            # Convert normalized values to actual temperature settings

            behavior_temp_settings = behavior_values * (self.env.upper_bound - self.env.lower_bound) + self.env.lower_bound
            reference_temp_settings = reference_values * (self.env.upper_bound - self.env.lower_bound) + self.env.lower_bound

            # 计算动作温度差值（cooler设定温度与其最近k个sensor目标温度的差值）
            # 使用当前的obs_graph
            current_obs_graph = self._create_sensor_cooler_graph(num_closest_sensors)
            
            diff_temp_behavior, diff_temp_reference = self._compute_action_temperature_differences_with_graph(
                behavior_temp_settings, 
                reference_temp_settings,
                current_obs_graph, 
                obs
            )
            
            # Create tags for each cooler [agent_id, policy_tag]
            cooler_tags = {}
            for cooler_idx, policy_name in enumerate(current_cooler_policies):
                if mask_all_tag or mask_episode_tag:
                    tag = self.tag_mapping['unknown']
                else:

                    tag = self.tag_mapping.get(policy_name, self.tag_mapping['unknown'])

                cooler_tags[cooler_idx] = [tag]
            for agent_idx in cooler_tags.keys():
                if not tags_dict.get(agent_idx, []):
                    tags_dict[agent_idx] = []
                tags_dict.get(agent_idx, []).extend(cooler_tags[agent_idx])

            # 行为动作：[温度设定值, 开关状态]
            behavior_action_array = np.array([behavior_temp_settings, behavior_switch], dtype=np.float32)

            # 标签动作：[温度设定值, 开关状态]
            reference_action_array = np.array([reference_temp_settings, reference_switch], dtype=np.float32)

            # diff_actions包含[温度差值, 开关状态]，与actions_behavior/label保持相同结构


            diff_behavior_action_array = np.array([diff_temp_behavior, behavior_switch], dtype=np.float32)
            diff_reference_action_array = np.array([diff_temp_reference, reference_switch], dtype=np.float32)

            # Check temperature failure
            temperature_failed = self._check_temperature_failure(obs)
            # 存储数据    
            observations.append(temperature_ori)
            diff_observations.append(temperature_deviations)
            actions_behavior.append(behavior_action_array)
            actions_label.append(reference_action_array)
            rewards.append(reward_behavior_action)
            diff_actions_behavior.append(diff_behavior_action_array) 
            diff_actions_label.append(diff_reference_action_array)    
            resets.append(1 if temperature_failed else 0)
            
            obs_graph.append(current_obs_graph)  # num_closest_sensors

            agent_graph.append(self._create_cooler_cooler_graph(num_closest_sensors))  # num_closest_sensors
            # Update statistics
            total_reward += reward_behavior_action
            steps += 1

            # Episode end handling
            if temperature_failed:
                print(f"  IMMEDIATE RESET at step {steps} due to overheating!")
                overheating_resets += 1
                episode_count += 1
                
                # Update noise level after overheating episode
                self.update_noise_level()
                
                # Reset environment immediately
                if steps < max_steps:
                    obs, info = self.env.reset()
                    self._reset_lstm_states()
                    reference_episode_start = True
                    # Update episode masking
                    mask_episode_tag = random.random() < self.mask_episode_tag_prob
                    
                    print(f"  Environment reset completed. Continuing from step {steps}")
                continue  # Skip normal episode end handling
            
            
            if done or truncated:
                episode_count += 1
                
                # Update noise level after each episode
                self.update_noise_level()
                
                # Reset if more steps remain
                if steps < max_steps:
                    obs, info = self.env.reset()
                    self._reset_lstm_states()
                    reference_episode_start = True
                    # Update episode masking
                    mask_episode_tag = random.random() < self.mask_episode_tag_prob

        

        # 全局数据
        observations = np.array(observations, dtype=np.float32)  # Shape: (5040, heater)
        diff_observations = np.array(diff_observations, dtype=np.float32)  # Shape: (5040, heater)
        actions_behavior = np.array(actions_behavior, dtype=np.float32)  # Shape: (5040, 2, cooler)
        actions_label = np.array(actions_label, dtype=np.float32)  # Shape: (5040, 2, cooler)
        diff_actions_behavior = np.array(diff_actions_behavior, dtype=np.float32)  # Shape: (5040, 2, cooler) - 修改
        diff_actions_label = np.array(diff_actions_label, dtype=np.float32)  # Shape: (5040, 2, cooler) - 修改
        obs_graph = np.array(obs_graph[0], dtype=np.float32)   # Shape: (n_sensors, cooler)
        agent_graph = np.array(agent_graph[0], dtype=np.float32)    # Shape: (cooler, cooler)
        rewards = np.array(rewards, dtype=np.float32).reshape(-1, 1)  # Shape: (5040, 1)
        tags = np.array(list(tags_dict.values()), dtype=np.float32)  # Shape: (cooler, step)
        resets = np.array(resets, dtype=np.int32).reshape(-1, 1)   # Shape: (5040, 1)
        
        # Print statistics
        print(f"\nEpoch {epoch_id:06d} completed:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Episodes: {episode_count}")
        print(f"  Temperature failures (immediate resets): {overheating_resets}")
        print(f"  Normal episode ends: {episode_count - overheating_resets}")
        print(f"  Total resets in data: {np.sum(resets)}")
        print(f"  Final noise level: {self.noise_level:.4f}")
        print(f"  Total learning steps: {self.total_learning_steps}")
        
        # Print policy usage
        if self.per_step_random_policy:
            print(f"  Policy usage:")
            for policy_name, count in policy_usage_count.items():
                percentage = (count / steps) * 100 if steps > 0 else 0
                print(f"    {policy_name}: {count} steps ({percentage:.1f}%)")
        
        # Print data shapes for verification
        print(f"\nData shapes:")
        print(f"  observations: {observations.shape}")
        print(f"  diff_observations: {diff_observations.shape}")
        print(f"  actions_behavior: {actions_behavior.shape}")
        print(f"  actions_label: {actions_label.shape}")
        print(f"  diff_actions_behavior: {diff_actions_behavior.shape}")
        print(f"  diff_actions_label: {diff_actions_label.shape}")
        print(f"  rewards: {rewards.shape}")
        print(f"  tags: {tags.shape}")
        print(f"  resets: {resets.shape}")
        print(f"  obs_graph: {obs_graph.shape}")
        print(f"  agent_graph: {agent_graph.shape}")
        
        # Prepare data dictionary
        processed_data = {
            "observations": observations,
            "diff_observations": diff_observations,
            "actions_behavior": actions_behavior,
            "actions_label": actions_label,
            "diff_actions_behavior": diff_actions_behavior,  # 修改：现在形状为 (5040, 2, cooler)
            "diff_actions_label": diff_actions_label,        # 修改：现在形状为 (5040, 2, cooler)
            "tags": tags,
            "rewards": rewards,
            "resets": resets,
            "obs_graph": obs_graph,
            "agent_graph": agent_graph,
            # 保存原始的per-agent数据用于特殊需求
        }
        
        reorganized_data = self.reorganize_data_by_agent(processed_data)
        
        elapsed_time = time.time() - start_time
        print(f"  Generation time: {elapsed_time:.1f}s")
        
        return reorganized_data

    def save_multiagent_data(self, data, output_path, epoch_id):
        """Save multi-agent data with correct format"""
        epoch_path = Path(output_path) / f'record-{epoch_id:06d}'
        os.makedirs(epoch_path, exist_ok=True)
        
        # 保存聚合后的数据（用于兼容性）
        np.save(epoch_path / 'observations.npy', data["observations"])
        np.save(epoch_path / 'diff_observations.npy', data["diff_observations"])
        np.save(epoch_path / 'actions_behavior.npy', data["actions_behavior"])
        np.save(epoch_path / 'actions_label.npy', data["actions_label"])
        np.save(epoch_path / 'diff_actions_behavior.npy', data["diff_actions_behavior"])  # 修改：现在包含开关状态
        np.save(epoch_path / 'diff_actions_label.npy', data["diff_actions_label"])      # 修改：现在包含开关状态
        np.save(epoch_path / 'tags.npy', data["tags"])
        np.save(epoch_path / 'rewards.npy', data["rewards"])
        np.save(epoch_path / 'resets.npy', data["resets"])
        np.save(epoch_path / 'obs_graph.npy', data["obs_graph"])
        np.save(epoch_path / 'agent_graph.npy', data["agent_graph"])
        print(f"Saved data to {epoch_path}")
        
    def _reset_lstm_states(self):
        """Reset LSTM states for new episode"""
        self.behavior_lstm_states = None
        self.reference_lstm_states = None
        self.episode_start = True  # 标记新episode开始

        
    def _select_policy_for_step(self, policies_dict, current_policy=None):
        """Legacy method for backward compatibility"""
        policies = self._select_policies_for_coolers([current_policy] * self.n_coolers if current_policy else None)
        return policies[0] if policies else None
    
    def reorganize_data_by_agent(self, data, max_steps=5040):
        """
        重新组织数据，将agent作为第一维度
        
        目标格式:
        - observations: (n_agents, timesteps, features_per_agent)
        - actions_behavior/label: (n_agents, timesteps, 2) -> [温度设定值, 开关状态]
        - diff_actions_behavior/label: (n_agents, timesteps, 2) -> [温度差值, 开关状态] - 修改
        - tags: (n_agents, timesteps, 1) -> policy_tag for each agent
        - rewards: (timesteps, 1) -> 保持不变（全局奖励）
        - resets: (timesteps, 1) -> 保持不变（全局重置）
        """
        n_coolers = self.n_coolers
        n_sensors = self.n_sensors
        timesteps = len(data["observations"])
        
        print(f"\nReorganizing data by agent...")
        print(f"  Number of agents (coolers): {n_coolers}")
        print(f"  Number of sensors: {n_sensors}")
        print(f"  Number of timesteps: {timesteps}")
        
        # 1. 重组observations 和 diff_observations
        # 原始: (timesteps, n_sensors)
        # 目标: (n_sensors, timesteps, 1) - 每个传感器作为一个agent
        observations_by_agent = []
        for sensor_idx in range(n_sensors):
            sensor_data = []
            for t in range(timesteps):
                sensor_data.append([data["observations"][t, sensor_idx]])
            observations_by_agent.append(sensor_data)
        observations_by_agent = np.array(observations_by_agent, dtype=np.float32)
        
        diff_observations_by_agent = []
        for sensor_idx in range(n_sensors):
            sensor_data = []
            for t in range(timesteps):
                sensor_data.append([data["diff_observations"][t, sensor_idx]])
            diff_observations_by_agent.append(sensor_data)
        diff_observations_by_agent = np.array(diff_observations_by_agent, dtype=np.float32)
        
        # 2. 重组actions
        # 原始: (timesteps, 2, n_coolers)
        # 目标: (n_coolers, timesteps, 2)
        actions_behavior_by_agent = []
        actions_label_by_agent = []
        for cooler_idx in range(n_coolers):
            behavior_data = []
            label_data = []
            for t in range(timesteps):
                # 提取该cooler在时间步t的动作 [温度设定值, 开关状态]
                behavior_data.append([
                    data["actions_behavior"][t, 0, cooler_idx],  # 温度设定值
                    data["actions_behavior"][t, 1, cooler_idx]   # 开关状态
                ])
                label_data.append([
                    data["actions_label"][t, 0, cooler_idx],     # 温度设定值
                    data["actions_label"][t, 1, cooler_idx]      # 开关状态
                ])
                
            actions_behavior_by_agent.append(behavior_data)
            actions_label_by_agent.append(label_data)

        actions_behavior_by_agent = np.array(actions_behavior_by_agent, dtype=np.float32)
        actions_label_by_agent = np.array(actions_label_by_agent, dtype=np.float32)
        
        # 3. 重组动作差值数据 - 修改：现在包含[温度差值, 开关状态]
        # 原始: (timesteps, 2, n_coolers)
        # 目标: (n_coolers, timesteps, 2)
        diff_actions_behavior_by_agent = []
        diff_actions_label_by_agent = []
        
        for cooler_idx in range(n_coolers):
            diff_behavior_data = []
            diff_label_data = []
            for t in range(timesteps):
                # 提取该cooler在时间步t的差值动作 [温度差值, 开关状态]

                diff_behavior_data.append([
                    data["diff_actions_behavior"][t, 0, cooler_idx],  # 温度差值
                    data["diff_actions_behavior"][t, 1, cooler_idx]   # 开关状态
                ])
                diff_label_data.append([
                    data["diff_actions_label"][t, 0, cooler_idx],     # 温度差值
                    data["diff_actions_label"][t, 1, cooler_idx]      # 开关状态
                ])
            diff_actions_behavior_by_agent.append(diff_behavior_data)
            diff_actions_label_by_agent.append(diff_label_data)

        diff_actions_behavior_by_agent = np.array(diff_actions_behavior_by_agent, dtype=np.float32)
        diff_actions_label_by_agent = np.array(diff_actions_label_by_agent, dtype=np.float32)
        print("diff_actions_label_by_agent", diff_actions_label_by_agent)
        # 4. tags保持不变
        tags = data["tags"]
        
        # 5. 图结构保持不变（已经是正确格式）
        obs_graph = data["obs_graph"].T    # (n_coolers，n_sensors)  
        agent_graph = data["agent_graph"]  # (n_coolers, n_coolers)
        
        # 6. 全局数据保持不变
        rewards = data["rewards"]  # (timesteps, 1)
        resets = data["resets"]  # (timesteps, 1)
        
        # 创建重组后的数据字典
        reorganized_data = {
            "observations": observations_by_agent,  # (n_sensors, timesteps, 1)
            "diff_observations": diff_observations_by_agent,  # (n_sensors, timesteps, 1)
            "actions_behavior": actions_behavior_by_agent,  # (n_coolers, timesteps, 2)
            "actions_label": actions_label_by_agent,  # (n_coolers, timesteps, 2)
            "diff_actions_behavior": diff_actions_behavior_by_agent,  # (n_coolers, timesteps, 2) - 修改
            "diff_actions_label": diff_actions_label_by_agent,  # (n_coolers, timesteps, 2) - 修改
            "tags": tags,  # (n_coolers, timesteps)
            "rewards": rewards,  # (timesteps, 1) - 全局奖励
            "resets": resets,  # (timesteps, 1) - 全局重置
            "obs_graph": obs_graph,  # (n_coolers, n_sensors)
            "agent_graph": agent_graph,  # (n_coolers, n_coolers)
            # 保存元数据
            "metadata": {
                "n_agents": n_coolers,
                "n_sensors": n_sensors,
                "n_timesteps": timesteps,
            }
        }
        
        # 打印重组后的数据形状
        print(f"\nReorganized data shapes:")
        print(f"  observations: {reorganized_data['observations'].shape}")
        print(f"  diff_observations: {reorganized_data['diff_observations'].shape}")
        print(f"  actions_behavior: {reorganized_data['actions_behavior'].shape}")
        print(f"  actions_label: {reorganized_data['actions_label'].shape}")
        print(f"  diff_actions_behavior: {reorganized_data['diff_actions_behavior'].shape}")
        print(f"  diff_actions_label: {reorganized_data['diff_actions_label'].shape}")
        print(f"  tags: {reorganized_data['tags'].shape}")
        print(f"  rewards: {reorganized_data['rewards'].shape}")
        print(f"  resets: {reorganized_data['resets'].shape}")
        print(f"  obs_graph: {reorganized_data['obs_graph'].shape}")
        print(f"  agent_graph: {reorganized_data['agent_graph'].shape}")
        
        return reorganized_data
