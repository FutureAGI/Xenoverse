import sys
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numbers
import numpy as np
from copy import deepcopy

class HVACEnv(gym.Env):
    def __init__(self,
                 max_steps=20160,  # sec_per_iter * iter_per_step * max_steps / 86400 days
                 iter_per_step=200,
                 sec_per_iter=0.15,
                 set_lower_bound=16,
                 set_upper_bound=32,
                 verbose=False,
                 action_space_format='box',
                 include_heat_in_observation=True,
                 include_switch_in_observation=False,
                 include_last_action_in_observation=True,
                 include_timestep_in_observation=True,
                 no_switch_action=True,
                 reward_mode = 0 # mode 0: include energy & target(+) & fail; 
                                 # mode 1: include target(+-) & fail; 
                                 # mode 2: include energy & fail
                 ):
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1,), dtype=np.float32)
        self.action_space = None
        self.max_steps = max_steps
        self.failure_reward = -30
        self.overheat_reward = -10
        self.energy_reward_wht = -3.0 #-3.0   
        self.switch_reward_wht = -20.0 
        self.target_reward_wht = -0.5
        self.action_diff_wht = -0.1 
        self.base_reward = 1.0 # survive bonus
        self.iter_per_step = iter_per_step
        self.sec_per_iter = sec_per_iter
        self.sec_per_step = self.iter_per_step * self.sec_per_iter
        self.lower_bound = set_lower_bound
        self.upper_bound = set_upper_bound
        self.verbose = verbose
        self.warning_count_tolerance = 5
        self.last_fail_timestemp = -1
        self.action_space_format = action_space_format
        self.include_heat_in_observation = include_heat_in_observation
        self.include_switch_in_observation = include_switch_in_observation
        self.include_last_action_in_observation = include_last_action_in_observation
        self.include_timestep_in_observation = include_timestep_in_observation
        self.include_action_cost = False
        self.no_switch_action = no_switch_action
        self.return_normilized_obs = False
        self.random_start_t = False
        self.generate_record = False
        self.reward_mode = reward_mode
        # stat
        self.avg_cooler_power_per_step = 0.0
        self.avg_reward = 0.0
        self.over_heat_percentage = [0.0, 0.0, 0.0, 0.0] # percentage over 0, 2, 4, 6 degree
        self.over_cool_percentage = [0.0, 0.0, 0.0, 0.0] # percentage lower 0, -2, -4, -6 degree
        self.bad_switch_percentage = [0.0, 0.0] #  below boundary, over boundary
        self.fail_percentage = 0.0
        # training
        self.overheat_no_termiated_training_only = False

    def set_task(self, task, generate_task=False):
        for key in task:
            self.__dict__[key] = task[key]
        self.task_set = True
        self.heat_capacity = task.get('heat_capacity', []) 
        self.equipments = task.get('equipments', [])

        # triggers failure above this temperature
        self.failure_upperbound = np.mean(self.target_temperature + 6)
        
        n_coolers = len(self.coolers)
        n_sensors = len(self.sensors)
        n_heaters = len(self.equipments)
        
        if self.action_space_format == 'dict':
            self.action_space = Dict({
                "switch": gym.spaces.MultiBinary(n_coolers),
                "value": gym.spaces.Box(low=0, high=1, shape=(n_coolers,), dtype=np.float32)
            })
        else:
            if not self.no_switch_action: 
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(2*n_coolers,), dtype=np.float32) # Placeholder shape
            else:
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_coolers,), dtype=np.float32)

        self.cooler_topology = np.zeros((n_coolers, n_coolers))
        self.cooler_sensor_topology = np.zeros((n_coolers, n_sensors))
        for i in range(n_coolers):
            for j in range(i, n_coolers):
                distance = np.sqrt(np.sum((self.coolers[i].loc - self.coolers[j].loc) ** 2))
                self.cooler_topology[i, j] = distance
                self.cooler_topology[j, i] = distance
        for i,cooler in enumerate(self.coolers):
           for j,sensor in enumerate(self.sensors):
                self.cooler_sensor_topology[i, j] = np.sqrt(np.sum((cooler.loc - sensor.loc) ** 2))

        # calculate cross sectional area
        self.csa = self.cell_size * self.floor_height
        
        # 动态构建 observation_space
        obs_spaces = {}

        # 1. 基础观测：传感器读数
        sensor_low = -1 if self.return_normilized_obs else 10
        sensor_high = 1 if self.return_normilized_obs else 50
        obs_spaces["sensor_readings"] = gym.spaces.Box(
            low=sensor_low, high=sensor_high, shape=(n_sensors,), dtype=np.float32
        )
        
        # 2. 根据标志位添加其他观测
        if self.include_heat_in_observation:
            obs_spaces["heat_readings"] = gym.spaces.Box(
                low=0.0, high=50000.0, shape=(n_heaters,), dtype=np.float32
            )

        if self.include_switch_in_observation:
            obs_spaces["switch_action"] = gym.spaces.Box(
                low=0, high=1, shape=(n_coolers,), dtype=np.float32
            )

        if self.include_last_action_in_observation:
            obs_spaces["action_temp"] = gym.spaces.Box(
                low=-1, high=33, shape=(n_coolers,), dtype=np.float32
            )

        if self.include_timestep_in_observation:
            obs_spaces["timestep"] = gym.spaces.Box(
                low=0, high = 30000, shape=(1,), dtype=np.int32
            )

        # 创建最终的 observation_space
        self.observation_space = gym.spaces.Dict(obs_spaces)
        if self.verbose:
            print(self.observation_space)

        # calculate average total heat
        if not generate_task:
            self.avg_total_heat = self._calculate_average_total_heat(time_step_interval=30, max_time=604800)
        else:
            self.avg_total_heat = 10000

        # For overheat reset
        self.history_length = 1000
        self.cooler_state_history = {
            'current_rest_cooler_idx': [],
            # 'cooler_rest_start_time': [],
            # 'last_action_switch': [],
            # 'current_action_switch': []
        }

    def _calculate_average_total_heat(self, time_step_interval=30, max_time=604800):
        """
        计算环境的平均总发热量
        
        参数:
            env: HVAC 环境实例（已加载任务）
            time_step_interval: 采样间隔（秒）
            max_time: 最大时间（秒）
        
        返回:
            float: 平均总发热量（W）
        """
        time_steps = np.arange(0, max_time, time_step_interval)
        total_heat_sum = 0.0
        
        for t in time_steps:
            # 计算当前时刻所有设备的总发热量
            step_heat = sum(equipment(t)["heat"] for equipment in self.equipments)
            total_heat_sum += step_heat
    
        return total_heat_sum / len(time_steps)
    
    def _action_value_to_temp(self, action_value):
        return action_value * (self.upper_bound - self.lower_bound) + self.lower_bound
    
    def _action_temp_to_value(self, action_value):
        return (action_value - self.lower_bound) / (self.upper_bound - self.lower_bound)
    
    def _get_obs(self):
        # 基础观测值：传感器读数
        sensor_readings = np.array([sensor(self.state, self.t) for sensor in self.sensors], dtype=np.float32)
        
        # 动态构建 obs 字典，根据 observation_space 中的键
        obs = {}
        
        # 遍历 observation_space 中定义的所有键
        for key in self.observation_space.spaces.keys():
            if key == "sensor_readings":
                obs[key] = sensor_readings
                
            elif key == "heat_readings":
                # 计算热读数
                heat_readings = np.array([equipment(self.t)["heat"] for equipment in self.equipments], dtype=np.float32)
                obs[key] = heat_readings
                
            elif key == "action_temp":
                # 计算当前动作温度
                current_switch_sign = np.where(self.current_action["switch"] < 0.5, 0, 1)
                current_action_temp = self._action_value_to_temp(self.current_action["value"])
                off_coolers_mask = current_switch_sign == 0
                current_action_temp[off_coolers_mask] = -1.0
                obs[key] = current_action_temp
                
            elif key == "switch_action":
                # 计算开关动作观测
                last_switch_sign = np.where(self.last_action["switch"] < 0.5, 0, 1)
                last_switch_time = (self.t - self.cooler_last_switch_time)
                switch_obs = np.zeros(last_switch_time.shape, dtype=np.float32)
                switch_obs[(last_switch_time < 1800) & 
                        (self.cooler_last_switch_time > self.start_time)] = 1.0
                switch_obs[(last_switch_sign == 1) & (last_switch_time > 172800)] = 2.0
                obs[key] = switch_obs
            elif key == "timestep":
                # 增加时间步 t
                obs[key] = np.array([self.t/(self.iter_per_step * self.sec_per_iter)], dtype=np.int32)

        
        return obs
    
    def get_current_obs(self):
        return self.current_obs

    def _normalize_obs(self, obs):
        if self.include_switch_in_observation:
            n_sensor = len(self.sensors)
            obs[:n_sensor] = np.clip(obs[:n_sensor], 10, 50)
            obs[:n_sensor] = (obs[:n_sensor] - 30.0) / 20.0 # [-1,1]
            return obs
        return obs
    
    def set_return_normilized_obs(self, return_normilized_obs):
        self.return_normilized_obs = return_normilized_obs


    def _get_state(self):
        return np.copy(self.state)

    def _get_info(self):
        return {"state": self._get_state(), 
                "time": self.t, 
                "topology_cooler": np.copy(self.cooler_topology), "topology_cooler_sensor":np.copy(self.cooler_sensor_topology)}

    def set_random_start_t(self, random_start_t):
        self.random_start_t = random_start_t

    def set_generate_record(self, generate_record):
        self.generate_record = generate_record

    def set_overheat_no_termiated_training_only(self, overheat_no_termiated_training_only):
        self.overheat_no_termiated_training_only = overheat_no_termiated_training_only

    def _save_cooler_state_history(self):
        """保存当前 cooler 状态到历史队列"""
        self.cooler_state_history['current_rest_cooler_idx'].append(
            self.current_rest_cooler_idx
        )
        
        if len(self.cooler_state_history['current_rest_cooler_idx']) > self.history_length:
            for key in self.cooler_state_history:
                self.cooler_state_history[key].pop(0)

    def _restore_cooler_state_history(self, steps_back=100):
        """从历史队列恢复 cooler 状态"""
        history_len = len(self.cooler_state_history['current_rest_cooler_idx'])
        if history_len > 0:
            idx = max(0, history_len - steps_back)
            self.current_rest_cooler_idx = self.cooler_state_history['current_rest_cooler_idx'][idx]
            self.cooler_rest_start_time = np.zeros(self.n_coolers)
            self.cooler_rest_start_time[self.current_rest_cooler_idx] = self.t
            self.last_action["switch"] = np.ones(self.n_coolers, dtype=np.int8)
            self.current_action["switch"] = np.ones(self.n_coolers, dtype=np.int8)
            self.last_action["switch"][self.current_rest_cooler_idx] = int(0)
            self.current_action["switch"][self.current_rest_cooler_idx] = int(0)

    def _reset_cooler_state_history(self):
        """
        重置 cooler 状态历史队列
        
        参数:
            reset_values: bool, 默认 True
                True: 同时重置 cooler_last_switch_time, cooler_rest_start_time, cooler_last_state 为零
                False: 只清空历史队列，保留当前值
        """
        # 清空历史队列
        self.cooler_state_history = {
            'current_rest_cooler_idx': [],
            # 'cooler_rest_start_time': [],
            # 'last_action_switch': [],
            # 'current_action_switch': []
        }

    def reset(self, *args, **kwargs):
        self.state = np.full((self.n_width, self.n_length), self.ambient_temp)
        # Add some initial noise
        self.state = self.state + np.random.normal(0, 2.0, (self.n_width, self.n_length)) 
        self.max_t = int(self.max_steps * self.iter_per_step * self.sec_per_iter)
        if self.random_start_t:
            if self.generate_record:
                self.t = np.random.randint(0, self.max_steps) * self.iter_per_step * self.sec_per_iter
                self.start_time = self.t
                self._reset_cooler_state_history()
            elif self.last_fail_timestemp > 0 and self.t < self.max_t:
                steps_back= 100
                self.t = max(0, self.last_fail_timestemp - steps_back * self.iter_per_step * self.sec_per_iter)
                self.start_time = self.t
                self._need_restore_cooler_state = True
            else:
                self._reset_cooler_state_history() 
                if np.random.random() < 0.5:
                    # 方法1：累积分布逆函数
                    u = np.random.random()
                    start_ratio = 1 - np.sqrt(1 - u)
                    self.t = int(start_ratio * self.max_t)
                else:
                    # 方法2：权重采样（1/(s+1)^2）
                    weights = 1.0 / ((np.arange(self.max_t + 1) + 1) ** 2)
                    weights = weights / weights.sum()
                    self.t = np.random.choice(np.arange(self.max_t + 1), p=weights)
                self.start_time = self.t
        else:
            self.t = 0.0
            self.start_time = self.t

        self.episode_step = 0
        self.warning_count = 0
        self.current_heat_power = -1

        self.n_coolers = len(self.coolers)
        self.n_sensor = len(self.sensors)
        self.cooler_last_switch_time = np.zeros(self.n_coolers)
        self.cooler_rest_start_time = np.zeros(self.n_coolers)
        self.cooler_last_state = np.zeros(self.n_coolers)

        if self.control_type.lower() == 'temperature':
            self.default_action_value = self._action_temp_to_value(self.target_temperature)
            
            self.last_action = {
                "switch": np.ones(self.n_coolers, dtype=np.int8),
                "value": np.full(self.n_coolers, self.default_action_value, dtype=np.float32)
            }
            self.current_action = {
                "switch": np.ones(self.n_coolers, dtype=np.int8),
                "value": np.full(self.n_coolers, self.default_action_value, dtype=np.float32)
            }

            if hasattr(self, '_need_restore_cooler_state') and self._need_restore_cooler_state:
                self._restore_cooler_state_history(steps_back=100)
                self._need_restore_cooler_state = False

            if self.no_switch_action and not self.cooler_state_history['current_rest_cooler_idx']:
                self.current_rest_cooler_idx = 0
                self.cooler_rest_start_time[self.current_rest_cooler_idx] = self.t
                self.last_action["switch"] = np.ones(self.n_coolers, dtype=np.int8)
                self.current_action["switch"] = np.ones(self.n_coolers, dtype=np.int8)
                self.last_action["switch"][self.current_rest_cooler_idx] = int(0)
                self.current_action["switch"][self.current_rest_cooler_idx] = int(0)
        elif self.control_type.lower() == 'power':
            self.last_action = {"switch": np.array([0]), "value": np.array([0.0])}
            self.current_action = {"switch": np.array([0]), "value": np.array([0.0])}

        observation = self._get_obs()

        for i, cooler in enumerate(self.coolers):
            if hasattr(cooler, 'reset') and callable(cooler.reset):
                cooler.reset()

        return observation, self._get_info()
    
    def set_control_type(self, control_type):
        if(control_type.lower() == 'temperature' or control_type.lower() == 'power'):
            self.control_type = control_type
            for i, cooler in enumerate(self.coolers):
                cooler.set_control_type(control_type)
            print("control type set to: ", control_type)
        else:
            raise Exception(f"Unknown control type: {self.control_type}")
    
    def action_transfer(self, action):
        if(self.control_type.lower() == 'temperature'):
            return self._action_value_to_temp(np.clip(action["value"], 0.0, 1.0))
        elif(self.control_type.lower() == 'power'):
            return np.clip(action["value"], 0.0, 1.0)
        else:
            raise Exception(f"Unknown control type: {self.control_type}")

    def update_states(self, action, dt=0.1, n=600):
        if ('state' not in self.__dict__):
            raise Exception('Must call reset before step')

        static_chtc_array = np.copy(self.convection_coeffs)
        static_heat = np.zeros((self.n_width, self.n_length))
        equip_heat = []
        energy_costs = np.zeros(len(self.coolers), dtype=np.float32)
        cell_area = self.cell_size * self.cell_size
        for i, equipment in enumerate(self.equipments):

            eff = equipment(self.t)
            static_heat += eff["delta_energy"]
            static_chtc_array += eff["delta_chtc"]
            equip_heat.append(eff["heat"])

        # Heat convection
        # (nw + 1) * nl
        for i in range(n):
            net_heat = np.copy(static_heat)
            net_chtc = np.copy(static_chtc_array)
            cooler_control = self.action_transfer(action)
            for i, cooler in enumerate(self.coolers):
                eff = cooler(action["switch"][i], cooler_control[i], self.t,
                             building_state=self.state,
                             ambient_state=self.ambient_temp)
                net_heat += eff["delta_energy"]
                net_chtc += eff["delta_chtc"]
                energy_costs[i] += eff["power"] * dt
            state_exp = np.full((self.n_width + 2, self.n_length + 2), self.ambient_temp)
            state_exp[1:-1, 1:-1] = self.state
            horizontal = - (state_exp[1:, 1:-1] - state_exp[:-1, 1:-1]) * net_chtc[:, :-1, 0] * self.csa
            # nw * (nl + 1)
            vertical = - (state_exp[1:-1, 1:] - state_exp[1:-1, :-1]) * net_chtc[:-1, :, 1] * self.csa

            # calculate the heat transfer at ceil and floor
            floor_ceil_transfer = self.floorceil_chtc * cell_area * (self.ambient_temp - self.state)

            net_in = (horizontal[:-1, :] - horizontal[1:, :]) + (vertical[:, :-1] - vertical[:, 1:]) + floor_ceil_transfer

            self.state += (net_heat + net_in) / self.heat_capacity * dt
            self.t += dt
        def custom_round(x):
            return int(x + 0.5) if x >= 0 else int(x - 0.5)
        self.t = custom_round(self.t)
            
        avg_power = energy_costs / (dt * n)

        return equip_heat, net_chtc, avg_power
    
    def reward(self, observation, action, power):
        # mode 0: include energy & target(+) & fail; 
        # mode 1: include target(+-) & fail; 
        # mode 2: include energy & fail

        obs_arr = observation["sensor_readings"]

        # temperature cost
        # Notice lower temperature is punished with energy automatically
        obs_dev = np.clip(obs_arr - self.target_temperature, 0.0, 8.0)
        # Modified huber loss to balance the loss of target at different temperature range
        target_loss = np.maximum(np.sqrt(obs_dev), obs_dev, obs_dev ** 2 / 8.0)
        target_cost = self.target_reward_wht * np.mean(target_loss)

        # switch cost
        if self.no_switch_action:
            switch_cost = 0.0
        else:
            switch_cost = 0.0
            for i in range(len(action["switch"])):
                duration_time = self.t - self.cooler_last_switch_time[i]
                if abs(action["switch"][i] - self.cooler_last_state[i]) > 0.5:
                    if duration_time < 1800 and self.cooler_last_switch_time[i] > 0:
                        deficit_time = 1800 - duration_time
                        switch_cost += 0.0002 * deficit_time
                    self.cooler_last_switch_time[i] = self.t
                    self.cooler_last_state[i] = action["switch"][i]
                elif duration_time > 172800 and self.cooler_last_state[i] > 0:
                    excess_time = duration_time - 172800
                    switch_cost += 0.0001 * excess_time
            switch_cost = (self.switch_reward_wht * switch_cost) / self.n_coolers

        # power cost, connected to the AVERAGE POWER of each cooler
        energy_cost = self.energy_reward_wht * (self.avg_total_heat/self.current_heat_power)*(power / 10000)

        if self.reward_mode == 1:
            energy_cost = energy_cost * 1.25
            target_cost = target_cost * 0.75
        elif self.reward_mode == 2:
            energy_cost = energy_cost * 1.5
            target_cost = target_cost * 0.5
            
        hard_loss = (obs_arr > self.failure_upperbound).any()
        overheat = 0
        over_tolerace = 0
        overheat_cost = 0
        if(hard_loss and self.episode_step > 5):
            self.warning_count += 1
            overheat = 1
            overheat_cost = self.overheat_reward
            self.warning_count = min(self.warning_count_tolerance + 1, self.warning_count)
        else:
            self.warning_count -= 1
            self.warning_count = max(self.warning_count, 0)

        # action cost
        if self.include_action_cost:
            action_temp = self._action_value_to_temp(action["value"])
            action_diff = action_temp - self.target_temperature
            def calculate_penalty(diff):
                if diff < -5:
                    return (diff + 5) ** 2
                else:
                    return 0.0
            action_cost = self.action_diff_wht * np.mean(np.vectorize(calculate_penalty)(action_diff))
        else:
            action_cost = 0.0

        info = {"over_heat": overheat,
                "over_tolerace": over_tolerace, 
                "energy_cost": energy_cost, 
                "target_cost": target_cost, 
                "switch_cost": switch_cost,
                "action_cost": action_cost}

        if(self.warning_count > self.warning_count_tolerance):
            info["over_tolerace"] = 1
            self.last_fail_timestemp = self.t
            return self.failure_reward, True, info
        
        self._save_cooler_state_history()

        return (self.base_reward + target_cost + switch_cost + energy_cost + action_cost + overheat_cost,  
                False, info)
    
    def _unflatten_action(self, action_flat):
        """Converts the flattened Box action back to the dictionary format."""
        if not isinstance(action_flat, np.ndarray):
            action_flat = np.array(action_flat)
        n_coolers = len(self.coolers)
        # Ensure action_flat has the correct shape
        if not self.no_switch_action:
            expected_shape = (n_coolers * 2,)

            if action_flat.shape != expected_shape:
    
                # Attempt to reshape if it's a single vector environment's output
                if action_flat.ndim == 1 and action_flat.size == expected_shape[0]:
                    action_flat = action_flat.reshape(expected_shape)
                # Handle potential batch dimension from vectorized environments
                elif action_flat.ndim == 2 and action_flat.shape[0] == 1 and action_flat.shape[1] == expected_shape[0]:
                    action_flat = action_flat.reshape(expected_shape)

                else:
                    raise ValueError(f"Received flattened action with unexpected shape {action_flat.shape}. Expected {expected_shape}.")
            
            switch_continuous = action_flat[:n_coolers]
            value_continuous = action_flat[n_coolers:]
        else:
            expected_shape = (n_coolers,)
            if action_flat.shape != expected_shape:
                if action_flat.ndim == 1 and action_flat.size == expected_shape[0]*2:
                    shape = (2*n_coolers,)
                    action_flat = action_flat.reshape(shape)[n_coolers:]

            switch_continuous = self._update_swith_action()
            value_continuous = action_flat[:n_coolers]

        # Threshold switch part (e.g., > 0.5 is ON)
        switch_binary = (switch_continuous > 0.5).astype(np.int8)

        # Clip value part to ensure it's within [0, 1] (Box space should handle bounds, but good practice)
        value_clipped = np.clip(value_continuous, 0.0, 1.0)

        action_dict = {
            "switch": switch_binary,
            "value": value_clipped.astype(np.float32)
        }
        return action_dict

    def _update_swith_action(self):
        current_switch = deepcopy(self.last_action["switch"])
        current_rest = self.current_rest_cooler_idx
        need_switch = (self.t - self.cooler_rest_start_time[current_rest]) > 3600
        if need_switch:
            current_switch[current_rest] = int(1)
            if current_rest + 1 > self.n_coolers - 1:
                next_rest = 0
            else:
                next_rest = current_rest + 1
            current_switch[next_rest] = int(0)
            self.current_rest_cooler_idx = next_rest
            self.cooler_rest_start_time[self.current_rest_cooler_idx] = self.t
        return current_switch

    def _set_default_action_value(self, action):
        switch_binary = action["switch"]
        value = action["value"].copy()
        off_indices = switch_binary < 0.5
        value[off_indices] = self.default_action_value
        action["value"] = value
        return action

    def step(self, action):

        if isinstance(self.action_space, Dict):

            action = action
        else:
            action = self._unflatten_action(action)

        if(self.control_type.lower() == 'temperature'):
            action = self._set_default_action_value(action)

        self.episode_step += 1
        equip_heat, chtc_array, powers = self.update_states(action, dt=self.sec_per_iter, n=self.iter_per_step)
        self.current_action = deepcopy(action)
        observation = self._get_obs()
        self.current_obs = observation

        # calculate average power of each cooler
        average_power = np.mean(powers)
        self.current_heat_power = np.sum(np.copy(equip_heat))

        reward, terminated, info = self.reward(observation, action, average_power)
        truncated = self.t >= self.max_t
        self.last_action = deepcopy(action)

        if self.return_normilized_obs:
            normalized_obs = self._normalize_obs(observation)
        else:
            normalized_obs = observation
        
        info.update(self._get_info())
        info.update({
                "last_control": deepcopy(self.last_action),
                "heat_power": np.copy(equip_heat),
                "chtc_array": np.copy(chtc_array),
                "cool_power": powers,
                })

        if self.verbose:
            # print("self.verbose", self.verbose)

            cool_power = round(np.sum(info.get("cool_power", 0)),4)
            heat_power = round(np.sum(info.get("heat_power", 0)),4)
            over_tolerace = info["over_tolerace"] if isinstance(info["over_tolerace"], numbers.Real) else 0
            info_total = f"energy_cost: {round(info.get('energy_cost', 0), 4)}, " \
                         f"target_cost: {round(info.get('target_cost', 0), 4)}, " \
                         f"switch_cost: {round(info.get('switch_cost', 0), 4)}, " \
                         f"action_cost: {round(info.get('action_cost', 0), 4)}, " \
                         f"cool_power: {cool_power}, heat_power: {heat_power}"
            print(f"Step {self.episode_step} | over_tolerace:{over_tolerace} | Reward: {reward} | {info_total} ", flush=True)
                
        if self.overheat_no_termiated_training_only:
            terminated = False

        return normalized_obs, reward, terminated, truncated, info

    def sample_action(self, mode="random"):
        n_coolers = len(self.coolers)
        if mode == "random":
            return self._random_action()
        elif mode == "max":
            sampled_action = np.concatenate((
                np.zeros(n_coolers, dtype=np.float32),  
                np.zeros(n_coolers, dtype=np.float32)  
            ))
            return sampled_action
        elif mode== "constant":
            action_value = self._action_temp_to_value(self.target_temperature)
            sampled_action = np.concatenate((
                np.full(n_coolers, action_value, dtype=np.float32),
                np.full(n_coolers, action_value, dtype=np.float32)
            ))
            return sampled_action
        elif mode == "constant_conservative":
            action_value = self._action_temp_to_value(self.target_temperature - 5)
            sampled_action = np.concatenate((
                np.full(n_coolers, action_value, dtype=np.float32),
                np.full(n_coolers, action_value, dtype=np.float32)
            ))
            return sampled_action
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _random_action(self):
        if isinstance(self.action_space, Dict):
            return self.action_space.sample()

    def stat(self, normalized_obs, terminated, info, reward):
        # power stat
        current_step = self.episode_step
        current_cool_power = round(np.sum(info.get("cool_power", 0)),4)
        self.avg_cooler_power_per_step = (self.avg_cooler_power_per_step * (current_step - 1) + current_cool_power) / current_step
        self.avg_reward = (self.avg_reward * (current_step - 1) + reward) / current_step
        # over heat percentage
        if self.return_normilized_obs:
            sensor_reading = self._denormalize_obs(normalized_obs)
        else:
            sensor_reading = normalized_obs["sensor_readings"]
        current_over_heat_reading = sensor_reading - self.target_temperature
        # overheat
        current_count_overheat = []
        current_count_overheat.append(np.sum(current_over_heat_reading > 0))
        current_count_overheat.append(np.sum(current_over_heat_reading > 2))
        current_count_overheat.append(np.sum(current_over_heat_reading > 4))
        current_count_overheat.append(np.sum(current_over_heat_reading > 6))
        last_counts_overheat = [
                p * (current_step - 1) * self.n_sensor 
                for p in self.over_heat_percentage
            ]
        self.over_heat_percentage = [
            (last + current) / (current_step * self.n_sensor)
            for last, current in zip(last_counts_overheat, current_count_overheat)
        ]
        # overcool
        current_over_cool = sensor_reading - self.target_temperature
        current_count_overcool = []
        current_count_overcool.append(np.sum(current_over_cool < 0))
        current_count_overcool.append(np.sum(current_over_cool < -2))
        current_count_overcool.append(np.sum(current_over_cool < -4))
        current_count_overcool.append(np.sum(current_over_cool < -6))
        last_counts_overcool = [
                p * (current_step - 1) * self.n_sensor 
                for p in self.over_cool_percentage
            ]
        self.over_cool_percentage = [
            (last + current) / (current_step * self.n_sensor)
            for last, current in zip(last_counts_overcool, current_count_overcool)
        ]
        # bad switch stat
        if self.include_switch_in_observation and not self.no_switch_action:
            switch_obs = normalized_obs[-self.n_coolers:]
            last_switch_sign = np.where(self.copy_last_action["switch"] < 0.5, 0, 1)
            current_switch_sign = np.where(self.current_action["switch"] < 0.5, 0, 1)
            condition_too_soon = (switch_obs > 0.5) & (switch_obs < 1.5) & (last_switch_sign != current_switch_sign)
            count_switch_too_soon = np.sum(condition_too_soon)
            count_switch_too_late = np.sum(switch_obs > 1.5)
            self.bad_switch_percentage[0] = (self.bad_switch_percentage[0] * self.n_coolers * (current_step - 1) 
                                             + count_switch_too_soon) / (self.n_coolers * current_step)
            self.bad_switch_percentage[1] = (self.bad_switch_percentage[1] * self.n_coolers * (current_step - 1) 
                                             + count_switch_too_late) / (self.n_coolers * current_step)
        # fail stat
        self.fail_percentage = (self.fail_percentage * (current_step - 1) + int(terminated)) / current_step

        # print
        if (self.verbose):
            print(f"Step {current_step}:")
            print(f"  Avg Reward: {self.avg_reward:.4f} per step")
            print(f"  Avg Cooler Power: {self.avg_cooler_power_per_step:.4f} per step")
            
            print("  Overheat Statistics:")
            print(f"    current > 0°C: {current_count_overheat[0]} sensors, total ({self.over_heat_percentage[0]*100:.2f}%)")
            print(f"    current > 2°C: {current_count_overheat[1]} sensors, total ({self.over_heat_percentage[1]*100:.2f}%)")
            print(f"    current > 4°C: {current_count_overheat[2]} sensors, total ({self.over_heat_percentage[2]*100:.2f}%)")
            print(f"    current > 6°C: {current_count_overheat[3]} sensors, total ({self.over_heat_percentage[3]*100:.2f}%)")
            
            print("  Overcool Statistics:")
            print(f"    current < 0°C: {current_count_overcool[0]} sensors, total ({self.over_cool_percentage[0]*100:.2f}%)")
            print(f"    current < -2°C: {current_count_overcool[1]} sensors, total ({self.over_cool_percentage[1]*100:.2f}%)")
            print(f"    current < -4°C: {current_count_overcool[2]} sensors, total ({self.over_cool_percentage[2]*100:.2f}%)")
            print(f"    current < -6°C: {current_count_overcool[3]} sensors, total ({self.over_cool_percentage[3]*100:.2f}%)")

            if self.include_switch_in_observation and not self.no_switch_action:
                print("  Switch Statistics:")
                print(f"    Too Soon: {count_switch_too_soon} coolers ({self.bad_switch_percentage[0]*100:.2f}%)")
                print(f"    Too Late: {count_switch_too_late} coolers ({self.bad_switch_percentage[1]*100:.2f}%)")
            
            print(f"  Fail Percentage: {self.fail_percentage*100:.2f}%")
            print("-" * 50)

        step_data = {
            "reward": reward,
            "cooler_power": current_cool_power,
            "cool_0": current_count_overcool[0],
            "cool_-2": current_count_overcool[1],
            "cool_-4": current_count_overcool[2],
            "cool_-6": current_count_overcool[3],
            "hot_0": current_count_overheat[0],
            "hot_2": current_count_overheat[1],
            "hot_4": current_count_overheat[2],
            "hot_6": current_count_overheat[3],
            "fail": int(terminated)
        }
        return step_data


class HVACEnvDiscreteAction(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_resulotion_temp = 0.1
        
    def _discretize_action(self, action):
        """Discretize the action value to make it a multiple of the resolution."""
        if isinstance(self.action_space, gym.spaces.Dict):
            discretized_action = deepcopy(action)
            if 'value' in discretized_action:
                temp_value = self._action_value_to_temp(action['value'])
                discretized_temp = np.round(temp_value / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
                discretized_action['value'] = np.clip(
                    self._action_temp_to_value(discretized_temp),
                    0.0, 1.0
                )
            return discretized_action
        else:
            n_coolers = len(self.coolers)
            if not self.no_switch_action:
                switch_part = action[:n_coolers]
                value_part = action[n_coolers:]
            else:
                value_part = action[:n_coolers]
            
            temp_value = self._action_value_to_temp(value_part)
            discretized_temp = np.round(temp_value / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
            discretized_value = np.clip(
                self._action_temp_to_value(discretized_temp),
                0.0, 1.0
            )
            if not self.no_switch_action:
                return np.concatenate([switch_part, discretized_value])
            else:
                return discretized_value
    
    def step(self, action):
        discretized_action = self._discretize_action(action)

        self.copy_last_action = self.last_action.copy()
        normalized_obs, reward, terminated, truncated, info = super().step(discretized_action)

        step_stat = self.stat(normalized_obs, terminated, info, reward)
        info["step_stat"] = step_stat

        return normalized_obs, reward, terminated, truncated, info
    
class HVACEnvDiffAction(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_resulotion_temp = 0.1

        min_temp = -3.0
        max_temp = 3.0
        action_diff_resulotion = 0.5
        self.num_steps = int((max_temp - min_temp) / action_diff_resulotion) + 1
        self.discrete_values = np.linspace(min_temp, max_temp, self.num_steps)
        self.target_temp_offset = 3

        self.discrete_rl_action_space = False

        # 新增：动作历史缓冲区，用于滑动窗口 cost 计算
        self.action_history = []  # 存储 (step, abs_values_sum) 元组
        self.window_len = 10  # 默认窗口长度
        self.n_threshold = 5  # 默认阈值
 
    def set_too_cold_limit(self, too_cold_limit):
        self.too_cold_limit = too_cold_limit

    def set_task(self, task, discretize_rl_action_space=False, add_action_cost=False, too_cold_limit=True):
        """
        设置任务配置
        
        参数:
            task: 任务配置字典
            discretize_rl_action_space: 是否将 RL 的动作空间离散化
        """
        # 调用父类方法完成基础设置
        super().set_task(task)
        
        # 保存离散化标志
        self.discrete_rl_action_space = discretize_rl_action_space
        self.add_action_cost = add_action_cost
        self.too_cold_limit = too_cold_limit
        
        # 如果需要离散化，重新声明动作空间
        if discretize_rl_action_space:
            n_coolers = len(self.coolers)
            n_actions = len(self.discrete_values)
            
            if self.action_space_format == 'dict':
                # Dict 格式：switch (MultiBinary) + value (MultiDiscrete)
                self.action_space = gym.spaces.Dict({
                    "switch": gym.spaces.MultiBinary(n_coolers),
                    "value": gym.spaces.MultiDiscrete([n_actions] * n_coolers)
                })
            else:
                # Flat 格式
                if not self.no_switch_action:
                    # switch (n_coolers 个 0/1) + value (n_coolers 个离散值)
                    # 使用 MultiDiscrete 表示两个部分
                    # 注意：MultiDiscrete 只能有一个 nvec，所以需要合并
                    # 方案1：使用 Dict（推荐，更清晰）
                    self.action_space = gym.spaces.Dict({
                        "switch": gym.spaces.MultiBinary(n_coolers),
                        "value": gym.spaces.MultiDiscrete([n_actions] * n_coolers)
                    })
                    
                else:
                    # 只有 value 部分：n_coolers 个离散值
                    self.action_space = gym.spaces.MultiDiscrete([n_actions] * n_coolers)
            
            # print(f"[INFO] RL action space 已离散化: {self.action_space}")
            # print(f"  - 每个 cooler 有 {self.num_steps} 个离散选择")
            # print(f"  - 动作空间类型: {type(self.action_space)}")
    
    def _diff_action(self, action):
        """
        将动作转换为温度差值
        
        支持三种输入格式：
        1. 连续 Box 空间（0-1），需要转换为离散索引
        2. Dict 格式：{'switch': array, 'value': MultiDiscrete 数组}
        3. MultiDiscrete：整数数组，shape=(n_coolers,)
        """
        n_coolers = len(self.coolers)
        
        if isinstance(self.action_space, gym.spaces.MultiDiscrete) and np.issubdtype(action.dtype, np.integer):
            # MultiDiscrete 格式：直接是离散索引数组
            indices = action.astype(int)
            discrete_diff_value = self.discrete_values[indices]
            
            # 计算当前温度
            last_temp = self._action_value_to_temp(self.last_action['value'])
            current_temp = last_temp + discrete_diff_value
            
            # 温度限制
            if self.too_cold_limit:
                too_cold_mask = current_temp < (self.target_temperature - self.target_temp_offset)
                current_temp[too_cold_mask] = self.target_temperature - self.target_temp_offset
            
            # 离散化
            discretized_current_temp = np.round(current_temp / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
            current_action = np.clip(
                self._action_temp_to_value(discretized_current_temp),
                0.0, 1.0
            )
            
            # 构建返回值
            if not self.no_switch_action:
                switch_part = np.ones(n_coolers)  # 默认全部开启
                return np.concatenate([switch_part, current_action]), np.concatenate([switch_part, discrete_diff_value])
            else:
                return current_action, discrete_diff_value
        
        elif isinstance(self.action_space, gym.spaces.Dict) and isinstance(action, dict):
            # Dict 格式
            discretized_action = deepcopy(action)
            rl_origin_action = deepcopy(action)
            
            # 获取 switch 部分（如果有）
            if not self.no_switch_action and 'switch' in action:
                switch_part = action['switch']
            else:
                switch_part = np.ones(n_coolers)  # 默认全部开启
            
            # 获取 value 部分
            value_action = action['value']
            
            # 判断 value 的格式
            if isinstance(value_action, np.ndarray):
                # MultiDiscrete 或连续 Box 输出
                if self.discrete_rl_action_space:
                    # 输入是离散索引，直接使用
                    indices = value_action.astype(int)
                else:
                    # 输入是连续值，需要转换为离散索引
                    indices = np.clip(np.round(value_action * (self.num_steps - 1)), 0, self.num_steps - 1).astype(int)
                
                # 将索引转换为温度差值
                discrete_diff_value = self.discrete_values[indices]
                rl_origin_action['value'] = discrete_diff_value
                
                # 计算当前温度
                last_temp = self._action_value_to_temp(self.last_action['value'])
                current_temp = last_temp + discrete_diff_value
                
                # 温度限制
                if self.too_cold_limit:
                    too_cold_mask = current_temp < (self.target_temperature - self.target_temp_offset)
                    current_temp[too_cold_mask] = self.target_temperature - self.target_temp_offset
                
                # 离散化
                discretized_current_temp = np.round(current_temp / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
                current_action = np.clip(
                    self._action_temp_to_value(discretized_current_temp),
                    0.0, 1.0
                )
                
                discretized_action['value'] = current_action
                if not self.no_switch_action:
                    discretized_action['switch'] = switch_part
                
                return discretized_action, rl_origin_action
            else:
                raise ValueError(f"Unsupported value format: {type(value_action)}")
        else:
            n_coolers = len(self.coolers)
            if not self.no_switch_action:
                switch_part = action[:n_coolers]
                value_part = action[n_coolers:]
            else:
                value_part = action[:n_coolers]
            indices = np.clip(np.round(value_part * (self.num_steps - 1)), 0, self.num_steps - 1).astype(int)
            discrete_diff_value = self.discrete_values[indices]
            last_temp = self._action_value_to_temp(self.last_action['value'])
            current_temp = last_temp + discrete_diff_value
            # Set lowest control temp to target temp - 3
            if self.too_cold_limit:
                too_cold_mask = current_temp < (self.target_temperature - 3)
                current_temp[too_cold_mask] = self.target_temperature - 3
            discretized_current_temp = np.round(current_temp / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
            current_action = np.clip(
                self._action_temp_to_value(discretized_current_temp),
                0.0, 1.0
            )
            if not self.no_switch_action:
                return np.concatenate([switch_part, current_action]), np.concatenate([switch_part, discrete_diff_value])
            else:
                return current_action, discrete_diff_value
        
    def step(self, action, call_diff_action=True):
        if call_diff_action:
            discretized_action, rl_origin_action = self._diff_action(action)
        else:
            discretized_action = action
        self.copy_last_action = self.last_action.copy()
        normalized_obs, reward, terminated, truncated, info = super().step(discretized_action)

        if self.add_action_cost:
            action_cost = self._calculate_simplified_action_cost(rl_origin_action)
            action_cost_weight = -1
            reward = reward + action_cost_weight * action_cost
            info["action_cost"] = action_cost_weight * action_cost
            if self.verbose:
                print(f"Action cost: {action_cost_weight * action_cost}")

        step_stat = self.stat(normalized_obs, terminated, info, reward)
        info["step_stat"] = step_stat

        return normalized_obs, reward, terminated, truncated, info
    
    def _calculate_action_cost(self, rl_origin_action):
        """
        计算动作成本，鼓励动作平滑（不要波动太大）
        
        参数:
            rl_origin_action: 原始动作（来自 RL 的输出）
            
        返回:
            float: 动作成本的平均值（每个维度的平方绝对值的平均）
        """
        # 提取动作值
        if isinstance(rl_origin_action, dict):
            # 如果是 dict，提取 'value' 部分
            if 'value' in rl_origin_action:
                action_values = rl_origin_action['value']
            else:
                # 如果没有 'value' 键，提取所有值
                action_values = np.array(list(rl_origin_action.values()))
        elif isinstance(rl_origin_action, (list, tuple, np.ndarray)):
            # 如果是数组或列表，直接使用
            action_values = np.array(rl_origin_action)
        else:
            # 如果是标量
            action_values = np.array([rl_origin_action])
        
        # 确保是 numpy 数组并展平
        action_values = np.array(action_values).flatten()
        
        # 计算每个维度的成本（绝对值的平方）
        # 动作越大（无论正负），成本越高
        abs_values = np.abs(action_values)
        action_costs = np.where(
            abs_values <= 1.0,
            abs_values,           # 线性部分：|x|
            np.square(abs_values)  # 平方部分：x²
        )
        
        # 返回所有维度成本的平均值
        total_cost = np.mean(action_costs)
        
        return float(total_cost)
        
    def _calculate_window_action_cost(self, rl_origin_action):
        """
        计算滑动窗口动作成本（每个 agent 独立维护窗口）
        
        参数:
            rl_origin_action: 当前 timestep 的原始动作，shape=(n_agents,)
            
        返回:
            float: 所有 agent 的平均窗口动作成本
        """
        if rl_origin_action is None:
            return 0.0
        
        # 提取动作值
        if isinstance(rl_origin_action, dict):
            if 'value' in rl_origin_action:
                action_values = rl_origin_action['value']
            else:
                action_values = np.array(list(rl_origin_action.values()))
        elif isinstance(rl_origin_action, (list, tuple, np.ndarray)):
            action_values = np.array(rl_origin_action)
        else:
            action_values = np.array([rl_origin_action])
        
        # 确保是 numpy 数组并展平
        action_values = np.array(action_values).flatten()
        n_agents = len(action_values)
        
        # 确保 action_history 的长度与 n_agents 匹配
        if len(self.action_history) != n_agents:
            self.action_history = [[] for _ in range(n_agents)]
        
        # 计算每个 agent 的绝对值并更新其历史
        abs_values = np.abs(action_values)
        for i in range(n_agents):
            self.action_history[i].append(abs_values[i])
            
            # 只保留最近 window_len 个记录
            if len(self.action_history[i]) > self.window_len:
                self.action_history[i] = self.action_history[i][-self.window_len:]
        
        # 计算每个 agent 的 cost
        agent_costs = []
        for i in range(n_agents):
            # 检查窗口是否已满
            if len(self.action_history[i]) < self.window_len:
                agent_costs.append(0.0)
                continue
            
            # 统计窗口内非零动作的数量
            non_zero_count = sum(1 for val in self.action_history[i] if val > 1e-6)
            
            # 如果非零动作数大于阈值，计算窗口内所有值的总和
            if non_zero_count > self.n_threshold:
                total_abs_sum = sum(val for val in self.action_history[i] if val > 1e-6)
                agent_costs.append(float(total_abs_sum))
            else:
                agent_costs.append(0.0)
        
        # 返回所有 agent 的平均 cost
        return np.mean(agent_costs)
    
    def _calculate_window_fluctuation_penalty(self, rl_origin_action):
        """
        仅计算滑动窗口内的动作来回波动惩罚（核心：只惩罚反向动作，不限制非零）
        
        参数:
            rl_origin_action: 当前 timestep 的原始动作，shape=(n_agents,)
            
        返回:
            float: 所有 agent 的平均波动惩罚值（越大表示波动越严重）
        """
        if rl_origin_action is None:
            return 0.0
        
        # 1. 提取并标准化动作值（保留原始逻辑）
        if isinstance(rl_origin_action, dict):
            action_values = rl_origin_action['value'] if 'value' in rl_origin_action else np.array(list(rl_origin_action.values()))
        elif isinstance(rl_origin_action, (list, tuple, np.ndarray)):
            action_values = np.array(rl_origin_action)
        else:
            action_values = np.array([rl_origin_action])
        action_values = np.array(action_values).flatten()
        n_agents = len(action_values)
        
        # 2. 初始化/对齐每个agent的动作历史（存原始动作，用于判断方向）
        if len(self.action_history) != n_agents:
            self.action_history = [[] for _ in range(n_agents)]
        
        # 3. 遍历每个agent，统计窗口内的反向波动次数
        agent_fluctuation_penalty = []
        for i in range(n_agents):
            # 更新当前agent的动作历史（存原始值，不是绝对值）
            self.action_history[i].append(action_values[i])
            # 只保留最近window_len步（聚焦短期波动）
            if len(self.action_history[i]) > self.window_len:
                self.action_history[i] = self.action_history[i][-self.window_len:]
            
            window_filled_len = len(self.action_history[i])
            # 窗口至少有2步才可能有波动，否则惩罚为0
            if window_filled_len < 2:
                agent_fluctuation_penalty.append(0.0)
                continue
            
            # 核心：统计窗口内“非零反向动作”的次数（排除0的干扰）
            reverse_count = 0
            for j in range(1, window_filled_len):
                prev_action = self.action_history[i][j-1]
                curr_action = self.action_history[i][j]
                
                # 仅当两步动作都非零，且符号相反时，才算波动
                if (abs(prev_action) > 1e-6) and (abs(curr_action) > 1e-6) and (prev_action * curr_action < 0):
                    reverse_count += 1
            
            # 波动惩罚：反向次数 / 窗口长度（归一化，避免窗口长短影响）
            # 系数可调（比如2.0，让波动惩罚更明显）
            fluctuation_penalty = (reverse_count / window_filled_len)
            agent_fluctuation_penalty.append(float(fluctuation_penalty))
        
        # 4. 返回所有agent的平均波动惩罚
        return np.mean(agent_fluctuation_penalty)

    def _calculate_simplified_action_cost(self, rl_origin_action):
        """
        极简版动作成本设计：
        - 仅包含2个维度：数值化非零动作cost + 反向波动惩罚
        - 去掉无意义非零/连续单向惩罚，非零cost直接用动作数值计算
        - 无任何温度依赖，仅基于动作本身
        """
        if rl_origin_action is None:
            return 0.0
        
        # 1. 提取并标准化增量动作值（仅处理动作本身）
        if isinstance(rl_origin_action, dict):
            action_values = rl_origin_action['value'] if 'value' in rl_origin_action else np.array(list(rl_origin_action.values()))
        elif isinstance(rl_origin_action, (list, tuple, np.ndarray)):
            action_values = np.array(rl_origin_action)
        else:
            action_values = np.array([rl_origin_action])
        action_values = np.array(action_values).flatten()
        n_agents = len(action_values)
        
        # 2. 初始化动作历史（仅存动作值，无温度，用于计算反向波动）
        if len(self.action_history) != n_agents:
            self.action_history = [[] for _ in range(n_agents)]  # 每个agent独立的动作历史
        window_len = self.window_len  # 建议设为3~5步（聚焦短期反向波动）
        
        # 3. 逐agent计算成本（仅两个核心维度）
        agent_total_cost = []
        for i in range(n_agents):
            curr_delta = action_values[i]  # 当前增量动作（±1/±2/±3/0）
            
            # 更新动作历史（仅保留最近window_len步）
            self.action_history[i].append(curr_delta)
            if len(self.action_history[i]) > window_len:
                self.action_history[i] = self.action_history[i][-window_len:]
            hist = self.action_history[i]
            hist_len = len(hist)
            
            # ========== 维度1：数值化非零动作cost（核心，直接用数值计算） ==========
            # 设计逻辑：0动作cost=0；非零动作cost=动作绝对值 × 系数（可线性/非线性）
            # 线性计算（简单易调，推荐）：abs(curr_delta) * 惩罚系数
            # non_zero_cost = abs(curr_delta) * 0.4  # 系数0.4可微调，比如0.3/0.5
            # （可选）非线性计算（更惩罚大幅动作）：abs(curr_delta)**1.2 * 0.3
            non_zero_cost = (abs(curr_delta) ** 1.2) * 0.3
            
            # ========== 维度2：反向波动惩罚（仅保留，避免来回动） ==========
            fluctuation_penalty = 0.0
            if hist_len >= 2:  # 至少2步才可能有反向波动
                reverse_count = 0
                for j in range(1, hist_len):
                    prev_d = hist[j-1]
                    curr_d_j = hist[j]
                    # 仅惩罚非零反向动作（+2→-1、-3→+2等无意义波动）
                    if prev_d * curr_d_j < 0 and prev_d != 0 and curr_d_j != 0:
                        reverse_count += 1
                # 反向波动惩罚：反向次数 × 系数（与非零cost量级匹配）
                fluctuation_penalty = reverse_count * 0.6
            
            # ========== 总成本合并（两个维度等权重，可微调） ==========
            total_cost = 0.5 * non_zero_cost + 0.5 * fluctuation_penalty
            agent_total_cost.append(float(total_cost))
        
        # 4. 返回所有agent的平均成本（确保非负）
        avg_cost = np.mean(agent_total_cost)
        return max(avg_cost, 0.0)


    def _denormalize_obs(self, normalized_obs):
        if self.include_switch_in_observation:
            
            denormalized_obs = normalized_obs.copy()
            
            denormalized_obs = normalized_obs["sensor_readings"] * 20.0 + 30.0
            
            return denormalized_obs
        return normalized_obs
    
    def reset(self, *args, **kwargs):
        self.action_history = []
        return super().reset(*args, **kwargs)