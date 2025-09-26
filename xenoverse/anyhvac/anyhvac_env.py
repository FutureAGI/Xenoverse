import sys
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy
import numbers
import numpy as np
from copy import deepcopy

class HVACEnv(gym.Env):
    def __init__(self,
                 max_steps=20160,  # sec_per_iter * iter_per_step * max_steps / 86400 days
                 iter_per_step=150,
                 sec_per_iter=0.2,
                 set_lower_bound=16,
                 set_upper_bound=32,
                 verbose=False,
                 action_space_format='box',
                 include_heat_in_observation=False,
                 include_switch_in_observation=True,
                 include_last_action_in_observation=True,
                 no_switch_action=True,
                 reward_mode = 0 # mode 0: include energy & target(+) & fail; 
                                 # mode 1: include target(+-) & fail; 
                                 # mode 2: include energy & fail
                 ):
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1,), dtype=numpy.float32)
        self.action_space = None
        self.max_steps = max_steps
        self.failure_reward = -5
        self.overheat_reward = -1
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
        self.action_space_format = action_space_format
        self.include_heat_in_observation = include_heat_in_observation
        self.include_switch_in_observation = include_switch_in_observation
        self.include_last_action_in_observation = include_last_action_in_observation
        self.include_action_cost = False
        self.no_switch_action = no_switch_action
        self.return_normilized_obs = False
        self.random_start_t = False
        self.reward_mode = reward_mode
        # stat
        self.avg_cooler_power_per_step = 0.0
        self.avg_reward = 0.0
        self.over_heat_percentage = [0.0, 0.0, 0.0, 0.0] # percentage over 0, 2, 4, 6 degree
        self.over_cool_percentage = [0.0, 0.0, 0.0, 0.0] # percentage lower 0, -2, -4, -6 degree
        self.bad_switch_percentage = [0.0, 0.0] #  below boundary, over boundary
        self.fail_percentage = 0.0

    def set_task(self, task):
        for key in task:
            self.__dict__[key] = task[key]
        self.task_set = True
        self.heat_capacity = task.get('heat_capacity', []) 
        self.equipments = task.get('equipments', [])

        # triggers failure above this temperature
        self.failure_upperbound = numpy.mean(self.target_temperature + 10)
        
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
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(2*n_coolers,), dtype=numpy.float32) # Placeholder shape
            else:
                self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_coolers,), dtype=numpy.float32)

        self.cooler_topology = numpy.zeros((n_coolers, n_coolers))
        self.cooler_sensor_topology = numpy.zeros((n_coolers, n_sensors))
        for i,cooler_i in enumerate(self.coolers):
             for j,cooler_j in enumerate(self.coolers):
                  if (i > j):
                      self.cooler_topology[i,j] = numpy.sqrt(numpy.sum((cooler_i.loc - cooler_j.loc) ** 2))
        for i in range(n_coolers):
              for j in range(i + 1, n_coolers):
                   self.cooler_topology[i, j] = self.cooler_topology[j, i]
        for i,cooler in enumerate(self.coolers):
           for j,sensor in enumerate(self.sensors):
                self.cooler_sensor_topology[i, j] = numpy.sqrt(numpy.sum((cooler.loc - sensor.loc) ** 2))

        # calculate cross sectional area
        self.csa = self.cell_size * self.floor_height

        if self.include_heat_in_observation:
            obs_shape_dim = n_sensors
            obs_shape_dim += n_heaters
            low_bounds = np.full(obs_shape_dim, -273.0, dtype=np.float32)
            high_bounds = np.full(obs_shape_dim, 273.0, dtype=np.float32)
            low_bounds[n_sensors:] = 0.0
            high_bounds[n_sensors:] = 80000.0 
            self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, shape=(obs_shape_dim,), dtype=numpy.float32)
        elif self.no_switch_action and self.include_last_action_in_observation:
            obs_shape_dim = n_sensors + n_coolers
            self.observation_space = gym.spaces.Box(low=-1, high=50, shape=(obs_shape_dim,), dtype=numpy.float32)
        elif self.include_switch_in_observation and not self.include_last_action_in_observation:
            obs_shape_dim = n_sensors + n_coolers
            if self.return_normilized_obs:     
                self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(obs_shape_dim,), dtype=numpy.float32)
            else:
                self.observation_space = gym.spaces.Box(low=-1, high=50, shape=(obs_shape_dim,), dtype=numpy.float32)
        elif self.include_switch_in_observation and self.include_last_action_in_observation:
            obs_shape_dim = n_sensors + 2 * n_coolers
            self.observation_space = gym.spaces.Box(low=-1, high=50, shape=(obs_shape_dim,), dtype=numpy.float32)
        else:
            # observation space
            self.observation_space = gym.spaces.Box(low=10, high=50, shape=(n_sensors,), dtype=numpy.float32)

    def _get_obs(self):

        heater_progress = []
        sensor_readings = np.array([sensor(self.state, self.t) for sensor in self.sensors], dtype=np.float32)

        if self.include_heat_in_observation:

            static_chtc_array = numpy.copy(self.convection_coeffs)
            static_heat = numpy.zeros((self.n_width, self.n_length))
            equip_heat = []

            for i, equipment in enumerate(self.equipments):
                
                eff = equipment(self.t)  

                static_heat += eff["delta_energy"]
                static_chtc_array += eff["delta_chtc"]
                equip_heat.append(eff["heat"])

                normalized_episode_progress = np.clip(eff["heat"], 0.0, 30000.0)
                heater_progress.append(normalized_episode_progress) 

            heat_readings = np.array(heater_progress, dtype=np.float32)
            return np.concatenate((sensor_readings, heat_readings))

        elif self.no_switch_action and self.include_last_action_in_observation:
            current_switch_sign = numpy.where(self.current_action["switch"] < 0.5, 0, 1)
            current_action_temp = self.current_action["value"] * (self.upper_bound - self.lower_bound) + self.lower_bound
            off_coolers_mask = current_switch_sign == 0
            current_action_temp[off_coolers_mask] = -1.0
            return  np.concatenate((sensor_readings, current_action_temp))
        
        elif self.include_switch_in_observation and not self.include_last_action_in_observation:
            last_switch_sign = numpy.where(self.last_action["switch"] < 0.5, 0, 1)
            last_switch_time = (self.t - self.cooler_last_switch_time)
            switch_obs =  numpy.zeros(last_switch_time.shape, dtype=numpy.float32)
            switch_obs[(last_switch_time < 1800) & 
                       (self.cooler_last_switch_time > self.start_time)] = 1.0
            switch_obs[(last_switch_sign == 1) & (last_switch_time > 172800)] = 2.0
            return  np.concatenate((sensor_readings, switch_obs))
        
        elif self.include_switch_in_observation and self.include_last_action_in_observation:
            
            current_switch_sign = numpy.where(self.current_action["switch"] < 0.5, 0, 1)
            current_action_temp = self.current_action["value"] * (self.upper_bound - self.lower_bound) + self.lower_bound
            off_coolers_mask = current_switch_sign == 0
            current_action_temp[off_coolers_mask] = -1.0

            last_switch_sign = numpy.where(self.last_action["switch"] < 0.5, 0, 1)
            last_switch_time = (self.t - self.cooler_last_switch_time)
            switch_obs =  numpy.zeros(last_switch_time.shape, dtype=numpy.float32)
            switch_obs[(last_switch_time < 1800) & 
                       (self.cooler_last_switch_time > self.start_time)] = 1.0
            switch_obs[(last_switch_sign == 1) & (last_switch_time > 172800)] = 2.0

            return  np.concatenate((sensor_readings, current_action_temp, switch_obs))

        else:
            return sensor_readings
    
    def get_current_obs(self):
        return self.current_obs

    def _normalize_obs(self, obs):
        if self.include_switch_in_observation:
            n_sensor = len(self.sensors)
            obs[:n_sensor] = numpy.clip(obs[:n_sensor], 10, 50)
            obs[:n_sensor] = (obs[:n_sensor] - 30.0) / 20.0 # [-1,1]
            return obs
        return obs
    
    def set_return_normilized_obs(self, return_normilized_obs):
        self.return_normilized_obs = return_normilized_obs


    def _get_state(self):
        return numpy.copy(self.state)

    def _get_info(self):
        return {"state": self._get_state(), 
                "time": self.t, 
                "topology_cooler": numpy.copy(self.cooler_topology), "topology_cooler_sensor":numpy.copy(self.cooler_sensor_topology)}

    def set_random_start_t(self, random_start_t):
        self.random_start_t = random_start_t

    def reset(self, *args, **kwargs):
        self.state = numpy.full((self.n_width, self.n_length), self.ambient_temp)
        # Add some initial noise
        self.state = self.state + numpy.random.normal(0, 2.0, (self.n_width, self.n_length)) 
        if self.random_start_t:
            max_value = int(self.max_steps * self.iter_per_step * self.sec_per_iter)
            self.t = numpy.random.randint(0, max_value)
            self.start_time = self.t
        else:
            self.t = 0.0
            self.start_time = self.t
        self.sliding_t = 120 * np.random.randint(0, 2520, size=len(self.equipments), dtype=np.int32) # 发热功率每次reset滑窗  # 2520

        self.episode_step = 0
        self.warning_count = 0

        self.n_coolers = len(self.coolers)
        self.n_sensor = len(self.sensors)
        self.cooler_last_switch_time = np.zeros(self.n_coolers)
        self.cooler_last_state = np.zeros(self.n_coolers)

        if self.control_type.lower() == 'temperature':
            self.default_action_value = (self.target_temperature - self.lower_bound) / (self.upper_bound - self.lower_bound)
            self.last_action = {
                "switch": numpy.zeros(self.n_coolers, dtype=numpy.int8),
                "value": numpy.full(self.n_coolers, self.default_action_value, dtype=numpy.float32)
            }
            self.current_action = {
                "switch": numpy.zeros(self.n_coolers, dtype=numpy.int8),
                "value": numpy.full(self.n_coolers, self.default_action_value, dtype=numpy.float32)
            }
            if self.no_switch_action:
                self.current_rest_cooler_idx = 0
                self.last_action["switch"] = self.last_action["switch"] + int(1)
                self.current_action["switch"] = self.current_action["switch"] + int(1)
                self.last_action["switch"][self.current_rest_cooler_idx] = int(0)
                self.current_action["switch"][self.current_rest_cooler_idx] = int(0)
        elif self.control_type.lower() == 'power':
            self.last_action = {"switch": numpy.array([0]), "value": numpy.array([0.0])}
            self.current_action = {"switch": numpy.array([0]), "value": numpy.array([0.0])}

        observation = self._get_obs()

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
            return numpy.clip(action["value"], 0.0, 1.0) * (self.upper_bound - self.lower_bound) + self.lower_bound
        elif(self.control_type.lower() == 'power'):
            return numpy.clip(action["value"], 0.0, 1.0)
        else:
            raise Exception(f"Unknown control type: {self.control_type}")

    def update_states(self, action, dt=0.1, n=600):
        if ('state' not in self.__dict__):
            raise Exception('Must call reset before step')

        static_chtc_array = numpy.copy(self.convection_coeffs)
        static_heat = numpy.zeros((self.n_width, self.n_length))
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
            net_heat = numpy.copy(static_heat)
            net_chtc = numpy.copy(static_chtc_array)
            cooler_control = self.action_transfer(action)
            for i, cooler in enumerate(self.coolers):
                eff = cooler(action["switch"][i], cooler_control[i], self.t,
                             building_state=self.state,
                             ambient_state=self.ambient_temp)
                net_heat += eff["delta_energy"]
                net_chtc += eff["delta_chtc"]
                energy_costs[i] += eff["power"] * dt
            state_exp = numpy.full((self.n_width + 2, self.n_length + 2), self.ambient_temp)
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

        obs_arr = numpy.array(observation[:self.n_sensor])

        # temperature cost
        if self.reward_mode == 2:
            target_loss = 0.0
        elif self.reward_mode == 1:
            # No energy cost, need to add lower temperature punishment
            obs_dev = numpy.abs(numpy.clip(obs_arr - self.target_temperature, -8.0, 8.0))
            # Modified huber loss to balance the loss of target at different temperature range
            target_loss = numpy.maximum(numpy.sqrt(obs_dev), obs_dev, obs_dev ** 2 / 8.0)
        elif self.reward_mode == 0: 
            # Notice lower temperature is punished with energy automatically
            obs_dev = numpy.clip(obs_arr - self.target_temperature, 0.0, 8.0)
            # Modified huber loss to balance the loss of target at different temperature range
            target_loss = numpy.maximum(numpy.sqrt(obs_dev), obs_dev, obs_dev ** 2 / 8.0)

        target_cost = self.target_reward_wht * numpy.mean(target_loss)

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
        if self.reward_mode == 1:
            energy_cost = 0.0
        else:
            energy_cost = self.energy_reward_wht * (power / 10000)

        hard_loss = (obs_arr > self.failure_upperbound).any()
        overheat = 0
        over_tolerace = 0
        overheat_cost = 0
        if(hard_loss):
            self.warning_count += 1
            overheat = 1
            overheat_cost = self.overheat_reward
        else:
            self.warning_count -= 1
            self.warning_count = max(self.warning_count, 0)

        # action cost
        if self.include_action_cost:
            action_temp = action["value"] * (self.upper_bound - self.lower_bound) + self.lower_bound 
            action_diff = action_temp - self.target_temperature
            def calculate_penalty(diff):
                if diff < -5:
                    return (diff + 5) ** 2
                else:
                    return 0.0
            action_cost = self.action_diff_wht * numpy.mean(numpy.vectorize(calculate_penalty)(action_diff))
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
            return self.failure_reward, True, info
        
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
        need_switch = (self.t - self.cooler_last_switch_time[current_rest]) > 3600
        if need_switch:
            current_switch[current_rest] = int(1)
            if current_rest + 1 > self.n_coolers - 1:
                next_rest = 0
            else:
                next_rest = current_rest + 1
            current_switch[next_rest] = int(0)
            self.current_rest_cooler_idx = next_rest
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
        average_power = numpy.mean(powers)

        reward, terminated, info = self.reward(observation, action, average_power)
        truncated = self.episode_step >= self.max_steps
        self.last_action = deepcopy(action)

        if self.return_normilized_obs:
            normalized_obs = self._normalize_obs(observation)
        else:
            normalized_obs = observation
        
        info.update(self._get_info())
        info.update({
                "last_control": deepcopy(self.last_action),
                "heat_power": numpy.copy(equip_heat),
                "chtc_array": numpy.copy(chtc_array),
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
                
        return normalized_obs, reward, terminated, truncated, info

    def sample_action(self, mode="random"):
        if mode == "random":
            return self._random_action()
        elif mode == "pid":
            return self._pid_action()
        elif mode == "max":
            n_coolers = len(self.coolers)
            sampled_action = np.concatenate((
                np.zeros(n_coolers, dtype=np.float32),  
                np.zeros(n_coolers, dtype=np.float32)  
            ))
            return sampled_action
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _random_action(self):
        if isinstance(self.action_space, Dict):
            return self.action_space.sample()

    def _pid_action(self, pid_params=None):

        action = np.zeros(self.action_space.shape, dtype=self.action_space.dtype)
        n_coolers = len(self.coolers)
        # Set switch part (first n_coolers elements) - Treat as continuous 1.0 for "ON"
        action[:n_coolers] = 1.0
        # Set value part (next n_coolers elements)
        target_temp = self.target_temperature # Assuming single target temp for simplicity here
        lb = self.lower_bound
        ub = self.upper_bound


        if isinstance(self.target_temperature, (np.ndarray, list)):
            target_temp = np.mean(self.target_temperature)
            target_temp = int(24)
        else:
            target_temp = self.target_temperature
            target_temp = int(24)
        # Calculate desired value based on control type (assuming Temperature control for PID example)
        if self.control_type.lower() == 'temperature':

            a = (target_temp - lb) / (ub - lb)
            a = np.clip(a, 0.0, 1.0) # Clip to valid 0-1 range

        elif self.control_type.lower() == 'power':
            a = 0.5 # Placeholder for power control PID
        else:
            a = 0.0 # Default if control type unknown
        action[n_coolers:] = a

        return action
    def stat(self, normalized_obs, terminated, info, reward):
        # power stat
        current_step = self.episode_step
        current_cool_power = round(np.sum(info.get("cool_power", 0)),4)
        self.avg_cooler_power_per_step = (self.avg_cooler_power_per_step * (current_step - 1) + current_cool_power) / current_step
        self.avg_reward = (self.avg_reward * (current_step - 1) + reward) / current_step
        # over heat percentage
        if self.return_normilized_obs:
            sensor_reading = self._denormalize_obs(normalized_obs)[:self.n_sensor]
        else:
            sensor_reading = normalized_obs[:self.n_sensor]
        current_over_heat_reading = sensor_reading - self.target_temperature
        # overheat
        current_count_overheat = []
        current_count_overheat.append(numpy.sum(current_over_heat_reading > 0))
        current_count_overheat.append(numpy.sum(current_over_heat_reading > 2))
        current_count_overheat.append(numpy.sum(current_over_heat_reading > 4))
        current_count_overheat.append(numpy.sum(current_over_heat_reading > 6))
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
        current_count_overcool.append(numpy.sum(current_over_cool < 0))
        current_count_overcool.append(numpy.sum(current_over_cool < -2))
        current_count_overcool.append(numpy.sum(current_over_cool < -4))
        current_count_overcool.append(numpy.sum(current_over_cool < -6))
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
            last_switch_sign = numpy.where(self.copy_last_action["switch"] < 0.5, 0, 1)
            current_switch_sign = numpy.where(self.current_action["switch"] < 0.5, 0, 1)
            condition_too_soon = (switch_obs > 0.5) & (switch_obs < 1.5) & (last_switch_sign != current_switch_sign)
            count_switch_too_soon = numpy.sum(condition_too_soon)
            count_switch_too_late = numpy.sum(switch_obs > 1.5)
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


class HVACEnvDiscreteAction(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_resulotion_temp = 0.1
        
    def _discretize_action(self, action):
        """Discretize the action value to make it a multiple of the resolution."""
        if isinstance(self.action_space, gym.spaces.Dict):
            discretized_action = deepcopy(action)
            if 'value' in discretized_action:
                temp_value = action['value'] * (self.upper_bound - self.lower_bound) + self.lower_bound
                discretized_temp = np.round(temp_value / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
                discretized_action['value'] = np.clip(
                    (discretized_temp - self.lower_bound) / (self.upper_bound - self.lower_bound),
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
            
            temp_value = value_part * (self.upper_bound - self.lower_bound) + self.lower_bound
            discretized_temp = np.round(temp_value / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
            discretized_value = np.clip(
                (discretized_temp - self.lower_bound) / (self.upper_bound - self.lower_bound),
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

        if self.verbose:
            self.stat(normalized_obs, terminated, info, reward)

        return normalized_obs, reward, terminated, truncated, info
    
class HVACEnvDiffAction(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_resulotion_temp = 0.1

        min_temp = -3.0
        max_temp = 3.0
        action_diff_resulotion = 0.5
        self.num_steps = int((max_temp - min_temp) / action_diff_resulotion) + 1
        self.discrete_values = numpy.linspace(min_temp, max_temp, self.num_steps)
 
    def _diff_action(self, action):

        if isinstance(self.action_space, gym.spaces.Dict):
            discretized_action = deepcopy(action)
            if 'value' in discretized_action:
                indices = np.clip(np.round(action['value'] * (self.num_steps - 1)), 0, self.num_steps - 1).astype(int)
                discrete_diff_value = self.discrete_values[indices]
                last_temp = self.last_action['value'] * (self.upper_bound - self.lower_bound) + self.lower_bound
                current_temp = last_temp + discrete_diff_value
                discretized_current_temp = np.round(current_temp / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
                current_action = np.clip(
                    (discretized_current_temp - self.lower_bound) / (self.upper_bound - self.lower_bound),
                    0.0, 1.0
                )
                discrete_diff_value['value'] = current_action
                return discrete_diff_value
        else:
            n_coolers = len(self.coolers)
            if not self.no_switch_action:
                switch_part = action[:n_coolers]
                value_part = action[n_coolers:]
            else:
                value_part = action[:n_coolers]
            indices = np.clip(np.round(value_part * (self.num_steps - 1)), 0, self.num_steps - 1).astype(int)
            discrete_diff_value = self.discrete_values[indices]
            last_temp = self.last_action['value'] * (self.upper_bound - self.lower_bound) + self.lower_bound
            current_temp = last_temp + discrete_diff_value
            discretized_current_temp = np.round(current_temp / self.action_resulotion_temp).astype(int) * self.action_resulotion_temp
            current_action = np.clip(
                (discretized_current_temp - self.lower_bound) / (self.upper_bound - self.lower_bound),
                0.0, 1.0
            )
            if not self.no_switch_action:
                return np.concatenate([switch_part, current_action])
            else:
                return current_action
        
    def step(self, action):
        discretized_action = self._diff_action(action)
        self.copy_last_action = self.last_action.copy()
        normalized_obs, reward, terminated, truncated, info = super().step(discretized_action)

        if self.verbose:
            self.stat(normalized_obs, terminated, info, reward)

        return normalized_obs, reward, terminated, truncated, info
    
    
        

    def _denormalize_obs(self, normalized_obs):
        if self.include_switch_in_observation:
            
            denormalized_obs = normalized_obs.copy()
            
            denormalized_obs[:self.n_sensor] = normalized_obs[:self.n_sensor] * 20.0 + 30.0
            
            return denormalized_obs
        return normalized_obs