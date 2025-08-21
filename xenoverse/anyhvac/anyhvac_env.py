import sys
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy
import numbers
import numpy as np
from copy import deepcopy

class HVACEnv(gym.Env):
    def __init__(self,
                 max_steps=5040,  # 7 days
                 failure_upperbound=40, # triggers failure above this temperature
                 iter_per_step=600,# 600
                #  iter_per_step=6000,
                 sec_per_iter=0.2,
                 set_lower_bound=16,
                 set_upper_bound=32,
                 verbose=False,
                 action_space_format='box',
                 include_time_in_observation=False,
                 include_heat_in_observation=True,
                 ):
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1,), dtype=numpy.float32)

        self.action_space = None
        # print("24 self.action_space",self.action_space.shape)
 
        self.max_steps = max_steps
        self.failure_upperbound = failure_upperbound
        self.failure_reward = -100
        self.energy_reward_wht = -3.0   
        self.switch_reward_wht = -0.1  
        self.target_reward_wht = -0.25  # ranging from 0.0 to -2.00
        self.base_reward = 1.0 # survive bonus
        self.iter_per_step = iter_per_step
        self.sec_per_iter = sec_per_iter
        self.sec_per_step = self.iter_per_step * self.sec_per_iter
        self.lower_bound = set_lower_bound
        self.upper_bound = set_upper_bound
        self.verbose = verbose
        self.warning_count_tolerance = 5
        self.action_space_format = action_space_format
        self.include_time_in_observation = include_time_in_observation 
        self.include_heat_in_observation = include_heat_in_observation
        
    def set_task(self, task):
        for key in task:
            self.__dict__[key] = task[key]
        self.task_set = True
        self.heat_capacity = task.get('heat_capacity', []) 
        # print("task", task)
        self.equipments = task.get('equipments', []) 
        # 验证设备类型
        
        n_coolers = len(self.coolers)
        n_sensors = len(self.sensors)
        n_heaters = len(self.equipments)
        # 根据格式创建动作空间
        
        if self.action_space_format == 'dict':
            self.action_space = Dict({
                "switch": gym.spaces.MultiBinary(n_coolers),
                "value": gym.spaces.Box(low=0, high=1, shape=(n_coolers,), dtype=np.float32)
            })
        else:  # 默认使用Box格式
            self.action_space = gym.spaces.Box(low=0, high=1, shape=(2*n_coolers,), dtype=numpy.float32) # Placeholder shape

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

        if self.include_time_in_observation:
            obs_shape_dim = n_sensors
            obs_shape_dim += 1 # Add one dimension for normalized time (episode progress)
            low_bounds = np.full(obs_shape_dim, -273.0, dtype=np.float32)
            high_bounds = np.full(obs_shape_dim, 273.0, dtype=np.float32)
            # Last element is normalized episode progress [0, 1]
            low_bounds[n_sensors] = 0.0
            high_bounds[n_sensors] = 100.0
            
            self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, shape=(obs_shape_dim,), dtype=numpy.float32)
        elif self.include_heat_in_observation:
            obs_shape_dim = n_sensors
            obs_shape_dim += n_heaters
            low_bounds = np.full(obs_shape_dim, -273.0, dtype=np.float32)
            high_bounds = np.full(obs_shape_dim, 273.0, dtype=np.float32)
            low_bounds[n_sensors:] = 0.0
            high_bounds[n_sensors:] = 80000.0 
            self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, shape=(obs_shape_dim,), dtype=numpy.float32)
            

        else:
            # observation space and action space
            self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(n_sensors,), dtype=numpy.float32)

    def _get_obs(self):

        heater_progress = []
        sensor_readings = np.array([sensor(self.state, self.t) for sensor in self.sensors], dtype=np.float32)

        if self.include_time_in_observation:

            normalized_episode_progress = float(self.episode_step*100) / 5040
            timer_readings = np.clip(normalized_episode_progress, 0.0, 100.0)
            timer_readings = [np.float32(normalized_episode_progress)]

            return np.concatenate((sensor_readings, timer_readings))

        elif self.include_heat_in_observation:

            static_chtc_array = numpy.copy(self.convection_coeffs)
            static_heat = numpy.zeros((self.n_width, self.n_length))
            equip_heat = []

            for i, equipment in enumerate(self.equipments):
                
                eff = equipment(self.sliding_t[i] + self.t)  # 发热功率每次reset滑窗

                static_heat += eff["delta_energy"]
                static_chtc_array += eff["delta_chtc"]
                equip_heat.append(eff["heat"])

                normalized_episode_progress = np.clip(eff["heat"], 0.0, 30000.0)
                heater_progress.append(normalized_episode_progress) 

            heat_readings = np.array(heater_progress, dtype=np.float32)
            return np.concatenate((sensor_readings, heat_readings))

        else:
            return sensor_readings
    def _get_state(self):
        return numpy.copy(self.state)

    def _get_info(self):
        return {"state": self._get_state(), 
                "time": self.t, 
                "topology_cooler": numpy.copy(self.cooler_topology), "topology_cooler_sensor":numpy.copy(self.cooler_sensor_topology)}

    def reset(self, *args, **kwargs):
        self.state = numpy.full((self.n_width, self.n_length), self.ambient_temp)
        # Add some initial noise
        self.state = self.state + numpy.random.normal(0, 2.0, (self.n_width, self.n_length)) 
        self.t = 0.0
        self.sliding_t = 120 * np.random.randint(0, 2520, size=len(self.equipments), dtype=np.int32) # 发热功率每次reset滑窗  # 2520

        self.last_action = {"switch": numpy.array([0]), "value": numpy.array([0.0])}

        self.episode_step = 0
        self.warning_count = 0

        observation = self._get_obs()

        return observation, self._get_info()
    
    def action_transfer(self, action):
        if(self.control_type.lower() == 'temperature'):
            return numpy.clip(1.0 - action["value"], 0.0, 1.0) * (self.upper_bound - self.lower_bound) + self.lower_bound
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

            eff = equipment(self.sliding_t[i] + self.t) # 发热功率每次reset滑窗
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
        if self.include_time_in_observation:
            num_sensors = len(self.sensors)
            obs_arr = numpy.array(observation[:num_sensors]) # Get only sensor readings
        elif self.include_heat_in_observation:
            num_sensors = len(self.sensors)
            obs_arr = numpy.array(observation[:num_sensors]) # Get only sensor readings
            
        else:
            obs_arr = numpy.array(observation)
        

        # Only punish those overheats
        # Notice lower temperature is punished with energy automatically
        obs_dev = numpy.clip(obs_arr - self.target_temperature, 0.0, 8.0)
        # Modified huber loss to balance the loss of target at different temperature range
        target_loss = numpy.maximum(numpy.sqrt(obs_dev), obs_dev, obs_dev ** 2 / 8.0)

        # get max temperature deviation in each area
        # cal loss with max temperature deviation in each area
        # add severe punishment to overheat
        target_cost = self.target_reward_wht * numpy.mean(target_loss)
        switch_cost = self.switch_reward_wht * numpy.mean(numpy.abs(
                action["switch"] - self.last_action["switch"]))

        # reward is connected to the AVERAGE POWER of each cooler
        energy_cost = self.energy_reward_wht * (power / 10000)
        hard_loss = (obs_arr > self.failure_upperbound).any()

        if(hard_loss):
            self.warning_count += 1
        else:
            self.warning_count -= 1
            self.warning_count = max(self.warning_count, 0)

        info = {"fail_step_percrentage": hard_loss, "energy_cost": energy_cost, "target_cost": target_cost, "switch_cost": switch_cost}

        if(self.warning_count > self.warning_count_tolerance):
            return self.failure_reward, True, info
        
        return (self.base_reward + target_cost + switch_cost + energy_cost,  
                False, info)
    
    def _unflatten_action(self, action_flat):
        """Converts the flattened Box action back to the dictionary format."""
        if not isinstance(action_flat, np.ndarray):
            action_flat = np.array(action_flat)
        n_coolers = len(self.coolers)
        # Ensure action_flat has the correct shape
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

        # Threshold switch part (e.g., > 0.5 is ON)
        switch_binary = (switch_continuous > 0.5).astype(np.int8)

        # Clip value part to ensure it's within [0, 1] (Box space should handle bounds, but good practice)
        value_clipped = np.clip(value_continuous, 0.0, 1.0)

        action_dict = {
            "switch": switch_binary,
            "value": value_clipped.astype(np.float32)
        }
        return action_dict


    def step(self, action):

        if isinstance(self.action_space, Dict):

            action = action
        else:
            action = self._unflatten_action(action)

        self.episode_step += 1
        equip_heat, chtc_array, powers = self.update_states(action, dt=self.sec_per_iter, n=self.iter_per_step)
        observation = self._get_obs()

        # calculate average power of each cooler
        average_power = numpy.mean(powers)

        reward, terminated, info = self.reward(observation, action, average_power)
        truncated = self.episode_step >= self.max_steps
        self.last_action = deepcopy(action)

        info.update(self._get_info())
        info.update({
                "last_control": deepcopy(self.last_action),
                "heat_power": numpy.copy(equip_heat),
                "chtc_array": numpy.copy(chtc_array),
                "cool_power": powers,
                })

        if self.verbose:
            # print("self.verbose", self.verbose)

            cool_power = round(np.mean(info.get("cool_power", 0)),4)
            heat_power = round(np.mean(info.get("heat_power", 0)),4)
            fail_step_percrentage = info["fail_step_percrentage"] if isinstance(info["fail_step_percrentage"], numbers.Real) else 0
            info_total = f"energy_cost: {round(info.get('energy_cost', 0),4)}, target_cost: {round(info.get('target_cost', 0),4)}, switch_cost: {round(info.get('switch_cost', 0),4)},cool_power: {cool_power}, heat_power: {heat_power}"
            print(f"Step {self.episode_step} | fail_step_percrentage:{fail_step_percrentage} | Reward: {reward} | {info_total}| cool_power: {cool_power:.2f} | heat_power:{heat_power:.2f} ", flush=True)
            
            # print(f"step:{self.episode_step}, Reward:{reward}, terminated:{terminated},\nmax-temperature:{numpy.max(observation)}, avg-temperature:{numpy.mean(observation)}, avg-power:{average_power:.5f}, ambient_temperature:{self.ambient_temp}, cool_power:{cool_power}, heat_power:{heat_power}\n")
                
        return observation, reward, terminated, truncated, info

    def sample_action(self, mode="random"):
        if mode == "random":
            return self._random_action()
        elif mode == "pid":
            return self._pid_action()
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
