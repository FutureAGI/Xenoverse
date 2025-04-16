import sys
import gymnasium as gym
from gymnasium.spaces import Dict, Box, Discrete
import numpy
import numpy as np
from copy import deepcopy

class HVACEnv(gym.Env):
    def __init__(self,
                 max_steps=5040,  # 7 days
                 failure_upperbound=80, # triggers failure above this temperature
                 iter_per_step=600,
                 sec_per_iter=0.2,
                 set_lower_bound=16,
                 set_upper_bound=32,
                 verbose=False):
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1,), dtype=numpy.float32)
        self.action_space = Dict({
                "switch": gym.spaces.MultiBinary(1, seed=42),
                "value": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=numpy.float32)
        })
        self.max_steps = max_steps
        self.failure_upperbound = failure_upperbound
        self.failure_reward = -100
        self.energy_reward_wht = -0.375  # ranging from 0.0 to -0.375
        self.switch_reward_wht = -0.375  # ranging from 0.0 to -0.375
        self.target_reward_wht = -0.25  # ranging from 0.0 to -2.00
        self.base_reward = 1.0 # survive bonus
        self.iter_per_step = iter_per_step
        self.sec_per_iter = sec_per_iter
        self.sec_per_step = self.iter_per_step * self.sec_per_iter
        self.lower_bound = set_lower_bound
        self.upper_bound = set_upper_bound
        self.verbose = verbose
        self.warning_count_tolerance = 5

    def set_task(self, task):
        for key in task:
            self.__dict__[key] = task[key]
        self.task_set = True

        # cacluate topology
        n_coolers = len(self.coolers)
        n_sensors = len(self.sensors)

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

        # observation space and action space
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(n_sensors,), dtype=numpy.float32)
        self.action_space = Dict({
                "switch": gym.spaces.MultiBinary(n_coolers, seed=42),
                "value": gym.spaces.Box(low=0, high=1, shape=(n_coolers,), dtype=numpy.float32)
        })

    def _get_obs(self):
         return [sensor(self.state, self.t) for sensor in self.sensors]
    
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
        self.t = 0
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

            net_in = (horizontal[:-1, :] - horizontal[1:, :]) + (vertical[:, :-1] - vertical[:, 1:])

            self.state += (net_heat + net_in) / self.heat_capacity * dt
            self.t += dt

        avg_power = energy_costs / (dt * n)

        return equip_heat, net_chtc, avg_power
    
    def reward(self, observation, action, power):
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

        if(self.warning_count > self.warning_count_tolerance):
            return self.failure_reward, True
        
        return (self.base_reward + target_cost + switch_cost + energy_cost,  
                False)

    def step(self, action):
        self.episode_step += 1
        equip_heat, chtc_array, powers = self.update_states(action, dt=self.sec_per_iter, n=self.iter_per_step)
        observation = self._get_obs()

        # calculate average power of each cooler
        average_power = numpy.mean(powers)

        reward, terminated = self.reward(observation, action, average_power)
        truncated = self.episode_step >= self.max_steps
        self.last_action = deepcopy(action)

        info = self._get_info()
        info.update({
                "last_control": deepcopy(self.last_action),
                "heat_power": numpy.copy(equip_heat),
                "chtc_array": numpy.copy(chtc_array),
                "cool_power": powers,
                })
        if self.verbose:
            print(f"step:{self.episode_step}, reward:{reward}, terminated:{terminated},\nmax-temperature:{numpy.max(observation)}, avg-temperature:{numpy.mean(observation)}, avg-power:{average_power:.5f}, ambient_temperature:{self.ambient_temp}\n")
        return observation, reward, terminated, truncated, info
