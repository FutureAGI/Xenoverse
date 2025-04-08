import sys
import gymnasium as gym
import numpy

class HVACEnv(gym.Env):
    def __init__(self, 
                 max_steps=86400,
                 target_temperature=28,
                 upper_limit=80,
                 lower_limit=-273,
                 power_normalizer=10000,
                 iter_per_step=600,
                 sec_per_iter=0.2):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1, ), dtype=numpy.float32)
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1, ), dtype=numpy.float32)
        self.max_steps = max_steps
        self.target_temperature = target_temperature
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.power_normalizer = power_normalizer
        self.failure_reward = -100
        self.energy_loss = 1.0e-5
        self.switch_loss = 1.0e-6
        self.t_loss = 1.0e-3
        self.iter_per_step = iter_per_step
        self.sec_per_iter = sec_per_iter


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

        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_coolers, ), dtype=numpy.float32)
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(n_sensors, ), dtype=numpy.float32)
    
    def _get_obs(self):
         return [sensor(self.state) for sensor in self.sensors]
    
    def _get_state(self):
        return numpy.copy(self.state)
    
    def _get_info(self):
        return {"state": self._get_state(), "time": self.t, "topology_cooler": numpy.copy(self.cooler_topology), "topology_cooler_sensor":numpy.copy(self.cooler_sensor_topology)}

    def reset(self):
        self.state = numpy.full((self.n_width, self.n_length), self.ambient_temp)
        # Add some initial noise
        self.state = self.state + numpy.random.normal(0, 2.0, (self.n_width, self.n_length))
        self.t = 0
        self.last_action = [0 for _ in self.coolers]

        observation = self._get_obs()

        # consistent with gymnasium
        return observation, self._get_info()

    def update_states(self, action, dt=0.1, n=600):
        if('state' not in self.__dict__):
            raise Exception('Must call reset before step')

        static_chtc_array = numpy.copy(self.convection_coeffs)
        static_heat = numpy.zeros((self.n_width, self.n_length))
        equip_heat = []
        for i,equipment in enumerate(self.equipments):
            eff = equipment(self.t)
            static_heat += eff["delta_energy"]
            static_chtc_array += eff["delta_chtc"]
            equip_heat.append(eff["heat"])

        # Heat convection
        # (nw + 1) * nl
        for i in range(n):
            net_heat = numpy.copy(static_heat)
            net_chtc = numpy.copy(static_chtc_array)
            for i,cooler in enumerate(self.coolers):
                eff = cooler(action[i] * self.power_normalizer, self.t, 
                            building_state=self.state, 
                            ambient_state=self.ambient_temp)
                net_heat += eff["delta_energy"]
                net_chtc += eff["delta_chtc"]

            state_exp = numpy.full((self.n_width + 2, self.n_length + 2), self.ambient_temp)
            state_exp[1:-1, 1:-1] = self.state
            horizontal = - (state_exp[1:, 1:-1] - state_exp[:-1, 1:-1]) * net_chtc[:, :-1, 0] * self.csa
            # nw * (nl + 1)
            vertical = - (state_exp[1:-1, 1:] - state_exp[1:-1, :-1]) * net_chtc[:-1, :, 1] * self.csa

            net_in = (horizontal[:-1, :] - horizontal[1:, :]) + (vertical[:, :-1] - vertical[:, 1:])

            self.state += (net_heat + net_in) / self.heat_capacity * dt

            self.t += dt
        return equip_heat, net_chtc

    def reward_v1(self, observation, action): # v1 juedges the temperature in all the cells
        soft_loss = numpy.mean((self.state - self.target_temperature) ** 2)
        hard_loss = (self.state > self.upper_limit).any() or (self.state < self.lower_limit).any()
        if(hard_loss):
            return self.failure_reward, True
        return (- soft_loss * self.t_loss 
                - numpy.mean(numpy.abs(action - self.last_action)) * self.switch_loss
                - numpy.mean(action) * self.energy_loss), False
    
    def reward_v2(self, observation, action): # v2 judges the temperature in the sensors
        obs_arr = numpy.array(observation)
        soft_loss = numpy.mean((obs_arr - self.target_temperature) ** 2)
        hard_loss = (obs_arr > self.upper_limit).any() or (obs_arr < self.lower_limit).any()
        if(hard_loss):
            return self.failure_reward, True
        return (- soft_loss * self.t_loss 
                - numpy.mean(numpy.abs(action - self.last_action)) * self.switch_loss
                - numpy.mean(action) * self.energy_loss), False

    def step(self, action):
        action = numpy.clip(action, 0, 1)
        equip_heat, chtc_array = self.update_states(action, dt=self.sec_per_iter, n=self.iter_per_step)
        observation = self._get_obs()
        # control the temperature at sensors only
        reward, terminated = self.reward_v2(observation, action)

        truncated = (self.t >= self.max_steps)
        self.last_action = numpy.copy(action)
        info = self._get_info()
        info.update({"last_control": numpy.copy(self.last_action)})
        info.update({"heat_power": numpy.copy(equip_heat)})
        info.update({"chtc_array": numpy.copy(chtc_array)})

        return observation, reward, terminated, truncated, info