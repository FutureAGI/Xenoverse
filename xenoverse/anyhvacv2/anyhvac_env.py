import sys
import gym
import numpy
import numpy as np


class HVACEnv(gym.Env):
    def __init__(self,
                 max_steps=86400,
                 target_temperature=28,
                 upper_limit=80,
                 lower_limit=-273,
                 iter_per_step=600,
                 sec_per_iter=0.2,
                 set_lower_bound=16,
                 set_upper_bound=32,
                 tolerance=1,
                 verbose=False):
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=numpy.float32)
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(1,), dtype=numpy.float32)
        self.max_steps = max_steps
        self.target_temperature = target_temperature
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.failure_reward = -100
        self.energy_loss = 1.0e-5
        self.switch_loss = 1.0e-3
        self.t_loss = 1.0e-3
        self.base_reward = 1
        self.iter_per_step = iter_per_step
        self.sec_per_iter = sec_per_iter
        self.lower_bound = set_lower_bound
        self.upper_bound = set_upper_bound
        self.area_divider = None
        self.tolerance = tolerance
        self.verbose = verbose

    def set_task(self, task):
        for key in task:
            self.__dict__[key] = task[key]
        self.task_set = True

        # cacluate topology
        n_coolers = len(self.coolers)
        n_sensors = len(self.sensors)

        self.topology = numpy.zeros((n_coolers, n_coolers))
        for i, cooler_i in enumerate(self.coolers):
            for j, cooler_j in enumerate(self.coolers):
                if (i > j):
                    self.topology[i, j] = numpy.sqrt(numpy.sum((cooler_i.loc - cooler_j.loc) ** 2))
        for i in range(n_coolers):
            for j in range(i + 1, n_coolers):
                self.topology[i, j] = self.topology[j, i]

        # calculate cross sectional area
        self.csa = self.cell_size * self.floor_height
        self.area_divider.ramdom_target(self.target_temperature, 1.0) # ramdom target temperature for each area
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(n_coolers * 2,), dtype=numpy.float32)
        self.observation_space = gym.spaces.Box(low=-273, high=273, shape=(n_sensors,), dtype=numpy.float32)

    def get_observation(self):
        return [sensor(self.state) for sensor in self.sensors]

    def reset(self):
        self.state = numpy.full((self.n_width, self.n_length), self.ambient_temp)
        # Add some initial noise
        self.state = self.state + numpy.random.normal(0, 2.0, (self.n_width, self.n_length))
        self.t = 0
        self.last_action = numpy.zeros(self.action_space.shape[0])
        self.episode_step = 0

        observation = self.get_observation()

        return observation

    def action_transfer(self, action):
        # transfer to working status and set temperature
        lb = self.lower_bound
        ub = self.upper_bound
        switches = [0 if a < 0.5 else 1 for a in action[::2]]
        temps = [lb + (ub - lb) * a for a in action[1::2]]
        return [a for a in zip(switches, temps)]

    def update_states(self, action, dt=0.1, n=600):
        if ('state' not in self.__dict__):
            raise Exception('Must call reset before step')

        static_chtc_array = numpy.copy(self.convection_coeffs)
        static_heat = numpy.zeros((self.n_width, self.n_length))
        equip_heat = []
        energy = 0
        for i, equipment in enumerate(self.equipments):
            eff = equipment(self.t)
            static_heat += eff["delta_energy"]
            static_chtc_array += eff["delta_chtc"]
            equip_heat.append(eff["heat"])

        # Heat convection
        # (nw + 1) * nl
        t_action = self.action_transfer(action)
        for i in range(n):
            net_heat = numpy.copy(static_heat)
            net_chtc = numpy.copy(static_chtc_array)
            for i, cooler in enumerate(self.coolers):
                eff = cooler(t_action[i], self.t,
                             building_state=self.state,
                             ambient_state=self.ambient_temp)
                net_heat += eff["delta_energy"]
                net_chtc += eff["delta_chtc"]
                energy += eff["power"] * dt
            state_exp = numpy.full((self.n_width + 2, self.n_length + 2), self.ambient_temp)
            state_exp[1:-1, 1:-1] = self.state
            horizontal = - (state_exp[1:, 1:-1] - state_exp[:-1, 1:-1]) * net_chtc[:, :-1, 0] * self.csa
            # nw * (nl + 1)
            vertical = - (state_exp[1:-1, 1:] - state_exp[1:-1, :-1]) * net_chtc[:-1, :, 1] * self.csa

            net_in = (horizontal[:-1, :] - horizontal[1:, :]) + (vertical[:, :-1] - vertical[:, 1:])

            self.state += (net_heat + net_in) / self.heat_capacity * dt

            self.t += dt
        return equip_heat, net_chtc, energy

    def reward(self, observation, action, energy):  # v1 juedges the temperature in all the cells
        obs_arr = numpy.array(observation)

        # get max temperature deviation in each area
        dev_matrix = self.area_divider.cal_max_temp_deviation(self.sensors, self.state)
        dev_matrix = abs(dev_matrix / self.tolerance)
        dev_matrix[dev_matrix < 1] = 0
        # cal loss with max temperature deviation in each area
        soft_loss = numpy.mean(dev_matrix ** 2)
        # soft_loss = numpy.mean((self.state - self.target_temperature) ** 2)
        hard_loss = (obs_arr > self.upper_limit).any() or (obs_arr < self.lower_limit).any()
        normalized_energy = energy / (self.sec_per_iter * self.iter_per_step)
        # print(f'processed dev_matrix:{dev_matrix}')
        # print(
        #     f"soft_loss:{soft_loss}, rated soft_loss:{soft_loss * self.t_loss}, energy:{energy}, normalized energy:{normalized_energy}, rated energy:{normalized_energy / len(self.coolers) * self.energy_loss}, switchloss:{numpy.mean(numpy.abs(action - self.last_action)) * self.switch_loss}")
        if (hard_loss):
            return self.failure_reward, True
        return self.base_reward +(- soft_loss * self.t_loss
                - numpy.mean(numpy.abs(action - self.last_action)) * self.switch_loss
                - (normalized_energy / len(self.coolers)) * self.energy_loss), False

    def step(self, action):
        self.episode_step += 1
        action = numpy.clip(action, 0, 1)
        equip_heat, chtc_array, energy = self.update_states(action, dt=self.sec_per_iter, n=self.iter_per_step)
        observation = self.get_observation()
        reward, done = self.reward(observation, action, energy)
        done = done or (self.episode_step >= self.max_steps)
        self.last_action = numpy.copy(action)
        info = {"topology": numpy.copy(self.topology),
                "gt_temperature": numpy.copy(self.state),
                "last_control": numpy.copy(self.last_action),
                "heat_power": numpy.copy(equip_heat),
                "chtc_array": numpy.copy(chtc_array),
                "energy": energy,
                "area_divider": self.area_divider}
        if self.verbose:
            print(f"step:{self.episode_step},reward:{reward}, done:{done},\nobservation:{observation}")
        return observation, reward, done, info
