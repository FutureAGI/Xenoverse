import numpy

from xenoverse.anyhvacv2.anyhvac_env import HVACEnv
from xenoverse.anyhvacv2.anyhvac_env_vis import HVACEnvVisible

class HVACSolverGTPID(object):
    def __init__(self, env):
        for key, val in env.__dict__.items():
            if not key.startswith('_'):
                setattr(self, key, val)
        self.env = env
        self.corr_sensor_cooler = []
        for sensor in self.sensors:
            nx, ny = sensor.nloc
            px, py = sensor.loc
            cooler_whts = numpy.asarray([cooler.cooler_diffuse[nx, ny] for cooler in self.coolers])
            while(numpy.sum(cooler_whts) < 1.0e-6):
                cooler_whts *=10.0
                cooler_whts += 1.0e-12
            self.corr_sensor_cooler.append(cooler_whts)
        self.corr_sensor_cooler /= numpy.clip(numpy.sum(self.corr_sensor_cooler, axis=1, keepdims=True), a_min=1e-6, a_max=None)
        self.cooler_int = numpy.zeros(len(self.coolers))
        self.minimum_action = numpy.ones(len(self.coolers)) * 0.01
        self.last_action = numpy.copy(self.minimum_action)
        self.acc_diff = numpy.zeros(len(self.sensors))
        self.last_observation = numpy.array(self.env._get_obs())
        self.ki = 2.0e-2
        self.kp = 5.0e-3
        self.kd = 5.0e-3
        self.delta_t = self.sec_per_step / 60



    def policy(self, observation):

        if isinstance(self.target_temperature, (list, numpy.ndarray)) and numpy.array(observation).ndim == 1:
            target_temp_arr = numpy.array(self.target_temperature)
            if target_temp_arr.ndim > 0 and target_temp_arr.shape[0] != numpy.array(observation).shape[0] :
                
                if target_temp_arr.size == 1 :
                    effective_target_temp = target_temp_arr.item()
                else: 

                    effective_target_temp = numpy.mean(target_temp_arr) 
            elif target_temp_arr.ndim == 0 : # scalar
                effective_target_temp = target_temp_arr
            else: 
                effective_target_temp = target_temp_arr

        else: 
            effective_target_temp = self.target_temperature

        current_observation_arr = numpy.array(observation)

        # diff calculation

        diff = effective_target_temp - current_observation_arr

        if self.last_observation.shape != current_observation_arr.shape:
            self.last_observation = numpy.zeros_like(current_observation_arr) # Re-initialize if shape mismatch

        last_diff = effective_target_temp - self.last_observation

        # Ensure self.acc_diff has the same shape as diff
        if self.acc_diff.shape != diff.shape:
            self.acc_diff = numpy.zeros_like(diff) # Re-initialize if shape mismatch
        self.acc_diff += diff
        # d_e calculation: This seems to result in a per-sensor error signal vector
        d_e = - (self.kp * diff - self.kd * (diff - last_diff) / self.delta_t + self.ki * self.acc_diff)
        action_values_continuous = numpy.matmul(d_e, self.corr_sensor_cooler)
        switch_continuous = (action_values_continuous > -0.05).astype(numpy.float32)
        # Value part: Clipped continuous values
        value_clipped = numpy.clip(action_values_continuous, 0.0, 1.0)
        self.last_action = numpy.concatenate((switch_continuous, value_clipped)) # Store the flat action
        self.last_observation = numpy.copy(current_observation_arr)
        n_coolers = len(self.coolers)
        flat_action = numpy.zeros(2 * n_coolers, dtype=numpy.float32)
        flat_action[:n_coolers] = switch_continuous
        flat_action[n_coolers:] = value_clipped

        return flat_action