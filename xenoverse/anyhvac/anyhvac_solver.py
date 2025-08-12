import numpy

from xenoverse.anyhvac.anyhvac_env import HVACEnv
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible

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
        diff = self.target_temperature - numpy.array(observation)
        last_diff = self.target_temperature - self.last_observation
        self.acc_diff += diff

        d_e =  - (self.kp * diff - self.kd * (diff - last_diff) / self.delta_t + self.ki * self.acc_diff)
        action = numpy.matmul(d_e, self.corr_sensor_cooler)
        switch = (action > -0.05).astype('int8')
        value = numpy.clip(action, 0.0, 1.0)
        self.last_action = action
        self.last_observation = numpy.copy(observation)
        return {'switch': switch, 'value': value}