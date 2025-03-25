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
            self.corr_sensor_cooler.append([cooler.cooler_diffuse[nx, ny] for cooler in self.coolers])
        self.corr_sensor_cooler /= numpy.clip(numpy.sum(self.corr_sensor_cooler, axis=1, keepdims=True), a_min=1e-6, a_max=None)
        self.cooler_int = numpy.zeros(len(self.coolers))
        self.minimum_action = numpy.ones(len(self.coolers)) * 0.01
        self.last_action = numpy.copy(self.minimum_action)
        self.acc_diff = numpy.zeros(len(self.sensors))
        self.last_observation = numpy.array(self.env.get_observation())
        self.ki = 1.0e-2
        self.kp = 1.0e-3
        self.kd = 1.0e-3

    def policy(self):
        sensors = numpy.array(self.env.get_observation())
        diff = self.target_temperature - sensors
        last_diff = self.target_temperature - self.last_observation
        self.acc_diff += diff

        d_e =  - (self.kp * diff + self.kd * (diff - last_diff) + self.ki * self.acc_diff)
        action = numpy.matmul(d_e, self.corr_sensor_cooler)
        self.last_action = action
        self.last_observation = numpy.copy(sensors)
        return action

