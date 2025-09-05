import numpy

from xenoverse.anyhvac.anyhvac_env import HVACEnv
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible

class HVACSolverGTPID:
    def __init__(self, env):

        self.env = env

        

        required_attrs = [
            'sensors', 'coolers', 'target_temperature', 
            'sec_per_step', 'lower_bound', 'upper_bound'
        ]
        
        for attr in required_attrs:
            if not hasattr(env, attr):
                raise AttributeError(f"Missing required attribute: {attr}")
            setattr(self, attr, getattr(env, attr))
        
        

        
        
        self.corr_sensor_cooler = []
        for sensor in self.sensors:
            nx, ny = sensor.nloc
            # px, py = sensor.loc
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

    def _extract_sensor_readings(self, observation_with_time):
        """
        Extracts only the sensor readings from the observation vector,
        which might include a time component.
        """
        obs_array = numpy.array(observation_with_time)
        if obs_array.shape[0] > len(self.sensors):
            return obs_array[:len(self.sensors)]
        elif obs_array.shape[0] == len(self.sensors): 
            return obs_array

    def policy(self, observation):
        # 兼容observation含有t的情况
        current_sensor_readings = self._extract_sensor_readings(observation)
        # print(current_sensor_readings.shape, current_sensor_readings)
        effective_target_temp = self.target_temperature

        # current_observation_arr = numpy.array(observation)
        current_observation_arr = numpy.array(current_sensor_readings)

        # diff calculation

        diff = effective_target_temp - current_observation_arr
        # print("diff",diff)
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
    def policy_mask(self, observation, mask=None):
        # 兼容observation含有t的情况
        current_sensor_readings = self._extract_sensor_readings(observation)
        effective_target_temp = self.target_temperature
        current_observation_arr = numpy.array(current_sensor_readings)

        # 处理mask参数
        n_coolers = len(self.coolers)
        if mask is None:
            mask = numpy.ones(n_coolers, dtype=bool)  # 默认所有节点都受控
        elif len(mask) != n_coolers:
            raise ValueError(f"Mask size {len(mask)} doesn't match number of coolers {n_coolers}")
        
        # 检测mask变化并重置整个PID状态
        if not hasattr(self, 'last_mask') or self.last_mask is None:
            self.last_mask = numpy.copy(mask)
            print("init mask = ", mask)
        
        mask_changed = not numpy.array_equal(mask, self.last_mask)
        if mask_changed:
            # 当mask变化时，重置整个PID状态
            self.acc_diff = numpy.zeros_like(self.acc_diff)  # 重置积分项
            # self.last_observation = numpy.zeros_like(current_observation_arr)  # 重置上一次观测值
            self.last_mask = numpy.copy(mask)
            # print("mask changes: ", mask)

        # 计算温度差异
        diff = effective_target_temp - current_observation_arr
        
        # 初始化历史数据（如果形状不匹配）
        if self.last_observation.shape != current_observation_arr.shape:
            self.last_observation = numpy.zeros_like(current_observation_arr)
        
        last_diff = effective_target_temp - self.last_observation
        
        # 初始化累积误差（如果形状不匹配）
        if self.acc_diff.shape != diff.shape:
            self.acc_diff = numpy.zeros_like(diff)
        
        # 更新PID误差项
        self.acc_diff += diff
        
        # 计算PID控制信号
        d_e = - (self.kp * diff - self.kd * (diff - last_diff) / self.delta_t + self.ki * self.acc_diff)
        
        # 只计算受控节点的动作值
        active_corr_matrix = self.corr_sensor_cooler[:, mask]
        active_action_values = numpy.matmul(d_e, active_corr_matrix)
        
        # 创建完整尺寸的动作数组
        action_values_continuous = numpy.zeros(n_coolers, dtype=numpy.float32)
        action_values_continuous[mask] = active_action_values
        
        # 计算开关信号（只对受控节点）
        switch_continuous = numpy.zeros(n_coolers, dtype=numpy.float32)
        active_switch = (active_action_values > -0.05).astype(numpy.float32)
        switch_continuous[mask] = active_switch
        
        # 裁剪连续动作值（只对受控节点）
        value_clipped = numpy.zeros(n_coolers, dtype=numpy.float32)
        active_value_clipped = numpy.clip(active_action_values, 0.0, 1.0)
        value_clipped[mask] = active_value_clipped
        
        # 确保不受控节点的动作值为0
        non_controlled = ~mask
        switch_continuous[non_controlled] = 0.0
        value_clipped[non_controlled] = 0.0
        
        # 更新历史状态
        self.last_action = numpy.concatenate((switch_continuous, value_clipped))
        self.last_observation = numpy.copy(current_observation_arr)
        
        # 构建最终动作向量
        flat_action = numpy.zeros(2 * n_coolers, dtype=numpy.float32)
        flat_action[:n_coolers] = switch_continuous
        flat_action[n_coolers:] = value_clipped

        return flat_action
