import numpy
from typing import Tuple
import os
import json
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
        print("observation_with_time", observation_with_time)
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


class HVACSolverLOCPID(HVACSolverGTPID):
    def __init__(self, env):
        super().__init__(env)
        
        self.corr_sensor_cooler = []
        
        sensor_positions = numpy.array([sensor.nloc for sensor in self.sensors])
        cooler_positions = numpy.array([cooler.nloc for cooler in self.coolers])
        
        dist_matrix = numpy.zeros((len(self.sensors), len(self.coolers)))
        for i, s_pos in enumerate(sensor_positions):
            for j, c_pos in enumerate(cooler_positions):
                dist = numpy.sqrt((s_pos[0]-c_pos[0])**2 + (s_pos[1]-c_pos[1])**2)
                dist_matrix[i, j] = dist
        
        dist_matrix = numpy.clip(dist_matrix, 1e-6, None)
        
        for i in range(len(self.sensors)):
            dist_weights = 1.0 / dist_matrix[i, :]
            
            while numpy.sum(dist_weights) < 1.0e-6:
                dist_weights *= 10.0
                dist_weights += 1.0e-12
            
            dist_weights /= numpy.clip(numpy.sum(dist_weights), a_min=1e-6, a_max=None)
            self.corr_sensor_cooler.append(dist_weights)
        
        self.corr_sensor_cooler = numpy.array(self.corr_sensor_cooler)

class HVACSolverGridSearchPID(HVACSolverGTPID):
    """
    继承 HVACSolverGTPID，添加 Grid Search 优化 PID 参数功能
    
    使用方法:
        solver = HVACSolverGridSearchPID(env)
        best_params = solver.search(max_steps=1000)
        # 最优参数自动应用到 solver.kp, solver.ki, solver.kd
    """
    
    def __init__(self, env, 
                 kp_range: Tuple[float, float, int] = (1e-3, 5e-2, 10),
                 ki_range: Tuple[float, float, int] = (1e-2, 1e-1, 10),
                 kd_range: Tuple[float, float, int] = (1e-3, 2e-2, 10)):
        """
        参数:
            env: HVAC 环境实例
            kp_range: (min, max, num_points) 比例增益搜索范围
            ki_range: (min, max, num_points) 积分增益搜索范围
            kd_range: (min, max, num_points) 微分增益搜索范围
        """
        # 调用父类初始化
        super().__init__(env)
        
        # 生成参数网格
        self.kp_values = numpy.linspace(kp_range[0], kp_range[1], kp_range[2])
        self.ki_values = numpy.linspace(ki_range[0], ki_range[1], ki_range[2])
        self.kd_values = numpy.linspace(kd_range[0], kd_range[1], kd_range[2])
        
        # 搜索范围记录
        self.kp_range = kp_range
        self.ki_range = ki_range
        self.kd_range = kd_range
        
        # 存储搜索结果
        self.search_results = []
        self.best_params = None
        
        print(f"[HVACSolverGridSearchPID] 初始化完成")
        print(f"  - 当前 PID 参数: kp={self.kp:.4f}, ki={self.ki:.4f}, kd={self.kd:.4f}")
        print(f"  - 搜索空间大小: {len(self.kp_values) * len(self.ki_values) * len(self.kd_values)}")
        print(f"  - kp 范围: [{kp_range[0]:.4f}, {kp_range[1]:.4f}], {kp_range[2]} 个点")
        print(f"  - ki 范围: [{ki_range[0]:.4f}, {ki_range[1]:.4f}], {ki_range[2]} 个点")
        print(f"  - kd 范围: [{kd_range[0]:.4f}, {kd_range[1]:.4f}], {kd_range[2]} 个点")
    
    def _evaluate_params(self, kp, ki, kd, max_steps):
        """
        评估单组参数的性能
        
        参数:
            kp, ki, kd: PID 参数
            max_steps: 最大测试步数
            
        返回:
            metrics: 性能指标字典
        """
        # 临时保存当前参数
        original_kp, original_ki, original_kd = self.kp, self.ki, self.kd
        original_acc_diff = numpy.copy(self.acc_diff)
        original_last_observation = numpy.copy(self.last_observation)
        
        # 设置测试参数
        self.kp, self.ki, self.kd = kp, ki, kd
        
        # 重置环境
        obs = self.env.reset()[0]
        self.acc_diff = numpy.zeros(len(self.sensors))
        self.last_observation = numpy.zeros(len(self.sensors))
        
        # 记录数据
        temp_errors = []
        rewards = []
        overshoots = []
        n_coolers = len(self.coolers)
        for step in range(max_steps):
            # 使用父类的 policy 方法
            action = 1 - self.policy(obs["sensor_readings"])[n_coolers:]
            
            # 执行动作
            obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 记录温度误差
            if isinstance(obs, dict) and 'sensor_readings' in obs:
                sensor_readings = obs['sensor_readings']
                error = numpy.abs(sensor_readings - self.target_temperature)
                temp_errors.extend(error.tolist())
                
                # 统计超调
                overshoot = numpy.sum(sensor_readings > self.target_temperature + 2.0)
                overshoots.append(overshoot)
            
            rewards.append(reward)
            
            if done:
                pass
                # break
        
        # 计算性能指标
        temp_errors = numpy.array(temp_errors)
        metrics = {
            'kp': kp,
            'ki': ki,
            'kd': kd,
            'mean_error': numpy.mean(temp_errors) if len(temp_errors) > 0 else float('inf'),
            'max_error': numpy.max(temp_errors) if len(temp_errors) > 0 else float('inf'),
            'std_error': numpy.std(temp_errors) if len(temp_errors) > 0 else float('inf'),
            'error_within_2C': numpy.mean(temp_errors <= 2.0) * 100 if len(temp_errors) > 0 else 0,
            'error_within_1C': numpy.mean(temp_errors <= 1.0) * 100 if len(temp_errors) > 0 else 0,
            'mean_reward': numpy.mean(rewards) if rewards else 0,
            'mean_overshoot': numpy.mean(overshoots) if overshoots else 0,
        }
        
        # 恢复原始参数
        self.kp, self.ki, self.kd = original_kp, original_ki, original_kd
        self.acc_diff = original_acc_diff
        self.last_observation = original_last_observation
        
        return metrics
    
    def search(self, max_steps=1000, save_path=None, apply_best=True):
        """
        执行网格搜索，找到最优 PID 参数
        
        参数:
            max_steps: 每组参数测试步数
            save_path: 结果保存路径（可选）
            apply_best: 是否自动应用最优参数
            
        返回:
            best_params: 最优参数字典
        """
        import itertools
        import json
        from datetime import datetime
        
        total_combinations = len(self.kp_values) * len(self.ki_values) * len(self.kd_values)
        
        print(f"\n[Grid Search] 开始搜索，共 {total_combinations} 组参数")
        print("=" * 60)
        
        count = 0
        best_score = float('inf')
        
        # 遍历所有参数组合
        for kp, ki, kd in itertools.product(self.kp_values, self.ki_values, self.kd_values):
            count += 1
            
            # 评估参数
            metrics = self._evaluate_params(kp, ki, kd, max_steps)
            self.search_results.append(metrics)
            
            # 计算综合得分（越小越好）
            score = (0.5 * metrics['mean_error'] + 
                    0.3 * (100 - metrics['error_within_2C']) / 10 + 
                    0.2 * metrics['max_error'])
            metrics['score'] = score
            
            # 更新最优参数
            if score < best_score:
                best_score = score
                self.best_params = metrics.copy()
                print(f"[{count}/{total_combinations}] 新最优: kp={kp:.4f}, ki={ki:.4f}, kd={kd:.4f}")
                print(f"    得分: {score:.4f}, 平均误差: {metrics['mean_error']:.4f}°C, "
                      f"±2°C: {metrics['error_within_2C']:.1f}%")
            
            # 每 10 组打印进度
            if count % 10 == 0:
                print(f"[进度] {count}/{total_combinations} 完成")
        
        # 输出最终结果
        print("\n" + "=" * 60)
        print("[Grid Search] 搜索完成！")
        print("=" * 60)
        print(f"最优参数:")
        print(f"  kp = {self.best_params['kp']:.6f}")
        print(f"  ki = {self.best_params['ki']:.6f}")
        print(f"  kd = {self.best_params['kd']:.6f}")
        print(f"\n性能指标:")
        print(f"  平均误差: {self.best_params['mean_error']:.4f}°C")
        print(f"  最大误差: {self.best_params['max_error']:.4f}°C")
        print(f"  标准差: {self.best_params['std_error']:.4f}°C")
        print(f"  ±1°C范围内: {self.best_params['error_within_1C']:.1f}%")
        print(f"  ±2°C范围内: {self.best_params['error_within_2C']:.1f}%")
        
        # 自动应用最优参数
        if apply_best and self.best_params:
            self.kp = self.best_params['kp']
            self.ki = self.best_params['ki']
            self.kd = self.best_params['kd']
            print(f"\n已应用最优参数到当前实例")
        
        # 保存结果
        if save_path:
            self._save_results(save_path)
        
        return self.best_params
    
    def _save_results(self, save_path):
        """保存搜索结果到 JSON 文件"""
        import json
        from datetime import datetime
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        result_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'best_params': self.best_params,
            'all_results': self.search_results,
            'search_space': {
                'kp_range': self.kp_range,
                'ki_range': self.ki_range,
                'kd_range': self.kd_range,
                'kp_values': self.kp_values.tolist(),
                'ki_values': self.ki_values.tolist(),
                'kd_values': self.kd_values.tolist(),
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n结果已保存到: {save_path}")
    
    def apply_params(self, kp=None, ki=None, kd=None):
        """
        手动应用 PID 参数
        
        参数:
            kp, ki, kd: PID 参数（None 则保持不变）
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd
        print(f"已更新参数: kp={self.kp:.6f}, ki={self.ki:.6f}, kd={self.kd:.6f}")