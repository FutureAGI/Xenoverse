import sys
import numpy
import numpy as np
from numpy import random as rnd
from xenoverse.utils import RandomFourier
from .hvac_config import *

class BaseNodes(object):
    def __init__(self, nw, nl, cell_size, cell_walls,
                 min_dist=0.5, avoidance=None, 
                 **kwargs):
        self.nw = nw
        self.nl = nl
        self.dw = nw * cell_size
        self.dl = nl * cell_size
        self.cell_size = cell_size
        self.cell_walls = cell_walls

        for key, val in kwargs.items():
            setattr(self, key, val)

        # 随机节点坐标
        self.loc = numpy.array([rnd.randint(0, self.dw),
                                rnd.uniform(0, self.dl)])
        if (avoidance is not None):  # 随机位置保持最小距离
            while True:
                mdist = 1e+10
                for node in avoidance:
                    dist = ((node.loc - self.loc) ** 2).sum() ** 0.5
                    if (dist < mdist):
                        mdist = dist
                if (mdist < min_dist):
                    self.loc = numpy.array([rnd.randint(0, self.dw),
                                            rnd.uniform(0, self.dl)])
                else:
                    break
        # 节点坐标转换为单元位置
        self.cloc = self.loc / self.cell_size
        # 取整
        self.nloc = self.cloc.astype(int)

    def __repr__(self):
        res_str = f"{type(self).__name__}({self.loc[0]:.1f},{self.loc[1]:.1f})\n"
        for key, val in self.__dict__.items():
            res_str += f"  {key}: {val}\n"
        return res_str

class BaseSensor(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # period of the sensor noise drift
        period = rnd.randint(SENSOR_DRIFT_PERIOD_LOW * 60,
                             SENSOR_DRIFT_PERIOD_HIGH * 60)  
        # drift of the sensor reading, simulating the sensor drift in real world, with a random period and scale
        self.drift_periodical = RandomFourier(ndim=1, 
                                              max_order=SENSOR_DRIFT_FOURIER_MAX_ORDER, 
                                              max_item=SENSOR_DRIFT_FOURIER_MAX_ITEMS, 
                                              max_steps=period,
                                              box_size=min(rnd.exponential(scale=SENSOR_DRIFT_MEAN),
                                                           SENSOR_DRIFT_UPPER_BOUND))

    def __call__(self, state, t):
        # 计算单元格内中心偏移量
        d_loc = self.cloc - self.nloc - 0.5

        """
        [3.2,4.6]
        [3,4]
        [2,4]
        [3,5]
            
        """
        # 计算最近的2个单元格坐标
        sgrid = numpy.floor(d_loc).astype(int) + self.nloc
        dgrid = sgrid + 1

        # 限制坐标范围
        sn = numpy.clip(sgrid, 0, [self.nw - 1, self.nl - 1])
        dn = numpy.clip(dgrid, 0, [self.nw - 1, self.nl - 1])

        # 最近的4个cell状态
        vss = state[sn[0], sn[1]]
        vdd = state[dn[0], dn[1]]
        vsd = state[sn[0], dn[1]]
        vds = state[dn[0], sn[1]]

        # 计算差值系数（表示和每个区域的距离）
        k = d_loc - numpy.floor(d_loc)
        
        # ground truth temperature
        gt_t = float(vss * (1 - k[0]) * (1 - k[1])
                     + vds * k[0] * (1 - k[1])
                     + vsd * (1 - k[0]) * k[1]
                     + vdd * k[0] * k[1])
        
        drift = self.drift_periodical(t)[0]

        return gt_t + drift

class CoolerVentilator(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cooler 专用参数
        self.power_eff_vent = rnd.uniform(COOLER_VENT_EFFICIENCY_LOW, COOLER_VENT_EFFICIENCY_HIGH)
        self.cooler_eer_base = rnd.uniform(COOLER_EER_BASE_LOW, COOLER_EER_BASE_HIGH)
        self.cooler_eer_decay_start = rnd.uniform(COOLER_EER_DECAY_START_HIGH, COOLER_EER_DECAY_START_HIGH)
        self.cooler_eer_zero_point = rnd.uniform(COOLER_EER_ZERO_POINT_LOW, COOLER_EER_ZERO_POINT_HIGH)
        self.cooler_eer_reverse = rnd.uniform(COOLER_EER_REVERSE_LOW, COOLER_EER_REVERSE_HIGH)
        self.cooler_diffuse_sigma = rnd.uniform(COOLER_SPACE_INSTANT_DIFFUSION_LOW, COOLER_SPACE_INSTANT_DIFFUSION_HIGH)

        # Cooler 专用扩散矩阵
        self.cooler_diffuse, self.cooler_vent_diffuse = wind_diffuser(
            self.cell_walls, self.loc,
            self.cell_size, self.cooler_diffuse_sigma)

    def step(self, power_cool, power_vent, building_state=None, ambient_state=None):
        """Cooler 的 step 方法"""
        if (building_state is not None):
            temp_diff = ambient_state - building_state[tuple(self.nloc)]
        else:
            temp_diff = 2.0

        if (temp_diff < 0):
            cooler_efficiency = self.cooler_eer_reverse
        elif (temp_diff < self.cooler_eer_decay_start):
            cooler_efficiency = self.cooler_eer_base
        elif (temp_diff < self.cooler_eer_zero_point):
            factor = (self.cooler_eer_zero_point - temp_diff) / (
                    self.cooler_eer_zero_point - self.cooler_eer_decay_start)
            cooler_efficiency = self.cooler_eer_base * factor
        else:
            cooler_efficiency = 0.0

        delta_energy = - cooler_efficiency * self.cooler_diffuse * power_cool

        delta_chtc = self.cooler_vent_diffuse * power_vent * self.power_eff_vent

        return {"delta_energy": delta_energy,
                "delta_chtc": delta_chtc,
                "power": power_cool + power_vent}

class HeaterVentilator(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.heat_diffuse_sigma = rnd.uniform(HEAT_SPACE_INSTANT_DIFFSION_LOW, HEAT_SPACE_INSTANT_DIFFUSION_HIGH)
        self.heat_diffuse, self.heat_vent_diffuse = wind_diffuser(
            self.cell_walls, self.loc,
            self.cell_size, self.heat_diffuse_sigma)

    def step(self, power_heat, power_vent):
        
        delta_energy = self.heat_diffuse * power_heat
        delta_chtc = self.heat_vent_diffuse * power_vent if hasattr(self, 'power_eff_vent') else 0

        return {"delta_energy": delta_energy,
                "delta_chtc": delta_chtc,
                "heat": power_heat,
                "power": power_vent}

class HeatCurve:
    def __init__(self, *args, **kwargs):
        if("period_range" in kwargs):
            self.period = rnd.randint(*kwargs["period_range"])
            self.period = self.period * 60
        else:
            self.period = rnd.randint(HEAT_SOURCE_PERIOD_RANGE_LOW * 60, 
                                      HEAT_SOURCE_PERIOD_RANGE_HIGH * 60)  # period of the heat source 
            self.period = self.period * 60
        if("heat_variant_scale" in kwargs):
            self.heat_variant_scale = rnd.uniform(*kwargs["heat_variant_scale"])
        else:
            self.heat_variant_scale = rnd.uniform(HEAT_SOURCE_VARIANT_SCALE_LOW, 
                                                    HEAT_SOURCE_VARIANT_SCALE_HIGH)
        
        if("heat_base_range" in kwargs):
            self.heat_base = rnd.uniform(*kwargs["heat_base_range"])
        else:

            self.heat_base = rnd.uniform(BASE_HEAT_SOURCE_PERIOD_RANGE_LOW, 
                                         BASE_HEAT_SOURCE_PERIOD_RANGE_HIGH)

        self.heat_periodical = RandomFourier(ndim=1, max_order=HEAT_SOURCE_FOURIER_MAX_ORDER, 
                                             max_item=HEAT_SOURCE_FOURIER_MAX_ITEM, 
                                             max_steps=self.period, 
                                             box_size=self.heat_variant_scale)
        
    def power_heat(self, t):
        return numpy.clip(self.heat_base + numpy.clip(self.heat_periodical(t)[0], 0, None), 
                          None, MAX_HEAT_SOURCE_POWER)

class HeaterUnc(HeaterVentilator):
    """
    Support defining the following parameters:
    - period_range: tuple, e.g. (86400, 604800) the range of period for heat source variation
    - heat_base_range: tuple, e.g. (200.0, 1600.0) the range of base heat for heat source
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if("base_heater" in kwargs):
            self.base_heater = kwargs["base_heater"]
            self.base_factor = rnd.uniform(HEAT_SOURCE_BASE_FACTOR_LOW, 
                                           HEAT_SOURCE_BASE_FACTOR_HIGH)
        else:
            self.base_heater = None
        
        self.heat_curve = HeatCurve(*args, **kwargs)

    def power_heat(self, t):
        if self.base_heater is not None:
            base_heat = self.base_heater.power_heat(t)
            power_heat = self.heat_curve.power_heat(t)
            return self.base_factor * base_heat + (1-self.base_factor) * power_heat
        else:
            return self.heat_curve.power_heat(t)

    def __call__(self, t):
        heat = self.power_heat(t)
        res = super().step(power_heat=heat, power_vent=0)
        return res

class Cooler(CoolerVentilator):
    """
    set power indirectly with set temperature and return temperature
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulate different temperature control strategy of Coolers
        self.temp_diff_decay_ub = rnd.uniform(COOLER_DIFF_DECAY_UB_LOW, COOLER_DIFF_DECAY_UB_HIGH)
        self.temp_diff_decay_lb = rnd.uniform(COOLER_DIFF_DECAY_LB_LOW, COOLER_DIFF_DECAY_LB_HIGH)

        self.max_cooling_power = rnd.uniform(COOLER_MAX_COOLING_POWER_LOW, COOLER_MAX_COOLING_POWER_HIGH)
        self.power_vent_min = rnd.uniform(COOLER_POWER_VENT_MIN_LOW, COOLER_POWER_VENT_MIN_HIGH)
        self.min_cooling_power = self.power_vent_min
        if (rnd.random() < COOLER_VENT_FIXED_RATIO_FACTOR):
            self.power_vent_ratio = rnd.uniform(COOLER_VENT_FIXED_RATIO_LOW, COOLER_VENT_FIXED_RATIO_HIGH)  # fixed ventilator ratio
        else:
            self.power_vent_ratio = 0.0
            self.power_vent_min = rnd.uniform(COOLER_VENT_FIXED_POWER_LOW, COOLER_VENT_FIXED_POWER_HIGH)  # fixed ventilator power

        # drift of return sensors
        max_bound = min(32 - kwargs["target_temperature"] - 2, 6)
        min_bound = -max_bound
        cooler_sensor_dift_std = kwargs["cooler_sensor_dift_std"]
        self.cooler_sensor_drift = RealisticSensorNoise(gaussian_std=cooler_sensor_dift_std,
                                                        min_bound=min_bound, 
                                                        max_bound=max_bound)
        
    def set_control_type(self, control_type):
        self.control_type = control_type

    # 根据设定温度和回风温度计算制冷功率
    def temperature_control(self, switch, value, t, building_state=None, ambient_state=None):
        # calculate the return temperature as the target for control
        env_temp = self.calc_return_temperature(building_state, t)
        self.return_temperature = env_temp
        if switch == 0:
            return super().step(power_cool=0, 
                                power_vent=0,
                                building_state=building_state, 
                                ambient_state=ambient_state)
        

        set_temp = value
        temp_diff = env_temp - set_temp
        ratio = 0

        # Proportional controller
        if temp_diff > self.temp_diff_decay_ub:
            ratio = 1
        elif temp_diff < self.temp_diff_decay_lb:
            ratio = 0
        else:
            ratio = (temp_diff - self.temp_diff_decay_lb) / (self.temp_diff_decay_ub - self.temp_diff_decay_lb)
        power_all = (self.max_cooling_power - self.min_cooling_power) * ratio + self.min_cooling_power

        power_vent = min(max(self.power_vent_ratio * power_all, self.power_vent_min), power_all)
        power_cool = power_all - power_vent
        #print(temp_diff, power_cool, power_vent)

        return super().step(power_cool=power_cool, 
                            power_vent=power_vent,
                            building_state=building_state, 
                            ambient_state=ambient_state)
    
    def power_control(self, switch, value, t, building_state=None, ambient_state=None):
        if(switch == 0):
            power_cool, power_vent = 0, 0
        else:
            power_all = (self.max_cooling_power - self.min_cooling_power) * value + self.min_cooling_power
            power_vent = min(max(self.power_vent_ratio * power_all, self.power_vent_min), power_all)
            power_cool = power_all - power_vent

        return super().step(power_cool=power_cool, 
                            power_vent=power_vent, 
                            building_state=building_state, 
                            ambient_state=ambient_state)

    def __call__(self, *args, **kwargs):
        if(self.control_type.lower()=="power"):
            return self.power_control(*args, **kwargs)
        elif(self.control_type.lower()=="temperature"):
            return self.temperature_control(*args, **kwargs)

    def calc_return_temperature(self, state, t):
        d_loc = self.cloc - self.nloc - 0.5

        sgrid = numpy.floor(d_loc).astype(int) + self.nloc
        dgrid = sgrid + 1

        # 限制坐标范围
        sn = numpy.clip(sgrid, 0, [self.nw - 1, self.nl - 1])
        dn = numpy.clip(dgrid, 0, [self.nw - 1, self.nl - 1])
        vss = state[sn[0], sn[1]]
        vdd = state[dn[0], dn[1]]
        vsd = state[sn[0], dn[1]]
        vds = state[dn[0], sn[1]]
        k = d_loc - numpy.floor(d_loc)

        gt_t = float(vss * (1 - k[0]) * (1 - k[1])
                     + vds * k[0] * (1 - k[1])
                     + vsd * (1 - k[0]) * k[1]
                     + vdd * k[0] * k[1])
        
        # Add temperature drifting to return temperature measure
        noise_t = self.cooler_sensor_drift(t,gt_t)

        return noise_t
    
    def reset(self):
        self.cooler_sensor_drift.reset()

def wind_diffuser(cell_wall, src, cell_size, sigma):
    # 空气扩散计算
    src_grid = src / cell_size  # 扩散源的网格坐标
    diffuse_queue = [src_grid]  # 扩散源队列
    neighbor = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    nx, ny, _ = cell_wall.shape  # 墙网格尺寸（按cell分）
    diffuse_mat = numpy.zeros((nx - 1, ny - 1))  # 初始化扩散矩阵
    diffuse_wall = numpy.zeros((nx, ny, 2))  # 初始化墙体扩散矩阵
    diffuse_mat[int(src_grid[0]), int(src_grid[1])] = 1.0  # 扩散源位置系数1.0

    while len(diffuse_queue) > 0:
        loc = diffuse_queue.pop(0)  # 当前计算的扩散源
        ci, cj = int(loc[0]), int(loc[1])  # 扩散源的网格坐标
        for i, j in neighbor:  # 遍历四个方向

            # 领格行列索引
            ni = ci + i
            nj = cj + j

            if (ni < 0 or nj < 0 or ni >= nx - 1 or nj >= ny - 1):
                continue


            # 墙体行列索引
            wi = ci + max(i, 0)
            wj = cj + max(j, 0)

            w = int(i == 0)  # 墙体方向

            if (cell_wall[wi, wj, w]):  # 墙体存在 跳过
                continue

            # calculate cell diffuse factor

            # 计算扩散位置和邻点中心的距离
            dist = numpy.sum(((loc - numpy.array([ni + 0.5, nj + 0.5])) * cell_size / sigma) ** 2)

            # 计算扩散系数
            k = numpy.exp(-dist) * diffuse_mat[ci, cj]

            if (k > diffuse_mat[ni, nj]):  # 更新扩散系数 如果大于当前值
                diffuse_mat[ni, nj] = k
                if (k > 1.0e-3):  # 如果扩散系数大于1.0e-3 加入扩散队列
                    diffuse_queue.append(numpy.array([ni + 0.5, nj + 0.5]))

            # calculate wall diffuse factor
            dist = numpy.sum(((loc - numpy.array([0.5 * ni + 0.5 * ci, 0.5 * nj + 0.5 * cj])) * cell_size / sigma) ** 2)
            k = numpy.exp(-dist) * diffuse_mat[ci, cj]
            # 更新墙体扩散系数
            if (k > diffuse_wall[wi, wj, w]):
                diffuse_wall[wi, wj, w] = k

    diffuse_mat /= numpy.sum(diffuse_mat)  # 归一化
    return diffuse_mat, diffuse_wall

class RealisticSensorNoise:
    """
    真实传感器噪声生成器（带延迟和平滑）
    
    物理模型：
    - 传输延迟：30-120秒（队列）
    - 热惯性：一阶低通滤波（时间常数 = 延迟时间/3）
    - 基础偏差：固定偏移
    
    实际数据特征：
    - 左侧真值变化快（30秒步长）
    - 右侧读数变化缓慢平滑
    - 有明显延迟和惯性
    """
    
    def __init__(self, 
                 gaussian_mean=0.5, 
                 gaussian_std=1.5, 
                 min_bound=-6.0, 
                 max_bound=6.0,
                 sign_positive_prob=0.65,
                 delay_range=(5, 30)):
        """
        参数:
            gaussian_mean: 正态分布均值
            gaussian_std: 正态分布标准差
            min_bound: 拒绝采样最小边界
            max_bound: 拒绝采样最大边界
            sign_positive_prob: 正值概率
            delay_range: 延迟范围（秒），(min, max)
        """
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.sign_positive_prob = sign_positive_prob
        
        # 1. 生成基础偏差（通过拒绝采样）
        rejection_count = 0
        while True:
            sample = rnd.normal(gaussian_mean, gaussian_std)
            if min_bound <= sample <= max_bound:
                if rnd.random() < sign_positive_prob:
                    self.base_bias = abs(sample)
                else:
                    self.base_bias = -abs(sample)
                break
            rejection_count += 1
        
        # 2. 生成随机延迟
        self.delay_seconds = rnd.uniform(*delay_range)
        
        # 3. 计算滤波器系数（时间常数 = 延迟/3）
        self.time_constant = self.delay_seconds / 3.0  # 经验值
        
        # 4. 初始化状态
        self.input_queue = []          # 输入队列（延迟用）
        self.last_output = None
        self.step_count = 0
        self.last_call_time = None
        self.total_time_elapsed = 0.0
    
    def __call__(self, t, true_temperature):
        """
        获取时刻 t 的传感器读数
        
        参数:
            t: 时间步
            true_temperature: 真实温度值
            
        返回:
            float: 传感器读数（延迟 + 平滑）
        """
        # 计算与上次调用的时间间隔
        if self.last_call_time is None:
            dt = 0.0
        else:
            dt = t - self.last_call_time

        if self.last_output is None:
            self.last_output = true_temperature + self.base_bias
        
        self.last_call_time = t
        self.total_time_elapsed += dt
        
        # 计算当前输入（真值 + 基础偏差）
        current_input = true_temperature + self.base_bias
        
        # 将输入加入时间戳队列
        self.input_queue.append((t, current_input))
        
        # 清理过期的队列数据（超过延迟时间）
        cutoff_time = t - self.delay_seconds
        while len(self.input_queue) > 0 and self.input_queue[0][0] < cutoff_time:
            self.input_queue.pop(0)
        
        # 获取延迟后的输入（队列头部）
        if len(self.input_queue) > 0:
            delayed_input = self.input_queue[0][1]
        else:
            delayed_input = current_input
        
        # 一阶低通滤波（使用实际时间间隔）
        if dt > 0:
            # 计算滤波系数：α = exp(-dt / τ)
            alpha = np.exp(-dt / self.time_constant)
        else:
            alpha = 1.0  # 如果是第一次调用，不滤波
        
        # 应用滤波
        output = alpha * self.last_output + (1 - alpha) * delayed_input
        
        # 更新状态
        self.last_output = output
        
        return output
    
    def reset(self):
        """重置传感器状态"""
        self.input_queue = []
        self.last_output = 0.0
        self.last_call_time = None
        self.total_time_elapsed = 0.0