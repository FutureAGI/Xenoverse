import sys
import numpy
import numpy as np
from numpy import random as rnd
from xenoverse.utils import RandomFourier

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
        return f"{type(self).__name__}({self.loc[0]:.1f},{self.loc[1]:.1f})\n"


class BaseSensor(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # period of the sensor noise drift
        period = rnd.randint(100000, 300000000)  
        # drift of the temperature
        self.drift_periodical = RandomFourier(ndim=1, 
                                              max_order=3, 
                                              max_item=3, 
                                              max_steps=period,
                                              box_size=rnd.uniform(0.05, 0.5))

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

class BaseVentilator(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wall_offset = numpy.array([[-0.5, 0], [0, -0.5]])  # 墙相对位置

        self.power_eff_vent = rnd.uniform(1.0, 10.0)  # 功率效率
        self.cooler_eer_base = rnd.uniform(2.0, 5.0)  # cooler effect
        self.cooler_eer_decay_start = rnd.uniform(8.0, 15.0)  # 制冷效率衰减起点
        self.cooler_eer_zero_point = rnd.uniform(16, 24)  # 制冷效率为0的点
        self.cooler_eer_reverse = rnd.uniform(5.0, 10.0)  # 随机生成一个介于 5.0 和 10.0 之间的浮动值，表示当温度差为负时的冷却效率。

        # Impact Range
        self.cooler_decay = rnd.uniform(1.0, 4.0)
        self.heat_decay = rnd.uniform(0.5, 1.0)

        self.cooler_diffuse, self.cooler_vent_diffuse = wind_diffuser(
            self.cell_walls, self.loc,
            self.cell_size, self.cooler_decay)
        self.heat_diffuse, self.heat_vent_diffuse = wind_diffuser(
            self.cell_walls, self.loc,
            self.cell_size, self.heat_decay)

    def power_heat(self, t):
        return 0.0

    def step(self, power_cool, power_vent, time, building_state=None, ambient_state=None):
        heat = self.power_heat(time)
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

        # 计算能力变化矩阵
        delta_energy = - cooler_efficiency * self.cooler_diffuse * power_cool \
                       + self.heat_diffuse * heat

        # 计算由于通风引起的热传递系数变化
        delta_chtc = self.cooler_vent_diffuse * power_vent * self.power_eff_vent

        return {"delta_energy": delta_energy,
                "delta_chtc": delta_chtc,
                "heat": heat,
                "power": power_cool + power_vent}

class HeaterUnc(BaseVentilator):
    """
    Support defining the following parameters:
    - period_range: tuple, e.g. (86400, 604800) the range of period for heat source variation
    - heat_variant_scale: tuple, e.g. (3200, 12000) the range of scale for heat source variation
    - heat_base_range: tuple, e.g. (200.0, 1600.0) the range of base heat for heat source
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if("base_heater" in kwargs):
            self.base_heater = kwargs["base_heater"]
            self.base_factor = rnd.uniform(0.2, 0.8)
        else:
            self.base_heater = None
        if("period_range" in kwargs):
            self.period = rnd.randint(*kwargs["period_range"])
        if(self.base_heater == None):
            self.period = rnd.randint(43200, 86400)  # period of the heat source, 0.5 ~ 1 days
        else:
            self.period = rnd.randint(86400, 604800)  # period of the heat source, 1 ~ 7 days 
        if("heat_variant_scale" in kwargs):
            self.heat_variant_scale = rnd.uniform(*kwargs["heat_variant_scale"])
        else:
            self.heat_variant_scale = rnd.uniform(0.1, 0.5)
        
        if("heat_base_range" in kwargs):
            self.heat_base = rnd.uniform(*kwargs["heat_base_range"])
        else:

            self.heat_base = rnd.uniform(2000.0, 4000.0)
        
        

        self.heat_periodical = RandomFourier(ndim=1, max_order=5, max_item=8, max_steps=self.period, box_size=rnd.uniform(3200, 6800))

        self.heat_base = rnd.uniform(1000.0, 2000.0)

    def power_heat(self, t):
        # 根据t随机生成一个发热量
        return numpy.clip(self.heat_base + numpy.clip(self.heat_periodical(t)[0], 0, None), None, 8000)

    def __call__(self, t):
        res = super().step(0, 0, t)
        if(self.base_heater is not None):
            # if there is a base heater, use it
            # neglect the power and chtc change
            base = self.base_heater(t) 
            res["heat"] = base["heat"] * self.base_factor + res["heat"] * (1 - self.base_factor)
            res["delta_energy"] = base["delta_energy"] * self.base_factor + res["delta_energy"] * (1 - self.base_factor)
        return res

class Cooler(BaseVentilator):
    """
    set power indirectly with set temperature and return temperature
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Simulate different temperature control strategy of Coolers
        self.temp_diff_decay_ub = rnd.uniform(0.01, 2.0)
        self.temp_diff_decay_lb = rnd.uniform(-2.0, -0.01)

        self.max_cooling_power = 10000
        self.power_vent_min = rnd.uniform(50, 100)
        self.min_cooling_power = self.power_vent_min
        if (rnd.random() < 0.5):
            self.power_vent_ratio = rnd.uniform(0.005, 0.01)  # fixed ventilator ratio
        else:
            self.power_vent_ratio = 0.0
            self.power_vent_min = rnd.uniform(50, 100)  # fixed ventilator power

        # drift of return sensors
        period = rnd.randint(100000, 300000000)
        self.drift_periodical = RandomFourier(ndim=1, 
                                              max_order=3, 
                                              max_item=3, 
                                              max_steps=period,
                                              box_size=rnd.uniform(0.05, 0.5))
        
    def set_control_type(self, control_type):
        self.control_type = control_type

    # 根据设定温度和回风温度计算制冷功率
    def temperature_control(self, switch, value, t, building_state=None, ambient_state=None):
        # calculate the return temperature as the target for control
        env_temp = self.calc_return_temperature(building_state, t)
        if switch == 0:
            return super().step(0, 0, t, building_state=building_state, ambient_state=ambient_state)

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

        return super().step(power_cool, power_vent, t, building_state=building_state, ambient_state=ambient_state)
    
    def power_control(self, switch, value, t, building_state=None, ambient_state=None):
        if(switch == 0):
            power_cool, power_vent = 0, 0
        else:
            power_all = (self.max_cooling_power - self.min_cooling_power) * value + self.min_cooling_power
            power_vent = min(max(self.power_vent_ratio * power_all, self.power_vent_min), power_all)
            power_cool = power_all - power_vent

        return super().step(power_cool, power_vent, t, building_state=building_state, ambient_state=ambient_state)

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
        drift_t = self.drift_periodical(t)[0]

        return gt_t + drift_t

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
            if (i < 0 or j < 0 or i >= nx or j >= ny):
                continue

            # 领格行列索引
            ni = ci + i
            nj = cj + j

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
