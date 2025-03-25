import sys
import numpy
from numpy import random as rnd
from xenoverse.utils import RandomFourier

class BaseNodes(object):
    def __init__(self, nw, nl, cell_size, cell_walls,
                 min_dist=0.5, avoidance=None):
        self.nw = nw
        self.nl = nl
        self.dw = nw * cell_size
        self.dl = nl * cell_size
        self.cell_size = cell_size
        self.cell_walls = cell_walls

        self.loc = numpy.array([rnd.randint(0, self.dw),
                rnd.uniform(0, self.dl)])
        if(avoidance is not None):
            while True:
                mdist = 1e+10
                for node in avoidance:
                    dist = ((node.loc - self.loc)**2).sum() ** 0.5
                    if(dist < mdist):
                        mdist = dist
                if(mdist < min_dist):
                    self.loc = numpy.array([rnd.randint(0, self.dw),
                            rnd.uniform(0, self.dl)])
                else:
                    break
        
        self.cloc = self.loc / self.cell_size
        self.nloc = self.cloc.astype(int)
    
    def __repr__(self):
        return f"{type(self).__name__}({self.loc[0]:.1f},{self.loc[1]:.1f})\n"

class BaseSensor(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, state):
        d_loc = self.cloc - self.nloc - 0.5
        sgrid = numpy.floor(d_loc).astype(int) + self.nloc
        dgrid = sgrid + 1
        sn = numpy.clip(sgrid, 0, [self.nw - 1, self.nl - 1])
        dn = numpy.clip(dgrid, 0, [self.nw - 1, self.nl - 1])
        vss = state[sn[0], sn[1]]
        vdd = state[dn[0], dn[1]]
        vsd = state[sn[0], dn[1]]
        vds = state[dn[0], sn[1]]
        k = d_loc - numpy.floor(d_loc)
        return float(vss * (1-k[0]) * (1-k[1])
            + vds * k[0] * (1-k[1])
            + vsd * (1-k[0]) * k[1] 
            + vdd * k[0] * k[1])

class BaseVentilator(BaseNodes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.wall_offset = numpy.array([[-0.5, 0], [0, -0.5]])  
        
        self.power_eff_vent = rnd.uniform(0.5, 1.0)
        self.cooler_eer_base = rnd.uniform(2.0, 5.0)  # cooler effect
        self.cooler_eer_decay_start = rnd.uniform(8.0, 15.0)
        self.cooler_eer_zero_point = rnd.uniform(16, 24) 
        self.cooler_eer_reverse = rnd.uniform(5.0, 10.0)        

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
        if(building_state is not None):
            temp_diff = ambient_state - building_state[*self.nloc]
        else:
            temp_diff = 2.0

        if(temp_diff < 0):
            cooler_efficiency = self.cooler_eer_reverse
        elif(temp_diff < self.cooler_eer_decay_start):
            cooler_efficiency = self.cooler_eer_base
        elif(temp_diff < self.cooler_eer_zero_point):
            factor = (self.cooler_eer_zero_point - temp_diff) / (self.cooler_eer_zero_point - self.cooler_eer_decay_start)
            cooler_efficiency = self.cooler_eer_base * factor
        else:
            cooler_efficiency = 0.0

        delta_energy = - cooler_efficiency * self.cooler_diffuse * power_cool \
                        + self.heat_diffuse * heat
        delta_chtc = self.cooler_vent_diffuse * power_vent * self.power_eff_vent

        return {"delta_energy": delta_energy, 
                "delta_chtc": delta_chtc,
                "heat": heat}

class HeaterUnc(BaseVentilator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        period = rnd.randint(86400, 604800) # period of the heat source
        self.heat_periodical = RandomFourier(ndim=1, max_order=128, max_item=8, max_steps=period, box_size=rnd.uniform(3200, 12000))
        self.heat_base = rnd.uniform(200.0, 1600.0)

    def power_heat(self, t):
        return numpy.clip(self.heat_base + numpy.clip(self.heat_periodical(t)[0], 0, None), None, 20000)
    
    def __call__(self, t):
        return super().step(0, 0, t)

class Cooler(BaseVentilator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if(rnd.random() < 0.5):
            self.power_ratio = rnd.uniform(0.1, 0.2) # fixed ventilator ratio
            self.power_residual = 500
        else:
            self.power_residual = rnd.uniform(500, 1500) # fixed ventilator power
            self.power_ratio = 0.0

    def __call__(self, power, t, building_state=None, ambient_state=None):
        power_vent = min(max(self.power_ratio * power, self.power_residual), power)
        power_cool = power - power_vent

        return super().step(power_cool, power_vent, t, building_state=building_state, ambient_state=ambient_state)
    
def wind_diffuser(cell_wall, src, cell_size, sigma):
    src_grid = src / cell_size
    diffuse_queue = [src_grid]
    neighbor = [(-1, 0), (0, -1), (1, 0), (0, 1)]
    nx, ny, _ = cell_wall.shape
    diffuse_mat = numpy.zeros((nx - 1, ny - 1))
    diffuse_wall = numpy.zeros((nx, ny, 2))
    diffuse_mat[int(src_grid[0]), int(src_grid[1])] = 1.0

    while len(diffuse_queue) > 0:
        loc = diffuse_queue.pop(0)
        ci, cj = int(loc[0]), int(loc[1])
        for i, j in neighbor:
            if(i < 0 or j < 0 or i >= nx or j >= ny):
                continue

            ni = ci + i
            nj = cj + j
            wi = ci + max(i, 0)
            wj = cj + max(j, 0)

            w = int(i == 0)
            if(cell_wall[wi, wj, w]):
                continue

            # calculate cell diffuse factor
            dist = numpy.sum(((loc - numpy.array([ni + 0.5, nj + 0.5])) * cell_size / sigma) ** 2)
            k = numpy.exp(-dist) * diffuse_mat[ci, cj]
            if(k > diffuse_mat[ni, nj]):
                diffuse_mat[ni, nj] = k
                if(k > 1.0e-3):
                    diffuse_queue.append(numpy.array([ni + 0.5, nj + 0.5]))
            
            # calculate wall diffuse factor
            dist = numpy.sum(((loc - numpy.array([0.5 * ni + 0.5 * ci, 0.5 * nj + 0.5 * cj])) * cell_size / sigma) ** 2)
            k = numpy.exp(-dist) * diffuse_mat[ci, cj]
            if(k > diffuse_wall[wi, wj, w]):
                diffuse_wall[wi, wj, w] = k

    diffuse_mat /= numpy.sum(diffuse_mat)
    return diffuse_mat, diffuse_wall