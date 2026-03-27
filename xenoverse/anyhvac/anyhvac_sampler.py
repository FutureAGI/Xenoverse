import sys
import numpy
from numpy import random as rnd
import random
import copy
from .anyhvac_utils import BaseSensor, HeatCurve, HeaterUnc, Cooler, wind_diffuser
from .hvac_config import *


def HVACTaskSampler(control_type='Temperature',
                    target_temperature=None):
    nw = rnd.randint(ROOM_W_GRID_SIZE_LOW, ROOM_W_GRID_SIZE_HIGH) # width of the building, in cell number
    nl = rnd.randint(ROOM_L_GRID_SIZE_LOW, ROOM_L_GRID_SIZE_HIGH) # length of the building, in cell number
    cell_size = rnd.uniform(CELL_SIZE_LOW, CELL_SIZE_HIGH) # size of each cell, in meter
    floor_height = rnd.uniform(FLOOR_HEIGHT_LOW, FLOOR_HEIGHT_HIGH) # height of the building, in meter

    dw = nw * cell_size
    dl = nl * cell_size

    area = dw * dl # area of the building
    cell_volume = floor_height * cell_size * cell_size # volume of each cell

    chtc_array = numpy.random.uniform(INTERNAL_CHTC_LOW, INTERNAL_CHTC_HIGH, size=(nw + 1, nl + 1, 2)) # Convective Heat Transfer Coefficients
    hc_array = numpy.random.uniform(GRID_HC_LOW, GRID_HC_HIGH, size=(nw, nl)) * cell_volume # heat capacity inside the building
    wall_chtc = rnd.uniform(WALL_CHTC_LOW, WALL_CHTC_HIGH)
    chtc_array[0, :, 0] = wall_chtc
    chtc_array[nw, :, 0] = wall_chtc
    chtc_array[:, 0, 1] = wall_chtc
    chtc_array[:, nl, 1] = wall_chtc
    cell_walls = chtc_array < 5.0
    floorceil_chtc = rnd.uniform(FLOOR_CHTC_LOW, 
                                 FLOOR_CHTC_HIGH)

    n_sensors = max(int(area * rnd.uniform(SENSOR_DENSITY_LOW, SENSOR_DENSITY_HIGH)), 1)
    n_heaters = max(int(area * rnd.uniform(HEATER_DENSITY_LOW, HEATER_DENSITY_HIGH)), 1)
    n_coolers = max(int(area * rnd.uniform(COOLER_DENSITY_LOW, COOLER_DENSITY_HIGH)), 1)


    pts, wht = zip(*AMBIENT_TEMPERATURE_HIGH)
    sum_wht = sum(wht)
    eps = rnd.uniform(0.0, sum_wht)
    
    t_b = AMBIENT_TEMPERATURE_LOW
    for pt, w in AMBIENT_TEMPERATURE_HIGH:
        if eps < w:
            t_ambient = rnd.uniform(t_b, pt)
            break
        eps -= w
        t_b = pt

    print(f"Sample Ambient Temperature: {t_ambient}")

    if(target_temperature is not None):
        pass
    else:
        target_temperature = round(rnd.uniform(TARGET_TEMPERATURE_LOW, TARGET_TEMPERATURE_HIGH) * 2) / 2

    cooler_sensor_dift_std = rnd.uniform(1, 2)

    sensors = []
    equipments = []
    coolers = []
    timer = []
    for i in range(n_sensors):
        sensors.append(BaseSensor(nw, nl, cell_size, cell_walls, min_dist=1.2,
                    avoidance=sensors))
    base_heater = HeatCurve()

    for i in range(n_heaters):
        heater = HeaterUnc(nw, nl, cell_size, cell_walls, min_dist=1.2, avoidance=equipments, base_heater=base_heater)
        timer.append(heater.heat_curve.period)
        equipments.append(heater)
        hc_array[equipments[-1].nloc[0], equipments[-1].nloc[1]] += rnd.uniform(EQUIPMENT_HC_LOW, EQUIPMENT_HC_HIGH)
    for i in range(n_coolers):
        coolers.append(Cooler(nw, nl, cell_size, cell_walls, min_dist=min(cell_size, 2.0),
                    avoidance=coolers,
                    control_type=control_type,
                    target_temperature=target_temperature,
                    cooler_sensor_dift_std=cooler_sensor_dift_std))

    unify_cooler_coefficent = 0
    sample_ratio = random.uniform(0.0, 1.0)
    if sample_ratio < 0.4:
        base_cooler = coolers[0]
        for idx in range(n_coolers):
            UNIFY_COOLER_COEFFICIENT(base_cooler, coolers[idx])
        unify_cooler_coefficent = 1
    elif sample_ratio < 0.8:
        base_cooler = coolers[0]
        unify_ratio = random.uniform(0.7, 1.0)
        n_to_unify = max(1, int((n_coolers - 1) * unify_ratio))
        other_coolers = coolers[1:]
        coolers_to_unify = random.sample(other_coolers, n_to_unify)
        for cooler in coolers_to_unify:
            UNIFY_COOLER_COEFFICIENT(base_cooler, cooler)
        unify_cooler_coefficent = (n_to_unify + 1) / n_coolers

    print("unify_cooler_coefficent: ", unify_cooler_coefficent)

    return {
        'width': dw,
        'length': dl,
        'n_width': nw,
        'n_length': nl,
        'cell_size': cell_size,
        'floor_height': floor_height,
        'floorceil_chtc': floorceil_chtc,
        'sensors': sensors,
        'convection_coeffs': chtc_array,
        'heat_capacity': hc_array,
        'ambient_temp': t_ambient,
        'equipments': equipments,
        'coolers': coolers,
        'control_type': control_type,
        'target_temperature': target_temperature,
        'heater_timer': timer,
        'unify_cooler_coefficent': unify_cooler_coefficent
    }

def UNIFY_COOLER_COEFFICIENT(base_cooler, cooler):
    cooler.max_cooling_power = base_cooler.max_cooling_power
    cooler.power_vent_min = base_cooler.power_vent_min
    cooler.min_cooling_power = base_cooler.min_cooling_power
    cooler.power_vent_ratio = base_cooler.power_vent_ratio


    cooler.power_eff_vent = base_cooler.power_eff_vent
    cooler.cooler_eer_base = base_cooler.cooler_eer_base
    cooler.cooler_eer_decay_start = base_cooler.cooler_eer_decay_start
    cooler.cooler_eer_zero_point = base_cooler.cooler_eer_zero_point
    cooler.cooler_eer_reverse = base_cooler.cooler_eer_reverse
    cooler.cooler_diffuse_sigma = base_cooler.cooler_diffuse_sigma

    cooler.cooler_diffuse, cooler.cooler_vent_diffuse = wind_diffuser(
        cooler.cell_walls, cooler.loc,
        cooler.cell_size, cooler.cooler_diffuse_sigma)
