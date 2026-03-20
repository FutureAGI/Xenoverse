import sys
import numpy
from numpy import random as rnd
from .anyhvac_utils import BaseSensor, HeaterUnc, Cooler
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
    sensors = []
    equipments = []
    coolers = []
    timer = []
    for i in range(n_sensors):
        sensors.append(BaseSensor(nw, nl, cell_size, cell_walls, min_dist=1.2,
                    avoidance=sensors))
    base_heater = HeaterUnc(nw, nl, cell_size, cell_walls, min_dist=1.2,
                    avoidance=equipments)
    for i in range(n_heaters):
        heater = HeaterUnc(nw, nl, cell_size, cell_walls, min_dist=1.2, avoidance=equipments, base_heater=base_heater)
        timer.append(heater.period)
        equipments.append(heater)
        hc_array[equipments[-1].nloc[0], equipments[-1].nloc[1]] += rnd.uniform(EQUIPMENT_HC_LOW, EQUIPMENT_HC_HIGH)
    for i in range(n_coolers):
        coolers.append(Cooler(nw, nl, cell_size, cell_walls, min_dist=min(cell_size, 2.0),
                    avoidance=coolers,
                    control_type=control_type))
        
    if(target_temperature is not None):
        target_temperature = target_temperature
    else:
        if(rnd.choice([True, False])):
            target_temperature = rnd.uniform(TARGET_TEMPERATURE_LOW, TARGET_TEMPERATURE_HIGH)
        else:
            target_temperature = rnd.uniform(TARGET_TEMPERATURE_LOW, TARGET_TEMPERATURE_HIGH, size=(n_sensors))

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
    }
