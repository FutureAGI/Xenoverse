import numpy
import math
from numba import njit

PI_4 = 0.7853981
PI_2 = 1.5707963
PI = 3.1415926
t_PI = 6.2831852
PI2d = 57.29578

OFFSET_10 = numpy.asarray([0.5, 0.5], dtype="float64")
OFFSET_01 = numpy.asarray([-0.5, 0.5], dtype="float64")
OFFSET_m0 = numpy.asarray([-0.5, -0.5], dtype="float64")
OFFSET_0m = numpy.asarray([0.5, -0.5], dtype="float64")

DEFAULT_ACTION_SPACE_16 = [(0.0, 0.5), 
                        (0.05, 0.0), (-0.05, 0.0),
                        (0.1, 0.0), (-0.1, 0.0),
                        (0.2, 0.0), (-0.2, 0.0),
                        (0.3, 0.0), (-0.3, 0.0),
                        (0.5, 0.0), (-0.5, 0.0), 
                        (0.0, 1.0),
                        (0.05, 1.0), 
                        (-0.05, 1.0), 
                        (0.10, 1.0), 
                        (-0.10, 1.0), 
                        ]

DEFAULT_ACTION_SPACE_32 = [(0.0, 0.2), 
                        (0.02, 0.0), (-0.02, 0.0),
                        (0.05, 0.0), (-0.05, 0.0),
                        (0.1, 0.0), (-0.1, 0.0),
                        (0.2, 0.0), (-0.2, 0.0),
                        (0.3, 0.0), (-0.3, 0.0),
                        (0.4, 0.0), (-0.4, 0.0),
                        (0.5, 0.0), (-0.5, 0.0), 
                        (0.0, 0.5), (0.0, 1.0),
                        (0.02, 0.5), (0.02, 1.0), 
                        (-0.02, 0.5), (-0.02, 1.0),
                        (0.05, 0.5), (0.05, 1.0), 
                        (-0.05, 0.5), (-0.05, 1.0),
                        (0.10, 0.5), (0.10, 1.0), 
                        (-0.10, 0.5), (-0.10, 1.0),
                        (0.0, -0.2),
                        (0.1, -0.2), (-0.1, -0.2)
                        ]

@njit(cache=True)
def angle_normalization(t):
    while(t > PI):
        t -= t_PI
    while(t < -PI):
        t += t_PI
    return t

@njit(cache=True)
def nearest_point(pos, line_1, line_2):
    unit_ori = line_2 - line_1
    edge_norm = numpy.sqrt(numpy.sum(unit_ori * unit_ori))
    unit_ori /= max(1.0e-6, edge_norm)

    dist_1 = numpy.sum((pos - line_1) * unit_ori)
    if(dist_1 > edge_norm):
        return numpy.sqrt(numpy.sum((pos - line_2) ** 2)), numpy.copy(line_2)
    elif(dist_1 < 0):
        return numpy.sqrt(numpy.sum((pos - line_1) ** 2)), numpy.copy(line_1)
    else:
        line_p = line_1 + dist_1 * unit_ori
        return numpy.sqrt(numpy.sum((pos - line_p) ** 2)), numpy.copy(line_p)

@njit(cache=True)
def collision_force(dist_vec, cell_size, col_dist):
    dist = float(numpy.sqrt(numpy.sum(dist_vec * dist_vec)))
    eff_col_dist = col_dist / cell_size
    if(dist > 0.708 + eff_col_dist):
        return numpy.array([0.0, 0.0], dtype="float64")
    if(abs(dist_vec[0]) < 0.5 and abs(dist_vec[1]) < 0.5):
        return numpy.float64(0.50 / max(dist, 1.0e-6) * (0.708 + eff_col_dist - dist) * cell_size) * dist_vec
    x_pos = (dist_vec[0] + dist_vec[1] > 0)
    y_pos = (dist_vec[1] - dist_vec[0] > 0)
    if(x_pos and y_pos):
        dist, np = nearest_point(dist_vec, OFFSET_10, OFFSET_01)
    elif((not x_pos) and y_pos):
        dist, np = nearest_point(dist_vec, OFFSET_01, OFFSET_m0)
    elif((not x_pos) and (not y_pos)):
        dist, np = nearest_point(dist_vec, OFFSET_m0, OFFSET_0m)
    elif(x_pos and (not y_pos)):
        dist, np = nearest_point(dist_vec, OFFSET_0m, OFFSET_10)

    if(eff_col_dist < dist):
        return numpy.array([0.0, 0.0], dtype="float64")
    else:
        ori = dist_vec - np
        ori_norm = numpy.sqrt(numpy.sum(ori * ori))
        ori *= 1.0 / max(1.0e-6, ori_norm)
        return (0.50 * (eff_col_dist - dist) * cell_size) * ori

@njit(cache=True)
def vector_move_no_collision(ori, turn_rate, walk_speed, dt):
    d_theta = turn_rate * dt
    arc = walk_speed * dt
    c_theta = numpy.cos(ori)
    s_theta = numpy.sin(ori)
    c_dt = numpy.cos(0.5 * d_theta)
    s_dt = numpy.sin(0.5 * d_theta)

    n_ori = ori + d_theta
    # Shape it to [-PI, PI]
    n_ori = angle_normalization(n_ori)

    if(abs(d_theta) < 1.0e-8):
        d_x = c_theta * arc
        d_y = s_theta * arc
    else:
        # Turning Radius
        rad = walk_speed / turn_rate
        offset = 2.0 * s_dt * rad
        c_n = c_theta * c_dt - s_theta * s_dt
        s_n = c_theta * s_dt + s_theta * c_dt
        d_x = c_n * offset
        d_y = s_n * offset

    return n_ori, numpy.array([d_x, d_y], dtype="float64") 

@njit(cache=True)
def search_optimal_action(ori, targ1, targ2, candidate_action, delta_t):
    d_targ1 = numpy.array(targ1, dtype=numpy.float64)
    if(targ2 is not None):
        d_targ2 = numpy.array(targ2, dtype=numpy.float64)
    costs = []
    for action in candidate_action:
        tr = action[0] * PI
        ws = action[1]
        n_ori, n_loc = vector_move_no_collision(ori, tr, ws, delta_t)

        # The position error costs
        dist_loss = numpy.sum((n_loc - d_targ1) ** 2)
        dist = numpy.sqrt(dist_loss)
        cost = dist_loss

        # The action costs
        cost += 1.0e-4 * (action[0] ** 2 + action[1] ** 2)

        # The orientation costs
        targ1_ang = math.atan2(d_targ1[1], d_targ1[0])
        delta1_ang = angle_normalization(targ1_ang - n_ori)
        delta2_ang = delta1_ang
        if(targ2 is not None):  # Else try prepare for the next target
            targ2_ang = math.atan2(d_targ2[1], d_targ2[0])
            delta2_ang = angle_normalization(targ2_ang - n_ori)

        # Try to face the next target by looking ahead
        f= min(dist/0.2, 1.0)
        cost += delta1_ang * delta1_ang * f + delta2_ang * delta2_ang * (1 - f)
        costs.append(cost)
    return numpy.argmin(numpy.array(costs))

def vector_move_with_collision(ori, pos, turn_rate, walk_speed, delta_t, cell_walls, cell_size, col_dist):
    slide_factor = 0.20
    tmp_pos = numpy.copy(numpy.array(pos, dtype="float64"))
    
    t_prec = 0.01
    iteration = int(delta_t / t_prec)
    collision = 0.0

    for i in range(iteration + 1):
        t_res = min(delta_t - i * t_prec, t_prec)
        if(t_res < 1.0e-8):
            continue
        ori, offset = vector_move_no_collision(ori, turn_rate, walk_speed, t_res)
        exp_pos = tmp_pos + offset
        exp_cell = exp_pos / cell_size
        
        #consider the collision in new cell
        col_f = numpy.array([0.0, 0.0], dtype="float64")
        for i in range(-1, 2): 
            for j in range(-1, 2): 
                w_i = i + int(exp_cell[0])
                w_j = j + int(exp_cell[1])
                if(w_i > -1 and w_i < cell_walls.shape[0] and w_j > -1  and w_j < cell_walls.shape[1]):
                    if(cell_walls[w_i,w_j] > 0):
                        cell_deta = exp_cell - numpy.floor(exp_cell) - numpy.array([i + 0.5, j + 0.5], dtype="float32")
                        col_f += collision_force(cell_deta, cell_size, col_dist)
        tmp_pos = col_f + exp_pos
        collision += numpy.sqrt(numpy.sum(col_f ** 2))

    return ori, tmp_pos, collision
