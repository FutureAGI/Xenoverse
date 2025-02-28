import numpy
import pygame
import time
import random
from numba import njit
from xenoverse.mazeworld.envs.dynamics import PI, PI_4

class LandmarksRGB(object):
    def __init__(self):
        self._landmarks_rgb = dict()
        self._landmarks_rgb[0] = numpy.array([0, 255, 0], dtype="float32") # Green
        self._landmarks_rgb[1] = numpy.array([255, 0, 0], dtype="float32") # Red
        self._landmarks_rgb[2] = numpy.array([0, 0, 255], dtype="float32") # Blue
        self._landmarks_rgb[3] = numpy.array([0, 255, 255], dtype="float32") # 
        self._landmarks_rgb[4] = numpy.array([255, 0, 255], dtype="float32") #
        self._landmarks_rgb[5] = numpy.array([255, 255, 0], dtype="float32") #
        self._landmarks_rgb[6] = numpy.array([128, 128, 255], dtype="float32") #
        self._landmarks_rgb[7] = numpy.array([128, 255, 128], dtype="float32") #
        self._landmarks_rgb[8] = numpy.array([255, 128, 128], dtype="float32") #
        self._landmarks_rgb[9] = numpy.array([0, 96, 128], dtype="float32") #
        self._landmarks_rgb[10] = numpy.array([96, 0, 128], dtype="float32") #
        self._landmarks_rgb[11] = numpy.array([0, 128, 96], dtype="float32") #
        self._landmarks_rgb[12] = numpy.array([96, 128, 0], dtype="float32") #
        self._landmarks_rgb[13] = numpy.array([128, 96, 0], dtype="float32") #
        self._landmarks_rgb[14] = numpy.array([128, 0, 96], dtype="float32") #

    def color(self, i, opacity=0.0):
        real_color = (1 - opacity) * self._landmarks_rgb[i] + opacity * numpy.array([255, 255, 255])
        return pygame.Color(int(real_color[0]), int(real_color[1]), int(real_color[2]), 255)

    @property
    def rgb(self):
        return self._landmarks_rgb

    @property
    def rgb_npa(self):
        rgb_arr = numpy.zeros((len(self._landmarks_rgb), 3), dtype="float32")
        for i in self._landmarks_rgb:
            rgb_arr[i] = self._landmarks_rgb[i]
        return rgb_arr

lrgb_handler = LandmarksRGB()
landmarks_rgb = lrgb_handler.rgb
landmarks_color = lrgb_handler.color
landmarks_rgb_arr = lrgb_handler.rgb_npa

@njit(cache=True)
def DDA_2D(pos, i, j, cell_number, cell_size, cos_ori, sin_ori, cell_walls, cell_transparent, visibility_3D):
    eps = 1.0e-8
    if(cos_ori < 0):
        c_sign = -1
    else:
        c_sign = 1
    if(sin_ori < 0):
        s_sign = -1
    else:
        s_sign = 1
    delta_dist_x = abs(cell_size / eps) if abs(cos_ori) < eps else abs(cell_size / cos_ori)
    delta_dist_y = abs(cell_size / eps) if abs(sin_ori) < eps else abs(cell_size / sin_ori)
    d_x = ((i + 1) * cell_size - pos[0]) if cos_ori > 0 else (i * cell_size - pos[0])
    d_y = ((j + 1) * cell_size - pos[1]) if sin_ori > 0 else (j * cell_size - pos[1])
    side_dist_x = c_sign * (d_x / eps) if abs(cos_ori) < eps else d_x / cos_ori
    side_dist_y = s_sign * (d_y / eps) if abs(sin_ori) < eps else d_y / sin_ori
    delta_i = 1 if(cos_ori > 0) else -1
    delta_j = 1 if(sin_ori > 0) else -1
    hit_i = i
    hit_j = j
    hit_dist = 0.0
    hit_side = 0
    hit_transparent_list = []
    exposed_cell = [[i,j]]

    #Remove this part as we don't want to see colors when in landmark cells themselves
    #if(cell_transparent[hit_i, hit_j] > -1):
    #    if(side_dist_x < side_dist_y):
    #        hit_transparent_list.append((side_dist_x, hit_i, hit_j, 0, cell_transparent[hit_i, hit_j]))
    #    elif(side_dist_x > side_dist_y):
    #        hit_transparent_list.append((side_dist_y, hit_i, hit_j, 1, cell_transparent[hit_i, hit_j]))

    while hit_dist < visibility_3D:
        if(side_dist_x < side_dist_y):
            hit_i += delta_i
            side_dist_y -= side_dist_x
            hit_dist += side_dist_x
            if(cell_transparent[hit_i, hit_j] > -1):
                hit_transparent_list.append((hit_dist, hit_i, hit_j, 0, cell_transparent[hit_i, hit_j]))
            if(hit_i < 0 or hit_i >= cell_number):
                if(hit_j < 0 or hit_j >= cell_number):
                    hit_dist = 1.0e+6
                    break
            else:
                if(hit_dist <= visibility_3D * 0.60):
                    exposed_cell.append([hit_i, hit_j]) # The cell becomes seen to the agent
                if(cell_walls[hit_i, hit_j] > 0):
                    hit_side = 0
                    break
            side_dist_x = delta_dist_x
        else:
            hit_j += delta_j
            side_dist_x -= side_dist_y
            hit_dist += side_dist_y
            if(cell_transparent[hit_i, hit_j] > -1):
                hit_transparent_list.append((hit_dist, hit_i, hit_j, 1, cell_transparent[hit_i, hit_j]))
            if(hit_i < 0 or hit_i >= cell_number):
                if(hit_j < 0 or hit_j >= cell_number):
                    hit_dist = 1.0e+6
                    break
            else:
                if(hit_dist <= visibility_3D * 0.60):
                    exposed_cell.append([hit_i, hit_j]) # The cell becomes seen to the agent
                if(cell_walls[hit_i, hit_j] > 0):
                    hit_side = 1
                    break
            side_dist_y = delta_dist_y
    return hit_dist, hit_i, hit_j, hit_side, hit_transparent_list, exposed_cell


"""
Depict 3D Maze View
cell transparent: N X N array, where -1 represents empty, and 0~9 (max = 9 represents different colors of transparency)
"""

@njit(cache=True)
def interpolate(array, i, j, d, px, py):
    d2 = max(d ** 2, 1.0e-8)
    w,h,_ = array.shape
    ib = int(i)
    jb = int(j)
    sum_wht = 0.0
    sum_r = numpy.zeros(shape=(3), dtype="float32")

    for x in range(ib-1, ib+3):
        for y in range(jb-1, jb+3):
            dist = (((x - i) * px) ** 2 + ((y - j) * py) ** 2)
            wht = max(0.01, min(1.0 - 10 * dist / d2, 1.0))
            sum_wht += wht
            xv = x % w
            yv = y % h
            sum_r += wht * array[xv, yv]
    return sum_r / sum_wht

@njit(cache=True)
def maze_view(pos, ori, vision_height, cell_walls, cell_transparent, cell_texts, cell_size, texture_array, ground_text,
        ceil_text, ceil_height, text_size, visibility_3D, l_focal, vision_angle_h, resolution_h, resolution_v, transparent_rgb):
    vision_screen_half_size_h = numpy.tan(vision_angle_h / 2) * l_focal
    vision_screen_half_size_v = vision_screen_half_size_h * resolution_v / resolution_h
    pixel_size = 2.0 * vision_screen_half_size_h / resolution_h
    s_ori = numpy.sin(ori)
    c_ori = numpy.cos(ori)
    max_cell_i = cell_walls.shape[0]
    max_cell_j = cell_walls.shape[1]
    pixel_factor = pixel_size / l_focal
    cell_exposed = numpy.zeros_like(cell_walls)
    percell_textnum = cell_size / text_size
    height_texts = ceil_height / text_size

    FAR_RGB = numpy.array([1, 1, 1], dtype="float32")

    textg_pixel_size_x = text_size / ground_text.shape[0]
    textg_pixel_size_y = text_size / ground_text.shape[1]
    textc_pixel_size_x = text_size / ceil_text.shape[0]
    textc_pixel_size_y = text_size / ceil_text.shape[1]

    # prepare some maths
    rgb_array = numpy.zeros(shape=(resolution_h, resolution_v, 3), dtype="int32")
    rgb_array[:, :] = FAR_RGB
    cos_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    cos_abs_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    sin_abs_hp_array = numpy.zeros(shape=(resolution_h,), dtype="float32")
    tan_hp = (- 0.5 - resolution_h / 2) * pixel_factor
    for d_h in range(resolution_h):
        tan_hp += pixel_factor
        cos_hp = numpy.sqrt(1.0 / (1.0 + tan_hp ** 2))
        sin_hp = tan_hp * cos_hp
        sin_abs_hp_array[d_h] = sin_hp * c_ori + cos_hp * s_ori
        cos_abs_hp_array[d_h] = cos_hp * c_ori - sin_hp * s_ori
        cos_hp_array[d_h] = cos_hp
    
    # paint floor
    for d_v in range(resolution_v - 1, resolution_v//2, -1):
        # horizontal distance on screen
        v_screen = (d_v + 0.5) * pixel_size - vision_screen_half_size_v
        distance = vision_height / v_screen * l_focal
        light_incident = min(1.0, v_screen / l_focal)
        if(distance > visibility_3D):
            continue

        for d_h in range(resolution_h):
            eff_distance = distance / cos_hp_array[d_h]
            alpha = min(1.0, max(2.0 * eff_distance / visibility_3D - 1.0, 0.0)) * light_incident
            hit_x = eff_distance * cos_abs_hp_array[d_h] + pos[0]
            hit_y = eff_distance * sin_abs_hp_array[d_h] + pos[1]
            i = (hit_x / cell_size)
            j = (hit_y / cell_size)
            d_i = i - numpy.floor(i)
            d_j = j - numpy.floor(j)
            i = int(i)
            j = int(j)
            eff_ps = eff_distance * pixel_size / l_focal

            if(i < max_cell_i and i >= 0 and j < max_cell_j and j >= 0):
                #text_id = cell_texts[i,j]
                d_i *= percell_textnum
                d_j *= percell_textnum
                d_i -= numpy.floor(d_i)
                d_j -= numpy.floor(d_j)
                d_i *= ground_text.shape[0]
                d_j *= ground_text.shape[1]
                rgb_array[d_h, d_v, :] = numpy.clip(
                        light_incident * (alpha * FAR_RGB + (1.0 - alpha) * interpolate(ground_text, d_i, d_j, eff_ps, textg_pixel_size_x, textg_pixel_size_y)),
                        0, 255)

    # paint ceil
    for d_v in range(resolution_v//2):
        v_screen = vision_screen_half_size_v - (d_v + 0.5) * pixel_size
        distance = (ceil_height - vision_height) / v_screen * l_focal
        light_incident = min(1.0, v_screen / l_focal)
        if(distance > visibility_3D):
            continue

        for d_h in range(resolution_h):
            eff_distance = distance / cos_hp_array[d_h]
            alpha = min(1.0, max(2.0 * eff_distance / visibility_3D - 1.0, 0.0))
            hit_x = eff_distance * cos_abs_hp_array[d_h] + pos[0]
            hit_y = eff_distance * sin_abs_hp_array[d_h] + pos[1]
            i = (hit_x / cell_size)
            j = (hit_y / cell_size)
            d_i = i - numpy.floor(i)
            d_j = j - numpy.floor(j)
            i = int(i)
            j = int(j)
            eff_ps = eff_distance * pixel_size / l_focal

            if(i < max_cell_i and i >= 0 and j < max_cell_j and j >= 0):
                #text_id = cell_texts[i,j]
                d_i *= percell_textnum
                d_j *= percell_textnum
                d_i -= numpy.floor(d_i)
                d_j -= numpy.floor(d_j)
                d_i *= ceil_text.shape[0]
                d_j *= ceil_text.shape[1]
                rgb_array[d_h, d_v, :] = numpy.clip(
                    light_incident * (alpha * FAR_RGB + (1.0 - alpha) * interpolate(ceil_text, d_i, d_j, eff_ps, textc_pixel_size_x, textc_pixel_size_y)),
                    0, 255)
    
    # paint wall
    for d_h in range(resolution_h):
        i = int(pos[0] / cell_size)
        j = int(pos[1] / cell_size)
        hit_dist, hit_i, hit_j, hit_side, hit_transparent, exposed_cell = DDA_2D(
                pos, i, j, max_cell_i, cell_size, cos_abs_hp_array[d_h], sin_abs_hp_array[d_h], 
                cell_walls, cell_transparent, visibility_3D)
        for idx in exposed_cell:
            if(random.random() < 0.05):
                cell_exposed[idx[0],idx[1]] = 1
        hit_transparent.sort(key=lambda x:x[0], reverse=True)

        alpha = min(1.0, max(2.0 * hit_dist / visibility_3D - 1.0, 0.0))
        text_id = cell_texts[hit_i,hit_j]
        hit_pt_x = hit_dist * cos_abs_hp_array[d_h] + pos[0]
        hit_pt_y = hit_dist * sin_abs_hp_array[d_h] + pos[1]
        if(hit_side == 0): # hit vertical wall
           local_h = hit_pt_y / cell_size 
           local_h -= numpy.floor(local_h)
           light_incident = abs(cos_abs_hp_array[d_h])
        else:
           local_h = hit_pt_x / cell_size 
           local_h -= numpy.floor(local_h)
           light_incident = abs(sin_abs_hp_array[d_h])

        ratio = hit_dist * cos_hp_array[d_h] / l_focal
        if(abs(ratio) < 1.0e-8):
            if(ratio > 0):
                ratio = 1.0e-8
            else:
                ratio = -1.0e-8
        top_v = (ceil_height - vision_height) / ratio
        bot_v = vision_height / ratio

        v_s = max(0, int((vision_screen_half_size_v - top_v) / pixel_size))
        v_e = min(resolution_v, int((vision_screen_half_size_v + bot_v) / pixel_size))

        for d_v in range(v_s, v_e):
            local_v = (vision_screen_half_size_v - (d_v + 0.5) * pixel_size) * ratio + vision_height
            d_i = local_h * percell_textnum
            d_j = local_v / text_size
            d_i -= numpy.floor(d_i)
            d_j -= numpy.floor(d_j)
            d_i = int(texture_array[text_id].shape[0] * d_i)
            d_j = int(texture_array[text_id].shape[1] * d_j)
            text_pixel_size_x = text_size / texture_array[text_id].shape[0]
            text_pixel_size_y = text_size / texture_array[text_id].shape[1]

            eff_ps = eff_distance * pixel_size / l_focal

            rgb_array[d_h, d_v, :] = numpy.clip(
                    light_incident * (alpha * FAR_RGB + (1.0 - alpha) * interpolate(texture_array[text_id], d_i, d_j, eff_ps, text_pixel_size_x, text_pixel_size_y)),
                    0, 255)

        # Add those transparent
        for hit_dist, hit_i, hit_j, hit_side, trans_ID in hit_transparent:
            ratio = hit_dist * cos_hp_array[d_h] / l_focal
            if(abs(ratio) < 1.0e-8):
                if(ratio > 0):
                    ratio = 1.0e-8
                else:
                    ratio = -1.0e-8
            top_v = (ceil_height - vision_height) / ratio
            bot_v = vision_height / ratio
            transparent_factor = 0.30

            v_s = max(0, int((vision_screen_half_size_v - top_v) / pixel_size))
            v_e = min(resolution_v, int((vision_screen_half_size_v + bot_v) / pixel_size))
            alpha = min(1.0, max(2.0 * hit_dist / visibility_3D - 1.0, 0.0))
            for d_v in range(v_s, v_e):
                rgb_array[d_h, d_v] = numpy.clip(
                    (1.0 - transparent_factor) * rgb_array[d_h, d_v] + transparent_factor * ((1.0 - alpha) * transparent_rgb[trans_ID] + alpha * FAR_RGB),
                    0, 255)
        
    return rgb_array, cell_exposed

def paint_agent_arrow(scr, color, offset, pos, angle, l1, l2):
    A_ARR = PI / 3
    A_A = angle - (PI - A_ARR)
    A_B = angle + (PI - A_ARR)
    pos_S = (float(pos[0] + offset[0]), float(pos[1] + offset[1]))
    pos_A = (float(pos[0] + l1 * numpy.cos(A_A) + offset[0]), float(pos[1] + l1 * numpy.sin(A_A) + offset[1]))
    pos_B = (float(pos[0] + l1 * numpy.cos(A_B) + offset[0]), float(pos[1] + l1 * numpy.sin(A_B) + offset[1]))
    pos_C = (float(pos[0] + l2 * numpy.cos(angle) + offset[0]), float(pos[1] + l2 * numpy.sin(angle) + offset[1]))
    pygame.draw.polygon(scr, color, (pos_S, pos_A, pos_C, pos_B))
