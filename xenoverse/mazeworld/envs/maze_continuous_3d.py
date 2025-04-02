"""
Core File of Maze Env
"""
import os
import numpy
import pygame
import random
import time
from pygame import font
from numpy import random as npyrnd
from numpy.linalg import norm
from xenoverse.mazeworld.envs.dynamics import PI, PI_2, PI_4, PI2d, vector_move_with_collision
from xenoverse.mazeworld.envs.ray_caster_utils import maze_view
from xenoverse.mazeworld.envs.task_sampler import MAZE_TASK_MANAGER
from xenoverse.mazeworld.envs.maze_base import MazeBase
from xenoverse.mazeworld.envs.ray_caster_utils import landmarks_rgb, landmarks_rgb_arr, paint_agent_arrow

class MazeCoreContinuous3D(MazeBase):
    #Read Configurations
    def __init__(
            self,
            collision_dist=0.20, #collision distance
            visibility_3D=12.0, #agent vision range
            resolution_horizon = 320, #resolution in horizontal
            resolution_vertical = 320, #resolution in vertical
            max_steps = 5000,
            command_in_observation = False # whether instruction / command is in observation
        ):
        super(MazeCoreContinuous3D, self).__init__(
                collision_dist = collision_dist,
                visibility_3D = visibility_3D,
                resolution_horizon = resolution_horizon,
                resolution_vertical = resolution_vertical,
                max_steps = max_steps,
                command_in_observation = command_in_observation
                )

    def reset(self):

        #add the navigation guidance bar
        self._navbar_l = 0.50 * self.resolution_vertical
        self._navbar_w = 0.05 * self.resolution_horizon

        self._navbar_start_x = 0.25 * self.resolution_vertical
        self._navbar_start_y = 0.10 * self.resolution_vertical

        return super(MazeCoreContinuous3D, self).reset()

    def do_action(self, action, delta_t=1.0):
        turn_rate, walk_speed = action
        turn_rate = numpy.clip(turn_rate, -1, 1) * PI
        walk_speed = numpy.clip(walk_speed, -1, 1)

        self._agent_ori, self._agent_loc, collide = vector_move_with_collision(
                self._agent_ori, self._agent_loc, turn_rate, walk_speed, delta_t, 
                self._cell_walls, self._cell_size, self.collision_dist)
        self._collision_punish = self._collision_reward * collide
        self._agent_grid = self.get_loc_grid(self._agent_loc)
        reward, terminated, truncated = self.evaluation_rule()
        self.update_observation()

        return reward, terminated, truncated

    def render_init(self, view_size):
        super(MazeCoreContinuous3D, self).render_init(view_size)
        self._pos_conversion = self._render_cell_size / self._cell_size
        self._ori_size = 0.60 * self._pos_conversion

    def render_observation(self):
        # Paint Observation
        view_obs_surf = pygame.transform.scale(self._obs_surf, (self._view_size, self._view_size))
        self._screen.blit(view_obs_surf, (0, 0))


    def movement_control(self, keys):
        #Keyboard control cases
        turn_rate = None
        walk_speed = None
        time.sleep(0.01)
        if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:
            turn_rate = 0.0
            walk_speed = 0.0
            if keys[pygame.K_LEFT]:
                turn_rate -= 0.1
            if keys[pygame.K_RIGHT]:
                turn_rate += 0.1
            if keys[pygame.K_UP]:
                walk_speed += 0.5
            if keys[pygame.K_DOWN]:
                walk_speed -= 0.5
        if keys[pygame.K_SPACE]:
            turn_rate = 0.0
            walk_speed = 0.0
        return turn_rate, walk_speed

    def update_observation(self):
        self._observation, self._cell_exposed = maze_view(numpy.array(self._agent_loc, dtype=numpy.float32), self._agent_ori, self._agent_height, 
                self._cell_walls, self._cell_landmarks, self._cell_texts, self._cell_size, 
                MAZE_TASK_MANAGER.textlib_walls, MAZE_TASK_MANAGER.textlib_grounds[self._ground_text], MAZE_TASK_MANAGER.textlib_ceilings[self._ceiling_text],
                self._wall_height, 1.0, self.visibility_3D, 0.20, 
                self._fol_angle, self.resolution_horizon, self.resolution_vertical, landmarks_rgb_arr)
        if(self.command_in_observation):
            start_x = int(self._navbar_start_x)
            start_y = int(self._navbar_start_y)
            end_x = int(self._navbar_start_x + self._navbar_l)
            end_y = int(self._navbar_start_y + self._navbar_w)
            self._observation[start_x:end_x, start_y:end_y] = landmarks_rgb[self._command]

        self._command_rgb = landmarks_rgb[self._command]
        self._obs_surf = pygame.surfarray.make_surface(self._observation)

    def get_observation(self):
        return numpy.copy(self._observation.astype('uint8'))

    def get_info(self, info):
        info["command"] = self._command_rgb