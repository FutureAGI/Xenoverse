import sys
import numpy
from queue import Queue
from copy import deepcopy
from xenoverse.mazeworld.envs.dynamics import PI, DEFAULT_ACTION_SPACE_16, DEFAULT_ACTION_SPACE_32
from xenoverse.mazeworld.envs.maze_env import MazeWorldContinuous3D


class AgentBase(object):
    """
    Base class for agents
    Use this as parent to create new rule based agents
    """
    def __init__(self, **kwargs):
        self.render = False
        for k in kwargs:
            self.__dict__[k] = kwargs[k]
        if("maze_env" not in kwargs):
            raise Exception("Must use maze_env as arguments")

        # Initialize information
        self._cell_size = self.maze_env.maze_core._cell_size
        self._god_info = 1 - self.maze_env.maze_core._cell_walls + self.maze_env.maze_core._cell_landmarks
        self._landmarks_coordinates = self.maze_env.maze_core._landmarks_coordinates
        self._step_reward = self.maze_env.maze_core._step_reward
        self._nx, self._ny = self._god_info.shape
        self.neighbors = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        self._landmarks_visit = dict()
        self._short_term_memory = list()
        self._action_space = self.maze_env.list_actions

        if(self._action_space is None):
            raise Exception("For smart agents, maze environment must use Discrete16 or Discrete32")

        if("short_term_memory_size" not in kwargs):
            self.short_term_memory_size = 3
        if("memory_keep_ratio" not in kwargs):
            self.memory_keep_ratio = 1.0
        self._long_term_memory = numpy.zeros((self._nx, self._ny), dtype=numpy.int8)

        # Render
        if(self.render):
            self.render_init()

    def render_init(self):
        raise NotImplementedError()

    def valid_neighbors(self, center=None, self_included=False, mask_included=True):
        if(center is None):
            cx, cy = self.maze_env.maze_core._agent_grid
        else:
            cx, cy = center
        valid_neighbors = []
        if(self_included):
            valid_neighbors.append((0, 0))
        for dx, dy in self.neighbors:
            nx = cx + dx
            ny = cy + dy
            if(nx < 0 or nx >= self._nx or ny < 0 or ny >= self._ny):
                continue
            if(not self._mask_info[nx, ny] and not mask_included):
                continue
            if(self._god_info[nx, ny] < 0 and self._mask_info[nx, ny]):
                continue
            if(dx * dy == 0):
                valid_neighbors.append((dx, dy))
            else:
                if(self._god_info[nx, cy] > -1 and self._god_info[cx, ny] > -1 
                   and self._mask_info[nx, cy] and self._mask_info[cx, ny]):
                    valid_neighbors.append((dx, dy))
        return valid_neighbors

    def update_common_info(self):
        self._command = self.maze_env.maze_core._command

        # Update long and short term memory
        # Pop the eldest memory from short term memory and insert it to long term memory, but with losses.
        self._short_term_memory.append(numpy.copy(self.maze_env.maze_core._cell_exposed))
        if(len(self._short_term_memory) > self.short_term_memory_size):
            to_longterm = self._short_term_memory.pop(0)
            long_term_keep = (numpy.random.rand(self._nx, self._ny) < self.memory_keep_ratio).astype(numpy.int8)
            self._long_term_memory = numpy.logical_or(self._long_term_memory, to_longterm * long_term_keep)

        # Calculate the current memory: include the long term and short term memory
        self._mask_info = numpy.copy(self._long_term_memory)
        for i in range(len(self._short_term_memory)):
            self._mask_info = numpy.logical_or(self._mask_info, self._short_term_memory[i])
        
        self._agent_ori = self.maze_env.maze_core._agent_ori
        self._agent_loc = self.maze_env.maze_core._agent_loc
        self._cur_grid = deepcopy(self.maze_env.maze_core._agent_grid)
        self._cur_grid_float = deepcopy(self.maze_env.maze_core.get_loc_grid_float(self.maze_env.maze_core._agent_loc))
        lid = self._god_info[self._cur_grid[0], self._cur_grid[1]]
        if(lid > 0):
            self._landmarks_visit[lid - 1] = 0

    def policy(self, observation, r):
        raise NotImplementedError()

    def render_update(self, observation):
        raise NotImplementedError()

    def step(self, observation, r):
        self.update_common_info()
        action = self.policy(observation, r)
        if(self.render):
            self.render_update(observation)
        return action