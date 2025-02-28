import numpy
import math
import pygame
from .agent_base import AgentBase
from queue import Queue
from pygame import font
from xenoverse.mazeworld.envs.dynamics import PI
from xenoverse.mazeworld.envs.utils import conv2d_numpy
from xenoverse.mazeworld.envs.dynamics import search_optimal_action
from xenoverse.mazeworld.envs.ray_caster_utils import landmarks_color, landmarks_rgb, landmarks_rgb_arr, paint_agent_arrow

def convolve_exploration(exp_wht):
    """
    Add weight to the cells where its surroundings are not exposed (mask_info = 0)
    """
    kernel = numpy.ones((5,5))
    kernel[2, 2] = 1000
    res = conv2d_numpy(exp_wht, kernel, padding=2)
    return res

class SmartSLAMAgent(AgentBase):
    def render_init(self, view_size=(480, 480)):
        """
        Initialize a God View With Landmarks
        """
        font.init()
        self._font = font.SysFont("Arial", 18)

        #Initialize the agent drawing
        self._render_cell_size_x = view_size[0] / self._nx
        self._render_cell_size_y = view_size[1] / self._ny
        self._view_size = view_size
        self._window_size = (view_size[0] * 2, view_size[1])

        self._pos_conversion_x = self._render_cell_size_x / self.maze_env.maze_core._cell_size
        self._pos_conversion_y = self._render_cell_size_y / self.maze_env.maze_core._cell_size

        self._screen = pygame.Surface(self._window_size)
        self._screen = pygame.display.set_mode(self._window_size)
        pygame.display.set_caption("AgentRender")
        self._surf_god = pygame.Surface(view_size)
        self._surf_god.fill(pygame.Color("white"))

        for x in range(self._nx):
            for y in range(self._ny):
                if(self._god_info[x,y] < 0):
                    pygame.draw.rect(self._surf_god, pygame.Color("black"), (x * self._render_cell_size_x, y * self._render_cell_size_y,
                            self._render_cell_size_x, self._render_cell_size_y), width=0)

    def render_update(self, observation):
        # paint landmarks
        self._screen.blit(self._surf_god, (0, 0))
        for landmarks_id, (x,y) in enumerate(self._landmarks_coordinates):
            pygame.draw.rect(self._screen, landmarks_color(landmarks_id), 
                    (x * self._render_cell_size_x, y * self._render_cell_size_y,
                    self._render_cell_size_x, self._render_cell_size_y), width=0)

        # paint masks (mists)
        for x in range(self._nx):
            for y in range(self._ny):
                if(self._mask_info[x,y] < 1):
                    pygame.draw.rect(self._screen, pygame.Color("grey"), (x * self._render_cell_size_x, y * self._render_cell_size_y,
                            self._render_cell_size_x, self._render_cell_size_y), width=0)

        # paint agents
        agent_pos = [self._agent_loc[0] * self._pos_conversion_x, self._agent_loc[1] * self._pos_conversion_y]
        paint_agent_arrow(self._screen, pygame.Color("gray"), (0, 0), (agent_pos[0], agent_pos[1]), self._agent_ori, 
                0.4 * self._pos_conversion_x, 0.5 * self._pos_conversion_x)

        # paint target trajectory
        for i in range(0, len(self._path) - 1):
            factor = (i + 1) / len(self._path)
            p = self._path[i]
            n = self._path[i+1]
            p = [(p[0] + 0.5) * self._render_cell_size_x, (p[1] + 0.5) *  self._render_cell_size_y]
            n = [(n[0] + 0.5) * self._render_cell_size_x, (n[1] + 0.5) *  self._render_cell_size_y]
            pygame.draw.line(self._screen, pygame.Color(int(255 * factor), int(255 * (1 - factor)), 128, 255), p, n, width=1)

        # paint the first segment of the path
        p = self._path[0]
        n = self._cur_grid_float
        p = [int((p[0] + 0.5) * self._render_cell_size_x), int((p[1] + 0.5) *  self._render_cell_size_y)]
        n = [int(n[0] * self._render_cell_size_x), int(n[1] *  self._render_cell_size_y)]
        pygame.draw.line(self._screen, pygame.Color(0, 255, 128, 255), p, n, width=1)

        # paint observation
        obs_surf = pygame.surfarray.make_surface(observation)
        obs_surf = pygame.transform.scale(obs_surf, self._view_size)
        self._screen.blit(obs_surf, (self._view_size[0], 0))

        # display
        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()

    def update_cost_map(self, r_exp=0.25):
        # Calculate Shortest Distance using A*
        # In survival mode, consider the loss brought by rewards
        self._cost_map = 1e+6 * numpy.ones_like(self._god_info)
        refresh_list = Queue()
        cx,cy = self._cur_grid
        for dx,dy in self.valid_neighbors(center=(cx, cy), self_included=True, mask_included=False):
            i = dx + cx
            j = dy + cy
            # Initialize the neiboring costs
            deta = numpy.asarray([(i + 0.5) - self._cur_grid_float[0], (j + 0.5) - self._cur_grid_float[1]])
            dist = numpy.sqrt(numpy.sum(deta ** 2))
            ori = 1.0 - numpy.sum(deta / (dist + 1.0e-3) * numpy.array([numpy.cos(self._agent_ori), numpy.sin(self._agent_ori)]))
            ori_cost = 20.0 * ori * min(dist, 0.01)
            self._cost_map[i, j] = dist + ori_cost
            refresh_list.put((i, j))

        while not refresh_list.empty():
            o_x, o_y = refresh_list.get()
            for d_x, d_y in self.valid_neighbors(center=(o_x, o_y), self_included=False, mask_included=True):
                n_x = o_x + d_x
                n_y = o_y + d_y
                if(n_x >= self._nx or n_x < 0 or n_y >= self._ny or n_y < 0):
                    continue
                c_type = self._god_info[n_x, n_y]
                m_type = self._mask_info[n_x, n_y]
                dist_cost = numpy.sqrt(d_x ** 2 + d_y ** 2)
                if(c_type < 0 and m_type > 0):
                    continue
                elif(m_type < 1):
                    cost = 10 + dist_cost
                else:
                    cost = dist_cost
                if(self._cost_map[n_x, n_y] > self._cost_map[o_x, o_y] + cost):
                    self._cost_map[n_x, n_y] = self._cost_map[o_x, o_y] + cost
                    refresh_list.put((n_x, n_y))

    def policy(self, observation, r):
        self.update_cost_map()
        path_greedy = self.navigate_landmarks_navigate(self._command)
        path = path_greedy
        if(path_greedy is None):
            path_exp = self.exploration()
            if(path_exp is not None):
                path = path_exp
            else:
                print("[WARNING] Unexpected failure in exploration, might cause unexpected stop")
                path = [self._cur_grid]
        self._path = path
        return self.path_to_action(path)

    def path_to_action(self, path):
        if(len(path) < 0):
            raise Exception("Unexpected Error in retrieving path for the current agent")
        d_x = path[0][0] + 0.5 - self._cur_grid_float[0]
        d_y = path[0][1] + 0.5 - self._cur_grid_float[1]
        if(len(path) > 1):
            d_x2 = path[1][0] + 0.5 - self._cur_grid_float[0]
            d_y2 = path[1][1] + 0.5 - self._cur_grid_float[1]
            targ2 = (d_x2, d_y2)

        else:
            targ2 = None
        return search_optimal_action(self._agent_ori, (d_x, d_y), targ2, self._action_space, 1.0)

    def retrieve_path(self, cost_map, goal_idx):
        path = [(int(goal_idx[0]), int(goal_idx[1]))]
        cost = cost_map[goal_idx]
        sel_x, sel_y = goal_idx
        iteration = 0
        eff_targets = []
        for dx, dy in self.valid_neighbors(self_included=True, mask_included=False):
            eff_targets.append((self._cur_grid[0] + dx, self._cur_grid[1] + dy))
        while sel_x != self._cur_grid[0] or sel_y != self._cur_grid[1]:
            flag = False
            for t_x, t_y in eff_targets:
                if(sel_x == t_x and sel_y == t_y):
                    flag = True
            if(flag):
                break
            iteration += 1
            min_cost = cost
            min_x = -1
            min_y = -1
            for d_x, d_y in self.valid_neighbors(center=(sel_x, sel_y)):
                n_x = sel_x + d_x
                n_y = sel_y + d_y
                if(n_x < 0 or n_x > self._nx - 1 or n_y < 0 or n_y > self._ny - 1):
                    continue
                if(cost_map[n_x, n_y] > 1e+4):
                    continue
                if(cost_map[n_x, n_y] < min_cost):
                    # Check whether the location is in shortest path
                    # if not, directly link the agent to the target cell
                    min_cost = cost_map[n_x, n_y]
                    min_x = int(n_x)
                    min_y = int(n_y)
            if(min_x > -1): 
                sel_x, sel_y = (min_x, min_y)
                path.insert(0, (sel_x, sel_y))
                cost=cost_map[sel_x, sel_y]
            else:
                print("[WARNING] Unexpected error in path retrieving")
                break

        if(len(path) > 2):
            d_x = path[0][0] + 0.5 - self._cur_grid_float[0]
            d_y = path[0][1] + 0.5 - self._cur_grid_float[1]
            deta_s = numpy.sqrt(d_x ** 2 + d_y ** 2)
            d_x2 = path[1][0] + 0.5 - self._cur_grid_float[0]
            d_y2 = path[1][1] + 0.5 - self._cur_grid_float[1]
            deta_s2 = numpy.sqrt(d_x2 ** 2 + d_y2 ** 2)
            if(deta_s + cost_map[path[0][0], path[0][1]] > deta_s2 + cost_map[path[1][0], path[1][1]] and deta_s < 0.2):
                del path[0]
        return path

    def exploration(self):
        explore_wht = 1 - numpy.array(self._mask_info, dtype=numpy.int32)
        explore_wht = convolve_exploration(explore_wht)
        utility = self._cost_map - explore_wht
        if(numpy.min(utility) >= 0):
            return None 
        target_idx = numpy.unravel_index(numpy.argmin(utility), utility.shape)
        return self.retrieve_path(self._cost_map, target_idx)

    def navigate_landmarks_navigate(self, landmarks_id):
        idxes = numpy.argwhere(self._god_info == landmarks_id + 1)
        for idx in idxes:
            if(self._mask_info[idx[0], idx[1]] < 1):
                continue
            else:
                return self.retrieve_path(self._cost_map, tuple(idx))
        return None
