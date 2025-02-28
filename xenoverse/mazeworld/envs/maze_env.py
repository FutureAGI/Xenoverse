"""
Gym Environment For Maze3D
"""
import numpy
import gym
import pygame

from gym import error, spaces, utils
from gym.utils import seeding
from xenoverse.mazeworld.envs.maze_continuous_3d import MazeCoreContinuous3D
from xenoverse.mazeworld.envs.dynamics import DEFAULT_ACTION_SPACE_16, DEFAULT_ACTION_SPACE_32

class MazeWorldEnvBase(gym.Env):
    """
    All Maze World Environments Use This Base Class
    """
    def __init__(self, 
            maze_type,
            enable_render=True,
            render_scale=480,
            max_steps=5000,
            ):
        self.maze_type = maze_type
        self.enable_render = enable_render
        self.render_viewsize = render_scale

        self.need_reset = True
        self.need_set_task = True

    def set_task(self, task_config):
        self.maze_core.set_task(task_config)
        self.need_set_task = False

    def reset(self):
        if(self.need_set_task):
            raise Exception("Must call \"set_task\" before reset")
        state = self.maze_core.reset()
        if(self.enable_render):
            self.maze_core.render_init(self.render_viewsize)
            self.keyboard_press = pygame.key.get_pressed()
        info = {"steps": self.maze_core.steps}
        self.maze_core.get_info(info)
        self.need_reset = False
        self.key_done = False
        return state, info

    def action_control(self, action):
        raise NotImplementedError("Must implement the action control logic")

    def step(self, action=None):
        if(self.need_reset):
            raise Exception("Must \"reset\" before doing any actions")

        internal_action = self.action_control(action)
            
        # In keyboard control, process only continues when key is pressed
        info = {"steps": self.maze_core.steps}
        if(internal_action is None):
            return self.maze_core.get_observation(), 0, False, info 
        reward, done = self.maze_core.do_action(internal_action)
        self.maze_core.get_info(info)

        if(done):
            self.need_reset=True

        return self.maze_core.get_observation(), reward, done, info

    def render(self, mode="human"):
        if(mode != "human"):
            raise NotImplementedError("Only human mode is supported")
        if(self.enable_render):
            self.key_done, self.keyboard_press = self.maze_core.render_update()

    def get_local_map(self, map_range=8, resolution=(128, 128)):
        """
        Returns the localized god-view map relative to the agent's standpoint
        """
        return self.maze_core.get_local_map(map_range=map_range, resolution=resolution)

    def get_global_map(self, resolution=(128, 128)):
        """
        Returns the global god-view map
        """
        return self.maze_core.get_global_map(resolution=resolution)

    def get_target_location(self):
        """
        Acquire relative position of the target to the agent
        Return (Distance, Angle)
        """
        target_id = self.maze_core._commands_sequence[self.maze_core._commands_sequence_idx]
        target_grid = self.maze_core._landmarks_coordinates[target_id]
        deta_grid = numpy.zeros(shape=(2,), dtype=numpy.float32)
        deta_grid[0] = target_grid[0] - self.maze_core._agent_grid[0]
        deta_grid[1] = target_grid[1] - self.maze_core._agent_grid[1]
        angle = numpy.arctan2(deta_grid[1], deta_grid[0]) - self.maze_core._agent_ori
        if(angle < -numpy.pi):
            angle += 2 * numpy.pi
        elif(angle > numpy.pi):
            angle -= 2 * numpy.pi
        dist = numpy.sqrt(numpy.sum(deta_grid * deta_grid))
        return dist, angle

    def save_trajectory(self, file_name, view_size=480):
        if(not self.enable_render):
            self.maze_core.render_init(view_size)
        self.maze_core.render_trajectory(file_name)

class MazeWorldContinuous3D(MazeWorldEnvBase):
    def __init__(self, 
            enable_render=True,
            render_scale=480,
            max_steps = 5000,
            resolution=(320, 320),
            visibility_3D=12.0,
            command_in_observation=False,
            action_space_type="Discrete16",  # Must choose in Discrete16, Discrete32, Continuous
            ):
        super(MazeWorldContinuous3D, self).__init__(
            "Continuous3D",
            enable_render=enable_render,
            render_scale=render_scale,
            max_steps=max_steps
        )
        self.maze_core = MazeCoreContinuous3D(
                resolution_horizon = resolution[0],
                resolution_vertical = resolution[1],
                max_steps = max_steps,
                visibility_3D=visibility_3D,
                command_in_observation=command_in_observation
                )
        
        self.inner_action_list  = None
        if(action_space_type == "Discrete16"):
            self.action_space = spaces.Discrete(16)
            # Using Default Discrete Actions
            self.inner_action_list = DEFAULT_ACTION_SPACE_16
        elif(action_space_type == "Discrete32"):
            self.action_space = spaces.Discrete(32)
            # Using Default Discrete Actions
            self.inner_action_list = DEFAULT_ACTION_SPACE_32
        elif(action_space_type == "Continuous"):
            # Turning Left/Right and go backward / forward
            self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=numpy.float32)
        else:
            raise ValueError("Invalid Action Space Type {}. Can only accept Discrete16, Discrete32, Continuous".format(action_space_type))

        # observation is the x, y coordinate of the grid
        self.observation_space = spaces.Box(low=0, high=255, shape=(resolution[0], resolution[1], 3), dtype=numpy.uint8)

    def action_control(self, action):
        if(action is None): # Only when there is no action input can we use keyboard control
            pygame.time.delay(20) # 50 FPS
            tr, ws = self.maze_core.movement_control(self.keyboard_press)
        else:
            if(self.inner_action_list is not None):
                tr, ws = self.inner_action_list[action]
            else:
                tr, ws = action
        if(tr is None or ws is None):
            return None
        return [tr, ws]
    
    @property
    def list_actions(self):
        return self.inner_action_list