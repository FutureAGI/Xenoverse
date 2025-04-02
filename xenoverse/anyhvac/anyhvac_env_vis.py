import pygame
import numpy
import time
from .anyhvac_env import HVACEnv
from pygame import font

class HVACEnvVisible(HVACEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_region = 20

    def reset(self, *args, **kwargs):
        res = super().reset(*args, **kwargs)
        self.render_init(render_size=640)
        self.keyboard_press = pygame.key.get_pressed()
        return res

    def step(self, actions):
        actions = numpy.clip(actions, 0, 1)
        observation, reward, terminated, truncated, info = super().step(actions)
        keydone, _ = self.render_update(info["heat_power"], actions, info["chtc_array"])
        truncated = truncated or keydone
        return observation, reward, terminated, truncated, info

    def render_init(self, render_size=640):
        """
        Initialize a God View With Landmarks
        """
        font.init()
        self._font = font.SysFont("Arial", 18)
        self.render_size = render_size

        #Initialize the agent drawing
        self._render_cell_size = (self.render_size - self.empty_region) // max(self.n_width, self.n_length)
        self._render_w = self.n_width * self._render_cell_size
        self._render_h = self.n_length * self._render_cell_size
        self.render_origin_w = (self.render_size - self._render_w) // 2
        self.render_origin_h = self.render_size - (self.render_size - self._render_h) // 2

        self._screen = pygame.display.set_mode((self.render_size, self.render_size))
        self._screen.fill(pygame.Color("white"))

        pygame.display.set_caption("HVAC Render")

    def render_update(self, heaters, actuators, chtc):
        """
        Update the God View with new data
        """
        if not hasattr(self, "_screen"):
            raise RuntimeError("Render is not initialized yet.")
            

        def colorbar(v, vmin=-10, vmax=100):
            return int(max(0, min(1.0, (v - vmin) / (vmax - vmin))) * 255)
        
        def radius_normalizer(v, vmin=0, vmax=10000, min_pixels=1, max_pixels=10):
            return int(max(0, (v - vmin) / (vmax - vmin)) * (max_pixels - min_pixels) + min_pixels)

        # Paint ambient temerature
        r = colorbar(self.ambient_temp)
        self._screen.fill(pygame.Color(r, 0, 255 - r, 128))

        # paint room temperature
        for i in range(self.n_width):
            for j in range(self.n_length):
                x = self.render_origin_w + i * self._render_cell_size
                y = self.render_origin_h - (j + 1) * self._render_cell_size
                rect = pygame.Rect(x, y, self._render_cell_size, self._render_cell_size)
                r = colorbar(self.state[i][j])
                color = pygame.Color(r, 0, 255 - r, 128)
                pygame.draw.rect(self._screen, color, rect)

        # paint heaters
        for i, equip in enumerate(self.equipments):
            pixels = ((equip.loc / self.cell_size) * self._render_cell_size).astype(int)
            r = radius_normalizer(heaters[i], vmax=10000)
            xs = pixels[0] + self.render_origin_w
            ys = self.render_origin_h - pixels[1]
            pygame.draw.circle(self._screen, pygame.Color(255,0,0,255), (xs,ys), r, width=0)

        # paint coolers
        for i, cooler in enumerate(self.coolers):
            pixels = ((cooler.loc / self.cell_size) * self._render_cell_size).astype(int)
            r = radius_normalizer(actuators[i], vmin=0, vmax=1)
            xs = pixels[0] + self.render_origin_w
            ys = self.render_origin_h - pixels[1]
            pygame.draw.circle(self._screen, pygame.Color(0,255,0,255), (xs,ys),r, width=0)

        # paint chtc
        for i in range(self.n_width + 1):
            for j in range(self.n_length + 1):
                xs = self.render_origin_w + i * self._render_cell_size
                ys = self.render_origin_h - j * self._render_cell_size
                xe0 = self.render_origin_w + i * self._render_cell_size
                ye0 = self.render_origin_h - (j + 1) * self._render_cell_size
                xe1 = self.render_origin_w + (i + 1) * self._render_cell_size
                ye1 = self.render_origin_h - j * self._render_cell_size
                alpha0 = colorbar(chtc[i][j][0], vmin=0, vmax=50)
                alpha1 = colorbar(chtc[i][j][1], vmin=0, vmax=50)
                width0 = 1
                width1 = 1
                if(chtc[i][j][0] < 5):
                    alpha0 = 0
                    width0 = 5
                if(chtc[i][j][1] < 5):
                    alpha1 = 0
                    width1 = 5
                if(j < self.n_length):
                    pygame.draw.line(self._screen, pygame.Color(alpha0,alpha0,alpha0), (xs,ys), (xe0,ye0), width=width0)
                if(i < self.n_width):
                    pygame.draw.line(self._screen, pygame.Color(alpha1,alpha1,alpha1), (xs,ys), (xe1,ye1), width=width1)

        pygame.display.update()
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()

        return done, keys