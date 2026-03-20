import pygame
import numpy
import random
import time
from .anyhvac_env import HVACEnv, HVACEnvDiffAction, HVACEnvDiscreteAction
from pygame import font

import os
import subprocess
from datetime import datetime

class HVACEnvVisible(HVACEnvDiffAction):
    def __init__(self, *args, record_video=False, video_fps=30, frame_dir=None, video_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_region = 20
        self.record_video = record_video
        self.video_fps = video_fps
        self.frame_count = 0
        self.frame_dir = frame_dir
        self.video_path = video_path

    def reset(self, *args, **kwargs):
        res =super().reset(*args, **kwargs)
        self.render_init(render_size=640)
        self.keyboard_press = pygame.key.get_pressed()
        self.action_change_count = numpy.zeros(self.n_coolers, dtype=int)
        return res

    def step(self, actions):
        self._last_action = self.last_action
        observation, reward, terminated, truncated, info = super().step(actions)
        self._current_action = self.current_action

        keydone, _ = self.render_update(info["heat_power"], info['cool_power'], info["chtc_array"], )
        truncated = truncated or keydone
        return observation, reward, terminated, truncated, info

    def _create_cooler_cooler_graph(self, k_nearest_coolers: int = 3):
        """
        Create cooler-cooler relationship graph using KNN.

        Args:
            k_nearest_coolers (int): The number of nearest coolers to consider for each cooler.
                                     Defaults to 3.

        Returns:
            numpy.ndarray: The cooler-cooler relationship graph.
        """
        n_coolers = self.cooler_sensor_topology.shape[0]
        agent_graph = numpy.zeros((n_coolers, n_coolers), dtype=numpy.float32)
        
        # Get cooler positions
        # Assuming self.coolers is an iterable of objects with a 'loc' attribute
        cooler_positions = numpy.array([cooler.loc for cooler in self.coolers])
        
        # Compute pairwise distances
        for i in range(n_coolers):
            for j in range(n_coolers):
                if i != j:
                    dist = numpy.linalg.norm(cooler_positions[i] - cooler_positions[j])
                    agent_graph[i, j] = dist
        
        # Convert to KNN graph (k='k_nearest_coolers' nearest neighbors)
        k = min(k_nearest_coolers, n_coolers - 1)
        for i in range(n_coolers):
            # Get k nearest neighbors
            distances = agent_graph[i, :]
            nearest_indices = numpy.argsort(distances)[1:k+1]  # Exclude self
            
            # Set connections
            agent_graph[i, :] = 0
            agent_graph[i, nearest_indices] = 1
        
        # Make symmetric and add self-connections (if desired, currently commented out)
        # agent_graph = numpy.maximum(agent_graph, agent_graph.T)
        # numpy.fill_diagonal(agent_graph, 1.0) # Uncomment if self-connections are needed
        self.agent_graph = agent_graph
        return agent_graph

    def generate_random_colors(self, num_colors=5):
        colors = []
        for _ in range(num_colors):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append((r, g, b))
        colors.sort(key=lambda c: 0.299*c[0] + 0.587*c[1] + 0.114*c[2])
        return colors

    def render_init(self, render_size=640):
        """
        Initialize a God View With Landmarks and Collaboration View (side by side)
        """
        font.init()
        self._font = font.SysFont("Arial", 18)
        self.render_size = render_size
        self.color_list = self.generate_random_colors(20)

        # Initialize the agent drawing
        self._render_cell_size = (self.render_size - self.empty_region) // max(self.n_width, self.n_length)
        self._render_w = self.n_width * self._render_cell_size
        self._render_h = self.n_length * self._render_cell_size
        self.render_origin_w = (self.render_size - self._render_w) // 2
        self.render_origin_h = self.render_size - (self.render_size - self._render_h) // 2

        # Calculate total window size for both views side by side
        total_width = 2 * self.render_size
        total_height = self.render_size
        
        # Initialize the combined window
        self._screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("HVAC Render - God View & Collaboration")
        
        # Create surfaces for each view
        self._surface_god = pygame.Surface((self.render_size, self.render_size))
        self._surface_collab = pygame.Surface((self.render_size, self.render_size))
        
        # Fill backgrounds
        self._surface_god.fill(pygame.Color("white"))
        self._surface_collab.fill(pygame.Color("lightgray"))
        
        # Draw titles
        title_font = pygame.font.SysFont("Arial", 24)
        god_title = title_font.render("God View", True, (0, 0, 0))
        collab_title = title_font.render("Collaboration View", True, (0, 0, 0))
        
        self._surface_god.blit(god_title, (20, 10))
        self._surface_collab.blit(collab_title, (20, 10))
        
        # Draw separator line
        pygame.draw.line(self._surface_collab, (100, 100, 100), (0, 0), (0, self.render_size), 2)
        
        # Update the display
        self._screen.blit(self._surface_god, (0, 0))
        self._screen.blit(self._surface_collab, (self.render_size, 0))
        pygame.display.update()

        self.cooler_graph = self._create_cooler_cooler_graph()

    def render_update_god(self, heaters, actuators, chtc):
        if not hasattr(self, "_surface_god"):
            raise RuntimeError("Render is not initialized yet.")
        
        def colorbar(v, vmin=-10, vmax=100):
            return int(max(0, min(1.0, (v - vmin) / (vmax - vmin))) * 255)
        
        def radius_normalizer(v, vmin=0, vmax=10000, min_pixels=1, max_pixels=10):
            return int(max(0, (v - vmin) / (vmax - vmin)) * (max_pixels - min_pixels) + min_pixels)

        # Paint ambient temerature
        r = colorbar(self.ambient_temp)
        self._surface_god.fill(pygame.Color(r, 0, 255 - r, 128))

        # paint room temperature
        for i in range(self.n_width):
            for j in range(self.n_length):
                x = self.render_origin_w + i * self._render_cell_size
                y = self.render_origin_h - (j + 1) * self._render_cell_size
                rect = pygame.Rect(x, y, self._render_cell_size, self._render_cell_size)
                r = colorbar(self.state[i][j])
                color = pygame.Color(r, 0, 255 - r, 128)
                pygame.draw.rect(self._surface_god, color, rect)

        # paint heaters
        for i, equip in enumerate(self.equipments):
            pixels = ((equip.loc / self.cell_size) * self._render_cell_size).astype(int)
            r = radius_normalizer(heaters[i], vmax=10000)
            xs = pixels[0] + self.render_origin_w
            ys = self.render_origin_h - pixels[1]
            pygame.draw.circle(self._surface_god, pygame.Color(255,0,0,255), (xs,ys), r, width=0)

        # paint coolers
        for i, cooler in enumerate(self.coolers):
            pixels = ((cooler.loc / self.cell_size) * self._render_cell_size).astype(int)
            r = radius_normalizer(actuators[i], vmin=0, vmax=10000)
            xs = pixels[0] + self.render_origin_w
            ys = self.render_origin_h - pixels[1]
            pygame.draw.circle(self._surface_god, pygame.Color(0,255,0,255), (xs,ys),r, width=0)

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
                    pygame.draw.line(self._surface_god, pygame.Color(alpha0,alpha0,alpha0), (xs,ys), (xe0,ye0), width=width0)
                if(i < self.n_width):
                    pygame.draw.line(self._surface_god, pygame.Color(alpha1,alpha1,alpha1), (xs,ys), (xe1,ye1), width=width1)
    
    def render_update_collab(self, actuators, chtc):
        def colorbar(v, vmin=-10, vmax=100):
            return int(max(0, min(1.0, (v - vmin) / (vmax - vmin))) * 255)
        def radius_normalizer(v, vmin=0, vmax=10000, min_pixels=1, max_pixels=10):
            return int(max(0, (v - vmin) / (vmax - vmin)) * (max_pixels - min_pixels) + min_pixels)
        # Paint ambient temerature
        r = colorbar(self.ambient_temp)
        self._surface_collab.fill(pygame.Color(r, 0, 255 - r, 128))
        
        # paint room temperature
        for i in range(self.n_width):
            for j in range(self.n_length):
                x = self.render_origin_w + i * self._render_cell_size
                y = self.render_origin_h - (j + 1) * self._render_cell_size
                rect = pygame.Rect(x, y, self._render_cell_size, self._render_cell_size)
                r = colorbar(self.state[i][j])
                color = pygame.Color(r, 0, 255 - r, 128)
                pygame.draw.rect(self._surface_collab, color, rect)

        # paint coolers
        # for i, cooler in enumerate(self.coolers):
        #     pixels = ((cooler.loc / self.cell_size) * self._render_cell_size).astype(int)
        #     r = radius_normalizer(actuators[i], vmin=0, vmax=10000)
        #     xs = pixels[0] + self.render_origin_w
        #     ys = self.render_origin_h - pixels[1]
        #     pygame.draw.circle(self._surface_collab, pygame.Color(0,255,0,255), (xs,ys),r, width=0)

        # paint chtc
        for i in range(self.n_width + 1):
            for j in range(self.n_length + 1):
                xs = self.render_origin_w + i * self._render_cell_size
                ys = self.render_origin_h - j * self._render_cell_size
                xe0 = self.render_origin_w + i * self._render_cell_size
                ye0 = self.render_origin_h - (j + 1) * self._render_cell_size
                xe1 = self.render_origin_w + (i + 1) * self._render_cell_size
                ye1 = self.render_origin_h - j * self._render_cell_size

                if(chtc[i][j][0] < 5) and (j < self.n_length):
                    alpha0 = 0
                    width0 = 5
                    if (i == self.n_width) or (i == 0):
                        pygame.draw.line(self._surface_collab, pygame.Color(alpha0,alpha0,alpha0), (xs,ys), (xe0,ye0), width=width0)
                if(chtc[i][j][1] < 5) and (i < self.n_width):
                    alpha1 = 0
                    width1 = 5
                    if (j == self.n_length) or (j == 0 ):
                        pygame.draw.line(self._surface_collab, pygame.Color(alpha1,alpha1,alpha1), (xs,ys), (xe1,ye1), width=width1)
        
        # paint collab graph
        action_change = numpy.abs(self._current_action["value"] - self._last_action["value"]) > 0.0001
        
        for i, cooler in enumerate(self.coolers):
            pixels = ((cooler.loc / self.cell_size) * self._render_cell_size).astype(int)
            xs = pixels[0] + self.render_origin_w
            ys = self.render_origin_h - pixels[1]
            
            pygame.draw.circle(self._surface_collab, pygame.Color(0, 200, 0), (xs, ys), 8)
            
            # Draw cooler label
            cooler_label = self._font.render(f"{i}", True, (150, 150, 150))
            self._surface_collab.blit(cooler_label, (xs - 10, ys - 25))
            
            connected_coolers = numpy.where(self.cooler_graph[i])[0]
            # 修复：直接遍历 connected_coolers 数组中的索引
            
            for idx in connected_coolers:
                connected_cooler = self.coolers[idx]
                pixels_connected = ((connected_cooler.loc / self.cell_size) * self._render_cell_size).astype(int)
                xe = pixels_connected[0] + self.render_origin_w
                ye = self.render_origin_h - pixels_connected[1]
                
                # Draw dashed line between connected coolers
                dx = xe - xs
                dy = ye - ys
                distance = numpy.sqrt(dx**2 + dy**2)
                steps = int(distance / 5)  # Dash length
                
                if action_change[i]:
                    pygame.draw.line(self._surface_collab, pygame.Color(200, 200, 200), (xs, ys), (xe, ye), 2)
                else:
                    for step in range(steps):
                        if step % 3 == 0:  # Only draw every other segment
                            start_x = xs + dx * step / steps
                            start_y = ys + dy * step / steps
                            end_x = xs + dx * (step + 1) / steps
                            end_y = ys + dy * (step + 1) / steps
                            pygame.draw.line(
                                self._surface_collab, 
                                pygame.Color(100, 100, 100), 
                                (start_x, start_y), 
                                (end_x, end_y), 
                                1
                            )
            if action_change[i]:
                self.action_change_count[i] += 1
            
            print("self.action_change_count: ", self.action_change_count, "self.n_coolers: ", self.n_coolers)
            self.draw_color_gradient_point(self._surface_collab, xs, ys, self.action_change_count[i])

    def draw_color_gradient_point(self, surface, xs, ys, count):
        color_index = (count // 20) % len(self.color_list)
        next_color_index = (color_index + 1) % len(self.color_list)
        
        progress = (count % 20) / 20
        
        current_color = self.color_list[color_index]
        next_color = self.color_list[next_color_index]
        
        r = int(current_color[0] + (next_color[0] - current_color[0]) * progress)
        g = int(current_color[1] + (next_color[1] - current_color[1]) * progress)
        b = int(current_color[2] + (next_color[2] - current_color[2]) * progress)
        
        pygame.draw.circle(surface, (r, g, b), (xs, ys), 8)
                    
    
    def render_update(self, heaters, actuators, chtc):
        """
        Update the God View with new data
        """
        self.render_update_god(heaters, actuators, chtc)
        self.render_update_collab(actuators, chtc)
        self.last_actuators = actuators

        self._screen.blit(self._surface_god, (self.render_size, 0))
        self._screen.blit(self._surface_collab, (0, 0))
        pygame.display.update()
        
        if self.record_video:
            self._save_frame()
            
        done = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done=True
        keys = pygame.key.get_pressed()

        return done, keys
    
    def _init_video_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_dir = f"frames_{timestamp}"
        os.makedirs(self.frame_dir, exist_ok=True)

        self.video_path = f"hvac_simulation_{timestamp}.mp4"
        self.frame_count = 0

    def _save_frame(self):
        if not self.record_video or not self.frame_dir:
            return
            
        frame_path = os.path.join(self.frame_dir, f"frame_{self.frame_count:06d}.png")
        
        pygame.image.save(self._screen, frame_path)
        self.frame_count += 1

    def _create_video(self):
        if not self.record_video or not self.frame_dir or self.frame_count == 0:
            return
            
        try:
            cmd = [
                'ffmpeg',
                '-y',  
                '-framerate', str(self.video_fps),
                '-i', os.path.join(self.frame_dir, 'frame_%06d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
                self.video_path
            ]
            
            subprocess.run(cmd, check=True)
            print(f"视频已保存到: {self.video_path}")
            
        except Exception as e:
            print(f"创建视频失败: {e}")
        finally:
            for file in os.listdir(self.frame_dir):
                os.remove(os.path.join(self.frame_dir, file))
            os.rmdir(self.frame_dir)

    def close(self):
        if self.record_video:
            self._create_video()
        super().close()