import numpy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from .anymdp_env import AnyMDPEnv
from xenoverse.utils import pseudo_random_seed
from restools.plotting.plot_2D import savitzky_golay

class AnyMDPv2Visualizer(AnyMDPEnv):
    
    def set_task(self, task, **kwargs):
        # Set task will automatically reset all records
        super().set_task(task, **kwargs)
        self.observation_records = []
        self.inner_state_records = []
        self.action_records = []
        self.reward_records = []
    
    def color_spec(self, i):
        return [self.color_spec_type[i][idx] for idx in self.colors]

    def reset(self):
        obs, info = super().reset()
        self.observation_records.append(numpy.copy(obs))
        self.action_records.append(numpy.zeros((self.action_dim,)))
        self.inner_state_records.append(numpy.copy(self.inner_state))
        self.reward_records.append(0.0)

        return obs
    
    def step(self, action):
        result = super().step(action)
        # 判断返回结果的长度来兼容旧版和新版 Gym
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        self.observation_records.append(numpy.copy(obs))
        self.inner_state_records.append(numpy.copy(self.inner_state))
        self.action_records.append(numpy.copy(action))
        self.reward_records.append(reward)
        return obs, reward, done, info

    def visualize_and_save(self, filename=None):
        tsne = TSNE(n_components=2, random_state=pseudo_random_seed(),
                    perplexity=10, max_iter=500, learning_rate=100)
        if(filename is not None):
            file_name = file_name
        else:
            file_name = "./anymdp_visualizer_output.pdf"
        obs_arr = numpy.array(self.observation_records, dtype="float32")
        act_arr = numpy.array(self.action_records, dtype="float32")

        s_arr = numpy.array(self.inner_state_records)
        max_steps = len(self.inner_state_records)

        obs_tsne = tsne.fit_transform(numpy.array(obs_arr))
        act_tsne = tsne.fit_transform(numpy.array(act_arr))
        s_tsne = tsne.fit_transform(numpy.array(s_arr))
        
        plt.figure(figsize=(10, 8))
        # Show Observation T-SNE
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(obs_tsne[:, 0], obs_tsne[:, 1], c='black', s=10, alpha=0.2)
        plt.title("Observation", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show Action T-SNE
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(act_tsne[:, 0], act_tsne[:, 1], c='black', s=10, alpha=0.2)
        plt.title("Action", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show State T-SNE
        plt.subplot(2, 2, 3)
        scatter = plt.scatter(s_tsne[:, 0], s_tsne[:, 1], c=self.reward_records, cmap='viridis', s=10, alpha=0.2, marker='o')
        plt.colorbar()
        plt.title("States", fontsize=12, fontweight='bold', color='blue', pad=10)

        # Show Reward Curve
        plt.subplot(2, 2, 4)
        rewards_smooth = savitzky_golay(self.reward_records, window_size=99, order=3)
        scatter = plt.plot(numpy.arange(numpy.size(self.reward_records)), self.reward_records, c='red', alpha=0.2)
        scatter = plt.plot(numpy.arange(numpy.size(rewards_smooth)), rewards_smooth, c='red')
        plt.title("Reward", fontsize=12, fontweight='bold', color='blue', pad=10)
        plt.savefig(file_name)
