import numpy
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from copy import deepcopy
from scipy.stats import norm
from typing import Dict, Any, Optional
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible, HVACEnv
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID, HVACSolverLOCPID, HVACSolverGridSearchPID
from xenoverse.anyhvac.anyhvac_env import HVACEnvDiscreteAction, HVACEnvDiffAction
from xenoverse.utils import RandomFourier
import pickle 
from rl_trainer import HVACRLTester

class FourierNoiseGenerator:
    """
    Random Fourier noise generator

    Function
    1. Generate random Fourier noise
    2. Automatically calculate the noise range
    3. Provide normalized noise values

    Parameter
    ndim: Output dimension
    max_order: Control the period range
    max_item: Controls the number of superimposed items
    max_steps: Time series length
    box_size: Controls the amplitude
    """
    
    def __init__(self, ndim=2, max_order=8, max_item=3, max_steps=2000, box_size=0.5):
        """
        Initialize the noise generator
        """
        self.ndim = ndim
        self.max_order = max_order
        self.max_item = max_item
        self.max_steps = max_steps
        self.box_size = box_size
        
        # Create a random Fourier function
        self.rf = RandomFourier(
            ndim=ndim,
            max_order=max_order,
            max_item=max_item,
            max_steps=max_steps,
            box_size=box_size
        )
        
        # Calculate the noise range
        self.y_min, self.y_max = self._calculate_range()
    
    def _calculate_range(self, max_attempts=10):
        """
        Calculate the noise range and ensure it is valid
        """
        for attempt in range(max_attempts):
            t = numpy.arange(0, self.max_steps)
            y = self.rf(t)
            y_min, y_max = y.min(), y.max()
            
            if y_min != y_max:
                return y_min, y_max
            
            self.rf = RandomFourier(
                ndim=self.ndim,
                max_order=self.max_order,
                max_item=self.max_item,
                max_steps=self.max_steps,
                box_size=self.box_size
            )
        
        raise ValueError(f"The effective noise range cannot be generated. Number of attempts: {max_attempts}")
    
    def get_noise(self, t):
        """
        Obtain the normalized noise value

        Parameter
        t: Time point or time series

        Return
        Normalized noise value (0-1 range)
        """
        raw_noise = self.rf(t)
        normalized = (raw_noise - self.y_min) / (self.y_max - self.y_min)
        return normalized

    def generate_sequence(self):
        """
        Generate the noise values of the complete time series

        Return
        A normalized noise sequence of length max_steps
        """
        t = numpy.arange(0, self.max_steps)
        return self.get_noise(t)

class HVACActionNoiseFourier(object):
    def __init__(self, agent_num):

        self.agent_num = agent_num
        noise_num_factor = numpy.random.uniform(low=0.4, high=0.6)
        self.add_noise_agent_num = int(round(noise_num_factor * agent_num))
        add_inverse_noise_num = int(round(0.15 * self.add_noise_agent_num))
        self.noise_value_factor = numpy.random.uniform(low=0.25, high=0.75, size=self.add_noise_agent_num)

        print(f"self.add_noise_agent_num: {self.add_noise_agent_num}")
        print(f"self.noise_value_factor: {self.noise_value_factor}")
        self.fourier_mask = numpy.zeros(agent_num, dtype=bool)
        self.inverse_mask = numpy.zeros(agent_num, dtype=bool)
        indices = numpy.random.choice(agent_num, self.add_noise_agent_num, replace=False)
        self.fourier_mask[indices] = True
        available_indices = numpy.where(~self.fourier_mask)[0]
        inverse_indices = numpy.random.choice(available_indices, size=add_inverse_noise_num, replace=False)
        self.inverse_mask[inverse_indices] = True

        ndim = 1
        max_order = 32
        max_item = 3
        max_steps = 4000
        box_size = 0.5
        
        self.generators = [
            FourierNoiseGenerator(
                ndim=ndim,
                max_order=max_order,
                max_item=max_item,
                max_steps=max_steps,
                box_size=box_size
            ) for _ in range(self.add_noise_agent_num)
        ]
    def add_noise(self, current_step, action):
        noisy_action = action.copy()
        noise = []
        for i in range(self.add_noise_agent_num):
            noise.append(self.generators[i].get_noise(current_step))
        noise = numpy.array(noise).squeeze()
        noisy_action[self.fourier_mask] = self.noise_value_factor * noise + (1 - self.noise_value_factor) * action[self.fourier_mask]
        noisy_action[self.inverse_mask] =  1 - noisy_action[self.inverse_mask]
        return noisy_action

class HVACActionNoise(object):
    def __init__(self, T_ini, T_fin, T_decay_type, T_total_step, mask_change_step=100):
        
        self.T_ini = max(1, T_ini)
        self.T_fin = min(0, T_fin)
        self.T_decay_type = T_decay_type
        self.T_total_step = T_total_step
        self.mask_change_step = mask_change_step
        self.mask = []

        self.dT_linear = (self.T_fin - self.T_ini) / self.T_total_step
        self.dT_exp = numpy.exp((numpy.log(max(0.0001, self.T_fin)) - numpy.log(self.T_ini)) / self.T_total_step)

    
    def add_noise(self, current_step, action):
        if self.T_decay_type == "linear":
            Temp = self.T_ini + min(current_step, self.T_total_step) * self.dT_linear
        elif self.T_decay_type == "exponential":
            Temp = self.T_ini * (self.dT_exp ** min(current_step, self.T_total_step))
        n = len(action)
        k = int(round(Temp * n))
        if k == 0:
            return action
        if len(self.mask) == 0 or current_step % self.mask_change_step == 0:
            self.mask = numpy.zeros(n, dtype=bool)
            indices = numpy.random.choice(n, k, replace=False)
            self.mask[indices] = True
        noise = numpy.random.random(action.shape)
        noisy_action = action.copy()
        noisy_action[self.mask] = Temp * noise[self.mask] + (1 - Temp) * action[self.mask]
        return noisy_action

def plot_cooler_values(values, output_dir, output_name, n_coolers, show_plot=False):
    """
    绘制并保存多个冷却器的数值曲线图
    
    参数:
    values (list or numpy.array): 二维数组，形状为 (时间步数, 冷却器数量)
    output_dir (str): 输出文件保存目录
    n_coolers (int): 冷却器数量
    show_plot (bool): 是否显示绘图结果 (默认为True)
    
    返回:
    tuple: (csv文件路径, 图像文件路径)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为NumPy数组
    values_array = numpy.array(values)
    
    # 保存为CSV文件
    df = pd.DataFrame(values_array, columns=[f"cooler_{i}" for i in range(n_coolers)])
    csv_path = os.path.join(output_dir, "cooler_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"数据已保存至: {csv_path}")
    
    # 计算图像尺寸参数
    min_width_per_plot = 4
    min_height_per_plot = 2
    max_width = 24
    max_height = 36
    
    # 计算子图布局
    n_cols = min(4, n_coolers)
    n_rows = (n_coolers + n_cols - 1) // n_cols
    
    # 调整尺寸参数
    min_width_per_plot *= 4
    max_width *= 4
    
    # 计算最终图像尺寸
    fig_width = min(n_cols * min_width_per_plot, max_width)
    fig_height = min(n_rows * min_height_per_plot, max_height)
    
    # 创建图像
    plt.figure(figsize=(fig_width, fig_height))
    
    # 设置子图间距
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        wspace=0.3,
        hspace=0.4
    )
    
    # 绘制每个冷却器的曲线
    for i in range(n_coolers):
        ax = plt.subplot(n_rows, n_cols, i + 1)
        cooler_values = values_array[:, i]
        
        # 绘制曲线
        ax.plot(cooler_values, 'b-', linewidth=0.5)
        
        # 设置子图属性
        ax.set_title(f"Cooler {i}", fontsize=10)
        ax.set_xlabel("Time Step", fontsize=8)
        ax.set_ylabel("Value", fontsize=8)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 调整布局并保存图像
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=2.0)
    plot_path = os.path.join(output_dir, f"{output_name}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"图像已保存至: {plot_path}")
    
    # 显示图像
    if show_plot:
        plt.show()
    
    # 关闭图像释放内存
    plt.close()
    
    return csv_path, plot_path

def plot_action_prob_distributions(
    action_prob_array, 
    discrete_temp_diffs, 
    output_dir, 
    output_name="action_prob_distributions",
    show_plot=False
):
    """
    绘制多个 agent 在 n 步上的平均动作概率分布柱状图
    
    参数:
    action_prob_array (list): 存储每一步的概率分布，每个元素形状为 (n_coolers, num_steps)
    discrete_temp_diffs (numpy.ndarray): 离散化的温度差值数组
    output_dir (str): 输出文件保存目录
    output_name (str): 输出文件名前缀
    show_plot (bool): 是否显示绘图结果
    """
    import numpy
    import matplotlib.pyplot as plt
    import os
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换为 numpy 数组并计算 n 步平均 (形状: [n_steps, n_coolers, num_steps])
    # 转置为 [n_coolers, num_steps, n_steps] 以便按 agent 计算平均
    if len(action_prob_array) == 0:
        print("警告: action_prob_array 为空，无法绘图")
        return None
    
    # 转换为数组形状: (n_steps, n_coolers, num_steps)
    prob_array = numpy.array(action_prob_array)
    
    # 计算平均概率: 在步数维度上求平均，结果形状 (n_coolers, num_steps)
    avg_probs = numpy.mean(prob_array, axis=0)
    
    n_coolers = avg_probs.shape[0]
    n_actions = avg_probs.shape[1]
    
    # 计算子图布局
    n_cols = min(4, n_coolers)
    n_rows = (n_coolers + n_cols - 1) // n_cols
    
    # 创建图形
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    
    # 处理只有一个 agent 的情况
    if n_coolers == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # 设置颜色映射
    colors = plt.cm.viridis(numpy.linspace(0, 1, n_actions))
    
    # 为每个 agent 绘制柱状图
    for i in range(n_coolers):
        ax = axes[i]
        
        # 绘制柱状图
        bars = ax.bar(
            range(n_actions), 
            avg_probs[i], 
            color=colors, 
            alpha=0.8, 
            edgecolor='black', 
            linewidth=0.5
        )
        
        # 设置标题和标签
        ax.set_title(f'Agent {i} - Avarage Probability Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('action', fontsize=10)
        ax.set_ylabel('probability', fontsize=10)
        
        # 设置 x 轴刻度标签为实际的温度差值
        tick_step = max(1, n_actions // 10)  # 如果动作太多，只显示部分刻度
        ax.set_xticks(range(0, n_actions, tick_step))
        ax.set_xticklabels([f'{discrete_temp_diffs[j]:.1f}' for j in range(0, n_actions, tick_step)], 
                          rotation=45, ha='right')
        
        # 添加网格
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # 设置 y 轴范围
        ax.set_ylim(0, max(0.1, numpy.max(avg_probs[i]) * 1.1))
        
        # 在每个柱子上方显示概率值
        for bar, prob in zip(bars, avg_probs[i]):
            if prob > 0.01:  # 只显示概率大于 1% 的
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 隐藏多余的子图
    for i in range(n_coolers, len(axes)):
        axes[i].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    plot_path = os.path.join(output_dir, f'{output_name}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f'动作概率分布图已保存至: {plot_path}')
    
    # 显示图像
    if show_plot:
        plt.show()
    
    # 关闭图形释放内存
    plt.close()
    
    # 保存平均概率数据到 CSV
    csv_path = os.path.join(output_dir, f'{output_name}_avg_probs.csv')
    df = pd.DataFrame(
        avg_probs,
        columns=[f'action_{j}' for j in range(n_actions)],
        index=[f'agent_{i}' for i in range(n_coolers)]
    )
    df.to_csv(csv_path)
    print(f'平均概率数据已保存至: {csv_path}')
    
    return plot_path, csv_path


class ActionDistributionDiscretizer:
    """
    将连续高斯动作分布转换为离散化温度调整值分布的工具类
    """
    
    def __init__(self, env: 'HVACEnvDiffAction', need_effective_probs=False):
        self.env = env
        self.need_effective_probs = need_effective_probs
        
        # 从环境获取参数
        self.num_steps = env.num_steps
        self.discrete_temp_diffs = env.discrete_values.copy()
        self.action_resolution_temp = env.action_resulotion_temp
        self.target_temp_offset = getattr(env, 'target_temp_offset', 3.0)
        self._action_value_to_temp = env._action_value_to_temp
        
        # 验证环境方法存在
        if not hasattr(env, '_action_temp_to_value') or not hasattr(env, '_action_value_to_temp'):
            raise AttributeError("Environment must have _action_temp_to_value and _action_value_to_temp methods")
    
    def discretize_distribution(
        self, 
        mean: numpy.ndarray, 
        std: numpy.ndarray, 
        last_action_value: numpy.ndarray,
        target_temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        将连续高斯分布转换为离散温度调整值的分布
        """
        # 处理批量维度
        if mean.ndim == 2:
            if mean.shape[0] == 1:
                mean = mean[0]
                std = std[0]
            else:
                raise ValueError(f"Batch size > 1 not supported. Got shape {mean.shape}")
        
        # 确保是1D数组
        if mean.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {mean.shape}")
        
        if mean.shape != std.shape:
            raise ValueError(f"Shape mismatch: mean {mean.shape} vs std {std.shape}")
        
        if mean.shape != last_action_value.shape:
            raise ValueError(f"Shape mismatch: mean {mean.shape} vs last_action_value {last_action_value.shape}")
        
        # 计算离散温度差值的概率分布
        temp_diff_dist = self._compute_temp_diff_distribution(mean, std)

        if not self.need_effective_probs:
            return temp_diff_dist
        
        # 考虑温度约束，计算实际生效的温度差值分布
        effective_dist = self._compute_effective_distribution(
            temp_diff_dist, 
            last_action_value,
            target_temperature
        )
        
        return {**temp_diff_dist, **effective_dist}
    
    def _compute_temp_diff_distribution(
        self, 
        mean: numpy.ndarray, 
        std: numpy.ndarray
    ) -> Dict[str, numpy.ndarray]:
        """
        计算离散温度差值的概率分布
        """
        mean = numpy.asarray(mean, dtype=numpy.float64)
        std = numpy.asarray(std, dtype=numpy.float64)
        
        n_coolers = mean.shape[0]
        probs = numpy.zeros((n_coolers, self.num_steps))
        
        for i in range(n_coolers):
            mu_action = float(mean[i])
            sigma_action = float(std[i])
            
            if not numpy.isfinite(mu_action) or not numpy.isfinite(sigma_action):
                probs[i, self.num_steps // 2] = 1.0
                continue
            
            if sigma_action < 1e-6:
                # 确定性分布
                idx = numpy.clip(int(numpy.round(mu_action * (self.num_steps - 1))), 0, self.num_steps - 1)
                probs[i, idx] = 1.0
            else:
                # 计算每个索引对应的概率
                bin_half_width = 0.5 / (self.num_steps - 1)
                
                for j in range(self.num_steps):
                    a_center = j / (self.num_steps - 1)
                    
                    if self.num_steps == 1:
                        prob = 1.0
                    elif j == 0:
                        # 下边界 [0, a_center + bin_half_width]
                        prob = norm.cdf(a_center + bin_half_width, loc=mu_action, scale=sigma_action)
                    elif j == self.num_steps - 1:
                        # 上边界 [a_center - bin_half_width, 1]
                        prob = 1.0 - norm.cdf(a_center - bin_half_width, loc=mu_action, scale=sigma_action)
                    else:
                        # 内部点 [a_center - bin_half_width, a_center + bin_half_width]
                        lower = a_center - bin_half_width
                        upper = a_center + bin_half_width
                        prob = norm.cdf(upper, loc=mu_action, scale=sigma_action) - \
                               norm.cdf(lower, loc=mu_action, scale=sigma_action)
                    
                    probs[i, j] = max(0.0, prob)
        
        # 归一化（在循环结束后进行）
        for i in range(n_coolers):
            probs_sum = probs[i].sum()
            if probs_sum > 1e-10:
                probs[i] /= probs_sum
        
        # 计算期望温度差值
        expected_temp_diff = numpy.sum(probs * self.discrete_temp_diffs, axis=1)
        
        return {
            'temp_diff_probs': probs,
            'discrete_temp_diffs': self.discrete_temp_diffs,
            'expected_temp_diff': expected_temp_diff
        }
    
    def _compute_effective_distribution(
        self,
        temp_diff_dist: Dict[str, numpy.ndarray],
        last_action_value: numpy.ndarray,
        target_temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        计算实际生效的温度差值分布（考虑温度约束）
        """
        last_temp = self._action_value_to_temp(last_action_value)
        
        if target_temperature is None:
            target_temperature = self.env.target_temperature
        
        n_coolers = len(last_temp)
        temp_diff_probs = temp_diff_dist['temp_diff_probs']
        
        effective_temp_diffs = []
        effective_probs = []
        effective_mapping = []
        
        for cooler_idx in range(n_coolers):
            temp_diff_prob_map = {}
            
            for temp_idx, temp_diff in enumerate(self.discrete_temp_diffs):
                prob = temp_diff_probs[cooler_idx, temp_idx]
                if prob <= 0:
                    continue
                
                # 计算当前温度
                current_temp = last_temp[cooler_idx] + temp_diff
                
                # 应用温度约束
                min_allowed_temp = target_temperature - self.target_temp_offset
                if current_temp < min_allowed_temp:
                    effective_temp_diff = min_allowed_temp - last_temp[cooler_idx]
                else:
                    effective_temp_diff = temp_diff
                
                # 四舍五入到与环境相同的精度（分辨率 0.5）
                closest_idx = numpy.argmin(numpy.abs(self.discrete_temp_diffs - effective_temp_diff))
                effective_temp_diff_rounded = self.discrete_temp_diffs[closest_idx]
                
                # 累加概率
                temp_key = f"{effective_temp_diff_rounded:.1f}"
                temp_diff_prob_map[temp_key] = temp_diff_prob_map.get(temp_key, 0) + prob
            
            # 转换为数组
            items = list(temp_diff_prob_map.items())
            items.sort(key=lambda x: float(x[0]))
            
            values = [float(k) for k, _ in items]
            probs = [v for _, v in items]
            
            # 归一化
            probs = numpy.array(probs, dtype=numpy.float64)
            probs_sum = probs.sum()
            if probs_sum > 1e-10:
                probs /= probs_sum
            
            effective_temp_diffs.append(numpy.array(values))
            effective_probs.append(probs)
            effective_mapping.append(items)
        
        return {
            'effective_temp_diffs': numpy.array(effective_temp_diffs, dtype=object),
            'effective_probs': numpy.array(effective_probs, dtype=object),
            'effective_mapping': effective_mapping,
        }

if __name__ == "__main__":
    

    env = HVACEnvDiffAction(reward_mode = 1, verbose=True)
    # env = HVACEnvDiscreteAction(reward_mode = 1, verbose=True)
    TASK_CONFIG_PATH = "./models/0309/hvac_task_1074/hvac_task_1074.pkl" # "./models/hvac_task_3333/hvac_task_3333.pkl"
    RL_MODEL_PATH = "./models/0309/hvac_task_1057/rppo_reward_mode_1_stage2_generateF.zip"
    output_dir = "./models/0309/hvac_task_1057/stage2_GF/"
    discretize_rl_action_space=True
    add_action_cost = False
    too_cold_limit = False
    model = HVACRLTester(RL_MODEL_PATH, "rppo", "cpu")
    model.reset()
    model_test = HVACRLTester(RL_MODEL_PATH, "rppo", "cpu")
    model_test.reset()
    # try:
    with open(TASK_CONFIG_PATH, "rb") as f:
        task = pickle.load(f)
    print(f"Loaded existing task config from {TASK_CONFIG_PATH}")
    print("t_ambient", task['ambient_temp'])
    print("target_temperature", task['target_temperature'])
    try:
        env.set_task(task, 
                     discretize_rl_action_space=discretize_rl_action_space,
                     add_action_cost=add_action_cost,
                     too_cold_limit=too_cold_limit)
        print("set task using diff_action")
    except Exception as e:
        env.set_task(task)
        print("set task using discrete_action")


    
    print("target_temperature: ", task['target_temperature'])
    print("unify_cooler_coefficent: ", task['unify_cooler_coefficent'])
    # env.set_control_type("power")
    env.set_random_start_t(False)
    
    terminated, truncated = False,False
    obs = env.reset()[0]

    pid = HVACSolverGTPID(env)
    # pid_params_save_path = os.path.join(os.path.dirname(TASK_CONFIG_PATH), "pid_search_results.json")
    # best_params = pid.search(
    #     max_steps=10000,
    #     save_path=pid_params_save_path,
    #     apply_best=True  # 自动应用最优参数
    # )
    obs = env.reset()[0]

    max_steps = 10000
    current_stage = []
    steps = 0
    n_coolers = len(env.coolers)
    n_sensors = len(env.sensors)
    values = []
    values_label = []
    lstm_states = None
    cool_power_sum = 0.0
    cool_power_count = 0
    
    T_total_step = 10000
    T_ini = 0.5
    T_fin = 0.0
    mask_change_step = 100
    # T_total_step = int(min(T_total_step, max_steps/2) * random.uniform(0.5, 1.5))
    T_total_step = max_steps/2
    # T_decay_type = random.choice(["linear", "exponential"])
    T_decay_type = "linear"
    # T_ini = min(random.uniform(0.75, 1.25) * T_ini, 1.0)
    T_ini = 0.5
    # mask_change_step = int(random.uniform(0.75, 1.25) * mask_change_step)
    # add_noise = HVACActionNoise(T_ini, T_fin, T_decay_type, T_total_step, mask_change_step)
    add_noise = HVACActionNoiseFourier(n_coolers)

    actionDistributionDiscretizer = ActionDistributionDiscretizer(env)
    action_prob_array = []

    while steps < max_steps:
        # Max coolers power
        # action = env.sample_action(mode="max")

        # Constant temp setting
        # action = env.sample_action(mode="constant")

        # RL
        # action = model.predict(obs, deterministic=False)
        # action_deterministic = model_test.predict(obs, deterministic=True)
        # action_deterministic_value, _ = env._diff_action(action_deterministic)
        # action, dist_info, _ = model.predict_with_distribution(obs, deterministic=False)
        # action_prob_array.append(numpy.squeeze(dist_info['probs']))
        # action_prob_array.append(discrete_dist["temp_diff_probs"])
        
        
        # pid
        action = 1 - pid.policy(obs["sensor_readings"])[n_coolers:]

        # add noise
        # action = add_noise.add_noise(steps, action)
        
        obs, reward, terminated, truncated, info = env.step(action)
        env_action = deepcopy(env.last_action)
        switch = env_action["switch"]
        value = env_action["value"]
        for i in range(len(switch)):
            if switch[i]<0.5:
                value[i] = -1.0
                # action_deterministic_value[i] = -1.0
        print("switch: ", env_action["switch"])
        print("action: ", value * (env.upper_bound - env.lower_bound) + env.lower_bound)
        # print(value)
        values.append(value)
        # values_label.append(action_deterministic_value)

        # print("t: ",env.t)
        print(obs["sensor_readings"])
        if 'cool_power' in info:
            cool_power_sum += numpy.sum(info['cool_power'])
            cool_power_count += 1
        if terminated or truncated:
            print("overheat!!!!!!")
            # break
        current_stage.append(reward)
        if steps < 1:
            print("info.keys(): ", info.keys())
            info_sums = {key: 0.0 for key in info.keys()}
            info_counts = {key: 0 for key in info.keys()}
        for key, value in info.items():
            if isinstance(value, (int, float)):
                info_sums[key] += value
                info_counts[key] += 1
        
        steps += 1
        # print("sensors - ", obs, "\nactions - ", action, "\nrewards - ", reward, "ambient temperature - ", env.ambient_temp)
        if steps % 100 == 0:
            mean_reward = numpy.mean(current_stage)
            
            info_means = {
                key: info_sums[key] / count if (count := info_counts.get(key, 0)) != 0 else 0
                for key in info_sums
            }
            
            info_str = " | ".join([f"{k}:{v:.4f}" for k,v in info_means.items()])
            print(f"Step {steps} | Reward: {mean_reward:.2f} | {info_str}", flush=True)
            
            current_stage = []
            info_sums = {k:0.0 for k in info_sums}
            info_counts = {k:0 for k in info_counts}
    
    if cool_power_count > 0:
        avg_cool_power = cool_power_sum / cool_power_count
        print(f"\navg coolers power: {avg_cool_power:.2f} kW")
    else:
        print("\nno data")
    
    os.makedirs(output_dir, exist_ok=True)

    if len(action_prob_array)>0:
        # save action_prob_array
        prob_array_np = numpy.array(action_prob_array)
        numpy_path = os.path.join(output_dir, "action_prob_array.npy")
        numpy.save(numpy_path, prob_array_np)
        plot_action_prob_distributions(
            action_prob_array=action_prob_array,
            discrete_temp_diffs=actionDistributionDiscretizer.discrete_temp_diffs,  # 或者 env.discrete_values
            output_dir=output_dir,
            output_name="action_probabilities",
            show_plot=True
        )
    plot_cooler_values(values, output_dir, "behavior", n_coolers, show_plot=False)
    plot_cooler_values(values_label, output_dir, "label", n_coolers, show_plot=False)
    print("Finish!")


