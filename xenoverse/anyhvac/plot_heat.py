import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from xenoverse.anyhvac.anyhvac_env import HVACEnv  # 确保路径正确

def plot_equipment_heat_curves(task_file_path):
    # 创建保存图片的目录
    plot_dir = "./plot"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 加载任务文件
    with open(task_file_path, 'rb') as f:
        task = pickle.load(f)
    
    # 创建环境实例
    env = HVACEnv()
    env.set_task(task)
    
    # 时间范围 (0 到 5040 秒)，每秒一个点
    time_steps = np.arange(0, 648000, 30)  # 每秒一个点
    
    # 为每个设备创建发热曲线
    plt.figure(figsize=(15, 10))
    
    # 初始化总发热量数组
    total_heat_values = np.zeros(len(time_steps))
    
    for i, equipment in enumerate(env.equipments):
        heat_values = []
        for t in time_steps:
            # 调用设备的 power_heat 方法获取当前时间的发热量
            heat = equipment(t)["heat"]
            heat_values.append(heat)
        
        # 累加到总发热量
        total_heat_values += np.array(heat_values)
        
        # 绘制单个设备曲线
        plt.plot(time_steps / 30, heat_values, label=f'Equipment {i+1}')
    
    # 添加图表元素
    plt.title('Equipment Heat Output Over Time (Per Second)')
    plt.xlabel('Steps (30 seconds)')
    plt.ylabel('Heat Output (W)')
    plt.legend()
    plt.grid(True)
    
    # 保存图表到plot目录
    plt.savefig(os.path.join(plot_dir, 'equipment_heat_curves_per_second.png'), dpi=300)
    plt.show()
    
    # 创建总发热量图表
    plt.figure(figsize=(15, 8))
    plt.plot(time_steps / 30, total_heat_values, 'r-', linewidth=2, label='Total Heat Output')
    
    # 添加图表元素
    plt.title('Total Heat Output Over Time')
    plt.xlabel('Steps (30 seconds)')
    plt.ylabel('Total Heat Output (W)')
    plt.grid(True)
    
    # 添加最大和最小值的标记
    max_idx = np.argmax(total_heat_values)
    min_idx = np.argmin(total_heat_values)
    
    plt.annotate(f'Max: {total_heat_values[max_idx]:.0f} W', 
                 xy=(time_steps[max_idx]/30, total_heat_values[max_idx]),
                 xytext=(time_steps[max_idx]/30 + 100, total_heat_values[max_idx] * 0.95),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.annotate(f'Min: {total_heat_values[min_idx]:.0f} W', 
                 xy=(time_steps[min_idx]/30, total_heat_values[min_idx]),
                 xytext=(time_steps[min_idx]/30 + 100, total_heat_values[min_idx] * 1.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # 保存总发热量图表
    plt.savefig(os.path.join(plot_dir, 'total_heat_output.png'), dpi=300)
    plt.show()
    
    # 另外保存每个设备的单独图表到plot目录
    # for i, equipment in enumerate(env.equipments):
    #     plt.figure(figsize=(10, 6))
    #     heat_values = []
    #     for t in time_steps:
    #         heat = equipment(t)["heat"]
    #         heat_values.append(heat)
        
    #     plt.plot(time_steps, heat_values, color='blue')
    #     plt.title(f'Equipment {i+1} Heat Output Over Time')
    #     plt.xlabel('Time (seconds)')
    #     plt.ylabel('Heat Output (W)')
    #     plt.grid(True)
    #     plt.savefig(os.path.join(plot_dir, f'equipment_{i+1}_heat_curve.png'), dpi=300)
    #     plt.close()

if __name__ == "__main__":
    # 替换为你的任务文件路径
    task_file_path = "./task_file/hvac_task_config_0906.pkl"
    plot_equipment_heat_curves(task_file_path)
    print(f"所有图表已保存到 {os.path.abspath('./plot')} 目录")
