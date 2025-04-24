"""
AnyMDP Task Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
 

def task_visualizer(task, show_gui=True, file_path=None):
    """
    绘制一个 N x N 的网格图，格子之间用横线和纵线填充，格子内着色，带透明度，
    并在最下方和最左方分别标记给定的标签列表。

    参数：
    N : 网格的大小（N x N）
    x_labels : x 轴的标签列表（长度为 N）
    y_labels : y 轴的标签列表（长度为 N）
    """
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))

    # 设置坐标轴范围
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)

    # 设置坐标轴刻度
    ax.set_xticks(np.arange(0, N+1))
    ax.set_yticks(np.arange(0, N+1))

    # 设置坐标轴标签
    ax.set_xticklabels([''] + x_labels)  # 在开头添加一个空字符串，以匹配刻度位置
    ax.set_yticklabels([''] + y_labels)  # 在开头添加一个空字符串，以匹配刻度位置

    # 绘制网格线
    for i in range(N+1):
        ax.axhline(y=i, color='black', linewidth=0.8)
        ax.axvline(x=i, color='black', linewidth=0.8)

    # 填充格子颜色，带透明度
    for i in range(N):
        for j in range(N):
            rect = plt.Rectangle((i, j), 1, 1, facecolor='blue', alpha=0.3, edgecolor='none')
            ax.add_patch(rect)

    # 设置坐标轴
    plt.gca().invert_yaxis()  # 反转 y 轴，使 (0, 0) 在左上角
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    from xenoverse.anymdp import AnyMDPTaskSampler
    task = AnyMDPTaskSampler(128, 5, keep_metainfo=True)
    task_visualizer(task)