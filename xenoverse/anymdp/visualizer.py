"""
AnyMDP Task Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib.cm as cm
 

def task_visualizer(task, show_gui=True, file_path=None):
    if("state_embedding" not in task):
        raise Exception("No state embedding found in task")
    
    ns, sv = task["state_embedding"].shape
    ns1, na, ns2 = task["reward"].shape
    ns3, na1, ns4 = task["transition"].shape

    assert ns == ns1 == ns2 == ns3 == ns4
    assert na == na1

    projection_matrix = np.random.rand(sv, 3)

    s= task["state_embedding"]
    s_min = s.min(axis=0)
    s_max = s.max(axis=0)
    coordinates = np.matmul((s - s_min) / (s_max - s_min), projection_matrix)
    link_strength_normalized = np.mean(task["transition"], axis=1)
    rewards = np.mean(task["reward"], axis=(0, 1))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    norm = mcolors.Normalize(vmin=np.min(rewards), vmax=np.max(rewards))
    cmap = cm.viridis

    ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                         c=cmap(norm(rewards)), marker='o')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.2, aspect=5)
    cbar.set_label('Rewards')

    for i in range(ns):
        for j in range(ns):
            if i != j and link_strength_normalized[i,j] > 0.1:  
                x_values = [coordinates[i, 0], coordinates[j, 0]]
                y_values = [coordinates[i, 1], coordinates[j, 1]]
                z_values = [coordinates[i, 2], coordinates[j, 2]]
                c = max(0, float(1 - 2 * link_strength_normalized[i, j]))
                ax.plot(x_values, y_values, z_values, color=(c,c,c), linewidth=2)
       
    ax.set_title('AnyMDP Task Visualization')

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.zaxis.set_ticks_position('none')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    
    ax.grid(False)
    if show_gui:
        plt.show()
    if file_path is not None:
        plt.savefig(file_path)


if __name__ == '__main__':
    from xenoverse.anymdp import AnyMDPTaskSampler
    task = AnyMDPTaskSampler(128, 5, keep_metainfo=True)
    task_visualizer(task)