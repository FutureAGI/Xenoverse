"""
AnyMDP Task Visualization
"""

import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy
from xenoverse.anymdp.solver import update_value_matrix
from xenoverse.anymdp.anymdp_env import map_transition_reward


def anymdp_task_visualizer(task, 
                    need_lengends=True, 
                    need_ticks=True,
                    show_gui=True, 
                    file_path=None):
    # 创建一个图形和坐标轴
    fig, ax = plt.subplots(figsize=(8, 8))

    ns = task["ns"]
    na = task["na"]

    transition = task["transition"]
    reward = task["reward"]

    s_0 = task["s_0"]
    s_e = task["s_e"]

    state_mapping = task["state_mapping"]
    state_mapping = [str(state_mapping[i]) for i in range(ns)]

    vm = np.zeros((ns, na))
    vm = update_value_matrix(task["transition"], task["reward"], 0.99, vm)
    vsm = np.max(vm, axis=-1)

    if(need_ticks):
        ax.set_xticks(np.arange(- 0.5, ns + 0.5))
        ax.set_yticks(np.arange(- 0.5, ns + 0.5))
        ax.set_xticklabels([''] + state_mapping)
        ax.set_yticklabels([''] + state_mapping)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ax.set_xlim(0, ns)
    ax.set_ylim(0, ns)

    ax.tick_params(axis='both', which='both', length=0)

    trans_ss = np.mean(transition, axis=1)
    r_position = np.mean(reward, axis=(0, 1))

    for i in range(ns): # State From
        for j in range(ns): # State To
            alpha = min(trans_ss[i, j] * 5.0, 1.0)
            rect = plt.Rectangle((j, i), 1, 1, facecolor='grey', alpha=alpha, edgecolor='none')
            ax.add_patch(rect)

    # Start states
    for s in s_0:
        rect = plt.Rectangle((0, s), ns, 1, facecolor='green', alpha=0.25, edgecolor='none')
        ax.add_patch(rect)

    # End states
    for s in s_e:
        if(s >= ns-1):
            color = 'blue'
            alpha = 0.40
        else:
            color = 'red'
            alpha = 0.20

        rect = plt.Rectangle((0, s), ns, 1, facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(rect)
        rect = plt.Rectangle((s, 0), 1, ns, facecolor=color, alpha=alpha, edgecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('State ($t+1$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('State ($t$)', fontsize=12, fontweight='bold')

    lw = 24 / (ns + 16)
    for i in range(ns + 1):
        ax.axhline(y=i, color='black', linewidth=lw)
        ax.axvline(x=i, color='black', linewidth=lw)

    # Plot the value function
    nonpitfalls = np.array([i for i in range(ns) if i not in s_e])

    v_max = np.max(vsm[nonpitfalls])
    v_min = np.min(vsm[nonpitfalls])

    scale = (v_max - v_min) * 0.05
    ax_v = ax.twinx()
    ax_v.set_ylim(v_min - scale, v_max + scale)
    ax_v.plot(nonpitfalls + 0.5, vsm[nonpitfalls], color='black', marker='o', linestyle='-', linewidth=2.5)

    ax_v.set_ylabel('State Value Function', fontsize=12, fontweight='bold', color='black')
    ax_v.tick_params(axis='y', labelcolor='black')

    if(need_lengends):
        transition_patch = mpatches.Patch(color='grey', alpha=0.5, label='$\mathbb{E}_{a}[P(s_t,a,s_{t+1})]$')
        born_patch = mpatches.Patch(color='green', alpha=0.2, label='$\mathcal{S}_0$')
        pitfall_patch = mpatches.Patch(color='red', alpha=0.2, label='$\mathcal{S}_E$ (pitfalls)')
        goal_patch = mpatches.Patch(color='blue', alpha=0.4, label='$\mathcal{S}_E$ (goals)')

        ax.legend(handles=[transition_patch, born_patch, pitfall_patch, goal_patch], loc='center left', fontsize=10)

    # Show and save
    if(file_path is not None):
        plt.savefig(file_path + '.pdf', format='pdf')

    if(show_gui):
        plt.show()

def rearrange_states(task, K=5):
    trans_ss = numpy.sum(task["transition"], axis=1)
    ra_task = deepcopy(task)

    s_map = []
    for s in task["s_0"]:
        s_map.append(s)

    vm = numpy.zeros((task["ns"], task["na"]), dtype='float32')
    vm = update_value_matrix(task["transition"], task["reward"], 0.99, vm)
    vsm = numpy.max(vm, axis=-1)
    print(task["s_e"])

    while len(s_map) < task["ns"]:
        s_trans_sum = []
        for s in range(len(trans_ss)):
            if(s in s_map):
                continue
            p2s = numpy.mean(trans_ss[s_map, [s for _ in range(len(s_map))]], axis=0)
            if(p2s > 1.0e-6):
                s_trans_sum.append((s, vsm[s], p2s))
        s_sorted_trans = sorted(s_trans_sum, key=lambda x:x[2], reverse=True)
        s_sorted_trans = sorted(s_sorted_trans[:K], key=lambda x:x[1], reverse=False)
        s_map.append(s_sorted_trans[0][0])

    # make the goal last
    for s in task["s_e"]:
        if(numpy.sum(task["reward"][:, :, s] > 0) and s_map.index(s) > task['ns'] // 2 and s_map.index(s) != task["ns"] - 1): # mv the goal to the end
            s_map[-1], s_map[s_map.index(s)] = s_map[s_map.index(s)], s_map[-1]

    s_map_inv = list(range(task["ns"]))
    for i, s in enumerate(s_map):
        s_map_inv[s] = i

    ra_task["transition"] *= 0.0
    ra_task["reward"] *= 0.0
    ra_task["transition"], ra_task["reward"] = map_transition_reward(
                        task["transition"], 
                        task["reward"], 
                        ra_task["transition"], 
                        ra_task["reward"], 
                        s_map_inv)
    ra_task["state_mapping"] = s_map
    
    ra_task["s_0"] = []
    ra_task["s_e"] = []
    for s in task["s_0"]:
        ra_task["s_0"].append(s_map_inv[s])
    for s in task["s_e"]:
        ra_task["s_e"].append(s_map_inv[s])

    return ra_task

if __name__ == '__main__':
    from xenoverse.anymdp import AnyMDPTaskSampler, GarnetTaskSampler
    ns = 64
    na = 5
    #task = AnyMDPTaskSampler(ns, na, verbose=True)
    task = GarnetTaskSampler(ns, na, b=2, verbose=True)
    task = rearrange_states(task, K=5)
    anymdp_task_visualizer(task, need_ticks=False, 
                           need_lengends=False,
                           file_path=f'./vis_anymdp_ns{ns}na{na}')