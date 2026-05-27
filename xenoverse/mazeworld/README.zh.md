# MazeWorld

[English](README.md) | 中文

MazeWorld 是一个程序化生成的 3D 迷宫导航环境。每个任务都会随机化迷宫拓扑、纹理、导航目标、命令序列和物理尺度，因此很适合用于导航、探索、Meta-RL、上下文适应以及 agent-environment interaction 研究。

和很多主要依赖强 zero-shot 能力的 ObjectNav 基准不同，MazeWorld 更强调迭代交互、记忆能力和对当前环境的自适应。

<div style="width: 960; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-1.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-2.jpg" alt="Keyboard Demo">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Keyboard-Demo-3.jpg" alt="Keyboard Demo">
</div>

## 模块提供了什么

- 程序化生成的 3D 迷宫任务。
- 同时支持离散动作和连续动作的 Gymnasium 环境。
- 基于命令序列的导航任务。
- 任务重采样工具。
- 一个内置的 SLAM 风格基线 agent。
- 用于分析和可视化的局部地图与全局地图接口。
- 键盘演示与脚本化 demo。

## 安装

通过 PyPI 安装：

```bash
pip install "xenoverse[mazeworld]"
```

从源码安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[mazeworld]"
```

## 主要组件

主要公开入口如下：

- `xenoverse.mazeworld.MazeWorldContinuous3D`：主 3D 环境类。
- `xenoverse.mazeworld.MazeTaskSampler`：随机任务采样器。
- `xenoverse.mazeworld.Resampler`：任务重采样器。
- `xenoverse.mazeworld.agents.SmartSLAMAgent`：内置导航基线。

当前注册的环境 ID 是：

- `mazeworld-v2`

## 推荐使用流程

标准使用流程如下：

1. 创建环境。
2. 采样或加载一个任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`。
5. 通过 `step(...)` 运行环境。

重要说明：调用 `reset()` 之前必须先挂载任务。

## 快速开始

### 1. 创建环境

```python
import gymnasium as gym
import xenoverse.mazeworld
from xenoverse.mazeworld import MazeTaskSampler

env = gym.make(
    "mazeworld-v2",
    enable_render=False,
    action_space_type="Discrete16",
)
```

如果你想打开窗口渲染，把 `enable_render=True` 即可，但需要本地 GUI 环境。

### 2. 采样并设置任务

```python
task = MazeTaskSampler()
env.set_task(task)
obs, info = env.reset()
```

### 3. 用随机动作运行

```python
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## 一个任务里包含什么

`MazeTaskSampler(...)` 生成的是一个任务字典，里面会随机化以下内容：

- 迷宫墙体拓扑
- 起点位置
- landmark 位置
- 命令序列
- 单元格尺寸
- 墙高
- agent 高度
- 纹理
- 奖励设置
- 视场角

因此两个任务可能在几何结构、物理尺度、难度、视觉外观和目标顺序上都明显不同。

## 任务采样

任务采样器支持较多迷宫生成控制参数。

示例：

```python
from xenoverse.mazeworld import MazeTaskSampler

task = MazeTaskSampler(
    n_range=(9, 25),
    allow_loops=True,
    cell_size_range=(1.5, 4.5),
    wall_height_range=(2.0, 6.0),
    agent_height_range=(1.6, 2.0),
    landmarks_number_range=(5, 10),
    commands_sequence=200,
    wall_density_range=(0.2, 0.4),
)
```

常用参数：

- `n_range`：迷宫网格尺寸范围。内部会强制使用奇数尺寸。
- `allow_loops`：是否允许迷宫中存在环。
- `cell_size_range`：每个网格单元的物理尺寸。
- `wall_height_range`：墙体高度范围。
- `agent_height_range`：机器人或视角高度范围。
- `landmarks_number_range`：候选导航目标数量。
- `commands_sequence`：一次任务中的命令长度。
- `wall_density_range`：迷宫生成时使用的墙体密度。
- `step_reward`、`collision_reward`、`goal_reward`：奖励 shaping 相关参数。

比如固定生成 15x15、每个网格 2 米的迷宫：

```python
task = MazeTaskSampler(
    n_range=(15, 15),
    cell_size_range=(2.0, 2.0),
)
```

## 任务重采样

如果你想保留当前迷宫布局，但改变目标、命令序列或起始点，可以使用 `Resampler(...)`。

```python
from xenoverse.mazeworld import Resampler

new_task = Resampler(task)
```

常见用途：

- 保留迷宫几何结构，只重采样命令
- 保留几何结构，只重采样起点
- 可选地重采样 landmarks 或 landmark 颜色分配

这类操作适合在同一个场景中构造多个相关导航 episode。

## 观测与 Info

### 观测

MazeWorld 返回的是 agent 第一视角下的 RGB 图像观测。

默认情况下：

- observation space 是形状为 `(H, W, 3)` 的图像张量
- 数值类型是 `uint8`

具体分辨率由创建环境时传入的 `resolution` 参数决定。

### Info 字典

`reset()` 和 `step()` 会返回一些辅助信息，包括：

- `steps`：当前步数
- `command`：当前目标命令对应的 RGB 颜色

这个命令颜色指示当前应该到达哪个 landmark。

## 命令表示

MazeWorld 使用颜色编码的命令，而不是自然语言文本。当前目标是通过 landmark 颜色指定的。

如果你想把命令直接嵌入图像观测中，可以在创建环境时设置：

```python
env = gym.make(
    "mazeworld-v2",
    command_in_observation=True,
    enable_render=False,
)
```

你也可以直接从 `info["command"]` 读取当前命令颜色。

<div style="width: 480; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/CommandDemo.jpg" alt="command_in_observation">
</div>

## 动作空间

MazeWorld 通过 `action_space_type` 支持三种动作模式：

- `Discrete16`
- `Discrete32`
- `Continuous`

示例：

```python
env = gym.make(
    "mazeworld-v2",
    action_space_type="Discrete16",
    enable_render=False,
)
```

### 离散模式

- `Discrete16` 是默认选项。
- `Discrete32` 提供更大的离散动作集合。

这两种模式都支持内置 `SmartSLAMAgent`。

### 连续模式

在 `Continuous` 模式下，动作是一个二维向量：

- 第一维：转向控制
- 第二维：前进或后退速度

两个分量都会被裁剪到 `[-1, 1]`。

<div style="width: 240; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/Dynamics.jpg" alt="Robot Dynamics">
</div>

## 渲染

当 `enable_render=True` 时，调用 `env.render()` 会显示：

- 第一视角观测
- 全局地图
- 局部地图

示例：

```python
env = gym.make("mazeworld-v2", enable_render=True)
```

如果你想手动操控环境，可以使用后面提到的键盘 demo。

## 键盘 Demo

你可以通过下面的命令手动控制 MazeWorld：

```bash
python -m xenoverse.mazeworld.demo.keyboard_play_demo --help
```

常用参数包括：

- `--max_steps`
- `--visibility_3D`
- `--save_replay`
- `--verbose`

## 内置 Smart Agent

MazeWorld 提供了一个带有 SLAM 与规划风格的内置基线：

```python
from xenoverse.mazeworld.agents import SmartSLAMAgent

agent = SmartSLAMAgent(
    maze_env=env,
    memory_keep_ratio=0.25,
    render=False,
)

terminated = False
truncated = False
reward = 0.0

while not (terminated or truncated):
    action = agent.step(obs, reward)
    obs, reward, terminated, truncated, info = env.step(action)
```

说明：

- `memory_keep_ratio=1.0` 表示近乎完整的长期记忆保留。
- 更低的值可以模拟遗忘。
- 内置 agent 只支持离散动作空间，不支持 `Continuous`。
- 不建议同时启用 `agent.render=True` 和 `env.enable_render=True`，否则两个渲染窗口会竞争。

![Demonstration-Agent-Control](https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/AgentDemo.gif)

## Agent 与 Teacher 的使用场景

内置 smart agent 更适合被看作：

- 一个较强的启发式基线
- 一个轨迹生成器
- 一个 imitation learning 或数据集构造时的 teacher policy

它并不保证全局最优。

## 访问地图

你可以直接从环境获取局部地图和全局地图。

```python
local_map = env.get_local_map()
global_map = env.get_global_map()
```

重要细节：

- 这两个函数返回的都是一个二元组：`(pygame surface, numpy array)`
- 如果你只想拿 NumPy 数组，需要取第二个返回值

示例：

```python
local_surface, local_array = env.get_local_map()
global_surface, global_array = env.get_global_map()
```

局部地图是相对于 agent 当前视角对齐的，全局地图展示完整迷宫布局。

## 保存轨迹

在 episode 结束后，你可以保存 agent 轨迹可视化结果：

```python
env.save_trajectory("trajectory.png")
```

它会生成一张图，展示 agent 在迷宫地图上的运动路径。

<div style="width: 320; overflow: hidden;">
  <img src="https://github.com/FutureAGI/DataPack/blob/main/demo/mazeworld/TrajectoryDemo.png" alt="Robot Trajectory">
</div>

## 奖励

默认奖励结构通常包括：

- 每步奖励，通常为 `0`
- 到达当前目标时的正奖励
- 碰撞惩罚

你可以通过任务采样器参数自定义这些项，例如：

- `step_reward`
- `goal_reward`
- `collision_reward`

## 常见坑

- `env.reset()` 之前必须先调用 `env.set_task(task)`。
- `SmartSLAMAgent` 只支持 `Discrete16` 或 `Discrete32`。
- 不要同时打开 agent 侧渲染和环境侧渲染，除非你明确需要两个窗口。
- `get_local_map()` 和 `get_global_map()` 返回的是 `(surface, numpy_array)`，不是单独的数组。
- `command_in_observation=True` 会直接修改图像观测，在图像中叠加命令颜色条。

## 文件说明

- `envs/task_sampler.py`：迷宫任务生成与重采样。
- `envs/maze_env.py`：Gymnasium 包装层与对外环境 API。
- `envs/maze_continuous_3d.py`：3D 观测与运动核心。
- `agents/smart_slam_agent.py`：内置 SLAM 风格基线。
- `demo/keyboard_play_demo.py`：键盘手动演示。
- `demo/agent_play_demo.py`：smart agent 脚本演示。

## 参考文献

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}

@article{wang2025context,
  title={Context and Diversity Matter: The Emergence of In-Context Learning in World Models},
  author={Wang, Fan and Chen, Zhiyuan and Zhong, Yuxuan and Zheng, Sunjian and Shao, Pengtao and Yu, Bo and Liu, Shaoshan and Wang, Jianan and Ding, Ning and Cao, Yang and others},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  year={2026}
}
```
