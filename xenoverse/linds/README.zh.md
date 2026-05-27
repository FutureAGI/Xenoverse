# LinDS

[English](README.md) | 中文

LinDS 是一组程序化生成的线性动力系统环境。每个任务都会重新采样线性状态转移、观测模型、控制矩阵、偏置项、目标信号和奖励参数，因此很适合用于连续控制、轨迹跟踪、系统辨识和鲁棒性实验。

## 模块提供了什么

- 随机生成的线性动力系统任务。
- 固定维度和随机维度两类任务采样器。
- 一个用于执行采样任务的 Gymnasium 兼容环境。
- 可选的 padded observation/action 维度，方便训练需要固定输入输出大小的模型。
- 一个会记录轨迹的可视化环境。
- 位于 `solver.py` 中的 MPC 基线控制器。

## 底层模型

采样任务遵循线性状态空间结构。用连续时间形式表示，可理解为：

- `dx / dt = A x + B u + X`
- `o(t) = C x + Y`

实际运行时，环境会将系统离散化后再仿真。每个任务都会采样：

- `A`：状态转移矩阵
- `B`：控制矩阵
- `C`：观测矩阵
- `X`：状态偏置
- `Y`：观测偏置

奖励会鼓励观测值跟踪目标命令，同时惩罚过大的控制输入。

## 安装

通过 PyPI 安装：

```bash
pip install "xenoverse[linds]"
```

从源码安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[linds]"
```

## 主要组件

主要公开入口包括：

- `xenoverse.linds.LinearDSEnv`：基础环境。
- `xenoverse.linds.LinearDSSampler`：固定维度任务采样器。
- `xenoverse.linds.LinearDSSamplerRandomDim`：随机维度任务采样器。
- `xenoverse.linds.LinearDSVisualizer`：带轨迹记录和绘图能力的环境。
- `xenoverse.linds.LTISystemMPC`：MPC 基线控制器。
- `xenoverse.linds.dump_linds_task` / `load_linds_task`：任务序列化辅助函数。

当前注册的环境 ID 为：

- `linear-dynamics-v0`
- `linear-dynamics-v0-visualizer`

## 推荐使用流程

推荐工作流如下：

1. 创建环境。
2. 采样或加载一个任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`。
5. 用连续动作与环境交互。

重要说明：`reset()` 之前必须先用 `set_task(...)` 挂载任务。

## 快速开始

### 1. 采样固定维度任务

```python
import numpy as np
from xenoverse.linds import LinearDSEnv, LinearDSSampler

task = LinearDSSampler(
    state_dim=16,
    observation_dim=8,
    action_dim=8,
)

env = LinearDSEnv(
    pad_observation_dim=16,
    pad_action_dim=8,
    pad_command_dim=16,
)

env.set_task(task, use_pad_dim=True)
obs, info = env.reset()
```

### 2. 用随机动作运行

```python
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## 任务采样

### 固定维度

当你希望显式指定状态、观测和动作大小时，使用 `LinearDSSampler(...)`。

关键参数：

- `state_dim`：隐藏状态维度。
- `observation_dim`：观测维度。
- `action_dim`：动作维度。
- `seed`：可选随机种子。

### 随机维度

当你希望每次任务采样的维度都不一样时，使用 `LinearDSSamplerRandomDim(...)`。

示例：

```python
from xenoverse.linds import LinearDSSamplerRandomDim

task = LinearDSSamplerRandomDim(
    max_state_dim=16,
    max_observation_dim=16,
    max_action_dim=8,
)
```

这适合“环境外壳固定，但任务内在维度变化”的实验设置。

## Padding 机制

LinDS 一个很重要的设计点是支持 padded 维度。

这样设计的原因是：

- 很多序列模型或策略网络要求固定输入维度。
- 采样任务的真实维度可能每次不同。

当你在 `env.set_task(...)` 里设置 `use_pad_dim=True` 时：

- 观测会被零填充到 `pad_observation_dim`。
- 传给 `step(...)` 的动作必须满足 `pad_action_dim`。
- `info["command"]` 里的目标命令会被零填充到 `pad_command_dim`。

当 `use_pad_dim=False` 时：

- observation space 直接使用任务真实的 `observation_dim`。
- action space 直接使用任务真实的 `action_dim`。

例子：

- 真实观测维度是 `3`
- `pad_observation_dim=5`
- 返回观测会像 `[o1, o2, o3, 0, 0]`

重要说明：padding 只影响对外暴露的接口形状，不改变底层系统本身的真实维度。

## 观测、动作与 Info

### 观测

环境会根据当前隐藏状态和采样得到的观测模型返回一个连续观测向量。

取决于 `use_pad_dim`，这个向量要么是：

- 真实观测值
- 经过零填充后的观测值

### 动作

动作空间始终是连续的，并且每一维都被限制在 `[-1, 1]`。

如果开启 padding：

- 环境期望传入形状为 `(pad_action_dim,)` 的动作向量
- 只有前 `action_dim` 个分量会真正作用于系统

### Info 字典

`reset()` 和 `step()` 会返回一些有用的辅助信息，包括：

- `steps`：当前步数
- `command`：当前目标命令
- `command_type`：`reset()` 时返回的目标类型
- `error`：当前跟踪误差

## 目标类型与奖励

采样任务可能使用以下两种目标：

- `static_target`：固定命令向量
- `dynamic_target`：随时间变化的命令轨迹

奖励通常由以下部分组成：

- 基础奖励
- 与跟踪误差成比例的惩罚
- 控制输入幅值惩罚
- 当系统明显发散时的终止惩罚

直观上，奖励越高，表示系统越能在较低控制代价下跟踪目标。

## 终止机制

当系统足够不稳定时，episode 会提前终止，例如：

- 跟踪误差过大
- 观测幅值过大

达到 `max_steps` 时则会被截断。

## 使用可视化环境

可视化环境继承自 `LinearDSEnv`，会在运行过程中记录 observation、state、action 和 reward。

示例：

```python
import gymnasium as gym
from xenoverse.linds import LinearDSSamplerRandomDim

task = LinearDSSamplerRandomDim()
env = gym.make("linear-dynamics-v0-visualizer")

env.set_task(task)
obs, info = env.reset()
```

运行完成后，可以在可视化环境实例上调用 `visualize_and_save(...)`，将 reward 曲线或可选的 t-SNE 轨迹图保存下来。

## MPC 基线

`LTISystemMPC` 提供了一个基于模型预测控制的参考基线，它会直接使用环境离散化后的系统矩阵。

示例：

```python
from xenoverse.linds import LTISystemMPC

solver = LTISystemMPC(env, K=20, gamma=0.99)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    future_cmds = env.get_future_inner_cmds(K=solver.K)
    action = solver.solve(env.state, future_cmds)
    obs, reward, terminated, truncated, info = env.step(action)
```

说明：

- 这个基线依赖 `osqp`、`scipy` 等数值计算依赖。
- 它更适合作为参考控制器，而不是学习型基线。

## 保存和加载任务

模块提供了任务持久化辅助函数：

```python
from xenoverse.linds import dump_linds_task, load_linds_task
```

当你想在同一个采样系统上做可复现实验时，这会很有用。

## 常见坑

- `env.reset()` 之前必须先调用 `env.set_task(task)`。
- 如果 `use_pad_dim=True`，传给 `step(...)` 的动作形状必须匹配 `pad_action_dim`，而不只是 `action_dim`。
- 开启 padding 后，观测和命令向量尾部的 0 只是接口填充，不是任务本身的一部分。
- `pad_observation_dim`、`pad_action_dim` 和 `pad_command_dim` 必须足够大，才能容纳采样任务的真实维度。
- `LinearDSSamplerRandomDim(...)` 生成的真实维度可能远小于环境配置的 padded 维度。

## 文件说明

- `linds_env.py`：环境执行逻辑。
- `task_sampler.py`：任务采样器与任务序列化辅助函数。
- `solver.py`：MPC 基线与评估辅助代码。
- `visualizer.py`：轨迹记录与绘图。
- `test_ppo.py`：PPO 风格训练示例。
