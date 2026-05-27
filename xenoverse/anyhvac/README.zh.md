# AnyHVAC

[English](README.md) | 中文

AnyHVAC 是一组面向 HVAC 控制的程序化环境，主要模拟类似机房或室内空间中的温度调节问题。每次采样都会生成不同的房间布局、传感器位置、发热设备模式和制冷单元配置，因此很适合用于控制、强化学习和鲁棒性实验。

![AnyHVAC 可视化](https://github.com/FutureAGI/DataPack/blob/main/demo/anyhvac/hvac_video.gif)

## 模块提供了什么

- 随机生成的 HVAC 任务，包含不同的房间尺寸、墙体、传感器、冷却器和发热设备。
- 兼容 Gymnasium `reset()` / `step()` 接口的温控环境。
- 一个方便调试的可视化环境。
- 位于 `anyhvac_solver.py` 中的内置 PID 风格基线控制器。

## 安装

通过 PyPI 安装：

```bash
pip install "xenoverse[anyhvac]"
```

从源码安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[anyhvac]"
```

## 主要组件

常用入口如下：

- `xenoverse.anyhvac.anyhvac_env.HVACEnv`：基础 HVAC 环境。
- `xenoverse.anyhvac.anyhvac_env_vis.HVACEnvVisible`：带渲染的可视化环境。
- `xenoverse.anyhvac.anyhvac_sampler.HVACTaskSampler`：随机任务采样器。
- `xenoverse.anyhvac.anyhvac_solver.HVACSolverGTPID`：PID 风格基线控制器。

注意：`xenoverse.anyhvac` 包根路径会导出环境类，但任务采样器和求解器更稳妥的导入方式是直接从各自子模块导入。

## 推荐使用流程

AnyHVAC 最清晰的使用方式分四步：

1. 创建环境实例。
2. 采样或加载一个任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`，然后开始与环境交互。

重要说明：`reset()` 默认假设任务已经挂到环境上，因此第一次使用前应先执行 `set_task(...)`。

## 快速开始

### 1. 创建环境并采样任务

```python
from xenoverse.anyhvac.anyhvac_env import HVACEnv
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler

env = HVACEnv(
    max_steps=5040,
    iter_per_step=600,
    set_lower_bound=16,
    set_upper_bound=32,
    verbose=False,
)

task = HVACTaskSampler(
    control_type="Temperature",
    target_temperature=26.0,
)

env.set_task(task)
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

这是最稳妥的最小示例，因为动作格式会随着环境配置变化。

## 使用可视化环境

```python
from xenoverse.anyhvac.anyhvac_env_vis import HVACEnvVisible
from xenoverse.anyhvac.anyhvac_sampler import HVACTaskSampler

env = HVACEnvVisible(verbose=True)
task = HVACTaskSampler(control_type="Temperature")

env.set_task(task)
obs, info = env.reset()
```

可视化环境适合检查任务布局、温度扩散过程和冷却器行为。

## 任务内容说明

`HVACTaskSampler(...)` 返回的是一个任务字典，供 `env.set_task(task)` 使用。采样结果通常包含：

- 房间宽度和长度。
- 网格分辨率与层高。
- 环境温度。
- 传感器对象。
- 冷却器对象。
- 发热设备对象。
- 对流与热容参数。
- 目标温度。

由于任务是程序化生成的，不同样本中的传感器、冷却器和设备数量都会变化。

## 观测结构

调用 `set_task(...)` 之后，环境会动态构建字典形式的 observation space。具体键会受配置项影响，常见字段包括：

- `sensor_readings`：传感器温度读数。
- `heat_readings`：当前设备发热量。
- `action_temp`：上一时刻的冷却温度设定。
- `timestep`：当前时间步。

因此默认情况下，观测并不是一个单独的扁平向量。实际使用时建议直接查看 `env.observation_space`。

## 动作结构

动作空间同样是在 `set_task(...)` 之后才会确定，因为它依赖当前任务中冷却器的数量。

常见情况：

- 当 `no_switch_action=True`（默认）时，动作是 `Box`，每个冷却器对应一个归一化控制值。
- 当 `no_switch_action=False` 时，动作是 `Box`，同时包含每个冷却器的开关值和控制值。
- 当 `action_space_format="dict"` 时，动作会变成带有 `switch` 与 `value` 字段的字典。

在温控模式下，归一化动作值会被映射到 `[set_lower_bound, set_upper_bound]` 这个设定温度区间。

## 奖励与终止说明

环境奖励通常综合了以下因素：

- 是否接近目标温度。
- 能耗相关代价。
- 动作或开关切换惩罚。
- 严重过热时的失败惩罚。

具体权重取决于环境配置和 reward mode。当前代码里，失败温度阈值是根据采样得到的目标温度内部推导出来的，所以一些旧示例中的 `upper_limit`、`tolerance` 之类参数并不适用于当前实现。

## 内置 PID 基线

基线控制器位于 `xenoverse.anyhvac.anyhvac_solver`。

导入示例：

```python
from xenoverse.anyhvac.anyhvac_solver import HVACSolverGTPID
```

更适合把这些求解器当作参考实现，而不是无条件直接套用。原因是它们期望的动作格式不一定和所有环境配置完全一致，尤其当你修改以下选项时：

- `no_switch_action`
- `action_space_format`
- 具体使用的环境子类

如果你要使用内置求解器，先确认返回动作的形状与 `env.action_space` 一致。

## 常见坑

- 第一次 `reset()` 前先调用 `env.set_task(task)`。
- `HVACTaskSampler` 应从 `xenoverse.anyhvac.anyhvac_sampler` 导入，而不是默认假设包根路径已导出。
- 不要继续沿用旧文档中的 `anyhvac-v0`；当前仓库里注册的是 `anyhvac-v1`。
- 任务采样完成后再查看 `env.action_space`，因为动作维度取决于冷却器数量。
- 任务采样完成后再查看 `env.observation_space`，因为观测键取决于环境配置。

## 文件说明

- `anyhvac_env.py`：基础环境逻辑。
- `anyhvac_env_vis.py`：渲染与可视化环境。
- `anyhvac_sampler.py`：随机任务生成。
- `anyhvac_solver.py`：PID 风格基线控制器。
- `hvac_config.py`：采样范围与物理参数配置。
