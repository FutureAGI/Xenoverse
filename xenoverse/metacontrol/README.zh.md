# MetaControl

[English](README.md) | 中文

`xenoverse.metacontrol` 提供了一组基于 Gymnasium classic-control 与 MuJoCo humanoid 的程序化随机控制环境。这个模块的目标是在物理参数和身体结构持续变化的前提下，评估策略的鲁棒性、适应能力与迁移能力。

## 模块提供了什么

- 随机化 CartPole 动力学。
- 随机化 Acrobot 动力学。
- 通过生成 MuJoCo XML 文件实现的随机化 Humanoid 身体结构。
- 每类环境对应的简单任务采样器。
- 通过 `set_task(...)` 将采样任务注入环境实例的机制。

## 环境家族

当前模块注册了三个 Gymnasium 环境：

- `random-cartpole-v0`
- `random-acrobot-v0`
- `random-humanoid-v0`

## 安装与依赖

基础安装：

```bash
pip install xenoverse
```

本地开发安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

重要依赖说明：

- CartPole 和 Acrobot 依赖 Gymnasium classic-control。
- Humanoid 依赖 Gymnasium 的 MuJoCo 支持以及本地可用的 MuJoCo 运行环境。
- 当前仓库并不会自动帮你完整安装好 MuJoCo，因此 humanoid 实验通常还需要额外的本地环境配置。

## 主要公开 API

模块对外导出：

- `sample_cartpole`
- `sample_acrobot`
- `sample_humanoid`
- `get_humanoid_tasks`
- `RandomCartPoleEnv`
- `RandomAcrobotEnv`
- `RandomHumanoidEnv`

## 推荐使用流程

推荐工作流如下：

1. 创建环境。
2. 采样一个任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`。
5. 用 `step(...)` 与环境交互。

对于 CartPole 和 Acrobot，任务是参数字典。  
对于 Humanoid，任务是一个生成出来的 MuJoCo XML 文件路径。

## 快速开始

### Random CartPole

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_cartpole

env = gym.make("random-cartpole-v0")
task = sample_cartpole()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Random Acrobot

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_acrobot

env = gym.make("random-acrobot-v0")
task = sample_acrobot()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

### Random Humanoid

```python
import gymnasium as gym
import xenoverse.metacontrol
from xenoverse.metacontrol import sample_humanoid

env = gym.make("random-humanoid-v0")
task = sample_humanoid()
env.set_task(task)

obs, info = env.reset()
terminated, truncated = False, False
while not (terminated or truncated):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
```

## 1. Random CartPole

源码：

- [random_cartpole.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_cartpole.py)

任务采样器：

- `sample_cartpole(...)`

采样参数包括：

- 重力
- 小车质量
- 杆质量
- 杆长度

环境类：

- `RandomCartPoleEnv`

主要特性：

- 继承自 Gymnasium `CartPoleEnv`
- 支持 `set_task(task_config)`
- 支持可配置的 `frameskip`
- 支持通过 `reset_bounds_scale` 控制随机初始状态范围

实际使用说明：

- `step(...)` 会在内部重复执行同一个动作 `frameskip` 次，并累计奖励。

## 2. Random Acrobot

源码：

- [random_acrobot.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_acrobot.py)

任务采样器：

- `sample_acrobot(...)`

采样参数包括：

- 连杆长度
- 连杆质量
- 质心位置
- 重力

环境类：

- `RandomAcrobotEnv`

主要特性：

- 继承自 Gymnasium `AcrobotEnv`
- 重写了动力学方程，使采样物理参数真正影响运动
- 支持 `set_task(task_config)`
- 支持可配置的 `frameskip`
- 支持通过 `reset_bounds_scale` 控制随机初始状态范围

实际使用说明：

- 由于终止条件和随机几何参数耦合，不同任务之间难度可能变化很大。

## 3. Random Humanoid

源码：

- [random_humanoid.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/random_humanoid.py)
- [humanoid_xml_sampler.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/humanoid_xml_sampler.py)

任务采样器：

- `sample_humanoid(root_path=None, noise_scale=1.0)`

它返回的内容是：

- 一个生成好的 MuJoCo XML 文件路径

随机化内容包括：

- 四肢尺寸与比例
- 躯干和骨盆几何形状
- 关节 armature、damping、stiffness 与活动范围
- actuator gear 值
- 与身体形态相关的接触属性

环境类：

- `RandomHumanoidEnv`

主要特性：

- 继承自 Gymnasium `HumanoidEnv`
- `set_task(task)` 会基于采样 XML 文件重新加载模型
- healthy height range 会根据采样到的 torso 几何自动调整

实际使用说明：

- humanoid 任务不是像 CartPole / Acrobot 那样的参数字典，而是一个 XML 文件路径。

## 复用预生成 Humanoid 任务

如果你已经有一个保存了多个 humanoid XML 文件的目录，可以用下面的方式列出它们：

```python
from xenoverse.metacontrol import get_humanoid_tasks

tasks = get_humanoid_tasks("path/to/xml_dir")
```

这适合：

- 构建可复现的评测集
- 固定 train/test 任务划分
- 在同一组 body morphology 上重复实验

## Reset 与任务注入机制

三个环境都支持 `set_task(...)`，但任务载荷的格式不同：

- CartPole：标量物理参数字典
- Acrobot：标量物理参数字典
- Humanoid：生成好的 XML 文件路径

推荐顺序：

1. `env = gym.make(...)`
2. `task = sample_*()`
3. `env.set_task(task)`
4. `env.reset()`

## Frameskip 与随机初始状态

CartPole 和 Acrobot 都支持两个比较重要的环境构造参数：

- `frameskip`
- `reset_bounds_scale`

`frameskip`：

- 表示同一个动作会被连续执行多个模拟步
- 会改变实际控制频率

`reset_bounds_scale`：

- 控制随机初始状态范围
- 值越大，通常任务越难

当前仓库里注册的默认值是：

- CartPole: `frameskip=1`
- Acrobot: `frameskip=1`

## 如何选择环境家族

适用建议：

- 当你需要一个轻量级、动力学随机化的 classic-control 基准时，用 CartPole。
- 当你需要欠驱动 swing-up 行为并同时随机化物理结构时，用 Acrobot。
- 当你需要高维连续控制和随机 body morphology 时，用 Humanoid。

## 常见坑

- `env.reset()` 之前必须先执行 `env.set_task(...)`。
- 不要把 `sample_humanoid()` 当成其他采样器一样理解；它返回的是 XML 文件路径，不是参数字典。
- Humanoid 需要本地 MuJoCo 环境正常可用。
- 不同任务家族的 observation / action space 差异很大，策略不能直接互换。
- 如果你改大 `frameskip` 或初始状态随机范围，历史 baseline 的可比性可能会被破坏。

## 测试

仓库中包含一个基础 humanoid smoke test：

- [tests/test.py](/C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/tests/test.py)

目前这个测试主要覆盖 humanoid rollout，而不是对三个环境家族做完整验证。
