# AnyMDP

[English](README.md) | 中文

AnyMDP 是一组基于程序化随机生成的决策环境，核心覆盖 MDP 与 POMDP。它不是在单一固定转移图上评测智能体，而是为每个任务重新采样转移结构、奖励结构和观测结构，因此很适合用来研究泛化、上下文适应以及强化学习算法。

## 模块提供了什么

- 随机生成的 MDP 任务。
- 带观测发射模型的随机 POMDP 任务。
- 支持多 token 观测和多 token 动作的 POMDP 任务。
- 通过 `AnyMDPEnv` 提供的 Gymnasium 兼容环境接口。
- 面向完全可观测 MDP 的内置参考求解器。
- 一个用于检查任务结构的可视化工具。

## 安装

通过 PyPI 安装：

```bash
pip install "xenoverse[anymdp]"
```

从源码安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install ".[anymdp]"
```

## 预生成数据

原始 README 提到了两个可下载资源：

- 训练数据集：[Kaggle large training set](https://www.kaggle.com/datasets/anonymitynobody/omnirl-training-data-d-large)
- 评测任务和验证数据：[Kaggle evaluation set](https://www.kaggle.com/datasets/anonymitynobody/omnirl-evaluation)

这些资源是可选的。即使不下载外部数据，也可以直接使用当前仓库中的任务采样器。

## 主要组件

常用入口包括：

- `xenoverse.anymdp.AnyMDPEnv`：Gymnasium 环境类。
- `xenoverse.anymdp.AnyMDPTaskSampler`：标准随机 MDP 采样器。
- `xenoverse.anymdp.GarnetTaskSampler`：GARNET 风格随机 MDP 采样器。
- `xenoverse.anymdp.AnyPOMDPTaskSampler`：POMDP 采样器。
- `xenoverse.anymdp.MultiTokensAnyPOMDPTaskSampler`：多 token POMDP 采样器。
- `xenoverse.anymdp.AnyMDPSolverOpt`：使用真实转移和奖励矩阵的最优求解器。
- `xenoverse.anymdp.AnyMDPSolverMBRL`：面向 MDP 的 model-based RL 基线。
- `xenoverse.anymdp.AnyMDPSolverQ`：面向 MDP 的表格型 Q-learning 基线。
- `xenoverse.anymdp.anymdp_task_visualizer`：任务可视化工具。

当前包注册的环境 ID 是：

- `anymdp-v0`

## 推荐使用流程

推荐工作流如下：

1. 创建环境。
2. 采样一个任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`。
5. 通过 `step(...)` 与环境交互。

重要说明：`reset()` 依赖任务已经被设置完成。

## 快速开始

### 1. 标准 MDP

```python
import gymnasium as gym
import xenoverse.anymdp
from xenoverse.anymdp import AnyMDPTaskSampler

env = gym.make("anymdp-v0", max_steps=5000)

task = AnyMDPTaskSampler(
    state_space=128,
    action_space=5,
    min_state_space=None,
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

## 任务类型

AnyMDP 通过同一个 `env.set_task(task)` 入口支持三类任务。

### MDP

当你需要完全可观测的有限 MDP 时，使用 `AnyMDPTaskSampler(...)`。

常用参数：

- `state_space`：可观测状态空间上界。
- `action_space`：动作数。
- `min_state_space`：可选的活动状态数下界。
- `seed`：随机种子。
- `verbose`：是否输出采样诊断信息。

在 MDP 场景下：

- Observation space 为 `Discrete(ns)`。
- Action space 为 `Discrete(na)`。
- 返回观测就是当前状态 ID。

### POMDP

当你需要隐藏状态与随机观测发射模型时，使用 `AnyPOMDPTaskSampler(...)`。

示例：

```python
from xenoverse.anymdp import AnyPOMDPTaskSampler

task = AnyPOMDPTaskSampler(
    state_space=64,
    action_space=5,
    observation_space=32,
    density=0.1,
)

env.set_task(task)
obs, info = env.reset()
```

在 POMDP 场景下：

- Observation space 为 `Discrete(no)`。
- Action space 仍然是 `Discrete(na)`。
- 观测值来自观测模型采样，而不是直接暴露真实状态。

### 多 token POMDP

如果你需要多 token 观测和多 token 动作，使用 `MultiTokensAnyPOMDPTaskSampler(...)`。

示例：

```python
from xenoverse.anymdp import MultiTokensAnyPOMDPTaskSampler

task = MultiTokensAnyPOMDPTaskSampler(
    state_space=128,
    action_space=5,
    observation_space=32,
    observation_tokens=4,
    action_tokens=2,
    density=0.2,
)

env.set_task(task)
obs, info = env.reset()
```

在多 token 场景下：

- Observation space 为 `MultiDiscrete([no] * observation_tokens)`。
- Action space 为 `MultiDiscrete([na] * action_tokens)`。
- 每次 `step(...)` 会在一次环境步内应用一串 action token。

## 采样任务的结构

任务采样器返回的是一个字典，由 `env.set_task(task)` 消费。根据任务类型不同，常见字段包括：

- `ns`：可观测状态空间大小。
- `na`：动作空间大小。
- `max_steps`：该任务建议的 episode 长度。
- `state_mapping`：将活动状态子集嵌入到可观测状态空间中的映射。
- `transition`：活动状态上的转移张量。
- `reward`：奖励张量。
- `reward_noise`：奖励噪声张量。
- `s_0` 与 `s_0_prob`：起始状态及其采样概率。
- `s_e`：终止状态。
- `observation_transition`：POMDP 变体中的观测发射模型。
- `task_type`：`MDP`、`POMDP` 或 `MTPOMDP`。

一个重要实现细节是：环境允许把较小的活动状态集合嵌入到更大的可观测状态空间中，这就是 `state_mapping` 的作用。

## 观测与动作语义

在执行 `env.set_task(task)` 之后：

- MDP 任务直接返回类似状态的观测。
- POMDP 任务返回采样得到的观测。
- 多 token POMDP 任务返回 token 向量。

动作空间会随任务类型变化：

- MDP 和 POMDP 使用标量离散动作。
- 多 token POMDP 使用 `MultiDiscrete` 动作向量。

因此，面向普通 MDP 编写的示例代码不能不加修改地直接用于多 token 任务。

## 内置求解器

### 最优求解器

`AnyMDPSolverOpt` 会直接使用任务中的真实转移与奖励矩阵。

```python
from xenoverse.anymdp import AnyMDPSolverOpt

solver = AnyMDPSolverOpt(env)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    action = solver.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

它更适合作为 oracle 风格的参考基线。

### 可学习的 MDP 基线

`AnyMDPSolverMBRL` 和 `AnyMDPSolverQ` 是面向标准 MDP 任务设计的，不适用于一般 POMDP 或多 token POMDP。

示例：

```python
from xenoverse.anymdp import AnyMDPSolverMBRL

solver = AnyMDPSolverMBRL(env)
obs, info = env.reset()

terminated = False
truncated = False
while not (terminated or truncated):
    action = solver.policy(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)
    solver.learner(obs, action, next_obs, reward, terminated, truncated)
    obs = next_obs
```

说明：

- `AnyMDPSolverQ` 会显式断言任务类型必须是 `MDP`。
- 上述求解器示例默认观测本身足以用于控制，这只在标准 MDP 任务下成立。
- 对于 POMDP 变体，建议使用你自己的带记忆策略、循环网络策略，或参考 `test_ppo.py`。

## 可视化

模块对外暴露了 `visualizer.py` 中的 `anymdp_task_visualizer`。

它适合用来检查采样任务的：

- 转移结构
- Markov 链连通性
- 与 value function 相关的结构特征

如果你需要先判断一个采样任务是否合理，再进行训练，这通常是最直接的检查工具。

## 常见坑

- 在 `env.reset()` 之前必须先调用 `env.set_task(task)`。
- 不要假设活动状态数一定等于 `state_space`；采样器可能通过 `state_mapping` 只使用其中一部分状态。
- `AnyMDPSolverQ` 和 `AnyMDPSolverMBRL` 只适合标准 MDP。
- 多 token 任务使用向量动作，标量动作代码需要调整。
- 旧版 README 提到过 `Resampler`，但它不是当前仓库里稳定导出的 API。

## 文件说明

- `anymdp_env.py`：环境执行逻辑。
- `task_sampler.py`：MDP 与 POMDP 任务采样器。
- `anymdp_solver_opt.py`：访问真实模型的 oracle 求解器。
- `anymdp_solver_mbrl.py`：model-based RL 基线。
- `anymdp_solver_q.py`：表格型 Q-learning 基线。
- `visualizer.py`：任务可视化工具。
- `test_ppo.py`：适用于部分可观测场景的 PPO 示例。

## 参考文献

```bibtex
@inproceedings{wang2025towards,
  title={Towards Large-Scale In-Context Reinforcement Learning by Meta-Training in Randomized Worlds},
  author={Fan Wang and Pengtao Shao and Yiming Zhang and Bo Yu and Shaoshan Liu and Ning Ding and Yang Cao and Yu Kang and Haifeng Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=b6ASJBXtgP}
}
```
