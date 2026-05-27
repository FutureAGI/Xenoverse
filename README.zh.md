# Xenoverse

[English](README.md) | 中文

Xenoverse 是一个面向通用决策、上下文适应、元训练与开放世界评测的程序化环境集合。它不依赖单一固定 benchmark，而是通过多类随机世界生成机制，为智能体提供更强的分布多样性与泛化压力。

当前仓库已经包含随机 MDP/POMDP、线性动力系统、HVAC 控制、伪语言生成、3D 迷宫导航、随机化经典控制任务，以及位于 `xenoverse_agents` 下的 agent-oriented 科学环境扩展。

## 项目定位

- 强调环境多样性，而不是对单一任务集的记忆。
- 强调开放式评测，降低 benchmark 被过拟合的风险。
- 用统一的 `xenoverse` Python 命名空间组织多种环境。
- 为后续 agentic workflow 和科学推理环境预留扩展路径。

## 仓库包含的模块

| 模块 | 方向 | 核心内容 | 文档状态 |
| --- | --- | --- | --- |
| `xenoverse.anymdp` | 随机 MDP / POMDP | 随机转移、奖励、观测结构 | 有独立 README |
| `xenoverse.linds` | 线性动力系统 | 随机 LTI 控制任务 | 有独立 README |
| `xenoverse.anyhvac` | HVAC 控制 | 随机室内结构、传感器与设备布局 | 有独立 README |
| `xenoverse.metalang` | 合成语言 | 面向 ICL/序列学习的伪语言生成 | 有独立 README |
| `xenoverse.mazeworld` | 3D 导航 | 程序化迷宫与导航指令 | 有独立 README |
| `xenoverse.metacontrol` | 随机控制 | 随机 cartpole、acrobot、humanoid 任务 | 有独立 README |
| `xenoverse_agents.sci_agents` | 科学 Agent | 化学世界生成与环境交互 API | 有独立 README |
| `xenoverse.utils` | 通用工具 | 内部工具函数 | 内部模块 |

## 目录结构

```text
Xenoverse/
  README.md
  README.zh.md
  setup.py
  requirements.txt
  xenoverse/
    anyhvac/
    anymdp/
    linds/
    mazeworld/
    metacontrol/
    metalang/
    utils/
  xenoverse_agents/
    README.md
    ai_town/
      README.md
      AI_TOWN_DESIGN.md
    sci_agents/
      README.md
      demo.py
      generate_worlds.py
      environment/
      world_gen/
      worlds/
```

说明：

- `xenoverse/` 是当前可安装的 Python 包主体。
- `xenoverse_agents/` 是面向 agentic 扩展的工作区，目前包含科学 Agent 环境代码以及后续规划文档。

## 安装方式

### 从 PyPI 安装

```bash
pip install xenoverse
```

### 从源码安装

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

### 基础依赖

当前仓库声明的基础依赖包括：

- `gymnasium>=1.0.0`
- `numpy>=1.24.4`
- `Pillow>=6.2.2`
- `six>=1.12.0`
- `pygame>=2.6.0`
- `numba>=0.58.1`
- `scipy`

## 快速开始

大多数环境通过导入模块完成注册，典型用法如下：

```python
import gymnasium as gym
import xenoverse.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
observation, info = env.reset()
```

## 各模块说明

### 1. AnyMDP

路径：[xenoverse/anymdp](C:/Users/fanan/codes/Xenoverse/xenoverse/anymdp)

- 面向随机生成的 MDP、POMDP 和多 token POMDP。
- 主要接口：`AnyMDPEnv`、`AnyMDPTaskSampler`、`AnyPOMDPTaskSampler`、`MultiTokensAnyPOMDPTaskSampler`。
- 内置求解器：`AnyMDPSolverOpt`、`AnyMDPSolverMBRL`、`AnyMDPSolverQ`。
- Gym ID：`anymdp-v0`。

### 2. LinDS

路径：[xenoverse/linds](C:/Users/fanan/codes/Xenoverse/xenoverse/linds)

- 面向程序化生成的线性时不变动力系统。
- 主要接口：`LinearDSEnv`、`LinearDSSampler`、`LinearDSSamplerRandomDim`、`LTISystemMPC`。
- Gym ID：`linear-dynamics-v0`、`linear-dynamics-v0-visualizer`。

### 3. AnyHVAC

路径：[xenoverse/anyhvac](C:/Users/fanan/codes/Xenoverse/xenoverse/anyhvac)

- 面向随机室内环境下的 HVAC 控制。
- 主要接口：`HVACEnv`、`HVACEnvVisible`。
- Gym ID：`anyhvac-v1`、`anyhvac-visualizer-v1`。

### 4. MetaLang

路径：[xenoverse/metalang](C:/Users/fanan/codes/Xenoverse/xenoverse/metalang)

- 面向长上下文与上下文学习评测的伪语言生成。
- 主要接口：`MetaLangV1`、`MetaLangV2`、`MetaLMV3Env`、`TaskSamplerV1`、`TaskSamplerV2`、`TaskSamplerV3`。
- 当前代码中注册的 Gym ID：`meta-language-v3`。

### 5. MazeWorld

路径：[xenoverse/mazeworld](C:/Users/fanan/codes/Xenoverse/xenoverse/mazeworld)

- 面向 3D 程序化迷宫导航。
- 主要接口：`MazeWorldContinuous3D`、`MazeTaskSampler`、`Resampler`。
- 内置基线 Agent：`SmartSLAMAgent`。
- Gym ID：`mazeworld-v2`。

### 6. MetaControl

路径：[xenoverse/metacontrol](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol)

- 面向随机化的经典控制与 humanoid 风格控制任务。
- 当前注册环境：`random-cartpole-v0`、`random-acrobot-v0`、`random-humanoid-v0`。
- 从代码结构看，包含 cartpole、acrobot、humanoid 的随机参数采样与环境实现。

### 7. Sci Agents

路径：[xenoverse_agents/sci_agents](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents)

- 面向科学 Agent 工作流与程序化化学世界。
- 主要入口：`demo.py`、`generate_worlds.py`、`environment/api.py`、`world_gen/sampler.py`。
- 目标：生成有效化学世界、提供环境交互 API，并支撑后续 agentic 实验流程。

## 模块文档

当前可以直接阅读的说明文档：

- [AnyMDP README](C:/Users/fanan/codes/Xenoverse/xenoverse/anymdp/README.md)
- [AnyHVAC README](C:/Users/fanan/codes/Xenoverse/xenoverse/anyhvac/README.md)
- [LinDS README](C:/Users/fanan/codes/Xenoverse/xenoverse/linds/README.md)
- [MetaLang README](C:/Users/fanan/codes/Xenoverse/xenoverse/metalang/README.md)
- [MazeWorld README](C:/Users/fanan/codes/Xenoverse/xenoverse/mazeworld/README.md)
- [MetaControl README](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/README.md)
- [xenoverse_agents README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/README.md)
- [AI Town README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/ai_town/README.md)
- [Sci Agents README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/README.md)

## 适用研究方向

Xenoverse 适合以下方向：

- 元强化学习
- 上下文强化学习
- 开放世界评测
- 随机化控制
- 程序化任务分布下的泛化能力测试
- 基于生成环境的科学 Agent 工作流