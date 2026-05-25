# Xenoverse

English | [中文](#中文说明)

Xenoverse is a collection of procedurally generated environments for benchmarking general decision-making, in-context adaptation, meta-training, and open-world evaluation. Instead of training agents in a single fixed benchmark family, Xenoverse provides multiple randomized worlds with different dynamics, observation structures, and task-generation mechanisms.

This repository currently contains environment families for random MDPs/POMDPs, linear dynamical systems, HVAC control, pseudo-language generation, 3D maze navigation, randomized classic-control tasks, and an agent-oriented extension workspace under `xenoverse_agents`.

## Overview

### Why Xenoverse

- Diversity over memorization: general agents should be tested across many world-generation processes rather than a small fixed benchmark set.
- Open-ended evaluation: procedurally generated tasks reduce benchmark overfitting.
- Reusable infrastructure: environment families are packaged under a single Python namespace, `xenoverse`.
- Agentic extension path: the repository now also contains an agent-oriented branch workspace for environment-driven scientific agents.

### What Is Included

| Module | Domain | Core idea | Documentation status |
| --- | --- | --- | --- |
| `xenoverse.anymdp` | Random MDP / POMDP | Random transition, reward, and observation structures | Has module README |
| `xenoverse.linds` | Linear dynamical systems | Randomized LTI control tasks | Has module README |
| `xenoverse.anyhvac` | HVAC control | Random buildings, sensors, coolers, and equipment | Has module README |
| `xenoverse.metalang` | Synthetic language | Random pseudo-language generation for ICL/sequence learning | Has module README |
| `xenoverse.mazeworld` | 3D navigation | Procedurally generated mazes with navigation commands | Has module README |
| `xenoverse.metacontrol` | Randomized control | Random cartpole, acrobot, humanoid-style tasks | Has module README |
| `xenoverse_agents.sci_agents` | Scientific agents | Chemistry-world generation and environment APIs | Has module README |
| `xenoverse.utils` | Shared utilities | Internal helper functions | Internal module |

## Repository Structure

```text
Xenoverse/
  README.md
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

Notes:

- `xenoverse/` is the installable Python package.
- `xenoverse_agents/` is an agent-oriented extension workspace currently containing scientific-agent environment code and planning documents for future agentic projects.

## Installation

### Install from PyPI

```bash
pip install xenoverse
```

### Install from source

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

### Base dependencies

The repository currently declares the following core dependencies:

- `gymnasium>=1.0.0`
- `numpy>=1.24.4`
- `Pillow>=6.2.2`
- `six>=1.12.0`
- `pygame>=2.6.0`
- `numba>=0.58.1`
- `scipy`

## Quick Start

Most environments register themselves through module import. A typical usage pattern is:

```python
import gymnasium as gym
import xenoverse.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
observation, info = env.reset()
```

## Environment Families

### 1. AnyMDP

Path: [xenoverse/anymdp](C:/Users/fanan/codes/Xenoverse/xenoverse/anymdp)

- Focus: procedurally generated MDPs, POMDPs, and multi-token POMDPs.
- Main APIs: `AnyMDPEnv`, `AnyMDPTaskSampler`, `AnyPOMDPTaskSampler`, `MultiTokensAnyPOMDPTaskSampler`.
- Built-in solvers: `AnyMDPSolverOpt`, `AnyMDPSolverMBRL`, `AnyMDPSolverQ`.
- Gym ID: `anymdp-v0`.

### 2. LinDS

Path: [xenoverse/linds](C:/Users/fanan/codes/Xenoverse/xenoverse/linds)

- Focus: procedurally generated linear time-invariant dynamical systems.
- Main APIs: `LinearDSEnv`, `LinearDSSampler`, `LinearDSSamplerRandomDim`, `LTISystemMPC`.
- Gym IDs: `linear-dynamics-v0`, `linear-dynamics-v0-visualizer`.

### 3. AnyHVAC

Path: [xenoverse/anyhvac](C:/Users/fanan/codes/Xenoverse/xenoverse/anyhvac)

- Focus: randomized HVAC control in procedurally generated indoor settings.
- Main APIs: `HVACEnv`, `HVACEnvVisible`.
- Gym IDs: `anyhvac-v1`, `anyhvac-visualizer-v1`.

### 4. MetaLang

Path: [xenoverse/metalang](C:/Users/fanan/codes/Xenoverse/xenoverse/metalang)

- Focus: pseudo-language generation for long-context and in-context learning benchmarks.
- Main APIs: `MetaLangV1`, `MetaLangV2`, `MetaLMV3Env`, `TaskSamplerV1`, `TaskSamplerV2`, `TaskSamplerV3`.
- Gym ID currently registered in code: `meta-language-v3`.

### 5. MazeWorld

Path: [xenoverse/mazeworld](C:/Users/fanan/codes/Xenoverse/xenoverse/mazeworld)

- Focus: 3D navigation in procedurally generated mazes with discrete or continuous control.
- Main APIs: `MazeWorldContinuous3D`, `MazeTaskSampler`, `Resampler`.
- Built-in baseline agent: `SmartSLAMAgent`.
- Gym ID: `mazeworld-v2`.

### 6. MetaControl

Path: [xenoverse/metacontrol](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol)

- Focus: randomized classic-control and humanoid-style control tasks.
- Registered environments: `random-cartpole-v0`, `random-acrobot-v0`, `random-humanoid-v0`.
- Main code files indicate task samplers and randomized parameter generation for cartpole, acrobot, and humanoid systems.

### 7. Sci Agents

Path: [xenoverse_agents/sci_agents](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents)

- Focus: scientific-agent workflows over procedurally generated chemistry worlds.
- Main entrypoints: `demo.py`, `generate_worlds.py`, `environment/api.py`, `world_gen/sampler.py`.
- Purpose: generate valid chemistry worlds, simulate environment interactions, and support agentic experiment-style workflows.

## Module Documentation

The following module-level documents are available:

- [AnyMDP README](C:/Users/fanan/codes/Xenoverse/xenoverse/anymdp/README.md)
- [AnyHVAC README](C:/Users/fanan/codes/Xenoverse/xenoverse/anyhvac/README.md)
- [LinDS README](C:/Users/fanan/codes/Xenoverse/xenoverse/linds/README.md)
- [MetaLang README](C:/Users/fanan/codes/Xenoverse/xenoverse/metalang/README.md)
- [MazeWorld README](C:/Users/fanan/codes/Xenoverse/xenoverse/mazeworld/README.md)
- [MetaControl README](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol/README.md)
- [xenoverse_agents README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/README.md)
- [AI Town README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/ai_town/README.md)
- [Sci Agents README](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/README.md)

## Research Positioning

Xenoverse is well suited for:

- meta-reinforcement learning
- in-context reinforcement learning
- open-world evaluation
- domain-randomized control
- procedural task-distribution generalization
- agentic scientific workflows over generated environments

## References

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}

@inproceedings{wang2025towards,
  title={Towards Large-Scale In-Context Reinforcement Learning by Meta-Training in Randomized Worlds},
  author={Fan Wang and Pengtao Shao and Yiming Zhang and Bo Yu and Shaoshan Liu and Ning Ding and Yang Cao and Yu Kang and Haifeng Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=b6ASJBXtgP}
}

@article{fan2025putting,
  title={Putting the smarts into robot bodies},
  author={Fan, Wang and Liu, Shaoshan},
  journal={Communications of the ACM},
  volume={68},
  number={3},
  pages={6--8},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

---

## 中文说明

Xenoverse 是一个面向通用决策、上下文适应、元训练与开放世界评测的程序化环境集合。它不依赖单一固定 benchmark，而是通过多类随机世界生成机制，为智能体提供更强的分布多样性与泛化压力。

当前仓库已经包含随机 MDP/POMDP、线性动力系统、HVAC 控制、伪语言生成、3D 迷宫导航、随机化经典控制任务，以及位于 `xenoverse_agents` 下的 agent-oriented 科学环境扩展。

### 项目定位

- 强调环境多样性，而不是对单一任务集的记忆。
- 强调开放式评测，降低 benchmark 被过拟合的风险。
- 用统一的 `xenoverse` Python 命名空间组织多种环境。
- 为后续 agentic workflow 和科学推理环境预留扩展路径。

### 仓库包含的模块

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

### 目录结构

```text
Xenoverse/
  README.md
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

### 安装方式

从 PyPI 安装：

```bash
pip install xenoverse
```

从源码安装：

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

当前仓库声明的基础依赖包括：

- `gymnasium>=1.0.0`
- `numpy>=1.24.4`
- `Pillow>=6.2.2`
- `six>=1.12.0`
- `pygame>=2.6.0`
- `numba>=0.58.1`
- `scipy`

### 快速开始

大多数环境通过导入模块完成注册，典型用法如下：

```python
import gymnasium as gym
import xenoverse.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
observation, info = env.reset()
```

### 各模块说明

#### 1. AnyMDP

路径：[xenoverse/anymdp](C:/Users/fanan/codes/Xenoverse/xenoverse/anymdp)

- 面向随机生成的 MDP、POMDP 和多 token POMDP。
- 主要接口：`AnyMDPEnv`、`AnyMDPTaskSampler`、`AnyPOMDPTaskSampler`、`MultiTokensAnyPOMDPTaskSampler`。
- 内置求解器：`AnyMDPSolverOpt`、`AnyMDPSolverMBRL`、`AnyMDPSolverQ`。
- Gym ID：`anymdp-v0`。

#### 2. LinDS

路径：[xenoverse/linds](C:/Users/fanan/codes/Xenoverse/xenoverse/linds)

- 面向程序化生成的线性时不变动力系统。
- 主要接口：`LinearDSEnv`、`LinearDSSampler`、`LinearDSSamplerRandomDim`、`LTISystemMPC`。
- Gym ID：`linear-dynamics-v0`、`linear-dynamics-v0-visualizer`。

#### 3. AnyHVAC

路径：[xenoverse/anyhvac](C:/Users/fanan/codes/Xenoverse/xenoverse/anyhvac)

- 面向随机室内环境下的 HVAC 控制。
- 主要接口：`HVACEnv`、`HVACEnvVisible`。
- Gym ID：`anyhvac-v1`、`anyhvac-visualizer-v1`。

#### 4. MetaLang

路径：[xenoverse/metalang](C:/Users/fanan/codes/Xenoverse/xenoverse/metalang)

- 面向长上下文与上下文学习评测的伪语言生成。
- 主要接口：`MetaLangV1`、`MetaLangV2`、`MetaLMV3Env`、`TaskSamplerV1`、`TaskSamplerV2`、`TaskSamplerV3`。
- 当前代码中注册的 Gym ID：`meta-language-v3`。

#### 5. MazeWorld

路径：[xenoverse/mazeworld](C:/Users/fanan/codes/Xenoverse/xenoverse/mazeworld)

- 面向 3D 程序化迷宫导航。
- 主要接口：`MazeWorldContinuous3D`、`MazeTaskSampler`、`Resampler`。
- 内置基线 Agent：`SmartSLAMAgent`。
- Gym ID：`mazeworld-v2`。

#### 6. MetaControl

路径：[xenoverse/metacontrol](C:/Users/fanan/codes/Xenoverse/xenoverse/metacontrol)

- 面向随机化的经典控制与 humanoid 风格控制任务。
- 当前注册环境：`random-cartpole-v0`、`random-acrobot-v0`、`random-humanoid-v0`。
- 从代码结构看，包含 cartpole、acrobot、humanoid 的随机参数采样与环境实现。

#### 7. Sci Agents

路径：[xenoverse_agents/sci_agents](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents)

- 面向科学 Agent 工作流与程序化化学世界。
- 主要入口：`demo.py`、`generate_worlds.py`、`environment/api.py`、`world_gen/sampler.py`。
- 目标：生成有效化学世界、提供环境交互 API，并支撑后续 agentic 实验流程。

### 模块文档

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

### 适用研究方向

Xenoverse 适合以下方向：

- 元强化学习
- 上下文强化学习
- 开放世界评测
- 随机化控制
- 程序化任务分布下的泛化能力测试
- 基于生成环境的科学 Agent 工作流

### 参考文献

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}

@inproceedings{wang2025towards,
  title={Towards Large-Scale In-Context Reinforcement Learning by Meta-Training in Randomized Worlds},
  author={Fan Wang and Pengtao Shao and Yiming Zhang and Bo Yu and Shaoshan Liu and Ning Ding and Yang Cao and Yu Kang and Haifeng Wang},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025},
  url={https://openreview.net/forum?id=b6ASJBXtgP}
}

@article{fan2025putting,
  title={Putting the smarts into robot bodies},
  author={Fan, Wang and Liu, Shaoshan},
  journal={Communications of the ACM},
  volume={68},
  number={3},
  pages={6--8},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```
