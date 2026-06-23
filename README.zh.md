# Xenoverse

[English](README.md) | 中文

Xenoverse 是一个面向通用决策、上下文适应、元训练与开放世界评测的程序化环境集合。它不依赖单一固定 benchmark，而是将多种随机世界生成器统一组织在 `xenoverse` Python 包下。

## 项目概览

### 为什么使用 Xenoverse

- 强调环境分布多样性，而不是对单一任务集的记忆。
- 通过程序化任务生成降低 benchmark 过拟合风险。
- 使用统一命名空间和相近的任务采样模式组织多个环境家族。
- 同时包含面向化学 Agent 和 town-style agent 的环境扩展。

### 主要模块

| 模块 | 方向 | 用途 |
| --- | --- | --- |
| `xenoverse.anymdp` | 随机 MDP / POMDP | 随机状态转移、奖励与观测结构 |
| `xenoverse.linds` | 线性动力系统 | 随机化 LTI 控制任务 |
| `xenoverse.anyhvac` | HVAC 控制 | 程序化 HVAC 环境 |
| `xenoverse.metalang` | 合成语言 | 程序化伪语言生成 |
| `xenoverse.mazeworld` | 3D 导航 | 程序化迷宫导航 |
| `xenoverse.metacontrol` | 随机控制 | 随机化 cartpole、acrobot 和 humanoid 风格任务 |
| `xenoverse.chemverse` | 化学 Agent | 化学世界生成、工具驱动交互与路线评分 |
| `xenoverse.ai_town_env` | Town-style Agent | town 风格 agent 环境设计工作区 |

## 仓库结构

```text
Xenoverse/
  README.md
  README.zh.md
  setup.py
  requirements.txt
  xenoverse/
```

- `xenoverse/` 是当前可安装的 Python 包主体。
- 包代码与包级文档位于 `xenoverse/` 目录下。
- 当前包内包含 `anymdp`、`linds`、`anyhvac`、`metalang`、`mazeworld`、`metacontrol`、`chemverse`、`ai_town_env` 和 `utils`。

## 安装方式

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

核心依赖包括：

- `gymnasium>=1.0.0`
- `numpy>=1.24.4`
- `Pillow>=6.2.2`
- `six>=1.12.0`
- `pygame>=2.6.0`
- `numba>=0.58.1`
- `scipy`

## 快速开始

```python
import gymnasium as gym
import xenoverse.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
observation, info = env.reset()
```

## Chemverse

`xenoverse.chemverse` 是仓库中的化学 Agent 环境家族，提供程序化化学世界、工具式交互和基于 backend 的任务评测流程。

常用入口：

```bash
python -m xenoverse.chemverse.demo
python -m xenoverse.chemverse.generate_worlds --list
python -m xenoverse.chemverse.tests.test_backend
```

更完整的模块结构、backend 用法、评分机制和世界生成说明见 [xenoverse/chemverse/README.zh.md](xenoverse/chemverse/README.zh.md)。

## 适用研究方向

Xenoverse 适合以下方向：

- 元强化学习
- 上下文强化学习
- 开放世界评测
- 随机化控制
- 程序化任务分布下的泛化能力测试
- 基于生成环境的科研 Agent 工作流

## 包内文档

关于当前包结构、环境家族、模块入口和包级文档索引，请查看 [xenoverse/README.zh.md](xenoverse/README.zh.md)。
