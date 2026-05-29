# Xenoverse 包内说明

[English](README.md) | Chinese

本文档描述 `xenoverse/` 包目录下的当前结构。

## 包结构

```text
xenoverse/
  __init__.py
  README.md
  README.zh.md
  ai_town_env/
  anyhvac/
  anymdp/
  linds/
  mazeworld/
  metacontrol/
  metalang/
  sci_research_env/
  utils/
```

## 环境家族

| 模块 | 路径 | 用途 | 主要入口 |
| --- | --- | --- | --- |
| `xenoverse.anymdp` | `xenoverse/anymdp/` | 随机 MDP、POMDP 和多 token POMDP 环境 | `task_sampler.py`, `anymdp_env.py` |
| `xenoverse.linds` | `xenoverse/linds/` | 随机化线性动力系统 | `task_sampler.py`, `linds_env.py` |
| `xenoverse.anyhvac` | `xenoverse/anyhvac/` | 程序化 HVAC 控制任务 | `anyhvac_sampler.py`, `anyhvac_env.py` |
| `xenoverse.metalang` | `xenoverse/metalang/` | 合成语言生成任务 | `task_sampler.py`, `metalangv1.py`, `metalangv2.py`, `metalangv3.py` |
| `xenoverse.mazeworld` | `xenoverse/mazeworld/` | 3D 程序化迷宫导航 | `envs/task_sampler.py`, `envs/maze_continuous_3d.py` |
| `xenoverse.metacontrol` | `xenoverse/metacontrol/` | 随机化 classic-control 和 humanoid 风格任务 | `random_cartpole.py`, `random_acrobot.py`, `random_humanoid.py` |
| `xenoverse.sci_research_env` | `xenoverse/sci_research_env/` | 面向科研 Agent 的化学环境，支持工具驱动交互与路线评分 | `task_sampler.py`, `environment/session.py`, `environment/backend.py`, `demo.py` |
| `xenoverse.ai_town_env` | `xenoverse/ai_town_env/` | town 风格多 Agent 环境设计工作区 | `README.md`, `AI_TOWN_DESIGN.md` |
| `xenoverse.utils` | `xenoverse/utils/` | 共享内部工具 | `__init__.py`, `tools.py`, `grid_ops.py` |

## 说明

- 各模块的使用说明与代码位于同级目录中。
- 当前科研 Agent 包路径是 `xenoverse/sci_research_env/`。
- 当前 town-style 工作区路径是 `xenoverse/ai_town_env/`。
- 旧的 `xenoverse_agents/...` 路径描述不再适用于当前仓库布局。

## 模块文档

- [AnyMDP](anymdp/README.md)
- [LinDS](linds/README.md)
- [AnyHVAC](anyhvac/README.md)
- [MetaLang](metalang/README.md)
- [MazeWorld](mazeworld/README.md)
- [MetaControl](metacontrol/README.md)
- [Sci Research Env](sci_research_env/README.md)
- [AI Town Env](ai_town_env/README.md)
