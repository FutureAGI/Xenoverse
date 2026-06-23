# Xenoverse Package Guide

English | [Chinese](README.zh.md)

This document describes the current package layout under `xenoverse/`.

## Package Structure

```text
xenoverse/
  __init__.py
  README.md
  README.zh.md
  ai_town_env/
  anyhvac/
  anymdp/
  chemverse/
  linds/
  mazeworld/
  metacontrol/
  metalang/
  utils/
```

## Environment Families

| Module | Path | Purpose | Main entrypoints |
| --- | --- | --- | --- |
| `xenoverse.anymdp` | `xenoverse/anymdp/` | Random MDP, POMDP, and multi-token POMDP environments | `task_sampler.py`, `anymdp_env.py` |
| `xenoverse.linds` | `xenoverse/linds/` | Randomized linear dynamical systems | `task_sampler.py`, `linds_env.py` |
| `xenoverse.anyhvac` | `xenoverse/anyhvac/` | Procedural HVAC control tasks | `anyhvac_sampler.py`, `anyhvac_env.py` |
| `xenoverse.metalang` | `xenoverse/metalang/` | Synthetic language generation tasks | `task_sampler.py`, `metalangv1.py`, `metalangv2.py`, `metalangv3.py` |
| `xenoverse.mazeworld` | `xenoverse/mazeworld/` | 3D procedural maze navigation | `envs/task_sampler.py`, `envs/maze_continuous_3d.py` |
| `xenoverse.metacontrol` | `xenoverse/metacontrol/` | Randomized classic-control and humanoid-style tasks | `random_cartpole.py`, `random_acrobot.py`, `random_humanoid.py` |
| `xenoverse.chemverse` | `xenoverse/chemverse/` | Chemistry-agent environment with tool-driven interaction and route scoring | `task_sampler.py`, `environment/session.py`, `environment/backend.py`, `demo.py` |
| `xenoverse.ai_town_env` | `xenoverse/ai_town_env/` | Town-style multi-agent environment design workspace | `README.md`, `AI_TOWN_DESIGN.md` |
| `xenoverse.utils` | `xenoverse/utils/` | Shared internal utilities | `__init__.py`, `tools.py`, `grid_ops.py` |

## Notes

- Module-specific usage guides live beside the code in each subdirectory.
- The chemistry-agent package path is `xenoverse/chemverse/`.
- The public package name changed from `sci_research_env` to `chemverse`.
- Internal class names such as `SciResearchBackend` and `SciResearchEnv` are still kept for compatibility.
- The town-style workspace path is `xenoverse/ai_town_env/`.
- Older `xenoverse_agents/...` references do not match the current repository layout.

## Module Documentation

- [AnyMDP](anymdp/README.md)
- [LinDS](linds/README.md)
- [AnyHVAC](anyhvac/README.md)
- [MetaLang](metalang/README.md)
- [MazeWorld](mazeworld/README.md)
- [MetaControl](metacontrol/README.md)
- [Chemverse](chemverse/README.md)
- [AI Town Env](ai_town_env/README.md)
