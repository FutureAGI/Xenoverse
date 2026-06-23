# Chemverse

[English](README.md) | 中文

`xenoverse.chemverse` 是一个面向化学 Agent 的环境模块，核心建立在程序化生成的世界之上。智能体需要在未知的化学空间中探索，通过购买原料、执行实验、发现合成路线，最终找到具有药用价值的化合物。

## 命名与兼容性

- 包目录路径：`xenoverse/chemverse/`
- Python 导入路径：`xenoverse.chemverse`
- 旧包名：`xenoverse.sci_research_env`
- 当前仍保留的内部兼容类名：`SciResearchBackend`、`SciResearchEnv`、`SciResearchTaskSampler`

这次改名的目标是把对外模块命名统一到化学领域语义上，同时暂时保留现有运行时类名，避免影响已有调用方。

## 目录结构

- `world_gen/`：世界模型、采样器与校验器
- `environment/`：环境 API、仿真逻辑、成本模型与会话管理
- `generate_worlds.py`：批量生成和管理世界的 CLI
- `demo.py`：交互式演示 REPL
- `task_sampler.py`：带校验逻辑的任务采样器
- `tests/`：面向 backend 的测试
- `worlds/`：生成世界 JSON 文件的默认存储目录

## 快速开始

### 交互式 Demo

```bash
python -m xenoverse.chemverse.demo
```

该 demo 会先采样一个世界，打印任务描述和可用工具的 JSON 信息，然后进入交互式 REPL，供你输入函数调用并查看结果。

### 生成世界

```bash
# 在包内 worlds/ 目录下生成 10 个世界
python -m xenoverse.chemverse.generate_worlds --n 10

# 列出已保存的世界
python -m xenoverse.chemverse.generate_worlds --list

# 输出到自定义目录
python -m xenoverse.chemverse.generate_worlds --n 5 --output-dir /tmp/my_worlds
```

常用参数包括 `--seed`、`--complexity`、`--layer1-min`、`--layer1-max`、`--last-layer-min` 和 `--last-layer-max`。

### 复杂度等级

可以使用 `--complexity` 选择控制世界规模和难度的预设：

| 等级 | 层数 | 第一层化学品数 | 化学品总数约值 | 反应总数约值 |
| --- | --- | --- | --- | --- |
| `easy` | 3 | 4-8 | 10-15 | 15-25 |
| `medium` | 3-4 | 6-10 | 18-30 | 30-60 |
| `hard` | 4-5 | 8-14 | 35-80+ | 55-200+ |

```bash
python -m xenoverse.chemverse.generate_worlds --n 5 --complexity hard
```

如果同时提供了显式层参数，例如 `--layer1-min 12`，则这些参数会覆盖预设值。如果没有提供 `--complexity` 和显式层参数，则生成器回退到旧版默认范围。

## 通过代码管理世界

```python
from xenoverse.chemverse.generate_worlds import DEFAULT_WORLDS_DIR, list_worlds
from xenoverse.chemverse.world_gen import World

worlds = list_worlds()
for world_info in worlds:
    print(world_info["world_id"], world_info["num_chemicals"], world_info["num_reactions"])

world = World.load(worlds[0]["file_path"])
```

## 基于 Backend 的使用方式

```python
from xenoverse.chemverse.environment.backend import SciResearchBackend

backend = SciResearchBackend()
session = backend.handle_request(
    {
        "action": "sample_environment",
        "sampler_kwargs": {"seed": 7, "complexity_level": "hard"},
    }
)
session_id = session["session_id"]

result = backend.handle_request(
    {
        "action": "dispatch_function_call",
        "session_id": session_id,
        "function_call": {"name": "list_purchasable", "arguments": {}},
    }
)
```

对于外部 Agent，`backend` 是推荐的集成入口。它负责采样环境、管理会话，并接受 `{name, arguments}` 形式的函数调用载荷。

## 面向 Agent 的工具

Agent 通过以下函数工具与环境交互：

| 工具 | 作用 |
| --- | --- |
| `task_description` | 获取任务目标和成功标准 |
| `restate_task_goal` | 重复当前任务目标 |
| `recap_recent_activity` | 汇总最近实验活动 |
| `list_function_tools` | 列出全部可用工具 |
| `list_purchasable` | 列出可购买的基础化学品 |
| `purchase` | 购买基础化学品 |
| `get_inventory` | 查看当前库存 |
| `analyze_compound` | 分析库存中的化合物 |
| `list_equipment` | 查看可用实验设备 |
| `perform_reaction` | 按指定条件执行反应 |
| `estimate_cost` | 估算候选反应配置的成本 |
| `submit_solution` | 提交目标化合物进行评分 |
| `finish_experiment` | 提前结束当前会话 |

每个工具都包含 `brief`、详细 `description`、参数 schema 和示例调用。

## 提交与评分

Agent 通过 `submit_solution` 提交待评分的目标化合物：

```json
{
  "name": "submit_solution",
  "arguments": {
    "target_compound": "CompoundX"
  }
}
```

运行规则：

- 可以多次提交，最终成绩取最高的有效分数。
- 评分依赖药用价值、毒性、成本、产率与效率。
- 温度、压力、反应时长等条件会同时影响成本和产率。
- 成本模型依赖具体世界，每个世界都会随机采样不同系数。
- `estimate_cost` 是理解局部成本曲线的主要低风险工具。

面向 Agent 的返回结果经过脱敏处理，不会泄露隐藏世界状态。

## 评测 API

对于评测脚本和基准测试，backend 还暴露了一组不对行动 Agent 开放的方法：

```python
backend.eval_find_synthesis_routes(session_id, target_compound="X")
backend.eval_find_cheapest_medicinal_pathway(session_id, min_medicinal_value=3.0)
backend.eval_score_synthesis_route(session_id, target_compound="X", steps=[...])
backend.eval_score_synthesis_plan(session_id, target_compound="X", steps=[...])
backend.eval_get_best_submission(session_id)
backend.eval_export_world(session_id)
```

这些方法会返回内部细节，例如精确药用值、毒性、生成出的最优条件以及详细路径评分结果，适合离线分析。

## 成本模型

成本模型会考虑以下因素：

- 温度：偏离室温会产生能耗成本
- 压力：偏离 1 atm 会产生能耗和设备成本
- 时长：反应时间越长，设备占用成本越高
- 原料：原料成本取决于采购价格
- 毒性：危险材料会增加设备与处理成本
- 纯化：回收产物越多、相态越复杂，纯化成本越高

所有成本系数都会按世界随机采样，因此每个生成环境都有不同的操作特征。

## 测试

backend 相关测试可以直接运行：

```bash
python -m xenoverse.chemverse.tests.test_backend
```

## 依赖

该子模块下的 `requirements.txt` 当前列出了 `numpy` 和 `scipy`。
