# Sci Research Environment

[English](README.md) | 中文

`xenoverse.sci_research_env` 是一个面向化学研究任务的科学智能体环境，核心建立在程序化生成的世界之上。智能体需要在未知的化学空间中探索，通过购买原料、执行实验、发现合成路线，最终找到具有药用价值的化合物。

## 结构

- `world_gen/`：世界模型、采样器与校验器
- `environment/`：环境 API、仿真逻辑、成本模型、会话管理
- `generate_worlds.py`：用于批量生成和管理世界的命令行工具
- `demo.py`：交互式演示程序（REPL）
- `task_sampler.py`：带校验逻辑的任务采样
- `worlds/`：生成世界 JSON 文件的默认存储目录

## 快速开始

### 交互式演示

```bash
python -m xenoverse.sci_research_env.demo
```

该演示会先采样一个世界，打印任务描述和可用工具的 JSON 信息，然后进入交互式 REPL。你可以输入函数调用并查看返回结果。

### 生成世界

```bash
# 生成 10 个世界（保存到包内的 worlds/ 目录）
python -m xenoverse.sci_research_env.generate_worlds --n 10

# 列出所有已保存的世界
python -m xenoverse.sci_research_env.generate_worlds --list

# 指定自定义输出目录
python -m xenoverse.sci_research_env.generate_worlds --n 5 --output-dir /tmp/my_worlds
```

可用参数：`--seed`、`--complexity`、`--layer1-min`、`--layer1-max`、`--last-layer-min`、`--last-layer-max`。

#### 复杂度等级

使用 `--complexity` 选择一个预设难度，用来控制世界规模和任务复杂度：

| 等级 | 层数 | 第一层化学品数 | 化学品总数约值 | 反应总数约值 |
|------|------|----------------|----------------|--------------|
| `easy` | 3 | 4-8 | 10-15 | 15-25 |
| `medium` | 3-4 | 6-10 | 18-30 | 30-60 |
| `hard` | 4-5 | 8-14 | 35-80+ | 55-200+ |

```bash
python -m xenoverse.sci_research_env.generate_worlds --n 5 --complexity hard
```

如果同时提供了显式层参数（例如 `--layer1-min 12`），它们会覆盖复杂度预设值。如果既未提供 `--complexity`，也未提供显式层参数，则默认行为与旧版设置保持一致：层数取自 `[3,4,5]`，第一层大小为 `6-10`，最后一层大小为 `2-4`。

### 通过代码管理世界

```python
from xenoverse.sci_research_env.generate_worlds import list_worlds, DEFAULT_WORLDS_DIR
from xenoverse.sci_research_env.world_gen import World

# 列出所有已保存的世界
worlds = list_worlds()
for w in worlds:
    print(w["world_id"], w["num_chemicals"], w["num_reactions"])

# 加载指定世界
world = World.load(worlds[0]["file_path"])
```

## 基于 Backend 的使用方式

```python
from xenoverse.sci_research_env.environment.backend import SciResearchBackend

backend = SciResearchBackend()
session = backend.handle_request({
    "action": "sample_environment",
    "sampler_kwargs": {"seed": 7, "complexity_level": "hard"},
})
session_id = session["session_id"]

# 智能体分发函数调用
result = backend.handle_request({
    "action": "dispatch_function_call",
    "session_id": session_id,
    "function_call": {"name": "list_purchasable", "arguments": {}},
})
```

对于外部智能体，`backend` 是推荐的集成入口。它负责采样环境、管理会话，并接受 `{name, arguments}` 形式的函数调用载荷。

## 面向智能体的工具

智能体通过以下函数工具与环境交互：

| 工具 | 作用 |
|------|------|
| `task_description` | 获取任务目标和成功标准 |
| `restate_task_goal` | 重述当前任务目标 |
| `recap_recent_activity` | 汇总最近的实验行为 |
| `list_function_tools` | 列出所有可用工具 |
| `list_purchasable` | 列出可购买的基础化学品 |
| `purchase` | 购买基础化学品 |
| `get_inventory` | 查看当前库存 |
| `analyze_compound` | 分析库存中的化合物 |
| `list_possible_reactions` | 列出当前库存下可执行的反应 |
| `perform_reaction` | 在指定条件下执行反应 |
| `estimate_cost` | 估算某个候选反应配置的成本 |
| `submit_solution` | 提交合成方案并评分 |
| `get_transaction_log` | 查看完整会话活动日志 |

每个工具都包含 `brief` 简述、带参数说明的详细 `description`，以及两个完整 JSON 调用示例 `examples`。

## 提交与评分

智能体通过 `submit_solution` 提交解法，需要提供一个完整且具体的合成方案：

```json
{
  "name": "submit_solution",
  "arguments": {
    "target_compound": "CompoundX",
    "steps": [
      {
        "reactant_amounts": {"A": 10.0, "B": 10.0},
        "temperature_C": 85.0,
        "pressure_atm": 1.0,
        "duration_seconds": 1800.0,
        "catalyst_names": ["C"]
      }
    ]
  }
}
```

规则如下：

- 智能体可以多次提交，最终成绩取所有提交中的最高分。
- 分数由药用价值、毒性、成本、产率和效率共同决定。
- 反应条件（温度、压力、时长）会直接影响成本和产率。
- 成本模型依赖具体世界，不同世界会随机采样不同的系数。智能体可以通过 `estimate_cost` 探测成本曲线。

面向智能体的返回结果经过脱敏处理，不会泄露真实世界内部信息：

```json
{
  "success": true,
  "aggregate_score": 72.5,
  "verdict": "strong",
  "reasoning": ["target has strong medicinal potential", "cost efficiency is competitive"],
  "pathway_metrics": {"num_steps": 2, "target_yield_g": 3.2, "total_cost": 45.0, "efficiency_rating": "moderate"},
  "is_new_best": true,
  "best_score": 72.5
}
```

## 评测 API（仅 Backend 可用）

为了支持评测脚本和基准测试，backend 暴露了一组“上帝视角”方法，这些方法对智能体不可见：

```python
# 查找合成路线（完整反应图搜索）
backend.eval_find_synthesis_routes(session_id, target_compound="X")

# 查找全局最优的药用路径
backend.eval_find_cheapest_medicinal_pathway(session_id, min_medicinal_value=3.0)

# 使用完整真实信息评分
backend.eval_score_synthesis_route(session_id, target_compound="X", steps=[...])
backend.eval_score_synthesis_plan(session_id, target_compound="X", steps=[...])

# 获取智能体最佳提交（完整未脱敏评分卡）
backend.eval_get_best_submission(session_id)

# 导出完整世界数据
backend.eval_export_world(session_id)
```

这些方法会返回完整的内部细节，包括精确的药用价值、毒性、生成出的最优条件以及路径评估结果，适合离线分析使用。

## 成本模型

成本模型会考虑以下因素：

- **温度**：偏离室温（25°C）的升温或降温都会产生能耗成本，且制冷与加热是分开建模的。
- **压力**：偏离 1 atm 的高压或低压都会引入能耗与设备成本。
- **时长**：反应持续时间越长，设备占用成本越高。
- **原料**：原材料成本取决于购买价格。
- **毒性**：危险材料会增加设备与处理成本。
- **纯化**：与产物数量以及相态复杂度相关。

所有成本系数都会在每个世界中随机采样，因此每个环境的成本结构都不相同。智能体应使用 `estimate_cost` 来探测并学习该环境的成本特征。

## 依赖

`requirements.txt` 中列出的依赖为：`numpy`、`scipy`。
