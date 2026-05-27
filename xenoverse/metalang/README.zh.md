# MetaLang

[English](README.md) | 中文

`xenoverse.metalang` 提供了一组程序化生成的伪语言任务，用于研究长上下文学习、序列建模和上下文适应。这个模块包含多种不同的生成机制，从带噪声的重复模式，到随机 n-gram 伪语言模型，再到 query-answer 风格的交互式任务。

## 模块提供了什么

- `MetaLangV1`：基于重复噪声模式的序列生成。
- `MetaLangV2`：基于随机 n-gram 模型的伪语言生成。
- `MetaLMV3Env`：带奖励的交互式 query-answer 环境。
- `TaskSamplerV1`、`TaskSamplerV2`、`TaskSamplerV3` 三类任务采样器。
- 用于生成任务和合成序列数据的命令行脚本。

## 一个重要的接口区别

当前模块实际上同时包含两类接口：

- `MetaLangV1` 和 `MetaLangV2` 更像序列生成器，核心方法是 `set_task(...)`、`data_generator(...)` 和 `batch_generator(...)`。
- `MetaLMV3Env` 才是当前包里真正注册成 Gymnasium 环境的接口。

当前注册的 Gym 环境 ID 是：

- `meta-language-v3`

旧文档里类似 `gym.make("meta-language-v2")` 的写法不适用于当前仓库代码。

## 主要公开 API

模块对外导出：

- `MetaLangV1`
- `MetaLangV2`
- `MetaLMV3Env`
- `TaskSamplerV1`
- `TaskSamplerV2`
- `TaskSamplerV3`
- `metalang_generator`
- `metalang_generator_v3`

## 安装

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

## 任务家族

### 1. MetaLangV1

`MetaLangV1` 通过重复发射潜在模式并叠加噪声来生成序列。

适合的研究方向包括：

- 重复结构恢复
- 类 denoising 的序列预测
- 带有长程重复依赖的基准任务

对应采样器是：

- `TaskSamplerV1(...)`

关键参数包括：

- `n_vocab`：词表大小
- `n_patterns`：可重复潜在模式数量
- `n_gram`：平均模式长度控制
- `error_ratio`：噪声或破坏比例

### 2. MetaLangV2

`MetaLangV2` 基于随机化 n-gram 神经生成器构造伪语言序列。

适合的研究方向包括：

- 合成语言风格 token 序列
- 随机化局部依赖结构
- 在任务层面变化词表、embedding 和转移统计

对应采样器是：

- `TaskSamplerV2(...)`

关键参数包括：

- `n_vocab`：词表大小
- `n_emb`：embedding 维度
- `n_hidden`：隐藏层维度
- `n_gram`：上下文长度
- `_lambda`：softmax sharpness 或采样温度控制参数

### 3. MetaLMV3Env

`MetaLMV3Env` 是一个交互式环境，而不是被动序列生成器。它会生成 query 序列，让 agent 输出 answer 序列，并根据 answer 质量给出奖励。

适合的研究方向包括：

- 交互式序列决策
- query-answer 预测
- in-context reinforcement learning 风格任务

对应采样器是：

- `TaskSamplerV3(...)`

关键参数包括：

- `vocab_size`
- `embedding_size`
- `hidden_size`

## 推荐工作流

这个模块推荐两种使用方式：

### 工作流 A：把 V1 / V2 当作合成序列生成器使用

1. 创建生成器实例。
2. 用 `TaskSamplerV1` 或 `TaskSamplerV2` 采样任务。
3. 调用 `set_task(task)`。
4. 通过 `data_generator(...)` 或 `batch_generator(...)` 生成序列。

### 工作流 B：把 V3 当作交互式 Gym 风格环境使用

1. 创建环境。
2. 用 `TaskSamplerV3` 采样任务。
3. 调用 `env.set_task(task)`。
4. 调用 `env.reset()`。
5. 在 `env.step(action)` 中用 token 序列回答 query。

## 快速开始：MetaLangV1

```python
from xenoverse.metalang import MetaLangV1, TaskSamplerV1

generator = MetaLangV1(L=2048)
task = TaskSamplerV1(
    n_vocab=64,
    n_patterns=10,
    n_gram=64,
    error_ratio=0.1,
)

generator.set_task(task)
features, labels = generator.batch_generator(batch_size=8)
```

在 V1 中：

- `features` 是加噪后的输入序列
- `labels` 是从干净模式流派生出来的 next-token 风格目标

## 快速开始：MetaLangV2

```python
from xenoverse.metalang import MetaLangV2, TaskSamplerV2

generator = MetaLangV2(L=4096)
task = TaskSamplerV2(
    n_emb=16,
    n_hidden=64,
    n_vocab=256,
    n_gram=3,
    _lambda=5.0,
)

generator.set_task(task)
tokens = generator.batch_generator(batch_size=8)
```

在 V2 中：

- 输出是一个 batch 的 token 序列
- 生成结果由采样得到的随机 n-gram 模型决定

## 快速开始：MetaLMV3Env

```python
import gymnasium as gym
import xenoverse.metalang
from xenoverse.metalang import TaskSamplerV3

env = gym.make("meta-language-v3")
task = TaskSamplerV3(
    vocab_size=32,
    embedding_size=16,
    hidden_size=32,
)

env.set_task(task)
obs = env.reset()

action = env.policy(T=1.0)
next_obs, reward, terminated, truncated, info = env.step(action)
```

V3 的重要说明：

- `reset()` 返回的是 query token 序列。
- `step(action)` 期望输入一个 answer token 序列。
- 当前实现里正常运行时基本不会返回 `terminated=True`。
- `truncated` 由内部 step budget 控制。

## V3 中的观测与动作语义

在执行 `set_task(...)` 之后，`MetaLMV3Env` 会把空间定义成：

- observation space: `Sequence(Discrete(vocabulary))`
- action space: `Sequence(Discrete(vocabulary))`

实际语义是：

- observation = 一个 query 序列
- action = 一个 answer 序列
- reward = 基于 answer perplexity 相对好答案和差答案计算出的质量信号
- `info["label"]` = 目标答案标签序列

## 命令行数据生成

模块提供两个数据生成脚本：

- `generator.py`：用于 V1 和 V2
- `generator_v3.py`：用于 V3

## V1 / V2 的 CLI

生成任务：

```bash
python -m xenoverse.metalang.generator --sample_type tasks --version v2 --samples 100 --output tasks.pkl
```

从任务文件生成序列：

```bash
python -m xenoverse.metalang.generator --sample_type sequences --version v2 --task_file tasks.pkl --samples 1000 --output sequences.txt --output_type txt
```

边采样任务边生成序列：

```bash
python -m xenoverse.metalang.generator --sample_type sequences --version v2 --samples 1000 --output sequences.txt --output_type txt
```

`generator.py` 的重要参数包括：

- `--version {v1,v2}`
- `--sample_type {tasks,sequences}`
- `--task_file`
- `--vocab_size`
- `--embedding_size`
- `--hidden_size`
- `--patterns_number`
- `--error_rate`
- `--n_gram`
- `--lambda_weight`
- `--batch_size`
- `--sequence_length`
- `--samples`
- `--output_type {txt,npy}`
- `--output`

## V3 的 CLI

生成 V3 任务：

```bash
python -m xenoverse.metalang.generator_v3 --sample_type tasks --samples 10 --output tasks_v3.pkl
```

生成 V3 序列：

```bash
python -m xenoverse.metalang.generator_v3 --sample_type sequences --task_file tasks_v3.pkl --samples 10 --output data_v3.npy --output_type npy
```

`generator_v3.py` 的重要参数包括：

- `--sample_type {tasks,sequences}`
- `--datatype {QAR,QA,QARA}`
- `--task_file`
- `--vocab_size`
- `--embedding_size`
- `--hidden_size`
- `--sequence_length`
- `--samples`
- `--output_type {txt,npy}`
- `--output`

## 如何选择 V1、V2 和 V3

适用建议：

- 如果你想要带噪声的重复模式和长程 recurrence，用 V1。
- 如果你想要基于随机局部统计结构的伪语言生成，用 V2。
- 如果你想要带奖励的交互式 query-answer 环境，用 V3。

## 常见坑

- 不要继续使用 `gym.make("meta-language-v2")`；当前仓库只注册了 `meta-language-v3`。
- V1 和 V2 在当前代码里不是标准 Gym 环境，更适合作为生成器使用。
- 生成数据或调用 `reset()` 之前都要先执行 `set_task(...)`。
- V3 的 `reset()` 只返回 observation，不返回 `(observation, info)` 二元组。
- 在 V3 中，`info["label"]` 很适合做评估，但它不是 agent 的动作。

## 文件说明

- `metalangv1.py`：重复模式生成器。
- `metalangv2.py`：随机 n-gram 伪语言生成器。
- `metalangv3.py`：交互式 query-answer 环境。
- `task_sampler.py`：三类任务采样器。
- `generator.py`：V1/V2 命令行生成脚本。
- `generator_v3.py`：V3 命令行生成脚本。

## 参考文献

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}
```
