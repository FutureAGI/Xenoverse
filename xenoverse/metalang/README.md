# MetaLang

English | [中文](README.zh.md)

`xenoverse.metalang` provides procedurally generated pseudo-language tasks for studying long-context learning, sequence modeling, and in-context adaptation. The module includes multiple task families with different generation mechanisms, ranging from repeated noisy patterns to randomized n-gram language models and query-answer style interaction tasks.

## What This Module Provides

- `MetaLangV1`: repeated noisy pattern sequences.
- `MetaLangV2`: randomized n-gram pseudo-language generation.
- `MetaLMV3Env`: an interactive query-answer environment with rewards.
- `TaskSamplerV1`, `TaskSamplerV2`, and `TaskSamplerV3` for sampling tasks.
- Command-line generators for creating tasks and synthetic sequence datasets.

## Important API Distinction

This module currently mixes two styles of interfaces:

- `MetaLangV1` and `MetaLangV2` are sequence generators with `set_task(...)`, `data_generator(...)`, and `batch_generator(...)`.
- `MetaLMV3Env` is the only Gymnasium environment currently registered in the package.

The currently registered Gym environment ID is:

- `meta-language-v3`

Older examples that use `gym.make("meta-language-v2")` are not accurate for the current code in this repository.

## Main Public APIs

The module exports:

- `MetaLangV1`
- `MetaLangV2`
- `MetaLMV3Env`
- `TaskSamplerV1`
- `TaskSamplerV2`
- `TaskSamplerV3`
- `metalang_generator`
- `metalang_generator_v3`

## Installation

Base installation:

```bash
pip install xenoverse
```

Local development install:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

## Task Families

### 1. MetaLangV1

`MetaLangV1` generates sequences by repeatedly emitting latent patterns and corrupting them with noise.

This family is useful when you want:

- repeated-structure recovery
- denoising-style sequence prediction
- long-range dependency benchmarks with recurring motifs

The corresponding sampler is:

- `TaskSamplerV1(...)`

Key task parameters include:

- `n_vocab`: vocabulary size
- `n_patterns`: number of reusable latent patterns
- `n_gram`: average pattern length control
- `error_ratio`: corruption level

### 2. MetaLangV2

`MetaLangV2` generates pseudo-language from a randomized n-gram neural generator.

This family is useful when you want:

- synthetic language-like token streams
- randomized local dependency structures
- task-level variation over vocabulary, embeddings, and transition statistics

The corresponding sampler is:

- `TaskSamplerV2(...)`

Key task parameters include:

- `n_vocab`: vocabulary size
- `n_emb`: embedding size
- `n_hidden`: hidden size
- `n_gram`: context length
- `_lambda`: softmax sharpness / sampling temperature control

### 3. MetaLMV3Env

`MetaLMV3Env` is an interactive environment rather than a passive generator. It creates query sequences, lets an agent answer with a token sequence, and assigns reward based on answer quality.

This family is useful when you want:

- interactive sequence decision-making
- query-answer prediction
- in-context reinforcement learning style setups

The corresponding sampler is:

- `TaskSamplerV3(...)`

Key task parameters include:

- `vocab_size`
- `embedding_size`
- `hidden_size`

## Recommended Workflows

There are two recommended ways to use this module:

### Workflow A: use V1 or V2 as synthetic sequence generators

1. Create a generator instance.
2. Sample a task with `TaskSamplerV1` or `TaskSamplerV2`.
3. Call `set_task(task)`.
4. Generate sequences with `data_generator(...)` or `batch_generator(...)`.

### Workflow B: use V3 as an interactive Gym-style environment

1. Create the environment.
2. Sample a task with `TaskSamplerV3`.
3. Call `env.set_task(task)`.
4. Call `env.reset()`.
5. Answer each query with a token sequence through `env.step(action)`.

## Quick Start: MetaLangV1

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

In V1:

- `features` are noisy sequences
- `labels` are next-token style targets derived from the clean pattern stream

## Quick Start: MetaLangV2

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

In V2:

- the output is a batch of generated token sequences
- generation depends on the sampled random n-gram model

## Quick Start: MetaLMV3Env

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

Important notes for V3:

- `reset()` returns a query token sequence.
- `step(action)` expects a token sequence answer.
- the environment currently never returns `terminated=True` in normal stepping.
- the `truncated` flag is driven by the internal step budget.

## Observation and Action Semantics in V3

After `set_task(...)`, `MetaLMV3Env` configures:

- observation space: `Sequence(Discrete(vocabulary))`
- action space: `Sequence(Discrete(vocabulary))`

In practice:

- observation = a query sequence
- action = an answer sequence
- reward = a quality signal derived from answer perplexity relative to better and worse reference answers
- `info["label"]` = target answer label sequence

## Command-line Dataset Generation

The module provides two dataset-generation scripts:

- `generator.py` for V1 and V2
- `generator_v3.py` for V3

## CLI for V1 / V2

Generate tasks:

```bash
python -m xenoverse.metalang.generator --sample_type tasks --version v2 --samples 100 --output tasks.pkl
```

Generate sequences from a task file:

```bash
python -m xenoverse.metalang.generator --sample_type sequences --version v2 --task_file tasks.pkl --samples 1000 --output sequences.txt --output_type txt
```

Generate sequences while sampling tasks on the fly:

```bash
python -m xenoverse.metalang.generator --sample_type sequences --version v2 --samples 1000 --output sequences.txt --output_type txt
```

Important CLI options for `generator.py` include:

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

## CLI for V3

Generate V3 tasks:

```bash
python -m xenoverse.metalang.generator_v3 --sample_type tasks --samples 10 --output tasks_v3.pkl
```

Generate V3 sequences:

```bash
python -m xenoverse.metalang.generator_v3 --sample_type sequences --task_file tasks_v3.pkl --samples 10 --output data_v3.npy --output_type npy
```

Important CLI options for `generator_v3.py` include:

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

## Choosing Between V1, V2, and V3

Use:

- V1 when you want repeated patterns with corruption and long-range recurrence.
- V2 when you want pseudo-language generation with randomized local statistical structure.
- V3 when you want an interactive query-answer environment with reward-driven evaluation.

## Common Pitfalls

- Do not rely on `gym.make("meta-language-v2")`; the current repository registers only `meta-language-v3`.
- V1 and V2 are not standard Gym environments in the current codebase; use them as generators.
- Always call `set_task(...)` before generating data or calling `reset()`.
- V3 `reset()` returns only the observation, not an `(observation, info)` tuple.
- In V3, `info["label"]` is useful for evaluation but should not be confused with the agent action.

## File Guide

- `metalangv1.py`: repeated-pattern generator.
- `metalangv2.py`: randomized n-gram pseudo-language generator.
- `metalangv3.py`: interactive query-answer environment.
- `task_sampler.py`: task samplers for all three families.
- `generator.py`: V1/V2 command-line generation.
- `generator_v3.py`: V3 command-line generation.

## References

```bibtex
@article{wang2024benchmarking,
  title={Benchmarking General Purpose In-Context Learning},
  author={Wang, Fan and Lin, Chuan and Cao, Yang and Kang, Yu},
  journal={arXiv preprint arXiv:2405.17234},
  year={2024}
}
```
