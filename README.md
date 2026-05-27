# Xenoverse

English | [中文](README.zh.md)

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

Path: [xenoverse/anymdp](xenoverse/anymdp)

- Focus: procedurally generated MDPs, POMDPs, and multi-token POMDPs.
- Main APIs: `AnyMDPEnv`, `AnyMDPTaskSampler`, `AnyPOMDPTaskSampler`, `MultiTokensAnyPOMDPTaskSampler`.
- Built-in solvers: `AnyMDPSolverOpt`, `AnyMDPSolverMBRL`, `AnyMDPSolverQ`.
- Gym ID: `anymdp-v0`.

### 2. LinDS

Path: [xenoverse/linds](xenoverse/linds)

- Focus: procedurally generated linear time-invariant dynamical systems.
- Main APIs: `LinearDSEnv`, `LinearDSSampler`, `LinearDSSamplerRandomDim`, `LTISystemMPC`.
- Gym IDs: `linear-dynamics-v0`, `linear-dynamics-v0-visualizer`.

### 3. AnyHVAC

Path: [xenoverse/anyhvac](xenoverse/anyhvac)

- Focus: randomized HVAC control in procedurally generated indoor settings.
- Main APIs: `HVACEnv`, `HVACEnvVisible`.
- Gym IDs: `anyhvac-v1`, `anyhvac-visualizer-v1`.

### 4. MetaLang

Path: [xenoverse/metalang](xenoverse/metalang)

- Focus: pseudo-language generation for long-context and in-context learning benchmarks.
- Main APIs: `MetaLangV1`, `MetaLangV2`, `MetaLMV3Env`, `TaskSamplerV1`, `TaskSamplerV2`, `TaskSamplerV3`.
- Gym ID currently registered in code: `meta-language-v3`.

### 5. MazeWorld

Path: [xenoverse/mazeworld](xenoverse/mazeworld)

- Focus: 3D navigation in procedurally generated mazes with discrete or continuous control.
- Main APIs: `MazeWorldContinuous3D`, `MazeTaskSampler`, `Resampler`.
- Built-in baseline agent: `SmartSLAMAgent`.
- Gym ID: `mazeworld-v2`.

### 6. MetaControl

Path: [xenoverse/metacontrol](xenoverse/metacontrol)

- Focus: randomized classic-control and humanoid-style control tasks.
- Registered environments: `random-cartpole-v0`, `random-acrobot-v0`, `random-humanoid-v0`.
- Main code files indicate task samplers and randomized parameter generation for cartpole, acrobot, and humanoid systems.

### 7. Sci Agents

Path: [xenoverse_agents/sci_agents](xenoverse_agents/sci_agents)

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
