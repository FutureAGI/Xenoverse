# Xenoverse

English | [Chinese](README.zh.md)

Xenoverse is a collection of procedurally generated environments for benchmarking general decision-making, in-context adaptation, meta-training, and open-world evaluation. Instead of relying on one fixed benchmark family, the repository groups multiple randomized world generators under a single Python package, `xenoverse`.

## Overview

### Why Xenoverse

- Diversity over memorization: agents are evaluated across many world-generation processes rather than one static task set.
- Open-ended evaluation: procedural tasks reduce benchmark overfitting.
- Reusable infrastructure: environment families share one namespace and similar task-sampling workflows.
- Agent-oriented extensions: the repository also contains scientific-agent and town-style environment workspaces.

### Main Modules

| Module | Domain | Purpose |
| --- | --- | --- |
| `xenoverse.anymdp` | Random MDP / POMDP | Random transition, reward, and observation structures |
| `xenoverse.linds` | Linear dynamical systems | Randomized LTI control tasks |
| `xenoverse.anyhvac` | HVAC control | Procedural HVAC environments |
| `xenoverse.metalang` | Synthetic language | Procedural pseudo-language generation |
| `xenoverse.mazeworld` | 3D navigation | Procedural maze navigation |
| `xenoverse.metacontrol` | Randomized control | Random cartpole, acrobot, and humanoid-style tasks |
| `xenoverse.sci_research_env` | Scientific agents | Chemistry-world generation, tool-driven interaction, and route scoring |
| `xenoverse.ai_town_env` | Town-style agents | Agent-oriented town environment design workspace |

## Repository Layout

```text
Xenoverse/
  README.md
  README.zh.md
  setup.py
  requirements.txt
  xenoverse/
```

- `xenoverse/` is the installable Python package.
- Package code and package-level documentation live under `xenoverse/`.
- The current package includes `anymdp`, `linds`, `anyhvac`, `metalang`, `mazeworld`, `metacontrol`, `sci_research_env`, `ai_town_env`, and `utils`.

## Installation

Install from PyPI:

```bash
pip install xenoverse
```

Install from source:

```bash
git clone https://github.com/FutureAGI/xenoverse
cd xenoverse
pip install .
```

Core dependencies include:

- `gymnasium>=1.0.0`
- `numpy>=1.24.4`
- `Pillow>=6.2.2`
- `six>=1.12.0`
- `pygame>=2.6.0`
- `numba>=0.58.1`
- `scipy`

## Quick Start

```python
import gymnasium as gym
import xenoverse.anymdp

env = gym.make("anymdp-v0", max_steps=5000)
observation, info = env.reset()
```

## Research Positioning

Xenoverse is well suited for:

- meta-reinforcement learning
- in-context reinforcement learning
- open-world evaluation
- domain-randomized control
- procedural task-distribution generalization
- agentic scientific workflows over generated environments

## Package Documentation

For the current package structure, environment families, module entrypoints, and package documentation index, see [xenoverse/README.md](xenoverse/README.md).
