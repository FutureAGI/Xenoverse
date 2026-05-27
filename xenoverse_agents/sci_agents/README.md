# Sci Agents

`xenoverse_agents/sci_agents` is a scientific-agent environment package centered on procedurally generated chemistry worlds. It combines world generation, world validation, and an interaction API that can be used by agents to explore synthesis pathways, purchase materials, and search for useful compounds.

## Scope

This directory currently contains:

- world generation utilities under `world_gen/`
- a chemistry environment API under `environment/`
- a world generation CLI in `generate_worlds.py`
- a demonstration script in `demo.py`
- a `worlds/` directory for generated outputs

## Core Components

### World Generation

Files:

- [world_gen/sampler.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/world_gen/sampler.py)
- [world_gen/models.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/world_gen/models.py)
- [world_gen/validator.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/world_gen/validator.py)

Responsibilities:

- sample chemistry worlds with multiple layers
- construct chemicals and reactions
- validate that generated worlds satisfy structural constraints

The exported interfaces from `world_gen/__init__.py` are:

- `Chemical`
- `Reaction`
- `World`
- `WorldSampler`
- `WorldValidator`

### Environment API

Files:

- [environment/api.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/environment/api.py)
- [environment/simulator.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/environment/simulator.py)
- [environment/cost_model.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/environment/cost_model.py)
- [environment/templates.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/environment/templates.py)

Responsibilities:

- load a generated world into an interactive environment
- expose purchasable chemicals and pathway search
- simulate chemistry-related decision workflows

The exported environment entrypoint is:

- `ChemistryEnvironment`

### Demo and Batch Generation

Files:

- [demo.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/demo.py)
- [generate_worlds.py](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/generate_worlds.py)

`demo.py` samples a valid world, prints summary statistics, instantiates `ChemistryEnvironment`, purchases layer-1 chemicals, and searches for cost-effective medicinal pathways.

`generate_worlds.py` is a CLI for generating multiple validated chemistry worlds and saving them as JSON files.

## Dependencies

The local module requirements file currently declares:

- `numpy>=1.24.0`
- `scipy>=1.10.0`

See [requirements.txt](C:/Users/fanan/codes/Xenoverse/xenoverse_agents/sci_agents/requirements.txt).

## Quick Start

### Run the demo

From the `xenoverse_agents/sci_agents` directory:

```bash
python demo.py
```

### Generate multiple worlds

```bash
python generate_worlds.py --n 10 --output-dir worlds/
```

Optional parameters include:

- `--seed`
- `--layer1-min`
- `--layer1-max`
- `--last-layer-min`
- `--last-layer-max`

## Intended Use

This package is useful for:

- scientific-agent planning benchmarks
- environment-grounded reasoning
- reaction-pathway search experiments
- agent workflows that need explicit world structure instead of pure text tasks

## Future Extensions

Potential next steps:

- add a package-level CLI
- add formal tests for world validity and environment transitions
- expose a stable agent-facing interface for automated research loops
- integrate with broader multi-agent or town-style simulations
