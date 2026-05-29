# Sci Research

`xenoverse.sci_research` is a chemistry-oriented scientific-agent environment built around procedurally generated worlds. It now follows the same high-level repository pattern as other Xenoverse modules: sample a task first, then load that task into an environment session.

The public interface is agent-facing by design: it publishes the task description and tool schemas, but it does not expose the full compound space, reaction graph, or other internal world details directly.

## Structure

- `world_gen/`: world models, samplers, and validators
- `environment/`: environment API, simulation logic, cost model, and response templates
- `generate_worlds.py`: batch world-generation CLI
- `demo.py`: end-to-end demo script
- `worlds/`: generated JSON outputs

## Main Interfaces

- `SciResearchTaskSampler`: sample a validated task dict containing a procedural chemistry world
- `SciResearchBackend`: backend wrapper that samples environments, manages sessions, and dispatches function calls
- `ChemistryEnvironment`: task-driven session environment; call `set_task(...)` then `reset()`
- `SciResearchEnv`: explicit class name for the same task-driven environment
- `LegacyChemistryEnvironment`: compatibility alias for the old world-path constructor

## Quick Start

Run the demo from the repository root:

```bash
python -m xenoverse.sci_research.demo
```

Generate a batch of worlds:

```bash
python -m xenoverse.sci_research.generate_worlds --n 10 --output-dir xenoverse/sci_research/worlds/
```

Useful options:

- `--seed`
- `--layer1-min`
- `--layer1-max`
- `--last-layer-min`
- `--last-layer-max`

## Dependencies

Local requirements are listed in `requirements.txt` and currently include `numpy` and `scipy`.

## Notes

The package now supports both module execution (`python -m ...`) and direct script execution for `demo.py` and `generate_worlds.py`, which helps local debugging during development.

## Backend-Driven Usage

```python
from xenoverse.sci_research import SciResearchBackend

backend = SciResearchBackend()
session = backend.handle_request({
    "action": "sample_environment",
    "sampler_kwargs": {"seed": 7},
})
session_id = session["session_id"]

tools = session["observation"]["function_tools"]
result = backend.handle_request({
    "action": "dispatch_function_call",
    "session_id": session_id,
    "function_call": {
        "name": "list_purchasable",
        "arguments": {},
    },
})
```

The backend is the recommended integration point for external agents. It samples the environment, returns the function tool schemas, and accepts service-style requests plus OpenAI-style or plain `{name, arguments}` function-call payloads for execution.

## Agent-Facing Tools

Besides chemistry actions, the environment also exposes meta-tools that help an agent stay oriented without leaking hidden state:

- `restate_task_goal`
- `recap_recent_activity`
- `list_function_tools`
- `score_synthesis_route`
- `score_synthesis_plan`

`score_synthesis_route` is the preferred adjudication tool for agent-proposed routes. The agent only needs to submit a target compound plus a high-level route structure, and the environment auto-completes practical parameters before scoring the route against hidden ground truth.

`score_synthesis_plan` remains available when the agent wants to submit a fully specified experimental plan with explicit amounts and conditions.
