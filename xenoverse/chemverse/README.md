# Chemverse

English | [Chinese](README.zh.md)

`xenoverse.chemverse` is a chemistry-agent environment built around procedurally generated worlds. Agents explore an unknown chemical space by purchasing materials, running experiments, and discovering synthesis routes to medicinally valuable compounds.

## Naming and Compatibility

- Package path: `xenoverse/chemverse/`
- Python import path: `xenoverse.chemverse`
- Previous package name: `xenoverse.sci_research_env`
- Current internal compatibility names still include `SciResearchBackend`, `SciResearchEnv`, and `SciResearchTaskSampler`

The package rename standardizes the public module name around the chemistry domain, while keeping existing runtime class names stable for now.

## Structure

- `world_gen/`: world models, samplers, and validators
- `environment/`: environment API, simulation logic, cost model, and session management
- `generate_worlds.py`: batch world generation and management CLI
- `demo.py`: interactive demo REPL
- `task_sampler.py`: task sampling with validation
- `tests/`: backend-oriented tests
- `worlds/`: default storage for generated world JSON files

## Quick Start

### Interactive Demo

```bash
python -m xenoverse.chemverse.demo
```

The demo samples a world, prints the task description and available tools as JSON, then enters an interactive REPL where you type function calls and inspect results.

### Generate Worlds

```bash
# Generate 10 worlds in the package worlds/ directory
python -m xenoverse.chemverse.generate_worlds --n 10

# List saved worlds
python -m xenoverse.chemverse.generate_worlds --list

# Write worlds to a custom directory
python -m xenoverse.chemverse.generate_worlds --n 5 --output-dir /tmp/my_worlds
```

Supported options include `--seed`, `--complexity`, `--layer1-min`, `--layer1-max`, `--last-layer-min`, and `--last-layer-max`.

### Complexity Levels

Use `--complexity` to select a preset that controls world size and difficulty:

| Level | Layers | Layer-1 Chemicals | Approx Total Chemicals | Approx Reactions |
| --- | --- | --- | --- | --- |
| `easy` | 3 | 4-8 | 10-15 | 15-25 |
| `medium` | 3-4 | 6-10 | 18-30 | 30-60 |
| `hard` | 4-5 | 8-14 | 35-80+ | 55-200+ |

```bash
python -m xenoverse.chemverse.generate_worlds --n 5 --complexity hard
```

Explicit layer parameters such as `--layer1-min 12` override the selected preset. If neither `--complexity` nor explicit layer parameters are provided, the generator falls back to the legacy default ranges.

## Programmatic World Management

```python
from xenoverse.chemverse.generate_worlds import DEFAULT_WORLDS_DIR, list_worlds
from xenoverse.chemverse.world_gen import World

worlds = list_worlds()
for world_info in worlds:
    print(world_info["world_id"], world_info["num_chemicals"], world_info["num_reactions"])

world = World.load(worlds[0]["file_path"])
```

## Backend-Driven Usage

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

The backend is the recommended integration point for external agents. It samples environments, manages sessions, and accepts `{name, arguments}` function-call payloads.

## Agent-Facing Tools

The agent interacts with the environment through these function tools:

| Tool | Purpose |
| --- | --- |
| `task_description` | Get the task objective and success criteria |
| `restate_task_goal` | Repeat the current task objective |
| `recap_recent_activity` | Summarize recent experimental actions |
| `list_function_tools` | List all available tools |
| `list_purchasable` | List base chemicals available for purchase |
| `purchase` | Buy a base chemical |
| `get_inventory` | View the current inventory |
| `analyze_compound` | Analyze a compound in inventory |
| `list_equipment` | Inspect available reaction equipment |
| `perform_reaction` | Execute a reaction with specified conditions |
| `estimate_cost` | Estimate cost for a candidate reaction setup |
| `submit_solution` | Submit a target compound for scoring |
| `finish_experiment` | End the current session early |

Each tool includes a `brief`, a detailed `description`, parameter schema, and example payloads.

## Submission and Scoring

Agents submit solutions via `submit_solution`, declaring the target compound to score:

```json
{
  "name": "submit_solution",
  "arguments": {
    "target_compound": "CompoundX"
  }
}
```

Operational rules:

- Agents may submit multiple times; the highest valid score becomes the final result.
- Scoring depends on medicinal value, toxicity, cost, yield, and efficiency.
- Reaction conditions such as temperature, pressure, and duration affect both cost and yield.
- The cost model is world-dependent; coefficients are randomly sampled per world.
- `estimate_cost` is the main low-risk probe for understanding the local cost curve.

The agent-facing response is sanitized and does not leak the hidden world state.

## Evaluation API

For evaluation scripts and benchmarking, the backend exposes methods that are not available to the acting agent:

```python
backend.eval_find_synthesis_routes(session_id, target_compound="X")
backend.eval_find_cheapest_medicinal_pathway(session_id, min_medicinal_value=3.0)
backend.eval_score_synthesis_route(session_id, target_compound="X", steps=[...])
backend.eval_score_synthesis_plan(session_id, target_compound="X", steps=[...])
backend.eval_get_best_submission(session_id)
backend.eval_export_world(session_id)
```

These methods return internal details such as exact medicinal values, toxicity, generated optimal conditions, and detailed pathway evaluations for offline analysis.

## Cost Model

The cost model accounts for:

- Temperature: deviations from room temperature incur energy costs
- Pressure: deviations from 1 atm incur energy and equipment costs
- Duration: longer reactions increase equipment-time costs
- Materials: raw material costs are based on purchase prices
- Toxicity: hazardous materials increase equipment and handling costs
- Purification: complexity grows with the number and phases of recovered products

All cost coefficients are sampled per world, so each generated environment has a different operating profile.

## Testing

Run the backend-focused test module with:

```bash
python -m xenoverse.chemverse.tests.test_backend
```

## Dependencies

The local `requirements.txt` for this package currently lists `numpy` and `scipy`.
