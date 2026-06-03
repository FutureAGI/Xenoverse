# Sci Research Environment

`xenoverse.sci_research_env` is a chemistry-oriented scientific-agent environment built around procedurally generated worlds. Agents explore an unknown chemical space by purchasing materials, running experiments, and discovering synthesis routes to medicinally valuable compounds.

## Structure

- `world_gen/` — world models, samplers, and validators
- `environment/` — environment API, simulation logic, cost model, session management
- `generate_worlds.py` — batch world generation and management CLI
- `demo.py` — interactive demo (REPL)
- `task_sampler.py` — task sampling with validation
- `worlds/` — default storage for generated world JSON files

## Quick Start

### Interactive Demo

```bash
python -m xenoverse.sci_research_env.demo
```

The demo samples a world, prints the task description and available tools as JSON, then enters an interactive REPL where you type function calls and see results.

### Generate Worlds

```bash
# Generate 10 worlds (saved to package's worlds/ directory)
python -m xenoverse.sci_research_env.generate_worlds --n 10

# List all saved worlds
python -m xenoverse.sci_research_env.generate_worlds --list

# Custom output directory
python -m xenoverse.sci_research_env.generate_worlds --n 5 --output-dir /tmp/my_worlds
```

Options: `--seed`, `--complexity`, `--layer1-min`, `--layer1-max`, `--last-layer-min`, `--last-layer-max`.

#### Complexity Levels

Use `--complexity` to select a preset that controls world size and difficulty:

| Level | Layers | Layer-1 Chemicals | Approx Total Chemicals | Approx Reactions |
|-------|--------|-------------------|------------------------|------------------|
| `easy` | 3 | 4–6 | 10–15 | 15–25 |
| `medium` | 3–4 | 6–10 | 18–30 | 30–50 |
| `hard` | 4–6 | 8–14 | 35–60+ | 55–100+ |

```bash
python -m xenoverse.sci_research_env.generate_worlds --n 5 --complexity hard
```

Explicit layer parameters (e.g. `--layer1-min 12`) override the preset values when both are provided. When neither `--complexity` nor explicit layer params are given, the default behavior matches legacy settings (layers in [3,4,5], layer1 6–10, last layer 2–5).

### Programmatic World Management

```python
from xenoverse.sci_research_env.generate_worlds import list_worlds, DEFAULT_WORLDS_DIR
from xenoverse.sci_research_env.world_gen import World

# List all saved worlds
worlds = list_worlds()
for w in worlds:
    print(w["world_id"], w["num_chemicals"], w["num_reactions"])

# Load a specific world
world = World.load(worlds[0]["file_path"])
```

## Backend-Driven Usage

```python
from xenoverse.sci_research_env.environment.backend import SciResearchBackend

backend = SciResearchBackend()
session = backend.handle_request({
    "action": "sample_environment",
    "sampler_kwargs": {"seed": 7, "complexity_level": "hard"},
})
session_id = session["session_id"]

# Agent dispatches function calls
result = backend.handle_request({
    "action": "dispatch_function_call",
    "session_id": session_id,
    "function_call": {"name": "list_purchasable", "arguments": {}},
})
```

The backend is the recommended integration point for external agents. It samples environments, manages sessions, and accepts `{name, arguments}` function-call payloads.

## Agent-Facing Tools

The agent interacts with the environment through these function tools:

| Tool | Purpose |
|------|---------|
| `task_description` | Get the task objective and success criteria |
| `restate_task_goal` | Repeat the current task objective |
| `recap_recent_activity` | Summarize recent experimental actions |
| `list_function_tools` | List all available tools |
| `list_purchasable` | List base chemicals available for purchase |
| `purchase` | Buy a base chemical |
| `get_inventory` | View current inventory |
| `analyze_compound` | Analyze a compound in inventory |
| `list_possible_reactions` | List reactions possible from current inventory |
| `perform_reaction` | Execute a reaction with specified conditions |
| `estimate_cost` | Estimate cost for a candidate reaction setup |
| `submit_solution` | Submit a synthesis plan for scoring |
| `get_transaction_log` | View full session activity log |

Each tool includes a `brief` description, detailed `description` with parameter explanations, and two `examples` showing complete JSON call format.

## Submission and Scoring

Agents submit solutions via `submit_solution`, providing a fully specified synthesis plan:

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

Rules:
- Agents may submit multiple times; the **highest score** across all submissions is the final result.
- Scores are based on medicinal value, toxicity, cost, yield, and efficiency.
- Reaction conditions (temperature, pressure, duration) directly affect both cost and yield.
- The cost model is world-dependent — coefficients are randomly sampled per world. Agents can probe the cost curve using `estimate_cost`.

The agent-facing response is sanitized (no ground-truth leakage):

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

## Evaluation API (Backend-Only)

For evaluation scripts and benchmarking, the backend exposes god-view methods that are **not** accessible to agents:

```python
# Find optimal routes (full reaction graph search)
backend.eval_find_synthesis_routes(session_id, target_compound="X")

# Find globally optimal medicinal pathway
backend.eval_find_cheapest_medicinal_pathway(session_id, min_medicinal_value=3.0)

# Score with full ground-truth details
backend.eval_score_synthesis_route(session_id, target_compound="X", steps=[...])
backend.eval_score_synthesis_plan(session_id, target_compound="X", steps=[...])

# Get best agent submission (full unsanitized scorecard)
backend.eval_get_best_submission(session_id)

# Export full world data
backend.eval_export_world(session_id)
```

These methods return complete internal details (exact medicinal values, toxicity, generated optimal conditions, pathway evaluations) for offline analysis.

## Cost Model

The cost model considers:
- **Temperature**: deviations from room temperature (25°C) in both directions incur energy costs (cooling is distinct from heating).
- **Pressure**: deviations from 1 atm in both directions incur energy and equipment costs (vacuum and high-pressure).
- **Duration**: longer reactions incur equipment-time costs.
- **Materials**: raw material costs based on purchase prices.
- **Toxicity**: hazardous materials increase equipment and handling costs.
- **Purification**: depends on number of products and phase complexity.

All cost coefficients are randomly sampled per world, making the cost structure unique to each environment. Agents should use `estimate_cost` to probe and learn the cost characteristics.

## Dependencies

Listed in `requirements.txt`: `numpy`, `scipy`.
