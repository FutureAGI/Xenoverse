from __future__ import annotations

from typing import Any, Dict, Optional

from .world_gen.models import World
from .world_gen.sampler import WorldSampler
from .world_gen.validator import WorldValidator


def _public_task_brief() -> Dict[str, Any]:
    return {
        "title": "Medicinal Chemistry Exploration",
        "objective": (
            "Discover a chemically plausible route to a medically promising compound "
            "while controlling toxicity and experimental cost."
        ),
        "agent_instructions": [
            "Start by inspecting available functions and purchasable chemicals.",
            "Use experiments and tool calls to discover useful compounds and viable reaction routes.",
            "Do not assume the full chemical space or reaction graph is known upfront.",
            "Track experimental outcomes and refine your plan based on observed results.",
            "When requesting final adjudication, prefer score_synthesis_route unless you have a fully specified experiment plan.",
            "For score_synthesis_route, submit a target_compound and an ordered steps list where each step contains reactants and optional catalysts.",
            "Do not invent hidden compounds or hidden reactions; only rely on compounds and route structure supported by observed tool outputs.",
        ],
        "success_criteria": [
            "Identify compounds with strong medicinal potential.",
            "Prefer lower-toxicity and lower-cost routes when alternatives exist.",
            "Use tool outputs and experimental evidence rather than hidden assumptions.",
            "Submit synthesis routes in the expected tool format so they can be scored.",
        ],
    }


def _world_summary(world: World) -> Dict[str, Any]:
    return {
        "world_id": world.world_id,
        "seed": world.seed,
        "num_layers": world.num_layers,
        "num_chemicals": len(world.chemicals),
        "num_reactions": len(world.reactions),
    }


def SciResearchTaskSampler(
    seed: Optional[int] = None,
    layer1_min: int = 6,
    layer1_max: int = 10,
    last_layer_min: int = 2,
    last_layer_max: int = 5,
    world_id: Optional[str] = None,
    max_attempts: int = 50,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Sample a valid sci_research task and return a portable task dict."""
    base_seed = 0 if seed is None else int(seed)
    validator = WorldValidator()

    for attempt in range(max_attempts):
        current_seed = base_seed + attempt
        sampler = WorldSampler(
            seed=current_seed,
            layer1_min=layer1_min,
            layer1_max=layer1_max,
            last_layer_min=last_layer_min,
            last_layer_max=last_layer_max,
        )
        sampled_world_id = world_id or f"sci_world_{current_seed}"
        world = sampler.sample_world(sampled_world_id)
        valid, reason = validator.validate(world)
        if valid:
            return {
                "task_type": "SCI_RESEARCH",
                "task_name": "procedural_chemistry_world",
                "seed": current_seed,
                "public_task": _public_task_brief(),
                "world": world.to_dict(),
                "summary": _world_summary(world),
            }
        if verbose:
            print(f"Failed to sample valid sci_research world with seed={current_seed}: {reason}")

    raise RuntimeError(f"Could not generate a valid sci_research task after {max_attempts} attempts.")
