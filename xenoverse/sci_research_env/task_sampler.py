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
            "You are in an unfamiliar world whose chemistry is entirely different from the real world. "
            "Real-world chemical knowledge does NOT apply here — compound names, reactions, and properties "
            "bear no relation to anything you may have learned. You must discover everything empirically.",
            "Start by inspecting available functions and purchasable chemicals.",
            "Use experiments and tool calls to discover useful compounds and viable reaction routes.",
            "The compounds and reaction pathways in this world are yet to be fully discovered.",
            "Track experimental outcomes and refine your plan based on observed results.",
            "Do not invent compounds or reactions; only rely on what you observe through tool outputs.",
            "Use estimate_cost to probe the cost structure — temperature, pressure, and duration all affect cost.",
            "When ready, use submit_solution to submit your synthesis plan with fully specified conditions for each step.",
        ],
        "rules": [
            "You may call submit_solution multiple times. Your highest score across all submissions is your final result.",
            "Each submission must include: target_compound and steps (each step specifying reactant_amounts, temperature_C, pressure_atm, duration_seconds, and optional catalyst_names).",
            "The score is based on medicinal value, toxicity, cost, yield, and efficiency of your proposed plan.",
            "Choosing appropriate reaction conditions (temperature, pressure, duration) matters — suboptimal conditions increase cost and reduce yield.",
        ],
        "success_criteria": [
            "Identify compounds with strong medicinal potential.",
            "Prefer lower-toxicity and lower-cost routes when alternatives exist.",
            "Choose appropriate reaction conditions for each step.",
            "Use tool outputs and experimental evidence rather than assumptions.",
            "Submit a fully specified synthesis plan via submit_solution.",
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
    complexity_level: Optional[str] = None,
    layer1_min: Optional[int] = None,
    layer1_max: Optional[int] = None,
    last_layer_min: Optional[int] = None,
    last_layer_max: Optional[int] = None,
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
            complexity_level=complexity_level,
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
                "complexity_level": complexity_level,
                "public_task": _public_task_brief(),
                "world": world.to_dict(),
                "summary": _world_summary(world),
            }
        if verbose:
            print(f"Failed to sample valid sci_research world with seed={current_seed}: {reason}")

    raise RuntimeError(f"Could not generate a valid sci_research task after {max_attempts} attempts.")
