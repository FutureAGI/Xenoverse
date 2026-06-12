from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional

from .world_gen.models import World
from .world_gen.sampler import WorldSampler
from .world_gen.sampler_v2 import BackwardDesignSampler
from .world_gen.validator import WorldValidator
CONSTRAINT_RANGES_BY_COMPLEXITY = {
    "easy": {
        "max_toxicity": (3.5, 5.0),
        "min_medicinal": (1.0, 2.0),
        "min_yield_g": (0.5, 1.5),
        "max_time_seconds": (14400, 28800),
        "phase_constraint_prob": 0.3,
    },
    "medium": {
        "max_toxicity": (2.5, 4.0),
        "min_medicinal": (1.5, 3.0),
        "min_yield_g": (1.0, 3.0),
        "max_time_seconds": (28800, 57600),
        "phase_constraint_prob": 0.5,
    },
    "hard": {
        "max_toxicity": (2.0, 3.5),
        "min_medicinal": (2.5, 4.0),
        "min_yield_g": (2.0, 5.0),
        "max_time_seconds": (57600, 115200),
        "phase_constraint_prob": 0.7,
    },
}

DEFAULT_CONSTRAINT_RANGES = {
    "max_toxicity": (2.5, 5.0),
    "min_medicinal": (1.0, 3.0),
    "min_yield_g": (0.5, 3.0),
    "max_time_seconds": (28800, 57600),
    "phase_constraint_prob": 0.4,
}

PHASE_OPTIONS = ["liquid", "solid"]
PHASE_TEMP_RANGE = (20.0, 40.0)


def _sample_constraints(
    rng: np.random.RandomState,
    complexity_level: Optional[str] = None,
) -> Dict[str, Any]:
    ranges = CONSTRAINT_RANGES_BY_COMPLEXITY.get(
        complexity_level or "", DEFAULT_CONSTRAINT_RANGES
    )
    constraints: Dict[str, Any] = {
        "max_toxicity": round(float(rng.uniform(*ranges["max_toxicity"])), 1),
        "min_medicinal": round(float(rng.uniform(*ranges["min_medicinal"])), 2),
        "min_yield_g": round(float(rng.uniform(*ranges["min_yield_g"])), 2),
        "max_time_seconds": round(float(rng.uniform(*ranges["max_time_seconds"])), 0),
    }
    phase_prob = ranges.get("phase_constraint_prob", 0.4)
    if rng.random() < phase_prob:
        constraints["required_phase"] = str(rng.choice(PHASE_OPTIONS))
        constraints["phase_temp_C"] = round(float(rng.uniform(*PHASE_TEMP_RANGE)), 0)
    return constraints


def _verify_feasible_route(world: World, constraints: Dict[str, Any]) -> bool:
    """Check that at least one route satisfies all constraints (med, tox, yield, phase)."""
    from .environment.session import SciResearchEnv
    from .environment.simulator import state_at

    required_phase = constraints.get("required_phase")
    phase_temp = constraints.get("phase_temp_C", 25.0)
    min_yield = constraints["min_yield_g"]

    env = SciResearchEnv()
    env._world = world
    env._task = {"constraints": constraints}

    for per_m1_g in [10.0, 30.0, 60.0]:
        result = env.find_cheapest_medicinal_pathway(
            min_medicinal_value=constraints["min_medicinal"],
            max_toxicity=constraints["max_toxicity"],
            per_m1_g=per_m1_g,
            max_routes_per_target=5,
            max_steps=6,
        )
        if not result.get("found"):
            continue
        for candidate in result.get("all_candidates", []):
            summary = candidate.get("pathway_summary", {})
            yield_g = summary.get("target_yield_g", 0.0)
            if yield_g < min_yield:
                continue
            if required_phase:
                target_name = candidate.get("target", "")
                target_chem = next(
                    (c for c in world.chemicals.values() if c.name == target_name), None
                )
                if target_chem is None:
                    continue
                if state_at(target_chem, phase_temp, 1.0) != required_phase:
                    continue
            return True
    return False


def _public_task_brief(constraints: Dict[str, Any]) -> Dict[str, Any]:
    max_tox = constraints["max_toxicity"]
    min_med = constraints["min_medicinal"]
    min_yield = constraints["min_yield_g"]
    time_budget = constraints["max_time_seconds"]
    time_hours = time_budget / 3600.0
    required_phase = constraints.get("required_phase")
    phase_temp = constraints.get("phase_temp_C")

    summary_parts = [
        f"Target toxicity < {max_tox}",
        f"Medicinal value > {min_med}",
        f"Total yield > {min_yield}g",
        f"Time budget: {time_budget:.0f}s ({time_hours:.1f}h)",
    ]
    if required_phase:
        summary_parts.append(f"Must be {required_phase} at {phase_temp:.0f}°C")

    constraint_info: Dict[str, Any] = {
        "description": (
            "Your submission must satisfy ALL of the following hard constraints. "
            "Submissions violating any constraint are REJECTED (no score)."
        ),
        "max_toxicity": max_tox,
        "min_medicinal_value": min_med,
        "min_yield_g": min_yield,
        "max_time_seconds": time_budget,
        "summary": " | ".join(summary_parts),
    }
    if required_phase:
        constraint_info["required_phase"] = required_phase
        constraint_info["phase_temp_C"] = phase_temp

    hard_constraints_list = [
        f"  - Target compound toxicity must be BELOW {max_tox}",
        f"  - Target compound medicinal value must be ABOVE {min_med}",
        f"  - Total yield of target compound must be at least {min_yield}g (accumulated from all reactions)",
        f"  - Time budget: {time_budget:.0f}s ({time_hours:.1f}h)",
    ]
    if required_phase:
        hard_constraints_list.append(
            f"  - Target compound must be {required_phase.upper()} at {phase_temp:.0f}°C (1 atm)"
        )

    rules_list = [
        "Your score is the TOTAL experiment cost at submission time (all purchases + reactions).",
        "submit_solution checks: target compound properties + total yield accumulated in this session.",
        f"Hard constraints: toxicity < {max_tox}, medicinal > {min_med}, yield > {min_yield}g, time < {time_budget:.0f}s.",
        "Submissions failing ANY constraint are rejected.",
        f"Time budget: {time_budget:.0f}s. Reactions consume duration_seconds; analyses take 300s each.",
    ]
    if required_phase:
        rules_list.append(
            f"Phase constraint: target compound must be {required_phase} at {phase_temp:.0f}°C and 1 atm."
        )

    success_criteria = [
        f"Find a compound with medicinal value > {min_med} and toxicity < {max_tox}.",
        f"Produce at least {min_yield}g of the target compound through reactions.",
        "Minimize total experiment cost (every purchase and reaction counts).",
        "Use tool outputs and experimental evidence rather than assumptions.",
        "Submit via submit_solution when you have produced enough of a qualifying compound.",
    ]
    if required_phase:
        success_criteria.insert(1, f"The compound must be {required_phase} at {phase_temp:.0f}°C (1 atm).")

    return {
        "title": "Medicinal Chemistry Exploration",
        "objective": (
            f"Synthesize a compound that satisfies ALL constraints below. "
            f"Your score is the TOTAL experiment cost (all purchases + all reactions). Lower = better."
        ),
        "constraints": constraint_info,
        "scoring": {
            "metric": "total_experiment_cost",
            "direction": "lower is better",
            "description": (
                "Your score is the TOTAL cost of the entire experiment — every purchase and "
                "every reaction you perform counts toward your score. Minimize unnecessary "
                "exploration. Lower total cost = better score."
            ),
        },
        "time_budget": {
            "total_seconds": time_budget,
            "total_hours": round(time_hours, 1),
            "description": (
                f"You have {time_budget:.0f}s ({time_hours:.1f}h) of simulated lab time. "
                f"Each reaction consumes its duration_seconds. Each compound analysis takes 300s. "
                f"When time runs out, you can only submit or finish."
            ),
        },
        "agent_instructions": [
            "You are in an unfamiliar world whose chemistry is entirely different from the real world. "
            "Real-world chemical knowledge does NOT apply here — compound names, reactions, and properties "
            "bear no relation to anything you may have learned. You must discover everything empirically.",
            "Start by inspecting available functions and purchasable chemicals.",
            "Use experiments and tool calls to discover useful compounds and viable reaction routes.",
            "HARD CONSTRAINTS (all must be satisfied):",
            *hard_constraints_list,
            "SCORING: Total experiment cost = all purchases + all reactions. LOWER IS BETTER.",
            "Every action costs money. Minimize wasteful exploration.",
            "Track experimental outcomes and refine your plan based on observed results.",
            "Do not invent compounds or reactions; only rely on what you observe through tool outputs.",
            "Use estimate_cost to probe the cost structure — temperature, pressure, and duration all affect cost.",
            "When you have produced enough of a qualifying compound, use submit_solution to declare it.",
        ],
        "rules": rules_list,
        "success_criteria": success_criteria,
    }


def _world_summary(world: World) -> Dict[str, Any]:
    return {
        "world_id": world.world_id,
        "seed": world.seed,
        "num_layers": world.num_layers,
        "num_chemicals": len(world.chemicals),
        "num_reactions": len(world.reactions),
    }


UNSOLVABLE_PROBABILITY = 0.05


def SciResearchTaskSampler(
    seed: Optional[int] = None,
    complexity_level: Optional[str] = None,
    layer1_min: Optional[int] = None,
    layer1_max: Optional[int] = None,
    last_layer_min: Optional[int] = None,
    last_layer_max: Optional[int] = None,
    world_id: Optional[str] = None,
    max_attempts: int = 150,
    verbose: bool = False,
    use_backward_design: bool = True,
    force_unsolvable: Optional[bool] = None,
) -> Dict[str, Any]:
    """Sample a valid sci_research task and return a portable task dict.

    Guarantees that the generated world contains at least one synthesis route
    satisfying all three hard constraints (medicinal, toxicity, yield).

    When use_backward_design=True (default), uses the backward-design sampler
    that constructs the optimal path first and guarantees feasibility by
    construction. Falls back to forward sampler on failure.

    With ~5% probability (controlled by UNSOLVABLE_PROBABILITY), generates an
    unsolvable world where no compound meets all constraints. The agent must
    identify this and declare no_solution=True via finish_experiment.
    """
    base_seed = 0 if seed is None else int(seed)
    rng = np.random.RandomState(base_seed)
    constraints = _sample_constraints(rng, complexity_level)

    is_unsolvable = force_unsolvable if force_unsolvable is not None else (rng.random() < UNSOLVABLE_PROBABILITY)

    if is_unsolvable:
        sampler = BackwardDesignSampler(
            seed=base_seed,
            complexity_level=complexity_level or "easy",
            constraints=constraints,
        )
        sampled_world_id = world_id or f"sci_world_{base_seed}"
        world = sampler.sample_unsolvable_world(sampled_world_id)
        return {
            "task_type": "SCI_RESEARCH",
            "task_name": "procedural_chemistry_world",
            "seed": base_seed,
            "complexity_level": complexity_level,
            "constraints": constraints,
            "is_solvable": False,
            "public_task": _public_task_brief(constraints),
            "world": world.to_dict(),
            "summary": _world_summary(world),
        }

    validator = WorldValidator(
        max_toxicity=constraints["max_toxicity"],
        min_medicinal=constraints["min_medicinal"],
    )

    if use_backward_design:
        for attempt in range(max_attempts):
            current_seed = base_seed + attempt
            sampler = BackwardDesignSampler(
                seed=current_seed,
                complexity_level=complexity_level or "easy",
                constraints=constraints,
            )
            sampled_world_id = world_id or f"sci_world_{current_seed}"
            world = sampler.sample_world(sampled_world_id)

            valid, reason = validator.validate(world)
            if not valid:
                if verbose:
                    print(f"  attempt {attempt} (backward): validation failed: {reason}")
                continue

            if not _verify_feasible_route(world, constraints):
                if verbose:
                    print(f"  attempt {attempt} (backward): no route meets yield constraint ({constraints['min_yield_g']}g)")
                continue

            return {
                "task_type": "SCI_RESEARCH",
                "task_name": "procedural_chemistry_world",
                "seed": current_seed,
                "complexity_level": complexity_level,
                "constraints": constraints,
                "is_solvable": True,
                "public_task": _public_task_brief(constraints),
                "world": world.to_dict(),
                "summary": _world_summary(world),
            }

        if verbose:
            print("  backward-design exhausted, falling back to forward sampler")

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
        if not valid:
            if verbose:
                print(f"  attempt {attempt} (forward): basic validation failed: {reason}")
            continue

        if not _verify_feasible_route(world, constraints):
            if verbose:
                print(f"  attempt {attempt} (forward): no route meets yield constraint ({constraints['min_yield_g']}g)")
            continue

        return {
            "task_type": "SCI_RESEARCH",
            "task_name": "procedural_chemistry_world",
            "seed": current_seed,
            "complexity_level": complexity_level,
            "constraints": constraints,
            "is_solvable": True,
            "public_task": _public_task_brief(constraints),
            "world": world.to_dict(),
            "summary": _world_summary(world),
        }

    raise RuntimeError(
        f"Could not generate a valid sci_research task after {max_attempts} attempts "
        f"with constraints {constraints}."
    )
