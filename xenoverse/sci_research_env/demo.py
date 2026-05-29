#!/usr/bin/env python3
"""Sample a sci_research environment and interact with it only through the backend."""

from __future__ import annotations

import json
import os
import random
import tempfile
import time

try:
    from .environment import LegacyChemistryEnvironment, SciResearchBackend
except ImportError:
    from environment import LegacyChemistryEnvironment
    from environment.backend import SciResearchBackend


def sample_environment(max_attempts: int = 200):
    seed = (int(time.time() * 1000) ^ random.getrandbits(20)) & 0xFFFFFFFF
    backend = SciResearchBackend()
    session = backend.handle_request(
        {
            "action": "sample_environment",
            "sampler_kwargs": {"seed": seed, "max_attempts": max_attempts},
        }
    )
    return backend, session


def sep(title: str = "", width: int = 64) -> None:
    if title:
        bar = "=" * max(4, width - len(title) - 2)
        left = bar[: len(bar) // 2]
        right = bar[len(bar) // 2 :]
        print(f"\n{left} {title} {right}")
    else:
        print("=" * width)


def main() -> None:
    print("\nParallel Chemistry World Environment Sampler\n")
    print("Sampling a valid world...")
    backend, session = sample_environment()
    session_id = session["session_id"]
    summary = session["observation"]["public_state"]
    task_description = session["task_description"]

    sep("SESSION")
    print(f"Session ID : {session_id}")
    print(f"World ID   : {summary['world_id']}")
    print(f"Inventory  : {summary['inventory_size']} known compounds")
    print(f"History    : {summary['transaction_count']} logged actions")

    sep("TASK")
    print(f"Title      : {task_description.get('title', '')}")
    print(f"Objective  : {task_description.get('objective', '')}")
    print("Instructions:")
    for item in task_description.get("agent_instructions", []):
        print(f"  - {item}")
    print("Success criteria:")
    for item in task_description.get("success_criteria", []):
        print(f"  - {item}")

    print("\nBackend prompt:")
    print(session["tool_prompt"])

    sep("FUNCTION TOOLS")
    for tool in session["observation"]["function_tools"]:
        print(f"  {tool['function']['name']}: {tool['function']['description']}")

    sep("RESTATE GOAL")
    goal_response = backend.handle_request(
        {
            "action": "dispatch_function_call",
            "session_id": session_id,
            "function_call": {"name": "restate_task_goal", "arguments": {}},
        }
    )
    print(goal_response["task_description"]["objective"])

    sep("PURCHASABLE CHEMICALS")
    purchasable_response = backend.handle_request(
        {
            "action": "dispatch_function_call",
            "session_id": session_id,
            "function_call": {"name": "list_purchasable", "arguments": {}},
        }
    )
    purchasable = purchasable_response["result"]
    for name, info in sorted(purchasable.items(), key=lambda item: item[1]["price_per_gram"]):
        print(
            f"  {name:<18s} {info['price_per_gram']:.3f} cr/g "
            f"[{info['state_at_room_temp']}]"
        )

    for name in purchasable:
        backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": {
                    "name": "purchase",
                    "arguments": {"chemical_name": name, "amount_grams": 50.0},
                },
            }
        )

    sep("CHEAPEST MEDICINAL PATHWAY")
    result = backend.handle_request(
        {
            "action": "dispatch_function_call",
            "session_id": session_id,
            "function_call": {
                "name": "find_cheapest_medicinal_pathway",
                "arguments": {
                    "min_medicinal_value": 3.0,
                    "max_toxicity": 4.0,
                    "per_m1_g": 10.0,
                },
            },
        }
    )
    result = result["result"] if "result" in result else result
    print(f"Qualifying compounds : {result['num_qualifying_compounds']}")
    print(f"Routes evaluated     : {result['num_evaluated_routes']}")
    if not result.get("found"):
        print(result["message"])

    if result.get("found"):
        best = result["best_pathway"]
        summary = best["pathway_summary"]
        print(
            f"\nBest target : {best['target']}\n"
            f"Medicinal   : {best['medicinal_value']:.3f}\n"
            f"Toxicity    : {best['base_toxicity']:.3f}\n"
            f"Route steps : {best['route']['num_steps']}\n"
            f"Route from  : {', '.join(best['route']['m1_starting_materials'])}\n"
            f"Yield       : {summary['target_yield_g']:.4f} g\n"
            f"Cost        : {summary['total_cost']:.2f} cr\n"
            f"Cost/g      : {summary['cost_per_gram_target']:.2f} cr/g\n"
            f"Efficiency  : {summary['efficiency_rating']}"
        )

        internal_task = backend.handle_request(
            {"action": "export_internal_task", "session_id": session_id}
        )["task"]
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as handle:
            json.dump(internal_task["world"], handle)
            world_path = handle.name
        try:
            legacy_env = LegacyChemistryEnvironment(world_path)
            target_id = legacy_env._name_to_id(best["target"])
            chains = legacy_env._find_reaction_chains(target_id, max_routes=1, max_steps=8)
            if chains:
                steps = legacy_env._build_pathway_steps(chains[0], per_m1_g=10.0)
                score_response = backend.handle_request(
                    {
                        "action": "dispatch_function_call",
                        "session_id": session_id,
                        "function_call": {
                            "name": "score_synthesis_plan",
                            "arguments": {"target_compound": best["target"], "steps": steps},
                        },
                    }
                )
                sep("PLAN SCORE")
                print(f"Score    : {score_response['aggregate_score']}")
                print(f"Verdict  : {score_response['verdict']}")
                print(f"Reasons  : {', '.join(score_response['reasoning'])}")
        finally:
            os.unlink(world_path)

    sep("RECENT ACTIVITY")
    activity_response = backend.handle_request(
        {
            "action": "dispatch_function_call",
            "session_id": session_id,
            "function_call": {"name": "recap_recent_activity", "arguments": {"last_n": 5}},
        }
    )
    for item in activity_response["activities"]:
        print(f"  {item}")

    backend.handle_request({"action": "close_session", "session_id": session_id})

    sep()
    print("Done.")


if __name__ == "__main__":
    main()
