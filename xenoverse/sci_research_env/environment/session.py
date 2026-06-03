from __future__ import annotations

import math
import json
from typing import Any, Dict, List, Optional

from ..task_sampler import SciResearchTaskSampler
from ..world_gen.models import World
from .api import ChemistryEnvironment


_FUNCTION_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "task_description",
            "brief": "Get the task objective and success criteria.",
            "description": (
                "Returns the full task description including title, objective, agent instructions, "
                "and success criteria. Use this at any time to review what you need to accomplish."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "task_description", "arguments": {}},
                {"name": "task_description", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "restate_task_goal",
            "brief": "Repeat the currently published task objective.",
            "description": (
                "Returns the task objective text and operating guidance. "
                "No parameters required. Useful to re-read the goal mid-session."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "restate_task_goal", "arguments": {}},
                {"name": "restate_task_goal", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recap_recent_activity",
            "brief": "Summarize the most recent experimental actions.",
            "description": (
                "Returns a list of the most recent purchases, reactions, and other actions "
                "recorded in the transaction log. Parameters: last_n (optional integer, default 5) "
                "— the number of recent log entries to return. Must be >= 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of recent activity entries to return (default: 5).",
                    },
                },
                "additionalProperties": False,
            },
            "examples": [
                {"name": "recap_recent_activity", "arguments": {}},
                {"name": "recap_recent_activity", "arguments": {"last_n": 10}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_function_tools",
            "brief": "List all available function tools.",
            "description": (
                "Returns the complete list of currently available function tools with their names "
                "and descriptions. No parameters required."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "list_function_tools", "arguments": {}},
                {"name": "list_function_tools", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_purchasable",
            "brief": "List layer-1 chemicals available for purchase.",
            "description": (
                "Returns all layer-1 (base) chemicals that can be directly purchased, along with "
                "their price per gram and physical state at room temperature. No parameters required."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "list_purchasable", "arguments": {}},
                {"name": "list_purchasable", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "purchase",
            "brief": "Purchase a layer-1 chemical by name and amount.",
            "description": (
                "Purchases the specified amount (in grams) of a layer-1 chemical and adds it to "
                "inventory. Parameters: chemical_name (required string) — exact name of the chemical "
                "as listed by list_purchasable; amount_grams (required number, > 0) — grams to buy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chemical_name": {
                        "type": "string",
                        "description": "Exact name of the layer-1 chemical to purchase.",
                    },
                    "amount_grams": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "description": "Amount in grams to purchase (must be > 0).",
                    },
                },
                "required": ["chemical_name", "amount_grams"],
                "additionalProperties": False,
            },
            "examples": [
                {"name": "purchase", "arguments": {"chemical_name": "Ethanol", "amount_grams": 50.0}},
                {"name": "purchase", "arguments": {"chemical_name": "Sulfuric Acid", "amount_grams": 25.0}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "brief": "Return the current inventory.",
            "description": (
                "Returns all chemicals currently held in the agent's inventory, with their amounts "
                "in grams. No parameters required."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "get_inventory", "arguments": {}},
                {"name": "get_inventory", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_compound",
            "brief": "Analyze a compound in inventory.",
            "description": (
                "Returns detailed properties of a compound already present in inventory, including "
                "physical state, molecular weight, medicinal value, toxicity, and known reaction "
                "participation. Parameters: chemical_name (required string) — exact name of the "
                "compound to analyze; must already be in inventory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chemical_name": {
                        "type": "string",
                        "description": "Exact name of the compound to analyze (must be in inventory).",
                    },
                },
                "required": ["chemical_name"],
                "additionalProperties": False,
            },
            "examples": [
                {"name": "analyze_compound", "arguments": {"chemical_name": "Ethanol"}},
                {"name": "analyze_compound", "arguments": {"chemical_name": "Aspirin"}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_possible_reactions",
            "brief": "List reactions possible from current inventory.",
            "description": (
                "Returns all reactions that can be performed using chemicals currently available "
                "in inventory (including catalysts). No parameters required. Each reaction entry "
                "shows reactants, products, required catalysts, and suggested conditions."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "list_possible_reactions", "arguments": {}},
                {"name": "list_possible_reactions", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_reaction",
            "brief": "Execute a reaction with specified conditions.",
            "description": (
                "Runs a reaction consuming reactants from inventory under specified conditions. "
                "Parameters: reactant_amounts (required object) — mapping of chemical name to grams "
                "to consume; temperature_C (required number) — reaction temperature in Celsius; "
                "pressure_atm (required number) — pressure in atmospheres; duration_seconds (required "
                "number) — reaction duration in seconds; catalyst_names (optional array of strings) "
                "— catalyst names to use (not consumed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Mapping of chemical name to amount in grams to use as reactant.",
                    },
                    "temperature_C": {
                        "type": "number",
                        "description": "Reaction temperature in degrees Celsius.",
                    },
                    "pressure_atm": {
                        "type": "number",
                        "description": "Reaction pressure in atmospheres.",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Reaction duration in seconds.",
                    },
                    "catalyst_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of catalyst names (not consumed by the reaction).",
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
            "examples": [
                {
                    "name": "perform_reaction",
                    "arguments": {
                        "reactant_amounts": {"Ethanol": 10.0, "Acetic Acid": 10.0},
                        "temperature_C": 80.0,
                        "pressure_atm": 1.0,
                        "duration_seconds": 3600.0,
                        "catalyst_names": ["Sulfuric Acid"],
                    },
                },
                {
                    "name": "perform_reaction",
                    "arguments": {
                        "reactant_amounts": {"Hydrogen": 5.0, "Nitrogen": 15.0},
                        "temperature_C": 450.0,
                        "pressure_atm": 200.0,
                        "duration_seconds": 7200.0,
                    },
                },
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_cost",
            "brief": "Estimate cost for a candidate reaction setup.",
            "description": (
                "Estimates the total process cost (materials + energy) for a candidate reaction "
                "without actually executing it. Parameters are identical to perform_reaction: "
                "reactant_amounts (required object), temperature_C (required number), "
                "pressure_atm (required number), duration_seconds (required number), "
                "catalyst_names (optional array of strings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Mapping of chemical name to amount in grams.",
                    },
                    "temperature_C": {
                        "type": "number",
                        "description": "Reaction temperature in degrees Celsius.",
                    },
                    "pressure_atm": {
                        "type": "number",
                        "description": "Reaction pressure in atmospheres.",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Reaction duration in seconds.",
                    },
                    "catalyst_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of catalyst names.",
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
            "examples": [
                {
                    "name": "estimate_cost",
                    "arguments": {
                        "reactant_amounts": {"Ethanol": 10.0, "Acetic Acid": 10.0},
                        "temperature_C": 80.0,
                        "pressure_atm": 1.0,
                        "duration_seconds": 3600.0,
                    },
                },
                {
                    "name": "estimate_cost",
                    "arguments": {
                        "reactant_amounts": {"Benzene": 20.0},
                        "temperature_C": 150.0,
                        "pressure_atm": 5.0,
                        "duration_seconds": 1800.0,
                        "catalyst_names": ["Platinum"],
                    },
                },
            ],
        },
    },


    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "brief": "Submit a synthesis solution for scoring.",
            "description": (
                "Submit your proposed synthesis plan as a solution. The plan must fully specify "
                "each reaction step including reactant amounts, temperature, pressure, and duration. "
                "The system evaluates your plan based on real chemistry (cost, yield, medicinal value, "
                "toxicity) and returns a score. You may submit multiple times — your best score is "
                "kept as your final result. Parameters: target_compound (required string) — the "
                "compound you aim to synthesize; steps (required array of objects) — each step "
                "requires reactant_amounts (object mapping name to grams), temperature_C (number), "
                "pressure_atm (number), duration_seconds (number), and optional catalyst_names "
                "(array of strings)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_compound": {
                        "type": "string",
                        "description": "Name of the target compound to synthesize.",
                    },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reactant_amounts": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"},
                                    "description": "Mapping of reactant name to grams consumed.",
                                },
                                "temperature_C": {
                                    "type": "number",
                                    "description": "Reaction temperature in Celsius.",
                                },
                                "pressure_atm": {
                                    "type": "number",
                                    "description": "Reaction pressure in atmospheres.",
                                },
                                "duration_seconds": {
                                    "type": "number",
                                    "description": "Reaction duration in seconds.",
                                },
                                "catalyst_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Optional catalyst names (not consumed).",
                                },
                            },
                            "required": [
                                "reactant_amounts",
                                "temperature_C",
                                "pressure_atm",
                                "duration_seconds",
                            ],
                            "additionalProperties": False,
                        },
                        "description": "Ordered list of fully specified reaction steps.",
                    },
                },
                "required": ["target_compound", "steps"],
                "additionalProperties": False,
            },
            "examples": [
                {
                    "name": "submit_solution",
                    "arguments": {
                        "target_compound": "Aspirin",
                        "steps": [
                            {
                                "reactant_amounts": {"Salicylic Acid": 10.0, "Acetic Anhydride": 10.0},
                                "temperature_C": 85.0,
                                "pressure_atm": 1.0,
                                "duration_seconds": 1800.0,
                                "catalyst_names": ["Phosphoric Acid"],
                            }
                        ],
                    },
                },
                {
                    "name": "submit_solution",
                    "arguments": {
                        "target_compound": "Compound-Y",
                        "steps": [
                            {
                                "reactant_amounts": {"A": 20.0, "B": 15.0},
                                "temperature_C": 120.0,
                                "pressure_atm": 3.0,
                                "duration_seconds": 5400.0,
                            },
                            {
                                "reactant_amounts": {"C": 5.0},
                                "temperature_C": 60.0,
                                "pressure_atm": 1.0,
                                "duration_seconds": 900.0,
                            },
                        ],
                    },
                },
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_log",
            "brief": "Return the full transaction and reaction log.",
            "description": (
                "Returns the complete accumulated log of all purchases, reactions, analyses, and "
                "evaluations performed during the session. No parameters required."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "get_transaction_log", "arguments": {}},
                {"name": "get_transaction_log", "arguments": {}},
            ],
        },
    },
]


class SciResearchEnv(ChemistryEnvironment):
    """Task-driven sci_research environment with function-tool dispatch."""

    def __init__(
        self,
        world_path: Optional[str] = None,
        task: Optional[Dict[str, Any]] = None,
        world: Optional[World] = None,
    ):
        self._world: Optional[World] = None
        self._task: Optional[Dict[str, Any]] = None
        self._inventory: Dict[str, float] = {}
        self._transaction_log: List[Dict[str, Any]] = []
        self._synthesized: set = set()
        self._best_submission: Optional[Dict[str, Any]] = None

        if task is not None:
            self.set_task(task)
        elif world is not None:
            self.set_task({"task_type": "SCI_RESEARCH", "world": world.to_dict()})
        elif world_path is not None:
            super().__init__(world_path)
            self._task = {
                "task_type": "SCI_RESEARCH",
                "world": self._world.to_dict(),
            }

    def reset(self) -> Dict[str, Any]:
        if self._world is None:
            raise RuntimeError("No sci_research task loaded. Call set_task(...) first.")
        self._inventory = {}
        self._transaction_log = []
        self._synthesized = set()
        self._best_submission = None
        return {
            "task_type": "SCI_RESEARCH",
            "task_description": self.get_task_goal(),
            "public_state": self.public_state(),
            "function_tools": self.get_function_tools(),
        }

    def set_task(self, task: Dict[str, Any]) -> None:
        if task.get("task_type") != "SCI_RESEARCH":
            raise ValueError(f"Unsupported sci_research task_type: {task.get('task_type')}")
        world_payload = task.get("world")
        if world_payload is None:
            raise ValueError("SciResearch task is missing a 'world' payload.")
        self._world = World.from_dict(world_payload)
        self._task = task
        self._inventory = {}
        self._transaction_log = []
        self._synthesized = set()
        self._best_submission = None

    def get_task(self) -> Dict[str, Any]:
        if self._task is None:
            raise RuntimeError("No sci_research task loaded.")
        return self._task

    def task_summary(self) -> Dict[str, Any]:
        if self._world is None:
            raise RuntimeError("No sci_research task loaded.")
        return {
            "world_id": self._world.world_id,
            "seed": self._world.seed,
            "num_layers": self._world.num_layers,
            "num_chemicals": len(self._world.chemicals),
            "num_reactions": len(self._world.reactions),
        }

    def public_state(self) -> Dict[str, Any]:
        if self._world is None:
            raise RuntimeError("No sci_research task loaded.")
        return {
            "world_id": self._world.world_id,
            "inventory_size": len(self.get_inventory()),
            "transaction_count": len(self._transaction_log),
            "notes": [
                "Many compounds and reaction pathways are yet to be discovered.",
                "Use the available function tools to explore and experiment.",
            ],
        }

    def get_task_goal(self) -> Dict[str, Any]:
        task = self.get_task()
        return task.get("public_task", {})

    def get_function_tools(self) -> List[Dict[str, Any]]:
        return _FUNCTION_TOOLS

    def get_function_tools_prompt(self) -> str:
        return (
            "You are a research chemist working in a laboratory. "
            "The compounds and reaction pathways in this world are yet to be fully discovered. "
            "Start by reviewing your research objective, then purchase base materials, "
            "run experiments, and use observed results to discover promising synthesis routes. "
            "All tool arguments must be valid JSON objects matching the declared schemas. "
            "When you are ready to submit your synthesis plan, use submit_solution with fully "
            "specified conditions for each step: reactant_amounts, temperature_C, pressure_atm, "
            "duration_seconds, and optional catalyst_names. You may submit multiple times — "
            "your highest score across all submissions is kept as your final result. "
            "The evaluation considers all specified conditions including their impact on cost and yield."
        )

    def task_description(self) -> Dict[str, Any]:
        return {
            "success": True,
            "task_description": self.get_task_goal(),
        }

    def restate_task_goal(self) -> Dict[str, Any]:
        return {
            "success": True,
            "task_description": self.get_task_goal(),
        }

    def recap_recent_activity(self, last_n: int = 5) -> Dict[str, Any]:
        if last_n <= 0:
            return {"success": False, "message": "last_n must be positive."}
        recent = self._transaction_log[-last_n:]
        return {
            "success": True,
            "num_returned": len(recent),
            "activities": recent,
        }

    def list_function_tools(self) -> Dict[str, Any]:
        return {
            "success": True,
            "function_tools": self.get_function_tools(),
        }

    def _resolve_route_step_to_plan(
        self,
        route_step: Dict[str, Any],
        per_m1_g: float,
    ) -> Dict[str, Any]:
        reactant_names = route_step.get("reactants", [])
        catalyst_names = route_step.get("catalysts", [])
        if not reactant_names:
            raise ValueError("Each route step must include at least one reactant.")

        reactant_ids = []
        for name in reactant_names:
            cid = self._name_to_id(name)
            if cid is None:
                raise ValueError(f"Unknown reactant in route step: {name}")
            reactant_ids.append(cid)

        catalyst_ids = []
        for name in catalyst_names:
            cid = self._name_to_id(name)
            if cid is None:
                raise ValueError(f"Unknown catalyst in route step: {name}")
            catalyst_ids.append(cid)

        matches = []
        reactant_set = set(reactant_ids)
        catalyst_set = set(catalyst_ids)
        for rxn in self._world.reactions.values():
            rxn_reactants = {cid for cid, _ in rxn.reactants}
            if rxn_reactants != reactant_set:
                continue
            if catalyst_set and not catalyst_set.issubset(set(rxn.catalysts)):
                continue
            matches.append(rxn)

        if not matches:
            raise ValueError(
                "No known reaction matches the supplied reactants/catalysts combination."
            )

        rxn = max(matches, key=lambda item: abs(item.delta_G_kJ))
        reactant_amounts = {}
        for cid, _ in rxn.reactants:
            chem = self._world.chemicals[cid]
            reactant_amounts[chem.name] = per_m1_g if chem.layer == 1 else 0.1

        temperature_C, duration_seconds = self._optimal_temp_for_reaction(rxn)
        return {
            "reactant_amounts": reactant_amounts,
            "temperature_C": temperature_C,
            "pressure_atm": 1.0,
            "duration_seconds": duration_seconds,
            "catalyst_names": [self._id_to_name(cid) for cid in rxn.catalysts],
        }

    def _score_synthesis_route_full(
        self,
        target_compound: str,
        steps: List[Dict[str, Any]],
        per_m1_g: float = 10.0,
    ) -> Dict[str, Any]:
        if not steps:
            return {"success": False, "message": "Synthesis route must contain at least one step."}

        try:
            generated_plan = [
                self._resolve_route_step_to_plan(route_step, per_m1_g=per_m1_g)
                for route_step in steps
            ]
        except ValueError as exc:
            return {"success": False, "message": str(exc)}

        scorecard = self._score_synthesis_plan_full(target_compound=target_compound, steps=generated_plan)
        if not scorecard.get("success", False):
            return scorecard

        scorecard["submitted_route"] = {
            "target_compound": target_compound,
            "steps": steps,
        }
        scorecard["generated_plan"] = generated_plan
        return scorecard

    def _score_synthesis_plan_full(self, target_compound: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        target_id = self._name_to_id(target_compound)
        if target_id is None:
            return {"success": False, "message": f"Unknown target compound: {target_compound}"}
        if not steps:
            return {"success": False, "message": "Synthesis plan must contain at least one step."}

        eval_result = self.evaluate_pathway(target_compound, steps)
        if not eval_result.get("success", False):
            return eval_result

        chem = self._world.chemicals[target_id]
        summary = eval_result["summary"]
        target_yield_g = float(summary.get("target_yield_g", 0.0))
        total_cost = float(summary.get("total_cost", 0.0))
        medicinal_value = float(chem.medicinal_value)
        toxicity = float(chem.base_toxicity)
        route_steps = int(summary.get("num_steps", len(steps)))

        effective_value = max(target_yield_g * medicinal_value, 1e-9)
        cost_per_medicinal_unit = total_cost / effective_value

        medicinal_score = max(0.0, min(100.0, medicinal_value * 10.0))
        toxicity_score = max(0.0, min(100.0, 100.0 - toxicity * 10.0))
        yield_score = max(0.0, min(100.0, target_yield_g * 20.0))
        efficiency_score = max(0.0, min(100.0, float(summary.get("mass_efficiency", 0.0)) * 200.0))
        cost_score = 100.0 / (1.0 + cost_per_medicinal_unit / 25.0)
        step_penalty = min(20.0, max(0.0, route_steps - 1) * 3.0)

        aggregate_score = (
            0.30 * medicinal_score
            + 0.20 * toxicity_score
            + 0.20 * cost_score
            + 0.15 * yield_score
            + 0.15 * efficiency_score
            - step_penalty
        )
        aggregate_score = max(0.0, min(100.0, aggregate_score))

        if aggregate_score >= 85.0:
            verdict = "excellent"
        elif aggregate_score >= 70.0:
            verdict = "strong"
        elif aggregate_score >= 55.0:
            verdict = "promising"
        elif aggregate_score >= 40.0:
            verdict = "weak"
        else:
            verdict = "poor"

        reasons = []
        if medicinal_score >= 70.0:
            reasons.append("target has strong medicinal potential")
        elif medicinal_score < 35.0:
            reasons.append("target medicinal value is limited")

        if toxicity_score >= 70.0:
            reasons.append("toxicity is favorable")
        elif toxicity_score < 40.0:
            reasons.append("toxicity is a major concern")

        if cost_score >= 60.0:
            reasons.append("cost efficiency is competitive")
        elif cost_score < 25.0:
            reasons.append("cost per medicinal unit is high")

        if yield_score >= 50.0:
            reasons.append("predicted yield is solid")
        elif yield_score < 10.0:
            reasons.append("predicted yield is very low")

        if efficiency_score < 20.0:
            reasons.append("material efficiency is poor")

        scorecard = {
            "aggregate_score": round(aggregate_score, 2),
            "verdict": verdict,
            "components": {
                "medicinal_score": round(medicinal_score, 2),
                "toxicity_score": round(toxicity_score, 2),
                "cost_score": round(cost_score, 2),
                "yield_score": round(yield_score, 2),
                "efficiency_score": round(efficiency_score, 2),
                "step_penalty": round(step_penalty, 2),
            },
            "target_profile": {
                "target_compound": target_compound,
                "medicinal_value": round(medicinal_value, 3),
                "base_toxicity": round(toxicity, 3),
            },
            "pathway_metrics": {
                "num_steps": route_steps,
                "target_yield_g": round(target_yield_g, 4),
                "total_cost": round(total_cost, 2),
                "cost_per_medicinal_unit": round(cost_per_medicinal_unit, 4),
                "efficiency_rating": summary.get("efficiency_rating"),
            },
            "reasoning": reasons,
            "pathway_evaluation": eval_result,
        }

        self._transaction_log.append(
            {
                "type": "evaluation",
                "target_compound": target_compound,
                "num_steps": route_steps,
                "aggregate_score": scorecard["aggregate_score"],
                "verdict": verdict,
            }
        )
        return {"success": True, **scorecard}

    def submit_solution(self, target_compound: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        full = self._score_synthesis_plan_full(target_compound, steps)
        sanitized = self._sanitize_scorecard(full)
        if sanitized.get("success"):
            current_score = sanitized["aggregate_score"]
            best_score = self._best_submission["aggregate_score"] if self._best_submission else None
            is_new_best = best_score is None or current_score > best_score
            if is_new_best:
                self._best_submission = full
            sanitized["is_new_best"] = is_new_best
            sanitized["best_score"] = current_score if is_new_best else best_score
        return sanitized

    def get_best_submission(self) -> Optional[Dict[str, Any]]:
        return self._best_submission

    @staticmethod
    def _sanitize_scorecard(full: Dict[str, Any]) -> Dict[str, Any]:
        if not full.get("success", False):
            return full
        metrics = full.get("pathway_metrics", {})
        return {
            "success": True,
            "aggregate_score": full["aggregate_score"],
            "verdict": full["verdict"],
            "reasoning": full["reasoning"],
            "pathway_metrics": {
                "num_steps": metrics.get("num_steps"),
                "target_yield_g": metrics.get("target_yield_g"),
                "total_cost": metrics.get("total_cost"),
                "efficiency_rating": metrics.get("efficiency_rating"),
            },
        }

    def sample_task(self, **kwargs: Any) -> Dict[str, Any]:
        return SciResearchTaskSampler(**kwargs)

    def dispatch_function_call(self, function_call: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(function_call, dict):
            raise TypeError("function_call must be a dict.")

        if "function" in function_call and isinstance(function_call["function"], dict):
            payload = function_call["function"]
            tool_name = payload.get("name")
            arguments = payload.get("arguments", {})
        else:
            tool_name = (
                function_call.get("name")
                or function_call.get("tool_name")
                or function_call.get("function_name")
            )
            arguments = function_call.get("arguments", {})

        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise TypeError("Function call arguments must decode to a dict.")

        return self.call_tool(tool_name, arguments)

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self._world is None:
            raise RuntimeError("No sci_research task loaded. Call set_task(...) first.")

        args = arguments or {}
        dispatch = {
            "task_description": lambda: self.task_description(),
            "restate_task_goal": lambda: self.restate_task_goal(),
            "recap_recent_activity": lambda: self.recap_recent_activity(**args),
            "list_function_tools": lambda: self.list_function_tools(),
            "list_purchasable": lambda: self.list_purchasable(),
            "purchase": lambda: self.purchase(**args),
            "get_inventory": lambda: self.get_inventory(),
            "analyze_compound": lambda: self.analyze_compound(**args),
            "list_possible_reactions": lambda: self.list_possible_reactions(),
            "perform_reaction": lambda: self.perform_reaction(**args),
            "estimate_cost": lambda: self.estimate_cost(**args),


            "submit_solution": lambda: self.submit_solution(**args),
            "get_transaction_log": lambda: self.get_transaction_log(),
        }
        if tool_name not in dispatch:
            return {
                "success": False,
                "message": f"Unknown sci_research tool: {tool_name}",
                "available_tools": [tool["function"]["name"] for tool in _FUNCTION_TOOLS],
            }
        try:
            result = dispatch[tool_name]()
        except TypeError as exc:
            return {
                "success": False,
                "message": f"Invalid arguments for sci_research tool '{tool_name}': {exc}",
            }

        if isinstance(result, dict):
            return result
        return {"success": True, "result": result}
