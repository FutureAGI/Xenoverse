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
            "name": "restate_task_goal",
            "description": "Repeat the currently published task objective and operating guidance.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recap_recent_activity",
            "description": "Summarize the most recent purchases, reactions, and other experimental actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_function_tools",
            "description": "Return the list of currently available function tools and their descriptions.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_purchasable",
            "description": "List layer-1 chemicals available for direct purchase.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "purchase",
            "description": "Purchase a given amount of a layer-1 chemical.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chemical_name": {"type": "string"},
                    "amount_grams": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["chemical_name", "amount_grams"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_inventory",
            "description": "Return the current inventory accumulated by the agent.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_compound",
            "description": "Analyze a compound already present in inventory.",
            "parameters": {
                "type": "object",
                "properties": {"chemical_name": {"type": "string"}},
                "required": ["chemical_name"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_possible_reactions",
            "description": "List reactions currently possible from available inventory and catalysts.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_reaction",
            "description": "Run a reaction given reactant amounts and reaction conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "temperature_C": {"type": "number"},
                    "pressure_atm": {"type": "number"},
                    "duration_seconds": {"type": "number"},
                    "catalyst_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estimate_cost",
            "description": "Estimate process cost for a candidate reaction setup before executing it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                    },
                    "temperature_C": {"type": "number"},
                    "pressure_atm": {"type": "number"},
                    "duration_seconds": {"type": "number"},
                    "catalyst_names": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_synthesis_routes",
            "description": "Search reaction-network routes from purchasable chemicals to a target compound.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_compound": {"type": "string"},
                    "max_routes": {"type": "integer", "minimum": 1},
                    "max_steps": {"type": "integer", "minimum": 1},
                },
                "required": ["target_compound"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_cheapest_medicinal_pathway",
            "description": "Find the most cost-effective medicinal synthesis pathway in the sampled world.",
            "parameters": {
                "type": "object",
                "properties": {
                    "min_medicinal_value": {"type": "number"},
                    "max_toxicity": {"type": "number"},
                    "per_m1_g": {"type": "number"},
                    "max_routes_per_target": {"type": "integer", "minimum": 1},
                    "max_steps": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_synthesis_route",
            "description": "Evaluate an agent-proposed high-level synthesis route and auto-complete practical step parameters before scoring.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_compound": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reactants": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "catalysts": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["reactants"],
                            "additionalProperties": False,
                        },
                    },
                    "per_m1_g": {"type": "number"},
                },
                "required": ["target_compound", "steps"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_synthesis_plan",
            "description": "Evaluate an agent-proposed synthesis plan and score it using hidden ground-truth chemistry, cost, yield, medicinal value, and toxicity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_compound": {"type": "string"},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reactant_amounts": {
                                    "type": "object",
                                    "additionalProperties": {"type": "number"},
                                },
                                "temperature_C": {"type": "number"},
                                "pressure_atm": {"type": "number"},
                                "duration_seconds": {"type": "number"},
                                "catalyst_names": {
                                    "type": "array",
                                    "items": {"type": "string"},
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
                    },
                },
                "required": ["target_compound", "steps"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transaction_log",
            "description": "Return the accumulated transaction and reaction log.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
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
                "Underlying compounds and reaction connectivity are intentionally hidden.",
                "Use the available function tools to explore the environment experimentally.",
            ],
        }

    def get_task_goal(self) -> Dict[str, Any]:
        task = self.get_task()
        return task.get("public_task", {})

    def get_function_tools(self) -> List[Dict[str, Any]]:
        return _FUNCTION_TOOLS

    def get_function_tools_prompt(self) -> str:
        return (
            "You are interacting with xenoverse.sci_research through function tools. "
            "The task description is public, but the full compound space and reaction graph are hidden. "
            "First inspect the published goal and available function tools, then buy layer-1 materials, "
            "run experiments, and use observed results to discover promising synthesis routes. "
            "All tool arguments must be valid JSON objects matching the declared schemas. "
            "For final route adjudication, prefer score_synthesis_route. Its required format is: "
            "{target_compound: string, steps: [{reactants: [string, ...], catalysts: [string, ...]?}, ...]}. "
            "Use score_synthesis_plan only when you want to provide a fully specified experiment plan with "
            "reactant_amounts, temperature_C, pressure_atm, duration_seconds, and optional catalyst_names."
        )

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
                "Could not match the provided route step to a hidden reaction using the supplied reactants/catalysts."
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

    def score_synthesis_route(
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

        scorecard = self.score_synthesis_plan(target_compound=target_compound, steps=generated_plan)
        if not scorecard.get("success", False):
            return scorecard

        scorecard["submitted_route"] = {
            "target_compound": target_compound,
            "steps": steps,
        }
        scorecard["generated_plan"] = generated_plan
        return scorecard

    def score_synthesis_plan(self, target_compound: str, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            "find_synthesis_routes": lambda: self.find_synthesis_routes(**args),
            "find_cheapest_medicinal_pathway": lambda: self.find_cheapest_medicinal_pathway(**args),
            "score_synthesis_route": lambda: self.score_synthesis_route(**args),
            "score_synthesis_plan": lambda: self.score_synthesis_plan(**args),
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
