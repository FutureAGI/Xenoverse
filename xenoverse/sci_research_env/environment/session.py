from __future__ import annotations

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
            "brief": "Summarize recent experiment (reaction) records.",
            "description": (
                "Returns a list of the most recent reaction records, including successful reactions, "
                "failed reactions, and equipment failures. Does NOT include purchases or analyses. "
                "Parameters: last_n (optional integer, default 5) — the number of recent reaction "
                "entries to return. Must be >= 1."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "last_n": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Number of recent reaction entries to return (default: 5).",
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
            "brief": "List chemicals available for purchase.",
            "description": (
                "Returns all base chemicals that can be directly purchased, along with "
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
            "brief": "Purchase a chemical by name and amount.",
            "description": (
                "Purchases the specified amount (in grams) of a base chemical and adds it to "
                "inventory. Parameters: chemical_name (required string) — exact name of the chemical "
                "as listed by list_purchasable; amount_grams (required number, > 0) — grams to buy."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chemical_name": {
                        "type": "string",
                        "description": "Exact name of the chemical to purchase (must be listed in list_purchasable).",
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
            "name": "list_equipment",
            "brief": "List available reaction equipment.",
            "description": (
                "Returns all available equipment types with their properties: vessel type "
                "(open/sealed), thermal mode (isothermal/adiabatic/open_air/heating/cooling), "
                "pressure/temperature limits, and base cost per hour. CAUTION: Exceeding an "
                "equipment's temperature or pressure limits causes equipment failure and TOTAL "
                "LOSS of all materials. Isothermal equipment has finite heat removal capacity; "
                "highly exothermic reactions in large batches may overwhelm the thermostat. "
                "No parameters required."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            "examples": [
                {"name": "list_equipment", "arguments": {}},
            ],
        },
    },
    {
        "type": "function",
        "function": {
            "name": "perform_reaction",
            "brief": "Execute a reaction with specified conditions and equipment.",
            "description": (
                "Runs a reaction with ALL specified chemicals placed into the reaction vessel. "
                "The simulation proceeds step-by-step (5s intervals); any subset of chemicals that "
                "can react WILL react, and products may chain-react with remaining chemicals. "
                "Include both reactants AND catalysts in reactant_amounts with their gram amounts. "
                "Choose equipment to control vessel type (open=constant P, sealed=constant V) and "
                "thermal mode (isothermal=thermostat, adiabatic=insulated, open_air=natural convection, "
                "heating/cooling=ramping). "
                "In adiabatic/insulated equipment, exothermic reactions raise temperature; "
                "endothermic reactions lower it. In open_air equipment (open_beaker, reflux_condenser), "
                "temperature exchanges heat with the environment (25°C) via Newton's law. "
                "In sealed vessels, gas production increases pressure. "
                "In open vessels, gaseous products ESCAPE and are lost. "
                "Phase changes (melting/boiling) consume/release latent heat. "
                "Heterogeneous reactions (solid-liquid, gas-liquid) have reduced rates due to "
                "limited contact area. "
                "WARNING: If temperature or pressure exceeds equipment limits during the reaction, "
                "EQUIPMENT FAILURE occurs — ALL materials are destroyed with no recovery. "
                "After reaction: products >= 0.001g are purified and isolated into inventory (cost charged). "
                "Products < 0.001g are lost. Unreacted reactants AND catalysts are LOST by default. "
                "Set recover_reactants=true to pay purification and recover leftover materials. "
                "NOTE: Product identities are NOT revealed in the reaction result — you will only see "
                "observations (e.g. 'new substances formed'). Use get_inventory to see what compounds "
                "were isolated, then use analyze_compound (costs 5 credits + 300s) to learn their properties."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Mapping of chemical name to amount in grams. Include BOTH reactants and catalysts here.",
                    },
                    "temperature_C": {
                        "type": "number",
                        "description": "Initial reaction temperature in degrees Celsius.",
                    },
                    "pressure_atm": {
                        "type": "number",
                        "description": "Initial reaction pressure in atmospheres.",
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Reaction duration in seconds.",
                    },
                    "equipment": {
                        "type": "string",
                        "description": "Equipment to use (from list_equipment). Default: open_beaker.",
                    },
                    "heating_rate_C_per_s": {
                        "type": "number",
                        "description": "Temperature ramp rate in C/s (positive=heating, negative=cooling). Only applies for heating/cooling equipment.",
                    },
                    "vessel_volume_L": {
                        "type": "number",
                        "description": "Volume of the reaction vessel in liters. Affects pressure in sealed vessels. Default: 1.0.",
                    },
                    "recover_on_failure": {
                        "type": "boolean",
                        "description": "If true, pay purification cost to recover all materials when no reaction occurs (default: false — lost).",
                    },
                    "recover_reactants": {
                        "type": "boolean",
                        "description": "If true, pay purification cost to recover unreacted leftover reactants AND catalysts after reaction (default: false — lost in mixture).",
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
            "examples": [
                {
                    "name": "perform_reaction",
                    "arguments": {
                        "reactant_amounts": {"Ethanol": 10.0, "Acetic Acid": 10.0, "Sulfuric Acid": 1.0},
                        "temperature_C": 80.0,
                        "pressure_atm": 1.0,
                        "duration_seconds": 3600.0,
                        "equipment": "open_beaker",
                    },
                },
                {
                    "name": "perform_reaction",
                    "arguments": {
                        "reactant_amounts": {"Hydrogen": 5.0, "Nitrogen": 15.0},
                        "temperature_C": 450.0,
                        "pressure_atm": 200.0,
                        "duration_seconds": 7200.0,
                        "equipment": "autoclave",
                        "recover_reactants": True,
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
                "Estimates the total process cost (materials + energy + equipment + purification) for a "
                "candidate reaction without actually executing it. Equipment choice affects cost significantly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reactant_amounts": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Mapping of chemical name to amount in grams. Include both reactants and catalysts.",
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
                    "equipment": {
                        "type": "string",
                        "description": "Equipment type (from list_equipment). Default: open_beaker.",
                    },
                },
                "required": ["reactant_amounts", "temperature_C", "pressure_atm", "duration_seconds"],
                "additionalProperties": False,
            },
            "examples": [
                {
                    "name": "estimate_cost",
                    "arguments": {
                        "reactant_amounts": {"Ethanol": 10.0, "Acetic Acid": 10.0, "Sulfuric Acid": 1.0},
                        "temperature_C": 80.0,
                        "pressure_atm": 1.0,
                        "duration_seconds": 3600.0,
                    },
                },
                {
                    "name": "estimate_cost",
                    "arguments": {
                        "reactant_amounts": {"Benzene": 20.0, "Platinum": 0.5},
                        "temperature_C": 150.0,
                        "pressure_atm": 5.0,
                        "duration_seconds": 1800.0,
                    },
                },
            ],
        },
    },


    {
        "type": "function",
        "function": {
            "name": "submit_solution",
            "brief": "Declare target compound for scoring.",
            "description": (
                "Declare the compound you have synthesized as your solution. The system checks "
                "whether the compound satisfies all constraints (toxicity, medicinal value) and "
                "whether you have produced enough of it (total yield from all reactions in this "
                "session). Your score is the total experiment cost (all purchases + reactions). "
                "Lower cost = better score. You may submit multiple times to check different "
                "compounds. Parameters: target_compound (required string) — name of the compound "
                "you are proposing as your solution."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_compound": {
                        "type": "string",
                        "description": "Name of the compound to submit as solution.",
                    },
                },
                "required": ["target_compound"],
                "additionalProperties": False,
            },
            "examples": [
                {"name": "submit_solution", "arguments": {"target_compound": "Compound-X"}},
                {"name": "submit_solution", "arguments": {"target_compound": "Aspirin"}},
            ],
        },
    },

    {
        "type": "function",
        "function": {
            "name": "finish_experiment",
            "brief": "End the experiment session early.",
            "description": (
                "Signal that you have finished your exploration and do not wish to take any more actions. "
                "Call this when you believe you have submitted your best solution or determined that "
                "no further improvement is possible. If after thorough exploration you are confident "
                "that no combination of available chemicals can produce a compound meeting all task "
                "requirements, set no_solution=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you are finishing (optional).",
                    },
                    "no_solution": {
                        "type": "boolean",
                        "description": "Set to true if you conclude that no qualifying compound can be synthesized from available chemicals. Only declare this after exhaustive exploration.",
                    },
                },
                "additionalProperties": False,
            },
            "examples": [
                {"name": "finish_experiment", "arguments": {"reason": "Submitted best route, no further improvements possible."}},
                {"name": "finish_experiment", "arguments": {"reason": "Exhaustive exploration shows no compound meets medicinal and toxicity requirements.", "no_solution": True}},
                {"name": "finish_experiment", "arguments": {}},
            ],
        },
    },
]


ANALYSIS_TIME_SECONDS = 300.0
ANALYSIS_COST = 5.0
FAILED_REACTION_CLEANUP_COST = 3.0


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
        self._elapsed_time: float = 0.0
        self._total_cost: float = 0.0
        self._total_produced: Dict[str, float] = {}
        self._finished: bool = False
        self._declared_no_solution: bool = False

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
        self._elapsed_time = 0.0
        self._total_cost = 0.0
        self._total_produced = {}
        self._finished = False
        self._declared_no_solution = False
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
        self._elapsed_time = 0.0
        self._total_cost = 0.0
        self._total_produced = {}
        self._finished = False
        self._declared_no_solution = False

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
        budget = self._time_budget()
        return {
            "world_id": self._world.world_id,
            "inventory_size": len(self.get_inventory()),
            "transaction_count": len(self._transaction_log),
            "elapsed_time": round(self._elapsed_time, 1),
            "time_budget": budget,
            "time_remaining": round(max(0.0, budget - self._elapsed_time), 1),
            "total_experiment_cost": round(self._total_cost, 2),
            "notes": [
                "Many compounds and reaction pathways are yet to be discovered.",
                "Use the available function tools to explore and experiment.",
                f"Time budget: {budget:.0f}s. Reactions consume their duration; analysis takes {ANALYSIS_TIME_SECONDS:.0f}s (costs {ANALYSIS_COST:.0f} credits). Failed reactions incur a {FAILED_REACTION_CLEANUP_COST:.0f}-credit cleanup fee.",
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
            "All tool arguments must be valid JSON objects matching the declared schemas.\n\n"
            "=== CRITICAL RULES ===\n"
            "TIME BUDGET: You have a limited time budget. Each reaction consumes its duration_seconds "
            "of simulated time. Each compound analysis takes 300 seconds and costs 5 credits. "
            "Failed reactions incur a 3-credit cleanup fee. When time runs out, you "
            "can only submit or finish.\n\n"
            "SCORING: Your score is the TOTAL experiment cost (all purchases + all reaction costs "
            "including purification). Lower cost = better score.\n\n"
            "REACTION MATERIAL RULES:\n"
            "- When you perform a reaction, ALL specified chemicals (reactants + catalysts) are "
            "removed from inventory and placed into the reaction vessel.\n"
            "- Products >= 0.001g are automatically purified from the mixture and added to your "
            "inventory. A purification cost is charged for each product extracted.\n"
            "- Products < 0.001g are UNDETECTABLE and LOST — they do not enter your inventory "
            "and you will not learn their identity.\n"
            "- Unreacted leftover reactants AND catalysts are LOST IN THE MIXTURE by default. "
            "Set recover_reactants=true to pay purification cost and recover them.\n"
            "- If no reaction occurs at all (wrong combination), ALL materials are LOST. Set "
            "recover_on_failure=true to pay purification cost and recover them.\n"
            "- PURIFICATION IS EXPENSIVE — significantly more costly than re-buying raw chemicals. "
            "Plan reactions carefully to avoid waste.\n\n"
            "SOLVENT & DISSOLUTION:\n"
            "- Some purchasable chemicals are solvents (marked role: 'solvent'). They are very cheap.\n"
            "- Reactions REQUIRE a liquid medium to proceed. Without adequate dissolution, "
            "reactions are blocked entirely.\n"
            "- Add a solvent to dissolve solid reactants. More solvent = better dissolution = faster rate.\n"
            "- If a liquid reactant IS a solvent, it can serve as the reaction medium automatically.\n"
            "- The 'observations' field describes dissolution behavior (what dissolved, what didn't).\n"
            "- Do NOT heat above the solvent's boiling point or it will evaporate and the reaction stops.\n\n"
            "EQUIPMENT & CAPACITY:\n"
            "- Each equipment has a max_capacity_g (total mass limit).\n"
            "- Use list_equipment to check capacity and T/P limits. Larger equipment costs more.\n\n"
            "PRE-CHECK RULES (instant rejection, NO time or cost penalty):\n"
            "- Insufficient inventory, total mass < 1g, mass exceeds equipment capacity, or\n"
            "  temperature/pressure setting exceeds equipment limits.\n"
            "- These are checked BEFORE the reaction starts — no materials consumed, no time passes.\n\n"
            "MID-REACTION EXPLOSION (causes material loss):\n"
            "- If temperature/pressure rises DURING the reaction and exceeds equipment limits,\n"
            "  ALL materials are destroyed. Use rated equipment for vigorous reactions.\n\n"
            "OBSERVATIONS:\n"
            "- Reaction results include an 'observations' field with physical phenomena: "
            "boiling/evaporation, melting, dissolution, gas escape, temperature/pressure changes.\n"
            "- READ THESE CAREFULLY — they tell you what went wrong or right.\n"
            "- 'Vigorous ebullition' = compound boiled off. Lower temp or use sealed vessel.\n"
            "- 'Remained undissolved' = need more solvent or a different solvent.\n"
            "- 'Escaped as gas' = use sealed equipment or lower temperature.\n\n"
            "CATALYST RULES:\n"
            "- Catalysts are included in reactant_amounts with their gram amounts.\n"
            "- Catalysts ACCELERATE reactions but do NOT change equilibrium.\n"
            "- Catalysts are NOT consumed but remain in the mixture (pay purification to recover).\n"
            "- Adding catalyst increases mixture complexity → higher purification cost.\n"
            "- Trade-off: catalyst speeds up reaction (shorter duration) but adds purification cost.\n\n"
            "YIELD: submit_solution checks the TOTAL amount of target compound you have actually "
            "produced across all reactions in this session. Produce enough before submitting.\n\n"
            "EQUILIBRIUM: The system reports whether equilibrium was reached and how much more "
            "time would be needed if not. Use this to optimize reaction durations.\n\n"
            "STRATEGY: Minimize total cost by (1) including appropriate solvents, "
            "(2) planning reactions to avoid failures, (3) using efficient conditions, "
            "(4) avoiding unnecessary purification recovery, (5) buying only what you need, "
            "(6) checking equipment capacity before large batches."
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
        reaction_entries = [
            e for e in self._transaction_log
            if e.get("type") in ("reaction", "failed_reaction")
        ]
        recent = reaction_entries[-last_n:]
        return {
            "success": True,
            "num_returned": len(recent),
            "total_reactions": len(reaction_entries),
            "activities": recent,
        }

    def list_function_tools(self) -> Dict[str, Any]:
        return {
            "success": True,
            "function_tools": self.get_function_tools(),
        }



    def _get_constraints(self) -> Dict:
        if self._task and "constraints" in self._task:
            c = self._task["constraints"]
            result = {
                "max_toxicity": float(c.get("max_toxicity", 4.0)),
                "min_medicinal": float(c.get("min_medicinal", 1.0)),
                "min_yield_g": float(c.get("min_yield_g", 0.5)),
                "max_time_seconds": float(c.get("max_time_seconds", 28800)),
            }
            if "required_phase" in c:
                result["required_phase"] = str(c["required_phase"])
                result["phase_temp_C"] = float(c.get("phase_temp_C", 25.0))
            return result
        return {"max_toxicity": 4.0, "min_medicinal": 1.0, "min_yield_g": 0.5, "max_time_seconds": 28800.0}

    def _time_budget(self) -> float:
        return self._get_constraints()["max_time_seconds"]

    def _time_remaining(self) -> float:
        return max(0.0, self._time_budget() - self._elapsed_time)

    def _is_time_expired(self) -> bool:
        return self._elapsed_time >= self._time_budget()

    def _consume_time(self, seconds: float) -> None:
        self._elapsed_time += seconds

    def _add_cost(self, cost: float) -> None:
        self._total_cost += cost

    def _record_production(self, chemical_name: str, grams: float) -> None:
        self._total_produced[chemical_name] = self._total_produced.get(chemical_name, 0.0) + grams

    def get_total_produced(self, chemical_name: str) -> float:
        return self._total_produced.get(chemical_name, 0.0)



    def submit_solution(self, target_compound: str) -> Dict[str, Any]:
        from .simulator import state_at

        constraints = self._get_constraints()
        target_id = self._name_to_id(target_compound)
        if target_id is None:
            return {"success": False, "message": f"Unknown compound: {target_compound}"}

        chem = self._world.chemicals[target_id]
        toxicity = float(chem.base_toxicity)
        medicinal_value = float(chem.medicinal_value)
        total_yield = self.get_total_produced(target_compound)

        violations = []
        if toxicity >= constraints["max_toxicity"]:
            violations.append(f"Toxicity ({toxicity:.2f}) >= threshold ({constraints['max_toxicity']})")
        if medicinal_value < constraints["min_medicinal"]:
            violations.append(f"Medicinal value ({medicinal_value:.3f}) < required ({constraints['min_medicinal']})")
        if total_yield < constraints["min_yield_g"]:
            violations.append(f"Total yield ({total_yield:.4f}g) < required ({constraints['min_yield_g']}g)")

        required_phase = constraints.get("required_phase")
        if required_phase:
            phase_temp = constraints.get("phase_temp_C", 25.0)
            actual_phase = state_at(chem, phase_temp, 1.0)
            if actual_phase != required_phase:
                violations.append(
                    f"Phase mismatch: compound is {actual_phase} at {phase_temp:.0f}°C, "
                    f"but must be {required_phase}"
                )

        if violations:
            self._transaction_log.append({
                "type": "submission",
                "target_compound": target_compound,
                "verdict": "rejected",
                "violations": violations,
            })
            return {
                "success": True,
                "passed": False,
                "verdict": "rejected",
                "violations": violations,
                "target_compound": target_compound,
                "total_yield_so_far": round(total_yield, 4),
                "total_experiment_cost": round(self._total_cost, 2),
                "elapsed_time": round(self._elapsed_time, 1),
            }

        experiment_cost = round(self._total_cost, 2)
        is_new_best = (
            self._best_submission is None
            or experiment_cost < self._best_submission["total_experiment_cost"]
        )
        if is_new_best:
            self._best_submission = {
                "target_compound": target_compound,
                "total_experiment_cost": experiment_cost,
                "total_yield": round(total_yield, 4),
                "elapsed_time": round(self._elapsed_time, 1),
                "medicinal_value": round(medicinal_value, 3),
                "toxicity": round(toxicity, 3),
            }

        self._transaction_log.append({
            "type": "submission",
            "target_compound": target_compound,
            "verdict": "passed",
            "total_experiment_cost": experiment_cost,
            "total_yield": round(total_yield, 4),
        })

        return {
            "success": True,
            "passed": True,
            "verdict": "passed",
            "target_compound": target_compound,
            "total_yield": round(total_yield, 4),
            "total_experiment_cost": experiment_cost,
            "elapsed_time": round(self._elapsed_time, 1),
            "constraints_satisfied": {
                "toxicity": f"{toxicity:.2f} < {constraints['max_toxicity']}",
                "medicinal": f"{medicinal_value:.3f} > {constraints['min_medicinal']}",
                "yield": f"{total_yield:.4f}g > {constraints['min_yield_g']}g",
            },
            "is_new_best": is_new_best,
            "best_cost": self._best_submission["total_experiment_cost"],
        }

    def get_best_submission(self) -> Optional[Dict[str, Any]]:
        return self._best_submission



    def finish_experiment(self, reason: str = "", no_solution: bool = False) -> Dict[str, Any]:
        self._finished = True
        self._declared_no_solution = no_solution
        best = self._best_submission
        best_cost = best["total_experiment_cost"] if best else None
        self._transaction_log.append({
            "type": "finish",
            "reason": reason,
            "no_solution": no_solution,
            "total_experiment_cost": round(self._total_cost, 2),
            "best_cost": best_cost,
        })
        return {
            "success": True,
            "finished": True,
            "reason": reason or "Agent chose to end the experiment.",
            "declared_no_solution": no_solution,
            "has_passing_submission": best is not None,
            "best_cost": best_cost,
            "total_experiment_cost": round(self._total_cost, 2),
            "elapsed_time": round(self._elapsed_time, 1),
            "time_budget": self._time_budget(),
            "total_submissions": sum(1 for e in self._transaction_log if e.get("type") == "submission"),
        }

    def _purchase_tracked(self, **kwargs) -> Dict[str, Any]:
        result = self.purchase(**kwargs)
        if result.get("success") and "cost" in result:
            self._add_cost(result["cost"])
        return result

    def _analyze_tracked(self, **kwargs) -> Dict[str, Any]:
        time_needed = ANALYSIS_TIME_SECONDS
        if self._elapsed_time + time_needed > self._time_budget():
            return {
                "success": False,
                "message": f"Not enough time remaining for analysis. "
                           f"Need {time_needed:.0f}s, have {self._time_remaining():.0f}s remaining.",
                "time_remaining": round(self._time_remaining(), 1),
            }
        self._consume_time(time_needed)
        self._add_cost(ANALYSIS_COST)
        result = self.analyze_compound(**kwargs)
        result["analysis_cost"] = ANALYSIS_COST
        result["time_consumed"] = time_needed
        result["elapsed_time"] = round(self._elapsed_time, 1)
        result["time_remaining"] = round(self._time_remaining(), 1)
        return result

    def _reaction_tracked(self, **kwargs) -> Dict[str, Any]:
        import numpy as np

        duration = kwargs.get("duration_seconds", 0)
        if duration <= 0:
            return {"success": False, "message": "duration_seconds must be positive."}
        if self._elapsed_time + duration > self._time_budget():
            return {
                "success": False,
                "message": f"Not enough time remaining for this reaction. "
                           f"Need {duration:.0f}s, have {self._time_remaining():.0f}s remaining.",
                "time_remaining": round(self._time_remaining(), 1),
            }

        self._consume_time(duration)
        result = self.perform_reaction(**kwargs)

        if not result.get("success") and result.get("_no_time_loss"):
            self._elapsed_time -= duration
            result.pop("_no_time_loss", None)
            result["time_consumed"] = 0
            result["elapsed_time"] = round(self._elapsed_time, 1)
            result["time_remaining"] = round(self._time_remaining(), 1)
            return result

        if not result.get("success"):
            purification_cost = result.get("purification_cost", 0.0)
            if purification_cost > 0:
                self._add_cost(purification_cost)
            self._add_cost(FAILED_REACTION_CLEANUP_COST)
            result["cleanup_cost"] = FAILED_REACTION_CLEANUP_COST
            result["time_consumed"] = duration
            result["elapsed_time"] = round(self._elapsed_time, 1)
            result["time_remaining"] = round(self._time_remaining(), 1)
            return result

        cost_info = result.get("cost", {})
        if isinstance(cost_info, dict):
            total_reaction_cost = float(cost_info.get("total_cost", 0.0))
        else:
            total_reaction_cost = float(cost_info) if cost_info else 0.0
        self._add_cost(total_reaction_cost)

        products = result.pop("_products_g", {})
        for name, grams in products.items():
            if grams > 0:
                self._record_production(name, grams)

        k_eff = result.get("_k_eff")
        reached_eq = result.get("_reached_equilibrium", False)
        if k_eff is not None and k_eff > 1e-30:
            t_eq = 3.0 / k_eff
            if reached_eq:
                actual_eq_time = min(t_eq, duration)
                result["equilibrium_reached_at"] = round(actual_eq_time, 1)
                result["message"] = (
                    result.get("message", "") +
                    f" Equilibrium was reached at ~{actual_eq_time:.0f}s "
                    f"(you specified {duration:.0f}s)."
                )
            else:
                time_still_needed = max(0.0, t_eq - duration)
                result["time_to_equilibrium"] = round(time_still_needed, 1)
                result["message"] = (
                    result.get("message", "") +
                    f" Equilibrium NOT yet reached. Estimated ~{time_still_needed:.0f}s more needed."
                )

        result["time_consumed"] = duration
        result["elapsed_time"] = round(self._elapsed_time, 1)
        result["time_remaining"] = round(self._time_remaining(), 1)
        return result

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

        if self._finished and tool_name not in ("task_description", "restate_task_goal", "get_inventory"):
            return {"success": False, "message": "Experiment has ended. No further actions allowed."}

        if self._is_time_expired() and tool_name not in (
            "task_description", "restate_task_goal",
            "get_inventory", "finish_experiment", "submit_solution",
        ):
            return {
                "success": False,
                "message": f"Time budget exhausted ({self._time_budget():.0f}s). "
                           f"You may only submit_solution or finish_experiment.",
                "elapsed_time": round(self._elapsed_time, 1),
                "time_budget": self._time_budget(),
            }

        args = arguments or {}
        dispatch = {
            "task_description": lambda: self.task_description(),
            "restate_task_goal": lambda: self.restate_task_goal(),
            "recap_recent_activity": lambda: self.recap_recent_activity(**args),
            "list_function_tools": lambda: self.list_function_tools(),
            "list_equipment": lambda: self.list_equipment(),
            "list_purchasable": lambda: self.list_purchasable(),
            "purchase": lambda: self._purchase_tracked(**args),
            "get_inventory": lambda: self.get_inventory(),
            "analyze_compound": lambda: self._analyze_tracked(**args),

            "perform_reaction": lambda: self._reaction_tracked(**args),
            "estimate_cost": lambda: self.estimate_cost(**args),

            "submit_solution": lambda: self.submit_solution(**args),
            "finish_experiment": lambda: self.finish_experiment(**args),
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
