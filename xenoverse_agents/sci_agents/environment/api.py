from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from world_gen.models import Chemical, Reaction, World
from .simulator import simulate_reaction, state_at, R_kJ
from .cost_model import calculate_cost
from .templates import generate_response, _medicinal_hint, _toxicity_note, PATHWAY_EFFICIENCY_NOTE


def _approx_mw(mw: float) -> float:
    """Return MW with ±5% noise for analysis."""
    import random
    factor = 1.0 + random.uniform(-0.05, 0.05)
    return round(mw * factor, 1)


def _toxicity_level(tox: float) -> str:
    if tox < 2.5:
        return "low"
    elif tox < 5.0:
        return "medium"
    elif tox < 7.5:
        return "high"
    return "extreme"


def _biological_activity_level(med_value: float) -> str:
    if med_value < 1.0:
        return "low"
    elif med_value < 3.0:
        return "moderate"
    return "high"


class ChemistryEnvironment:
    def __init__(self, world_path: str):
        self._world: World = World.load(world_path)
        self._inventory: Dict[str, float] = {}
        self._transaction_log: List[Dict] = []
        self._synthesized: set = set()  # chemical ids that have been synthesized

    def _name_to_id(self, name: str) -> Optional[str]:
        for cid, chem in self._world.chemicals.items():
            if chem.name.lower() == name.lower():
                return cid
        return None

    def _id_to_name(self, cid: str) -> str:
        chem = self._world.chemicals.get(cid)
        return chem.name if chem else cid

    def list_purchasable(self) -> Dict:
        result = {}
        for cid, chem in self._world.chemicals.items():
            if chem.layer == 1:
                phase = state_at(chem, 25.0, 1.0)
                result[chem.name] = {
                    "name": chem.name,
                    "price_per_gram": round(chem.price_per_gram, 4),
                    "state_at_room_temp": phase,
                    "molecular_weight_approx": _approx_mw(chem.molecular_weight),
                }
        return result

    def purchase(self, chemical_name: str, amount_grams: float) -> Dict:
        cid = self._name_to_id(chemical_name)
        if cid is None:
            return {"success": False, "message": f"Unknown chemical: {chemical_name}"}

        chem = self._world.chemicals[cid]
        if chem.layer != 1:
            return {"success": False, "message": f"{chemical_name} is not available for direct purchase."}
        if amount_grams <= 0:
            return {"success": False, "message": "Amount must be positive."}

        cost = chem.price_per_gram * amount_grams
        self._inventory[cid] = self._inventory.get(cid, 0.0) + amount_grams

        entry = {
            "type": "purchase",
            "chemical": chem.name,
            "chemical_id": cid,
            "amount_g": amount_grams,
            "cost": round(cost, 2),
        }
        self._transaction_log.append(entry)

        phase = state_at(chem, 25.0, 1.0)
        msg = generate_response(
            "purchase_success",
            amount=amount_grams,
            name=chem.name,
            state=phase,
            cost=cost,
            toxicity=chem.base_toxicity,
        )
        return {"success": True, "message": msg, "cost": round(cost, 2)}

    def get_inventory(self) -> Dict:
        result = {}
        for cid, grams in self._inventory.items():
            if grams < 1e-6:
                continue
            chem = self._world.chemicals.get(cid)
            if chem is None:
                continue
            phase = state_at(chem, 25.0, 1.0)
            entry = {
                "name": chem.name,
                "amount_g": round(grams, 4),
                "state_at_room_temp": phase,
                "layer": chem.layer,
            }
            if chem.layer == 1:
                entry["price_per_gram"] = chem.price_per_gram
            result[chem.name] = entry
        return result

    def analyze_compound(self, chemical_name: str) -> Dict:
        cid = self._name_to_id(chemical_name)
        if cid is None:
            return {"success": False, "message": f"Unknown compound: {chemical_name}"}

        chem = self._world.chemicals[cid]
        if cid not in self._inventory or self._inventory[cid] < 1e-6:
            return {"success": False, "message": f"You don't have any {chemical_name} in your inventory."}

        phase = state_at(chem, 25.0, 1.0)
        med_level = _biological_activity_level(chem.medicinal_value)
        med_hint = _medicinal_hint(chem.medicinal_value)

        return {
            "success": True,
            "name": chem.name,
            "melting_point_C": round(chem.melting_point, 1),
            "boiling_point_C": round(chem.boiling_point, 1),
            "molecular_weight_approx": _approx_mw(chem.molecular_weight),
            "state_at_room_temp": phase,
            "toxicity_level": _toxicity_level(chem.base_toxicity),
            "toxicity_note": _toxicity_note(chem.base_toxicity),
            "biological_activity": med_level,
            "biological_activity_note": med_hint,
        }

    def list_possible_reactions(self) -> Dict:
        available_ids = {cid for cid, g in self._inventory.items() if g > 1e-6}
        result = {}

        for rid, rxn in self._world.reactions.items():
            reactant_ids = {cid for cid, _ in rxn.reactants}
            catalyst_ids = set(rxn.catalysts)

            if not reactant_ids.issubset(available_ids):
                continue

            # Check catalysts available (either in inventory or already have)
            if not catalyst_ids.issubset(available_ids):
                # Check if catalysts might be purchasable or in inventory
                missing_cats = catalyst_ids - available_ids
                if missing_cats:
                    continue

            products_known = all(pid in self._synthesized for pid, _ in rxn.products)

            reactant_names = [self._id_to_name(cid) for cid, _ in rxn.reactants]
            catalyst_names = [self._id_to_name(cid) for cid in rxn.catalysts]

            if products_known:
                product_desc = [
                    {"name": self._id_to_name(pid), "coefficient": coeff}
                    for pid, coeff in rxn.products
                ]
            else:
                product_desc = [
                    {"name": "unknown product", "coefficient": coeff}
                    for _, coeff in rxn.products
                ]

            result[rid] = {
                "reaction_id": rid,
                "reactants": [
                    {"name": self._id_to_name(cid), "coefficient": coeff}
                    for cid, coeff in rxn.reactants
                ],
                "catalysts_needed": catalyst_names,
                "products": product_desc,
                "conditions_hint": self._conditions_hint(rxn),
            }

        return result

    def _conditions_hint(self, rxn: Reaction) -> str:
        Ea = rxn.activation_energy_kJ
        if Ea < 45:
            temp_hint = "mild temperatures"
        elif Ea < 75:
            temp_hint = "moderate temperatures"
        else:
            temp_hint = "elevated temperatures"

        dG = rxn.delta_G_kJ
        if dG < -30:
            thermo_hint = "thermodynamically favorable"
        elif dG < 0:
            thermo_hint = "slightly favorable"
        else:
            thermo_hint = "requires driving conditions"

        return f"Requires {temp_hint}; reaction is {thermo_hint}."

    def perform_reaction(
        self,
        reactant_amounts: Dict[str, float],
        temperature_C: float,
        pressure_atm: float,
        duration_seconds: float,
        catalyst_names: Optional[List[str]] = None,
    ) -> Dict:
        name_to_id = {
            chem.name.lower(): cid
            for cid, chem in self._world.chemicals.items()
        }

        reactant_ids: Dict[str, float] = {}
        for name, grams in reactant_amounts.items():
            cid = name_to_id.get(name.lower())
            if cid is None:
                return {"success": False, "message": f"Unknown chemical: {name}"}
            reactant_ids[cid] = grams

        catalyst_ids: List[str] = []
        if catalyst_names:
            for cname in catalyst_names:
                cid = name_to_id.get(cname.lower())
                if cid is None:
                    return {"success": False, "message": f"Unknown catalyst: {cname}"}
                catalyst_ids.append(cid)

        # Check inventory sufficiency
        for cid, needed in reactant_ids.items():
            available = self._inventory.get(cid, 0.0)
            if available < needed - 1e-9:
                return {
                    "success": False,
                    "message": f"Insufficient {self._id_to_name(cid)}: need {needed:.2f}g, have {available:.2f}g",
                }

        # Find matching reactions
        matching = self._find_matching_reactions(reactant_ids, catalyst_ids)
        if not matching:
            return {"success": False, "message": generate_response("reaction_fail")}

        # Pick most thermodynamically driven
        rxn = max(matching, key=lambda r: abs(r.delta_G_kJ))

        result = simulate_reaction(
            rxn, self._world.chemicals, reactant_ids, temperature_C, pressure_atm, duration_seconds
        )

        # Update inventory
        for cid, consumed in result["consumed_g"].items():
            self._inventory[cid] = max(0.0, self._inventory.get(cid, 0.0) - consumed)

        for cid, produced in result["produced_g"].items():
            self._inventory[cid] = self._inventory.get(cid, 0.0) + produced
            self._synthesized.add(cid)

        for cid, produced in result["byproduct_g"].items():
            self._inventory[cid] = self._inventory.get(cid, 0.0) + produced
            self._synthesized.add(cid)

        products_str = ", ".join(
            f"{round(g, 3)}g of {self._id_to_name(cid)}"
            for cid, g in result["produced_g"].items()
            if g > 1e-6
        )
        if not products_str:
            products_str = "trace amounts"

        msg = generate_response(
            "reaction_success",
            duration=duration_seconds,
            temp=temperature_C,
            pressure=pressure_atm,
            conversion=result["conversion"],
            products_str=products_str,
            reached_equilibrium=result["reached_equilibrium"],
        )

        cost_info = calculate_cost(
            rxn, self._world.chemicals, reactant_ids, temperature_C, pressure_atm, duration_seconds
        )

        log_entry = {
            "type": "reaction",
            "reaction_id": rxn.id,
            "reactants": {self._id_to_name(cid): g for cid, g in reactant_ids.items()},
            "catalysts": [self._id_to_name(cid) for cid in catalyst_ids],
            "temperature_C": temperature_C,
            "pressure_atm": pressure_atm,
            "duration_s": duration_seconds,
            "conversion": round(result["conversion"], 4),
            "products_produced_g": {self._id_to_name(cid): round(g, 4) for cid, g in result["produced_g"].items()},
            "cost": cost_info,
        }
        self._transaction_log.append(log_entry)

        return {
            "success": True,
            "message": msg,
            "reaction_id": rxn.id,
            "conversion": round(result["conversion"], 4),
            "products_g": {self._id_to_name(cid): round(g, 4) for cid, g in result["produced_g"].items()},
            "byproducts_g": {self._id_to_name(cid): round(g, 4) for cid, g in result["byproduct_g"].items() if g > 1e-6},
            "cost": cost_info,
        }

    def _find_matching_reactions(
        self,
        reactant_ids: Dict[str, float],
        catalyst_ids: List[str],
    ) -> List[Reaction]:
        provided_reactants = set(reactant_ids.keys())
        provided_catalysts = set(catalyst_ids)
        # Also allow catalysts from inventory
        inventory_catalysts = {cid for cid, g in self._inventory.items() if g > 1e-6}
        all_catalysts = provided_catalysts | inventory_catalysts

        matches = []
        for rxn in self._world.reactions.values():
            rxn_reactants = {cid for cid, _ in rxn.reactants}
            rxn_catalysts = set(rxn.catalysts)

            if not rxn_reactants.issubset(provided_reactants):
                continue
            if not provided_reactants.issubset(rxn_reactants):
                continue
            if not rxn_catalysts.issubset(all_catalysts):
                continue

            # Check stoichiometric sufficiency: need at least a trace amount of each reactant
            sufficient = True
            for cid, coeff in rxn.reactants:
                chem = self._world.chemicals.get(cid)
                mw = chem.molecular_weight if chem else 100.0
                available_mol = reactant_ids.get(cid, 0.0) / mw
                if available_mol < 1e-9:
                    sufficient = False
                    break
            if sufficient:
                matches.append(rxn)

        return matches

    def estimate_cost(
        self,
        reactant_amounts: Dict[str, float],
        temperature_C: float,
        pressure_atm: float,
        duration_seconds: float,
        catalyst_names: Optional[List[str]] = None,
    ) -> Dict:
        name_to_id = {
            chem.name.lower(): cid
            for cid, chem in self._world.chemicals.items()
        }

        reactant_ids: Dict[str, float] = {}
        for name, grams in reactant_amounts.items():
            cid = name_to_id.get(name.lower())
            if cid is None:
                return {"success": False, "message": f"Unknown chemical: {name}"}
            reactant_ids[cid] = grams

        catalyst_ids: List[str] = []
        if catalyst_names:
            for cname in catalyst_names:
                cid = name_to_id.get(cname.lower())
                if cid:
                    catalyst_ids.append(cid)

        matching = self._find_matching_reactions(reactant_ids, catalyst_ids)
        if not matching:
            # Try to find a close match just for cost estimate
            return {"success": False, "message": "No matching reaction found for cost estimate."}

        rxn = max(matching, key=lambda r: abs(r.delta_G_kJ))
        cost = calculate_cost(
            rxn, self._world.chemicals, reactant_ids, temperature_C, pressure_atm, duration_seconds
        )
        cost["success"] = True
        cost["reaction_id"] = rxn.id
        return cost

    def get_transaction_log(self) -> List[Dict]:
        return list(self._transaction_log)

    # ------------------------------------------------------------------
    # Medicinal-pathway helpers
    # ------------------------------------------------------------------

    def _find_reaction_chains(
        self,
        target_id: str,
        max_routes: int = 5,
        max_steps: int = 4,
    ) -> List[List]:
        """Backward DFS from target_id; returns chains in *forward* (M1→target) order."""
        produces: Dict[str, List] = {}
        for rxn in self._world.reactions.values():
            for pid, _ in rxn.products:
                produces.setdefault(pid, []).append(rxn)

        chains: List[List] = []

        def dfs(needed: List[str], chain_rev: List, used: frozenset) -> None:
            if len(chains) >= max_routes:
                return
            non_m1 = [cid for cid in needed if self._world.chemicals[cid].layer > 1]
            if not non_m1:
                chains.append(list(chain_rev))
                return
            if len(chain_rev) >= max_steps:
                return
            resolve_id = non_m1[0]
            remaining = [cid for cid in needed if cid != resolve_id]
            for rxn in produces.get(resolve_id, []):
                if rxn.id in used:
                    continue
                new_needed = list(dict.fromkeys(remaining + [c for c, _ in rxn.reactants]))
                dfs(new_needed, chain_rev + [rxn], used | {rxn.id})

        dfs([target_id], [], frozenset())
        # chains are built target-first; reverse each to get M1→target forward order
        return [list(reversed(ch)) for ch in chains]

    def _optimal_temp_for_reaction(self, rxn) -> Tuple[float, float]:
        """Return (temp_C, duration_s) sized for ~95% approach to equilibrium.

        Solves Arrhenius for T such that k(T) = 0.01 s⁻¹  (gives k·t≈3 in 300 s),
        then computes exact duration for 95% conversion at that T.
        Both values are clamped to practical ranges.
        """
        A = 10.0 ** rxn.log_A_factor
        target_k = 0.01  # s⁻¹ — gives (1 − e⁻³) ≈ 95% in 300 s
        ratio = A / max(target_k, 1e-30)
        if ratio <= 1.0:
            T_K = 298.15
        else:
            T_K = rxn.activation_energy_kJ / (R_kJ * math.log(ratio))
        T_K = max(298.15, min(873.15, T_K))

        k_at_T = A * math.exp(-rxn.activation_energy_kJ / (R_kJ * T_K))
        # t such that 1 − exp(−k·t) = 0.95  →  t = −ln(0.05)/k ≈ 3/k
        duration_s = min(3600.0, max(60.0, 3.0 / max(k_at_T, 1e-30)))
        return round(T_K - 273.15, 1), round(duration_s, 1)

    def _build_pathway_steps(
        self,
        rxn_chain: List,
        per_m1_g: float,
    ) -> List[Dict]:
        """Build step dicts for evaluate_pathway from a Reaction chain.

        M1 reactants are set to *per_m1_g*; non-M1 intermediates are sized to
        whatever the previous step produced (via a lightweight pre-simulation).
        """
        virtual_pool: Dict[str, float] = {}
        steps = []
        for rxn in rxn_chain:
            reactant_ids: Dict[str, float] = {}
            reactant_names: Dict[str, float] = {}
            for cid, _ in rxn.reactants:
                chem = self._world.chemicals[cid]
                amt = per_m1_g if chem.layer == 1 else max(virtual_pool.get(cid, 0.0), 0.1)
                reactant_ids[cid] = amt
                reactant_names[chem.name] = amt

            T, dur = self._optimal_temp_for_reaction(rxn)
            P = 1.0

            # Pre-simulate to propagate outputs into virtual_pool for the next step
            sim = simulate_reaction(rxn, self._world.chemicals, reactant_ids, T, P, dur)
            for cid, consumed in sim["consumed_g"].items():
                virtual_pool[cid] = max(0.0, virtual_pool.get(cid, 0.0) - consumed)
            for cid, produced in sim["produced_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced

            steps.append({
                "reactant_amounts": reactant_names,
                "temperature_C": T,
                "pressure_atm": P,
                "duration_seconds": dur,
                "catalyst_names": [self._id_to_name(cid) for cid in rxn.catalysts],
            })
        return steps

    # ------------------------------------------------------------------
    # Pathway helpers
    # ------------------------------------------------------------------

    def _find_matching_reactions_with_pool(
        self,
        reactant_ids: Dict[str, float],
        catalyst_ids: List[str],
        virtual_pool: Dict[str, float],
    ) -> List:
        """Like _find_matching_reactions but uses virtual_pool for catalysts."""
        provided_reactants = set(reactant_ids.keys())
        all_catalysts = set(catalyst_ids) | {cid for cid, g in virtual_pool.items() if g > 1e-6}

        matches = []
        for rxn in self._world.reactions.values():
            if {cid for cid, _ in rxn.reactants} != provided_reactants:
                continue
            if not set(rxn.catalysts).issubset(all_catalysts):
                continue
            if all(
                reactant_ids.get(cid, 0.0) / (self._world.chemicals[cid].molecular_weight if cid in self._world.chemicals else 100.0) >= 1e-9
                for cid, _ in rxn.reactants
            ):
                matches.append(rxn)
        return matches

    def _pathway_nl_summary(
        self,
        summary: Dict,
        step_results: List[Dict],
        warnings: List[str],
    ) -> str:
        target_yield_g = summary["target_yield_g"]
        cost_per_gram = summary["cost_per_gram_target"]
        cpg_str = f"{cost_per_gram:.2f}" if isinstance(cost_per_gram, float) else cost_per_gram

        msg = generate_response(
            "pathway_summary",
            target=summary["target_compound"],
            num_steps=summary["num_steps"],
            yield_g=target_yield_g,
            efficiency=summary["mass_efficiency"],
            total_cost=summary["total_cost"],
            cost_per_gram=cpg_str,
            bottleneck=summary["bottleneck_step"],
            bottleneck_conv=summary["bottleneck_conversion"] or 0.0,
            atom_economy=summary["overall_atom_economy"],
            efficiency_rating=summary["efficiency_rating"],
        )
        if warnings:
            msg += f" Note: {warnings[0]}"
        return msg

    # ------------------------------------------------------------------
    # Public: evaluate_pathway
    # ------------------------------------------------------------------

    def evaluate_pathway(
        self,
        target_compound: str,
        steps: List[Dict],
    ) -> Dict:
        """Evaluate efficiency and cost of a multi-step synthesis pathway.

        Does NOT modify inventory — purely analytical.

        Parameters
        ----------
        target_compound : str
            Name of the desired final product.
        steps : list of dict
            Ordered reaction steps. Each step::

                {
                    "reactant_amounts": {"CompoundA": 10.0, "CompoundB": 5.0},
                    "temperature_C": 150.0,
                    "pressure_atm": 1.5,
                    "duration_seconds": 300.0,
                    "catalyst_names": ["CatalystX"],   # optional
                }

        Returns
        -------
        dict
            ``summary`` — aggregate metrics:
              - ``target_yield_g``: grams of target produced
              - ``mass_efficiency``: g target / g M1 input
              - ``cost_per_gram_target``: total cost / g target
              - ``overall_atom_economy``: product of per-step atom economies
              - ``bottleneck_step`` / ``bottleneck_conversion``
            ``step_results`` — per-step details (conversion, cost, products)
            ``warnings`` — supply-chain warnings (e.g. insufficient intermediate)
            ``virtual_pool_remaining`` — leftover compounds after the pathway
            ``message`` — natural-language summary
        """
        name_to_id = {c.name.lower(): cid for cid, c in self._world.chemicals.items()}

        target_id = name_to_id.get(target_compound.lower())
        if target_id is None:
            return {"success": False, "message": f"Unknown compound: '{target_compound}'"}
        if not steps:
            return {"success": False, "message": "Pathway must contain at least one step."}

        virtual_pool: Dict[str, float] = {}
        total_m1_cost = 0.0
        total_process_cost = 0.0
        total_time_s = 0.0
        step_results = []
        m1_totals: Dict[str, float] = {}
        warnings: List[str] = []

        for step_idx, step in enumerate(steps):
            step_num = step_idx + 1

            # Parse reactants
            step_reactant_ids: Dict[str, float] = {}
            for name, grams in step.get("reactant_amounts", {}).items():
                cid = name_to_id.get(name.lower())
                if cid is None:
                    return {"success": False, "message": f"Step {step_num}: unknown chemical '{name}'"}
                if grams <= 0:
                    return {"success": False, "message": f"Step {step_num}: amount for '{name}' must be positive"}
                step_reactant_ids[cid] = float(grams)

            if not step_reactant_ids:
                return {"success": False, "message": f"Step {step_num}: no reactants specified"}

            # Parse catalysts
            catalyst_ids: List[str] = []
            for cname in step.get("catalyst_names", []):
                cid = name_to_id.get(cname.lower())
                if cid is None:
                    return {"success": False, "message": f"Step {step_num}: unknown catalyst '{cname}'"}
                catalyst_ids.append(cid)

            T = float(step.get("temperature_C", 25.0))
            P = float(step.get("pressure_atm", 1.0))
            dur = float(step.get("duration_seconds", 60.0))

            # Find reaction — check virtual pool for catalysts first, then real inventory
            matching = self._find_matching_reactions_with_pool(step_reactant_ids, catalyst_ids, virtual_pool)
            if not matching:
                matching = self._find_matching_reactions(step_reactant_ids, catalyst_ids)
            if not matching:
                names = list(step.get("reactant_amounts", {}).keys())
                cats = step.get("catalyst_names", [])
                return {
                    "success": False,
                    "message": (
                        f"Step {step_num}: no reaction found for reactants {names}"
                        + (f" with catalysts {cats}" if cats else "") + "."
                    ),
                }

            rxn = max(matching, key=lambda r: abs(r.delta_G_kJ))

            # Warn if non-M1 reactants are under-supplied from the virtual pool
            for cid, needed in step_reactant_ids.items():
                chem = self._world.chemicals[cid]
                if chem.layer > 1:
                    available = virtual_pool.get(cid, 0.0)
                    if available < needed - 1e-6:
                        warnings.append(
                            f"Step {step_num}: {chem.name} needs {needed:.2f}g but only "
                            f"{available:.2f}g is available from earlier steps."
                        )

            # Accumulate M1 cost
            for cid, grams in step_reactant_ids.items():
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    m1_totals[cid] = m1_totals.get(cid, 0.0) + grams
                    total_m1_cost += chem.price_per_gram * grams

            # Simulate (no inventory side-effects)
            sim = simulate_reaction(rxn, self._world.chemicals, step_reactant_ids, T, P, dur)
            cost = calculate_cost(rxn, self._world.chemicals, step_reactant_ids, T, P, dur)
            total_process_cost += cost["total_cost"]
            total_time_s += dur

            # Propagate outputs through virtual pool
            for cid, consumed in sim["consumed_g"].items():
                virtual_pool[cid] = max(0.0, virtual_pool.get(cid, 0.0) - consumed)
            for cid, produced in sim["produced_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced
            for cid, produced in sim["byproduct_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced

            # Per-step atom economy: MW(desired products) / MW(all products incl. byproducts)
            desired_mw = sum(
                self._world.chemicals[pid].molecular_weight * c
                for pid, c in rxn.products if pid in self._world.chemicals
            )
            all_mw = desired_mw + sum(
                self._world.chemicals[pid].molecular_weight * c
                for pid, c in rxn.byproducts if pid in self._world.chemicals
            )
            step_ae = desired_mw / all_mw if all_mw > 0 else 1.0

            step_results.append({
                "step": step_num,
                "reaction_id": rxn.id,
                "temperature_C": T,
                "pressure_atm": P,
                "duration_s": dur,
                "conversion": round(sim["conversion"], 4),
                "reached_equilibrium": sim["reached_equilibrium"],
                "K_eq": round(sim["K_eq"], 4) if sim["K_eq"] < 1e6 else ">1e6",
                "k_eff": round(sim["k_eff"], 6),
                "products_g": {
                    self._id_to_name(cid): round(g, 4)
                    for cid, g in sim["produced_g"].items() if g > 1e-6
                },
                "atom_economy": round(step_ae, 4),
                "cost_breakdown": cost,
            })

        # Aggregate
        target_yield_g = virtual_pool.get(target_id, 0.0)
        total_m1_g = sum(m1_totals.values())
        total_cost = total_m1_cost + total_process_cost

        mass_efficiency = target_yield_g / total_m1_g if total_m1_g > 1e-9 else 0.0
        cost_per_gram: object = round(total_cost / target_yield_g, 2) if target_yield_g > 1e-9 else "N/A (no yield)"

        bottleneck = min(step_results, key=lambda s: s["conversion"]) if step_results else {}
        overall_ae = 1.0
        for s in step_results:
            overall_ae *= s["atom_economy"]

        if mass_efficiency > 0.30:
            rating = "excellent"
        elif mass_efficiency > 0.10:
            rating = "good"
        elif mass_efficiency > 0.05:
            rating = "moderate"
        elif mass_efficiency > 0.01:
            rating = "poor"
        else:
            rating = "very poor"

        summary = {
            "target_compound": target_compound,
            "num_steps": len(steps),
            "target_yield_g": round(target_yield_g, 4),
            "total_m1_input_g": round(total_m1_g, 4),
            "mass_efficiency": round(mass_efficiency, 4),
            "efficiency_rating": rating,
            "total_cost": round(total_cost, 2),
            "total_m1_cost": round(total_m1_cost, 2),
            "total_process_cost": round(total_process_cost, 2),
            "cost_per_gram_target": cost_per_gram,
            "total_time_seconds": round(total_time_s, 1),
            "overall_atom_economy": round(overall_ae, 4),
            "bottleneck_step": bottleneck.get("step"),
            "bottleneck_conversion": bottleneck.get("conversion"),
            "m1_chemicals_used": {
                self._id_to_name(cid): round(g, 4) for cid, g in m1_totals.items()
            },
        }

        return {
            "success": True,
            "message": self._pathway_nl_summary(summary, step_results, warnings),
            "summary": summary,
            "step_results": step_results,
            "warnings": warnings,
            "virtual_pool_remaining": {
                self._id_to_name(cid): round(g, 4)
                for cid, g in virtual_pool.items() if g > 1e-6
            },
        }

    # ------------------------------------------------------------------
    # Public: find_synthesis_routes
    # ------------------------------------------------------------------

    def find_synthesis_routes(
        self,
        target_compound: str,
        max_routes: int = 5,
        max_steps: int = 4,
    ) -> Dict:
        """Discover synthesis routes from M1 chemicals to *target_compound*.

        Performs a backward DFS through the reaction network.  Products of
        intermediate reactions are shown as ``"unknown compound"`` unless they
        have already been synthesised in this session.

        Parameters
        ----------
        target_compound : str
            Name of the desired product.
        max_routes : int
            Stop after this many distinct routes are found.
        max_steps : int
            Maximum reaction steps allowed per route (avoids combinatorial blow-up).

        Returns
        -------
        dict
            ``routes`` — list of route dicts, each containing:
              - ``num_steps``, ``m1_starting_materials``
              - ``steps`` — ordered list with ``reactants_needed``,
                ``catalysts_needed``, ``product``, ``conditions_hint``
            ``message`` — natural-language summary
        """
        name_to_id = {c.name.lower(): cid for cid, c in self._world.chemicals.items()}
        target_id = name_to_id.get(target_compound.lower())
        if target_id is None:
            return {"success": False, "message": f"Unknown compound: '{target_compound}'"}

        target_chem = self._world.chemicals[target_id]
        if target_chem.layer == 1:
            return {
                "success": True,
                "target": target_compound,
                "routes": [{
                    "route_id": 0,
                    "num_steps": 0,
                    "m1_starting_materials": [target_compound],
                    "steps": [],
                    "message": f"{target_compound} is a base chemical available for direct purchase.",
                }],
                "message": f"{target_compound} is a layer-1 chemical available for direct purchase.",
            }

        # Index: chem_id -> reactions that produce it
        produces: Dict[str, List] = {}
        for rxn in self._world.reactions.values():
            for pid, _ in rxn.products:
                produces.setdefault(pid, []).append(rxn)

        routes: List[Tuple[List[Dict], List[str]]] = []
        # each entry: (steps_reversed, m1_ids_at_terminal)

        def dfs(
            needed: List[str],          # chem IDs we still need to source
            steps_so_far: List[Dict],   # steps collected so far (reverse order)
            used_rxns: frozenset,       # reaction IDs already in route (no repeats)
        ) -> None:
            if len(routes) >= max_routes:
                return

            non_m1 = [cid for cid in needed if self._world.chemicals[cid].layer > 1]
            if not non_m1:
                # All needs are M1 — route is complete
                routes.append((list(steps_so_far), list(needed)))
                return
            if len(steps_so_far) >= max_steps:
                return

            # Resolve the first unresolved non-M1 compound
            resolve_id = non_m1[0]
            remaining = [cid for cid in needed if cid != resolve_id]

            for rxn in produces.get(resolve_id, []):
                if rxn.id in used_rxns:
                    continue
                rxn_reactant_ids = [cid for cid, _ in rxn.reactants]
                step_info = {
                    "reaction_id": rxn.id,
                    "reactants_needed": [self._id_to_name(cid) for cid in rxn_reactant_ids],
                    "catalysts_needed": [self._id_to_name(cid) for cid in rxn.catalysts],
                    "product_id": resolve_id,
                    "product": (
                        self._id_to_name(resolve_id)
                        if resolve_id in self._synthesized
                        else "unknown compound"
                    ),
                    "conditions_hint": self._conditions_hint(rxn),
                }
                # Merge: replace resolved_id with its reactants (deduplicate)
                new_needed = list(dict.fromkeys(remaining + rxn_reactant_ids))
                dfs(new_needed, steps_so_far + [step_info], used_rxns | {rxn.id})

        dfs([target_id], [], frozenset())

        if not routes:
            msg = generate_response("route_not_found", target=target_compound, max_steps=max_steps)
            return {"success": True, "target": target_compound, "routes": [], "message": msg}

        formatted: List[Dict] = []
        for i, (steps_rev, m1_ids) in enumerate(routes):
            # steps_rev is in backward order (target first); reverse for forward presentation
            steps_fwd = list(reversed(steps_rev))
            m1_names = sorted({self._id_to_name(cid) for cid in m1_ids})

            clean_steps = []
            for j, s in enumerate(steps_fwd):
                clean_steps.append({
                    "step": j + 1,
                    "reactants_needed": s["reactants_needed"],
                    "catalysts_needed": s["catalysts_needed"],
                    "product": s["product"],
                    "conditions_hint": s["conditions_hint"],
                })

            formatted.append({
                "route_id": i,
                "num_steps": len(steps_fwd),
                "m1_starting_materials": m1_names,
                "steps": clean_steps,
                "message": (
                    f"Route {i}: {len(steps_fwd)} step(s) to {target_compound}. "
                    f"M1 starting materials: {', '.join(m1_names)}."
                ),
            })

        min_steps = min(r["num_steps"] for r in formatted)
        m1_for_shortest = next(r["m1_starting_materials"] for r in formatted if r["num_steps"] == min_steps)
        msg = generate_response(
            "route_found",
            n=len(formatted),
            target=target_compound,
            min_steps=min_steps,
            m1_list=", ".join(m1_for_shortest),
        )
        return {"success": True, "target": target_compound, "routes": formatted, "message": msg}

    # ------------------------------------------------------------------
    # Public: find_cheapest_medicinal_pathway
    # ------------------------------------------------------------------

    def find_cheapest_medicinal_pathway(
        self,
        min_medicinal_value: float = 3.0,
        max_toxicity: float = 4.0,
        per_m1_g: float = 10.0,
        max_routes_per_target: int = 3,
        max_steps: int = 4,
    ) -> Dict:
        """Find the synthesis pathway with the lowest cost per unit medicinal value.

        Scans every compound in the world that satisfies the toxicity and medicinal
        value constraints, enumerates synthesis routes for each, simulates each
        route with auto-scaled amounts and heuristically optimal temperatures, then
        ranks results by::

            cost_per_medicinal_unit = total_cost / (target_yield_g × medicinal_value)

        This metric represents *credits per gram of therapeutically effective product*.

        Parameters
        ----------
        min_medicinal_value : float
            Minimum true medicinal value (0–10).  Default 3.0.
        max_toxicity : float
            Maximum base toxicity allowed (0–10).  Default 4.0.
        per_m1_g : float
            Reference grams of each M1 reactant per step; used to normalise
            cost comparisons across routes.  Default 10.0.
        max_routes_per_target : int
            How many distinct routes to evaluate per qualifying compound.
        max_steps : int
            Maximum depth of the backward route search.

        Returns
        -------
        dict
            ``best_pathway``  — the optimal candidate (see below)
            ``all_candidates`` — all evaluated candidates, sorted best-first
            ``num_qualifying_compounds`` — how many compounds passed the filter
            ``num_evaluated_routes`` — total routes evaluated
            ``message`` — natural-language summary

        Each candidate contains:
            ``target``, ``medicinal_value``, ``base_toxicity``,
            ``cost_per_medicinal_unit``, ``route`` (steps with conditions),
            ``pathway_summary`` (yield, efficiency, total cost, cost/g),
            ``step_results`` (per-step details from evaluate_pathway),
            ``warnings``.
        """
        qualifying = [
            chem for chem in self._world.chemicals.values()
            if chem.medicinal_value >= min_medicinal_value
            and chem.base_toxicity <= max_toxicity
        ]

        if not qualifying:
            return {
                "success": True,
                "found": False,
                "message": (
                    f"No compound in this world satisfies medicinal ≥ {min_medicinal_value} "
                    f"and toxicity ≤ {max_toxicity}. Try relaxing the constraints."
                ),
                "all_candidates": [],
                "num_qualifying_compounds": 0,
                "num_evaluated_routes": 0,
            }

        candidates = []

        for target_chem in qualifying:
            # Layer-1 compounds: no synthesis; direct purchase cost
            if target_chem.layer == 1:
                cpg = target_chem.price_per_gram
                cpm = cpg / target_chem.medicinal_value
                candidates.append({
                    "target": target_chem.name,
                    "medicinal_value": round(target_chem.medicinal_value, 3),
                    "base_toxicity": round(target_chem.base_toxicity, 3),
                    "cost_per_medicinal_unit": round(cpm, 4),
                    "route": {
                        "num_steps": 0,
                        "m1_starting_materials": [target_chem.name],
                        "steps": [],
                        "note": "Direct purchase — no synthesis required.",
                    },
                    "pathway_summary": {
                        "target_yield_g": 1.0,
                        "mass_efficiency": 1.0,
                        "efficiency_rating": "excellent",
                        "total_cost": round(cpg, 2),
                        "cost_per_gram_target": round(cpg, 2),
                        "total_m1_input_g": 1.0,
                        "total_time_seconds": 0.0,
                    },
                    "step_results": [],
                    "warnings": [],
                })
                continue

            # A route may need up to num_layers * 2 steps when a product requires
            # multiple intermediates each produced by a separate sub-path.
            effective_steps = max(max_steps, self._world.num_layers * 2)
            chains = self._find_reaction_chains(target_chem.id, max_routes_per_target, effective_steps)
            for chain in chains:
                steps = self._build_pathway_steps(chain, per_m1_g)
                eval_result = self.evaluate_pathway(target_chem.name, steps)
                if not eval_result["success"]:
                    continue

                summary = eval_result["summary"]
                yield_g = summary["target_yield_g"]
                total_cost = summary["total_cost"]
                if yield_g < 1e-6:
                    continue

                cost_per_gram = total_cost / yield_g
                cost_per_medicinal_unit = cost_per_gram / target_chem.medicinal_value

                # Build human-readable route steps
                route_steps = []
                for i, (rxn, step) in enumerate(zip(chain, steps)):
                    route_steps.append({
                        "step": i + 1,
                        "reactants": list(step["reactant_amounts"].keys()),
                        "catalysts": step["catalyst_names"],
                        "temperature_C": step["temperature_C"],
                        "pressure_atm": step["pressure_atm"],
                        "duration_seconds": step["duration_seconds"],
                        "conditions_hint": self._conditions_hint(rxn),
                    })

                m1_ids = {
                    cid
                    for rxn in chain
                    for cid, _ in rxn.reactants
                    if self._world.chemicals[cid].layer == 1
                }

                candidates.append({
                    "target": target_chem.name,
                    "medicinal_value": round(target_chem.medicinal_value, 3),
                    "base_toxicity": round(target_chem.base_toxicity, 3),
                    "cost_per_medicinal_unit": round(cost_per_medicinal_unit, 4),
                    "route": {
                        "num_steps": len(chain),
                        "m1_starting_materials": sorted(self._id_to_name(cid) for cid in m1_ids),
                        "steps": route_steps,
                    },
                    "pathway_summary": {
                        "target_yield_g": round(yield_g, 4),
                        "mass_efficiency": summary["mass_efficiency"],
                        "efficiency_rating": summary["efficiency_rating"],
                        "total_cost": round(total_cost, 2),
                        "cost_per_gram_target": round(cost_per_gram, 2),
                        "total_m1_input_g": summary["total_m1_input_g"],
                        "total_time_seconds": summary["total_time_seconds"],
                    },
                    "step_results": eval_result["step_results"],
                    "warnings": eval_result["warnings"],
                })

        if not candidates:
            return {
                "success": True,
                "found": False,
                "message": (
                    f"No feasible synthesis route found for compounds satisfying "
                    f"medicinal ≥ {min_medicinal_value}, toxicity ≤ {max_toxicity} "
                    f"within {max_steps} steps."
                ),
                "all_candidates": [],
                "num_qualifying_compounds": len(qualifying),
                "num_evaluated_routes": 0,
            }

        candidates.sort(key=lambda x: x["cost_per_medicinal_unit"])
        best = candidates[0]

        tox_label = _toxicity_level(best["base_toxicity"])
        msg = (
            f"Most cost-effective medicinal synthesis: {best['target']} "
            f"(medicinal value {best['medicinal_value']:.2f}/10, toxicity {best['base_toxicity']:.1f}/10 — {tox_label}). "
            f"Cost per medicinal unit: {best['cost_per_medicinal_unit']:.2f} cr. "
            f"{best['route']['num_steps']}-step route from "
            f"{', '.join(best['route']['m1_starting_materials'])}. "
            f"Reference yield {best['pathway_summary']['target_yield_g']:.3f}g "
            f"at {best['pathway_summary']['cost_per_gram_target']:.2f} cr/g "
            f"(efficiency: {best['pathway_summary']['efficiency_rating']}). "
            f"Total reference cost: {best['pathway_summary']['total_cost']:.2f} credits. "
            f"Evaluated {len(candidates)} candidate route(s) across {len(qualifying)} qualifying compound(s)."
        )

        return {
            "success": True,
            "found": True,
            "message": msg,
            "best_pathway": best,
            "all_candidates": candidates,
            "num_qualifying_compounds": len(qualifying),
            "num_evaluated_routes": len(candidates),
        }
