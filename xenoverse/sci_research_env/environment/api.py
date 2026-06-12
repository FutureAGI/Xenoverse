from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..world_gen.models import Chemical, Reaction, World
from .simulator import simulate_reaction, simulate_chain_reaction, state_at, R_kJ
from .cost_model import calculate_cost, estimate_reaction_cost
from .templates import generate_response, _medicinal_hint, _toxicity_note, PATHWAY_EFFICIENCY_NOTE


def _round_sig(value: float, sig: int = 4) -> float:
    """Round to significant digits — preserves info for small values."""
    if value == 0:
        return 0.0
    import math as _m
    digits = sig - int(_m.floor(_m.log10(abs(value)))) - 1
    return round(value, max(digits, 0))


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

    def list_equipment(self) -> Dict:
        catalog = self._world.equipment
        result = {}
        for name, spec in catalog.items():
            result[name] = {
                "description": spec["description"],
                "vessel_type": spec["vessel_type"],
                "thermal_mode": spec["thermal_mode"],
                "max_pressure_atm": spec["max_pressure_atm"],
                "max_temp_C": spec["max_temp_C"],
                "min_temp_C": spec["min_temp_C"],
                "max_capacity_g": spec.get("max_capacity_g", 500.0),
                "base_cost_per_hour": spec["base_cost_per_hour"],
            }
        return result

    def list_purchasable(self) -> Dict:
        result = {}
        for cid, chem in self._world.chemicals.items():
            if chem.layer == 1:
                phase = state_at(chem, 25.0, 1.0)
                entry = {
                    "name": chem.name,
                    "price_per_gram": round(chem.price_per_gram, 4),
                    "state_at_room_temp": phase,
                    "molecular_weight_approx": _approx_mw(chem.molecular_weight),
                }
                if chem.is_solvent:
                    entry["role"] = "solvent"
                result[chem.name] = entry
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
            if chem.layer > 1 and cid not in self._synthesized:
                continue
            phase = state_at(chem, 25.0, 1.0)
            entry = {
                "name": chem.name,
                "amount_g": round(grams, 4),
                "state_at_room_temp": phase,
            }
            if chem.layer == 1:
                entry["purchasable"] = True
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

        result = {
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
        if chem.is_solvent:
            result["role"] = "solvent"
        if chem.solubility:
            solubility_info = {}
            for sid, max_g in chem.solubility.items():
                solvent_chem = self._world.chemicals.get(sid)
                if solvent_chem:
                    solubility_info[solvent_chem.name] = round(max_g, 2)
            if solubility_info:
                result["solubility_g_per_100mL"] = solubility_info
        return result

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

    def _compute_purification_cost(self, reactant_ids: Dict[str, float], temperature_C: float, pressure_atm: float) -> float:
        from .cost_model import compute_purification_cost
        return compute_purification_cost(
            reactant_ids, self._world.chemicals, temperature_C, pressure_atm,
            detection_threshold=self.DETECTION_THRESHOLD_G,
        )

    DETECTION_THRESHOLD_G = 0.001

    def perform_reaction(
        self,
        reactant_amounts: Dict[str, float],
        temperature_C: float,
        pressure_atm: float,
        duration_seconds: float,
        equipment: Optional[str] = None,
        heating_rate_C_per_s: float = 0.0,
        vessel_volume_L: float = 1.0,
        recover_on_failure: bool = False,
        recover_reactants: bool = False,
    ) -> Dict:
        name_to_id = {
            chem.name.lower(): cid
            for cid, chem in self._world.chemicals.items()
        }

        all_amounts: Dict[str, float] = {}
        for name, grams in reactant_amounts.items():
            cid = name_to_id.get(name.lower())
            if cid is None:
                return {"success": False, "message": f"Unknown chemical: {name}"}
            all_amounts[cid] = grams

        for cid, needed in all_amounts.items():
            available = self._inventory.get(cid, 0.0)
            if available < needed - 1e-4:
                return {
                    "success": False,
                    "_no_time_loss": True,
                    "message": f"Insufficient {self._id_to_name(cid)}: need {needed:.4f}g, have {available:.4f}g",
                }
            if needed > available:
                all_amounts[cid] = available

        total_mass_g = sum(all_amounts.values())
        if total_mass_g < 1.0:
            return {
                "success": False,
                "_no_time_loss": True,
                "message": (
                    f"Total reactant mass {total_mass_g:.2f}g is below the minimum of 1g required "
                    f"to perform or observe a reaction. Increase amounts."
                ),
            }

        catalog = self._world.equipment
        if equipment and equipment not in catalog:
            return {"success": False, "_no_time_loss": True, "message": f"Unknown equipment: {equipment}. Available: {list(catalog.keys())}"}

        equip_name = equipment or "open_beaker"
        equip_spec = catalog[equip_name]
        max_capacity = equip_spec.get("max_capacity_g", 500.0)
        if total_mass_g > max_capacity:
            return {
                "success": False,
                "_no_time_loss": True,
                "message": (
                    f"Total mass {total_mass_g:.1f}g exceeds {equip_name} capacity of {max_capacity:.0f}g. "
                    f"Reduce amounts or use larger equipment."
                ),
            }

        max_T = equip_spec.get("max_temp_C", 5000.0)
        min_T = equip_spec.get("min_temp_C", -273.0)
        max_P = equip_spec.get("max_pressure_atm", 1000.0)
        if temperature_C > max_T or temperature_C < min_T:
            return {
                "success": False,
                "_no_time_loss": True,
                "message": (
                    f"Temperature {temperature_C:.1f}°C is outside {equip_name} range "
                    f"[{min_T:.0f}, {max_T:.0f}]°C. Adjust temperature or use different equipment."
                ),
            }
        if pressure_atm > max_P:
            return {
                "success": False,
                "_no_time_loss": True,
                "message": (
                    f"Pressure {pressure_atm:.2f} atm exceeds {equip_name} limit of {max_P:.0f} atm. "
                    f"Reduce pressure or use different equipment."
                ),
            }

        from .simulator import _find_applicable_reactions, _find_common_solvent, _dissolved_fraction

        dissolution_observations = self._compute_dissolution_observations(
            all_amounts, temperature_C, pressure_atm
        )

        applicable = _find_applicable_reactions(all_amounts, self._world.reactions)
        if not applicable:
            for cid, grams in all_amounts.items():
                self._inventory[cid] = max(0.0, self._inventory.get(cid, 0.0) - grams)

            purification_cost = 0.0
            if recover_on_failure:
                purification_cost = self._compute_purification_cost(all_amounts, temperature_C, pressure_atm)
                for cid, grams in all_amounts.items():
                    self._inventory[cid] = self._inventory.get(cid, 0.0) + grams

            lost_names = {self._id_to_name(cid): round(g, 2) for cid, g in all_amounts.items()}
            self._transaction_log.append({
                "type": "failed_reaction",
                "reactants_consumed": lost_names,
                "recovered": recover_on_failure,
                "purification_cost": purification_cost,
            })

            msg = generate_response("reaction_fail")
            if recover_on_failure:
                msg += (
                    f" Materials recovered via purification (cost: {purification_cost:.2f} credits). "
                    f"Materials returned to inventory."
                )
            else:
                msg += (
                    f" All materials were lost in the failed attempt. "
                    f"Lost: {lost_names}. "
                    f"Tip: set recover_on_failure=true to pay purification cost and recover materials."
                )

            no_rxn_result = {
                "gas_lost_g": {},
                "temperature_history": [],
                "consumed_g": {},
                "reactions_fired": {},
            }
            phenomena = self._generate_phenomena(
                all_amounts, no_rxn_result, temperature_C, pressure_atm,
                dissolution_observations, set(),
            )

            return {
                "success": False,
                "message": msg,
                "observations": phenomena or "No observable changes.",
                "reactants_lost": not recover_on_failure,
                "purification_cost": purification_cost,
                "dissolution": dissolution_observations or None,
            }

        result = simulate_chain_reaction(
            world=self._world,
            initial_amounts_g=all_amounts,
            temperature_C=temperature_C,
            pressure_atm=pressure_atm,
            duration_s=duration_seconds,
            equipment=equipment,
            heating_rate_C_per_s=heating_rate_C_per_s,
            vessel_volume_L=vessel_volume_L,
        )

        for cid, grams in all_amounts.items():
            self._inventory[cid] = max(0.0, self._inventory.get(cid, 0.0) - grams)

        if result.get("equipment_failure"):
            failure_reason = result.get("failure_reason", "Equipment limits exceeded")
            cost_info = {"reactant_cost": 0, "condition_cost": 0, "purification_cost": 0, "total_cost": 0}
            self._transaction_log.append({
                "type": "reaction",
                "reactants": {self._id_to_name(cid): g for cid, g in all_amounts.items()},
                "temperature_C": temperature_C,
                "pressure_atm": pressure_atm,
                "duration_s": duration_seconds,
                "equipment_failure": True,
                "failure_reason": failure_reason,
                "cost": cost_info,
            })
            return {
                "success": True,
                "message": (
                    f"EQUIPMENT FAILURE: {failure_reason}. "
                    f"All materials in the vessel were destroyed. "
                    f"Final temperature: {result['final_temperature_C']}°C, "
                    f"Final pressure: {result['final_pressure_atm']} atm."
                ),
                "equipment_failure": True,
                "failure_reason": failure_reason,
                "conversion": 0.0,
                "products_g": {},
                "byproducts_g": {},
                "reactants_recovered": None,
                "reactants_lost": {self._id_to_name(cid): round(g, 4) for cid, g in all_amounts.items()},
                "cost": cost_info,
                "final_temperature_C": result["final_temperature_C"],
                "final_pressure_atm": result["final_pressure_atm"],
                "equipment_used": result["equipment"],
            }

        final_pool = result["final_pool_g"]

        catalyst_ids_all: set = set()
        for rxn_id in result["reactions_fired"]:
            rxn = self._world.reactions[rxn_id]
            catalyst_ids_all.update(rxn.catalysts)

        leftover_g: Dict[str, float] = {}
        for cid, g in final_pool.items():
            if cid in all_amounts and cid not in result["net_produced_g"]:
                leftover_g[cid] = g
            elif cid in catalyst_ids_all:
                leftover_g[cid] = g

        observed_products: Dict[str, float] = {}
        observed_byproducts: Dict[str, float] = {}
        for cid, g in result["net_produced_g"].items():
            if g >= self.DETECTION_THRESHOLD_G:
                observed_products[cid] = g

        for cid, g in result["byproduct_g"].items():
            if g >= self.DETECTION_THRESHOLD_G and cid not in observed_products:
                observed_byproducts[cid] = g

        all_produced = dict(result["produced_g"])
        all_produced.update(result["byproduct_g"])
        unobserved_count = sum(
            1 for cid, g in all_produced.items()
            if 0 < g < self.DETECTION_THRESHOLD_G and cid not in observed_products and cid not in observed_byproducts
        )

        mixture_components = (
            len([g for g in leftover_g.values() if g >= self.DETECTION_THRESHOLD_G])
            + len(observed_products)
            + len(observed_byproducts)
        )

        from .cost_model import _phase_separation_factor, _purification_cost_per_component
        from .simulator import state_at as _state_at
        _mixture_phases = set()
        for cid in list(leftover_g.keys()) + list(observed_products.keys()) + list(observed_byproducts.keys()):
            if cid in self._world.chemicals:
                _mixture_phases.add(_state_at(self._world.chemicals[cid], temperature_C, pressure_atm))
        _pf = _phase_separation_factor(_mixture_phases)

        def _per_component_purification(grams: float) -> float:
            return _purification_cost_per_component(grams, mixture_components, _pf)

        product_purification_cost = 0.0
        for cid, g in observed_products.items():
            cost = _per_component_purification(g)
            product_purification_cost += cost
            self._inventory[cid] = self._inventory.get(cid, 0.0) + g
            self._synthesized.add(cid)

        for cid, g in observed_byproducts.items():
            cost = _per_component_purification(g)
            product_purification_cost += cost
            self._inventory[cid] = self._inventory.get(cid, 0.0) + g
            self._synthesized.add(cid)

        reactant_purification_cost = 0.0
        reactants_recovered = {}
        reactants_lost = {}
        for cid, g in leftover_g.items():
            if g < self.DETECTION_THRESHOLD_G:
                continue
            if recover_reactants:
                cost = _per_component_purification(g)
                reactant_purification_cost += cost
                self._inventory[cid] = self._inventory.get(cid, 0.0) + g
                reactants_recovered[self._id_to_name(cid)] = round(g, 4)
            else:
                reactants_lost[self._id_to_name(cid)] = round(g, 4)

        total_purification_cost = product_purification_cost + reactant_purification_cost

        n_observed = len(observed_products)
        total_product_mass = sum(observed_products.values())
        if n_observed > 0:
            products_str = (
                f"{n_observed} new substance(s) formed ({total_product_mass:.2f}g total)"
            )
            if unobserved_count > 0:
                products_str += f" (+ {unobserved_count} trace product(s) below detection limit)"
        elif unobserved_count > 0:
            products_str = f"{unobserved_count} trace product(s) below detection limit"
        else:
            products_str = "trace amounts below detection limit"

        chain_msg = ""
        if result["chain_reaction"]:
            num_rxns = len(result["reactions_fired"])
            chain_msg = f" Chain reaction detected: {num_rxns} distinct reactions occurred during the experiment."

        total_consumed_g = sum(result["consumed_g"].values())
        total_produced_g = sum(g for g in result["net_produced_g"].values() if g > 0)
        overall_conversion = total_consumed_g / max(sum(all_amounts.values()), 1e-9)
        overall_conversion = float(min(overall_conversion, 1.0))

        msg = generate_response(
            "reaction_success",
            duration=duration_seconds,
            temp=temperature_C,
            pressure=pressure_atm,
            conversion=overall_conversion,
            products_str=products_str,
            reached_equilibrium=result["converged"],
        )
        msg += chain_msg

        gas_escaped = {
            self._id_to_name(cid): round(g, 4)
            for cid, g in result.get("gas_lost_g", {}).items()
            if g >= self.DETECTION_THRESHOLD_G
        }
        if gas_escaped:
            msg += f" WARNING: Gaseous products escaped from open vessel: {gas_escaped}."

        if reactants_lost:
            msg += f" Unreacted materials lost in mixture: {reactants_lost}."
        if reactants_recovered:
            msg += f" Unreacted materials recovered via purification: {reactants_recovered}."

        if not result["reactions_fired"]:
            cost_info = {"reactant_cost": 0, "condition_cost": 0, "purification_cost": round(total_purification_cost, 2), "total_cost": round(total_purification_cost, 2)}
        else:
            primary_rxn_id = max(result["reactions_fired"], key=result["reactions_fired"].get)
            primary_rxn = self._world.reactions[primary_rxn_id]
            rxn_reactant_ids = {cid for cid, _ in primary_rxn.reactants}
            reactant_ids_for_cost: Dict[str, float] = {
                cid: g for cid, g in all_amounts.items() if cid not in catalyst_ids_all
            }

            cost_info = calculate_cost(
                primary_rxn, self._world.chemicals, reactant_ids_for_cost,
                temperature_C, pressure_atm, duration_seconds,
                self._world.cost_params, equipment=equipment,
                equipment_catalog=self._world.equipment,
            )
            estimated_purif = cost_info["purification_cost"]
            cost_info["purification_cost"] = round(total_purification_cost, 2)
            cost_info["total_cost"] = round(float(cost_info["total_cost"]) - estimated_purif + total_purification_cost, 2)

        log_entry = {
            "type": "reaction",
            "reactants": {self._id_to_name(cid): g for cid, g in all_amounts.items() if cid not in catalyst_ids_all},
            "catalysts": {self._id_to_name(cid): round(all_amounts.get(cid, 0.0), 4) for cid in catalyst_ids_all if all_amounts.get(cid, 0.0) > 0},
            "temperature_C": temperature_C,
            "pressure_atm": pressure_atm,
            "duration_s": duration_seconds,
            "conversion": round(overall_conversion, 4),
            "chain_reaction": result["chain_reaction"],
            "reactions_count": len(result["reactions_fired"]),
            "products_produced_g": {
                self._id_to_name(cid): round(g, 4) for cid, g in observed_products.items()
            },
            "reactants_recovered": reactants_recovered if recover_reactants else None,
            "reactants_lost": reactants_lost if reactants_lost else None,
            "unobserved_trace_products": unobserved_count,
            "cost": cost_info,
        }
        self._transaction_log.append(log_entry)

        phenomena = self._generate_phenomena(
            all_amounts, result, temperature_C, pressure_atm,
            dissolution_observations, catalyst_ids_all,
        )

        return {
            "success": True,
            "message": msg,
            "observations": phenomena or "No observable changes.",
            "conversion": round(overall_conversion, 4),
            "chain_reaction": result["chain_reaction"],
            "reactions_count": len(result["reactions_fired"]),
            "num_products_formed": n_observed,
            "total_product_mass_g": round(total_product_mass, 4),
            "num_byproducts_formed": len(observed_byproducts),
            "total_byproduct_mass_g": round(sum(observed_byproducts.values()), 4),
            "reactants_recovered": reactants_recovered if recover_reactants else None,
            "reactants_lost": reactants_lost if reactants_lost else None,
            "unobserved_trace_products": unobserved_count,
            "purification_cost": round(total_purification_cost, 2),
            "cost": cost_info,
            "final_temperature_C": result["final_temperature_C"],
            "final_pressure_atm": result["final_pressure_atm"],
            "gas_escaped_g": round(sum(
                g for g in result.get("gas_lost_g", {}).values()
                if g >= self.DETECTION_THRESHOLD_G
            ), 4) or None,
            "dissolution": dissolution_observations or None,
            "equipment_used": result["equipment"],
            "note": "Use get_inventory to see isolated products. Use analyze_compound to learn their properties.",
            "_products_g": {self._id_to_name(cid): round(g, 4) for cid, g in observed_products.items()},
        }

    def _generate_phenomena(
        self,
        all_amounts: Dict[str, float],
        result: Dict,
        temperature_C: float,
        pressure_atm: float,
        dissolution_observations: Dict,
        catalyst_ids: set,
    ) -> List[str]:
        from .simulator import state_at
        chemicals = self._world.chemicals
        phenomena = []

        for cid, g in all_amounts.items():
            if g < 0.01 or cid not in chemicals:
                continue
            chem = chemicals[cid]
            name = self._id_to_name(cid)
            phase = state_at(chem, temperature_C, pressure_atm)
            initial_phase = state_at(chem, 25.0, 1.0)

            if phase == "gas" and initial_phase != "gas":
                bp_adj = chem.boiling_point + 10 * __import__("numpy").log(max(0.01, pressure_atm))
                if temperature_C > bp_adj + 30:
                    phenomena.append(
                        f"{name} went into vigorous ebullition and completely evaporated upon contact with the heated vessel."
                    )
                elif temperature_C > bp_adj:
                    phenomena.append(
                        f"{name} began boiling and gradually evaporated at the reaction temperature."
                    )
            elif phase == "liquid" and initial_phase == "solid":
                phenomena.append(f"{name} melted into a liquid at the reaction temperature.")
            elif phase == "solid" and initial_phase == "liquid":
                phenomena.append(f"{name} solidified at the reaction temperature.")

        for solvent_name, obs in dissolution_observations.items():
            dissolved = obs.get("dissolved_g", {})
            undissolved = obs.get("undissolved_g", {})
            for chem_name, g in dissolved.items():
                if chem_name in undissolved:
                    und_g = undissolved[chem_name]
                    total = g + und_g
                    frac = g / total if total > 0 else 0
                    if frac < 0.3:
                        phenomena.append(
                            f"Observed {chem_name} dissolving only slightly in {solvent_name}; "
                            f"the bulk of the material ({und_g:.2f}g) settled at the bottom of the vessel."
                        )
                    elif frac < 0.7:
                        phenomena.append(
                            f"Observed {chem_name} partially dissolving in {solvent_name} — "
                            f"approximately {g:.2f}g went into solution while {und_g:.2f}g remained "
                            f"as suspended particles in the mixture."
                        )
                    else:
                        phenomena.append(
                            f"Observed {chem_name} dissolving almost completely in {solvent_name}, "
                            f"with only a small residue ({und_g:.2f}g) remaining undissolved."
                        )
                else:
                    phenomena.append(
                        f"Observed {chem_name} dissolving completely in {solvent_name}, "
                        f"forming a clear homogeneous solution."
                    )
            for chem_name, g in undissolved.items():
                if chem_name not in dissolved:
                    phenomena.append(
                        f"Observed {chem_name} refusing to dissolve in {solvent_name}; "
                        f"the material ({g:.2f}g) remained as a separate phase in the vessel."
                    )

        gas_lost = result.get("gas_lost_g", {})
        gas_lost_from_reactants = 0.0
        gas_lost_from_products = 0.0
        for cid, g in gas_lost.items():
            if g >= self.DETECTION_THRESHOLD_G and cid in chemicals:
                if cid in all_amounts:
                    name = self._id_to_name(cid)
                    initial_g = all_amounts.get(cid, 0.0)
                    ratio = g / max(initial_g, g, 1e-9)
                    if ratio > 0.9:
                        phenomena.append(
                            f"{name} escaped entirely as gas from the open vessel."
                        )
                    elif ratio > 0.3:
                        phenomena.append(
                            f"A significant portion of {name} ({g:.2f}g) escaped as gas from the vessel."
                        )
                    else:
                        phenomena.append(
                            f"Small bubbles of {name} were observed escaping from the vessel surface."
                        )
                    gas_lost_from_reactants += g
                else:
                    gas_lost_from_products += g
        if gas_lost_from_products > 0.01:
            phenomena.append(
                f"Gaseous products ({gas_lost_from_products:.2f}g) escaped from the open vessel and were lost."
            )

        temp_history = result.get("temperature_history", [])
        if len(temp_history) >= 2:
            t_start = temp_history[0]["temperature_C"]
            t_end = temp_history[-1]["temperature_C"]
            delta = t_end - t_start
            if delta > 50:
                phenomena.append(
                    f"The mixture temperature rose sharply from {t_start:.0f}°C to {t_end:.0f}°C "
                    f"(strongly exothermic reaction)."
                )
            elif delta > 10:
                phenomena.append(
                    f"The mixture warmed noticeably from {t_start:.0f}°C to {t_end:.0f}°C "
                    f"(exothermic reaction)."
                )
            elif delta < -50:
                phenomena.append(
                    f"The mixture cooled dramatically from {t_start:.0f}°C to {t_end:.0f}°C "
                    f"(strongly endothermic reaction)."
                )
            elif delta < -10:
                phenomena.append(
                    f"The mixture cooled noticeably from {t_start:.0f}°C to {t_end:.0f}°C "
                    f"(endothermic reaction)."
                )

        p_start = result.get("final_pressure_atm", pressure_atm)
        if result.get("temperature_history"):
            p_start_val = result["temperature_history"][0].get("pressure_atm", pressure_atm)
            p_end_val = result.get("final_pressure_atm", pressure_atm)
            if p_end_val > p_start_val * 1.5:
                phenomena.append(
                    f"Pressure in the vessel increased significantly "
                    f"from {p_start_val:.2f} to {p_end_val:.2f} atm due to gas generation."
                )

        conversion = sum(result.get("consumed_g", {}).values()) / max(sum(all_amounts.values()), 1e-9)
        if conversion > 0.95 and result.get("reactions_fired"):
            phenomena.append("The reaction went to near-completion; virtually all reactants were consumed.")
        elif conversion > 0.5 and result.get("reactions_fired"):
            phenomena.append("A moderate amount of reactants were consumed during the reaction.")
        elif conversion > 0.01 and result.get("reactions_fired"):
            phenomena.append("Only a small fraction of the reactants were consumed; the reaction progressed slowly.")

        net_produced = result.get("net_produced_g", {})
        products_formed = {cid: g for cid, g in net_produced.items() if g >= self.DETECTION_THRESHOLD_G}
        if products_formed:
            total_g = sum(products_formed.values())
            n_products = len(products_formed)
            product_phases = set()
            for cid in products_formed:
                if cid in chemicals:
                    product_phases.add(state_at(chemicals[cid], temperature_C, pressure_atm))

            if n_products == 1:
                phase_desc = next(iter(product_phases)) if product_phases else "unknown"
                if total_g > 5.0:
                    phenomena.append(
                        f"A substantial amount of a new {phase_desc} substance ({total_g:.2f}g) formed in the vessel."
                    )
                elif total_g > 0.5:
                    phenomena.append(
                        f"A new {phase_desc} substance ({total_g:.2f}g) appeared in the reaction mixture."
                    )
                else:
                    phenomena.append(
                        f"A small amount of a new {phase_desc} substance ({total_g:.3f}g) was detected."
                    )
            else:
                phase_list = sorted(product_phases)
                phase_desc = "/".join(phase_list) if phase_list else "unknown"
                if total_g > 5.0:
                    phenomena.append(
                        f"Multiple new substances ({n_products} distinct products, {total_g:.2f}g total) "
                        f"formed in the vessel. Phases observed: {phase_desc}."
                    )
                elif total_g > 0.5:
                    phenomena.append(
                        f"Several new substances ({n_products} products, {total_g:.2f}g total) "
                        f"appeared in the mixture ({phase_desc} phases)."
                    )
                else:
                    phenomena.append(
                        f"Trace amounts of {n_products} new substances ({total_g:.3f}g total) were detected."
                    )

            phenomena.append(
                "Products have been isolated via purification. "
                "Check inventory for new compounds; use analyze_compound to determine their properties."
            )

        return phenomena

    def _compute_dissolution_observations(
        self,
        pool: Dict[str, float],
        temperature_C: float,
        pressure_atm: float,
    ) -> Dict[str, Any]:
        from .simulator import _dissolved_fraction, state_at
        chemicals = self._world.chemicals
        solvents_in_pool = [
            cid for cid, g in pool.items()
            if g > 1e-9 and cid in chemicals and chemicals[cid].is_solvent
            and state_at(chemicals[cid], temperature_C, pressure_atm) == "liquid"
        ]
        if not solvents_in_pool:
            return {}

        non_solvents_in_pool = [
            cid for cid, g in pool.items()
            if g > 1e-9 and cid in chemicals and not chemicals[cid].is_solvent
        ]
        if not non_solvents_in_pool:
            return {}

        observations = {}
        for sid in solvents_in_pool:
            solvent_name = self._id_to_name(sid)
            dissolved = {}
            undissolved = {}
            for cid in non_solvents_in_pool:
                frac = _dissolved_fraction(cid, sid, pool, chemicals)
                chem_name = self._id_to_name(cid)
                g = pool.get(cid, 0.0)
                if frac >= 0.99:
                    dissolved[chem_name] = round(g, 3)
                elif frac > 0.01:
                    dissolved[chem_name] = round(g * frac, 3)
                    undissolved[chem_name] = round(g * (1.0 - frac), 3)
                else:
                    undissolved[chem_name] = round(g, 3)
            entry = {}
            if dissolved:
                entry["dissolved_g"] = dissolved
            if undissolved:
                entry["undissolved_g"] = undissolved
            if entry:
                observations[solvent_name] = entry
        return observations

    def _find_matching_reactions(
        self,
        all_amounts: Dict[str, float],
        subset_match: bool = False,
    ) -> List[Reaction]:
        provided_ids = set(all_amounts.keys())

        matches = []
        for rxn in self._world.reactions.values():
            rxn_reactants = {cid for cid, _ in rxn.reactants}
            rxn_catalysts = set(rxn.catalysts)
            rxn_all_needed = rxn_reactants | rxn_catalysts

            if not rxn_all_needed.issubset(provided_ids):
                continue
            if not subset_match and not provided_ids.issubset(rxn_all_needed):
                continue

            sufficient = True
            for cid, coeff in rxn.reactants:
                chem = self._world.chemicals.get(cid)
                mw = chem.molecular_weight if chem else 100.0
                available_mol = all_amounts.get(cid, 0.0) / mw
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
        equipment: Optional[str] = None,
    ) -> Dict:
        name_to_id = {
            chem.name.lower(): cid
            for cid, chem in self._world.chemicals.items()
        }

        all_amounts: Dict[str, float] = {}
        for name, grams in reactant_amounts.items():
            cid = name_to_id.get(name.lower())
            if cid is None:
                return {"success": False, "message": f"Unknown chemical: {name}"}
            all_amounts[cid] = grams

        cost = estimate_reaction_cost(
            self._world.chemicals, all_amounts, temperature_C, pressure_atm, duration_seconds,
            self._world.cost_params, equipment=equipment,
            equipment_catalog=self._world.equipment,
        )
        cost["success"] = True
        return cost



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
        Both values are clamped to practical ranges, solvent BP, and reactant BPs
        (to avoid gas-liquid heterogeneous penalties).
        """
        A = 10.0 ** rxn.log_A_factor
        target_k = 0.01
        ratio = A / max(target_k, 1e-30)
        if ratio <= 1.0:
            T_K = 298.15
        else:
            T_K = rxn.activation_energy_kJ / (R_kJ * math.log(ratio))
        T_K = max(298.15, min(873.15, T_K))

        max_solvent_bp_C = self._max_solvent_bp_for_reaction(rxn)
        T_C = T_K - 273.15
        if T_C > max_solvent_bp_C - 5.0:
            T_C = max_solvent_bp_C - 5.0

        min_reactant_bp = min(
            (self._world.chemicals[cid].boiling_point
             for cid, _ in rxn.reactants if cid in self._world.chemicals),
            default=9999.0,
        )
        if T_C > min_reactant_bp - 5.0:
            T_C = min_reactant_bp - 5.0

        T_C = max(25.0, T_C)
        T_K = T_C + 273.15

        k_at_T = A * math.exp(-rxn.activation_energy_kJ / (R_kJ * T_K))
        duration_s = min(3600.0, max(60.0, 3.0 / max(k_at_T, 1e-30)))
        return round(T_C, 1), round(duration_s, 1)

    def _max_solvent_bp_for_reaction(self, rxn) -> float:
        """Find the highest boiling point among solvents that can dissolve all non-solvent reactants."""
        reactant_ids = [cid for cid, _ in rxn.reactants]
        non_solvent_reactants = [
            cid for cid in reactant_ids
            if cid in self._world.chemicals and not self._world.chemicals[cid].is_solvent
        ]
        if not non_solvent_reactants:
            return 600.0

        solvent_reactant_ids = [
            cid for cid in reactant_ids
            if cid in self._world.chemicals and self._world.chemicals[cid].is_solvent
        ]
        best_bp = -273.0
        for sid in solvent_reactant_ids:
            all_dissolve = all(
                sid in self._world.chemicals[cid].solubility
                for cid in non_solvent_reactants
            )
            if all_dissolve:
                best_bp = max(best_bp, self._world.chemicals[sid].boiling_point)

        for cid, chem in self._world.chemicals.items():
            if not chem.is_solvent or cid in reactant_ids:
                continue
            all_dissolve = all(
                cid in self._world.chemicals[r].solubility
                for r in non_solvent_reactants
            )
            if all_dissolve:
                best_bp = max(best_bp, chem.boiling_point)

        return best_bp if best_bp > -273.0 else 600.0

    def _build_pathway_steps(
        self,
        rxn_chain: List,
        per_m1_g: float,
    ) -> List[Dict]:
        """Build step dicts for evaluate_pathway from a Reaction chain.

        M1 reactants and catalysts are set to *per_m1_g*; non-M1 intermediates
        are sized to whatever the previous step produced (via pre-simulation).
        Catalysts are included in reactant_amounts (same as agent API).
        """
        virtual_pool: Dict[str, float] = {}
        steps = []
        for rxn in rxn_chain:
            reactant_ids: Dict[str, float] = {}
            catalyst_g: Dict[str, float] = {}
            reactant_names: Dict[str, float] = {}
            for cid, _ in rxn.reactants:
                chem = self._world.chemicals[cid]
                amt = per_m1_g if chem.layer == 1 else max(virtual_pool.get(cid, 0.0), 0.1)
                reactant_ids[cid] = amt
                reactant_names[chem.name] = amt
            for cid in rxn.catalysts:
                chem = self._world.chemicals[cid]
                amt = per_m1_g * 0.1 if chem.layer == 1 else max(virtual_pool.get(cid, 0.0), 0.1)
                catalyst_g[cid] = amt
                reactant_names[chem.name] = amt

            T, dur = self._optimal_temp_for_reaction(rxn)
            P = 1.0

            solvent_addition = self._find_best_solvent_for_step(rxn, reactant_ids, catalyst_g, T)
            if solvent_addition:
                sid, sol_g = solvent_addition
                reactant_ids[sid] = reactant_ids.get(sid, 0.0) + sol_g
                reactant_names[self._world.chemicals[sid].name] = reactant_ids[sid]

            sim = simulate_reaction(rxn, self._world.chemicals, reactant_ids, T, P, dur,
                                    catalyst_amounts_g=catalyst_g)
            for cid, consumed in sim["consumed_g"].items():
                virtual_pool[cid] = max(0.0, virtual_pool.get(cid, 0.0) - consumed)
            for cid, produced in sim["produced_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced
            for cid, g in catalyst_g.items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + g

            steps.append({
                "reactant_amounts": reactant_names,
                "temperature_C": T,
                "pressure_atm": P,
                "duration_seconds": dur,
            })
        return steps

    def _find_best_solvent_for_step(
        self, rxn, reactant_ids: Dict[str, float], catalyst_g: Dict[str, float], temperature_C: float
    ) -> Optional[Tuple[str, float]]:
        from .simulator import _find_common_solvent, state_at
        pool = dict(reactant_ids)
        for cid, g in catalyst_g.items():
            pool[cid] = pool.get(cid, 0.0) + g
        solvent = _find_common_solvent(rxn, self._world.chemicals, pool, temperature_C, 1.0)
        if solvent and solvent not in ("__neat__", "__self__", None):
            return None
        if solvent in ("__neat__", "__self__"):
            return None

        reactant_ids_set = set(reactant_ids.keys()) | set(catalyst_g.keys())
        non_solvent_reactants = [
            cid for cid, _ in rxn.reactants
            if cid in self._world.chemicals and not self._world.chemicals[cid].is_solvent
        ]
        total_reactant_g = sum(reactant_ids.get(cid, 0.0) for cid in non_solvent_reactants)

        best_sid = None
        best_bp = -273.0
        for cid, chem in self._world.chemicals.items():
            if not chem.is_solvent:
                continue
            if state_at(chem, temperature_C, 1.0) != "liquid":
                continue
            all_dissolve = all(
                cid in self._world.chemicals[r].solubility
                for r in non_solvent_reactants if r in self._world.chemicals
            )
            if all_dissolve and chem.boiling_point > best_bp:
                best_sid = cid
                best_bp = chem.boiling_point

        if best_sid is None:
            return None
        solvent_g = max(total_reactant_g * 5.0, 50.0)
        return best_sid, solvent_g

    # ------------------------------------------------------------------
    # Pathway helpers
    # ------------------------------------------------------------------

    def _find_matching_reactions_with_pool(
        self,
        all_amounts: Dict[str, float],
        virtual_pool: Dict[str, float],
    ) -> List:
        """Like _find_matching_reactions but also considers virtual_pool contents."""
        combined = dict(all_amounts)
        for cid, g in virtual_pool.items():
            if g > 1e-6 and cid not in combined:
                combined[cid] = g
        return self._find_matching_reactions(combined)

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

            T = float(step.get("temperature_C", 25.0))
            P = float(step.get("pressure_atm", 1.0))
            dur = float(step.get("duration_seconds", 60.0))

            # Find reaction — also considers virtual pool contents
            matching = self._find_matching_reactions_with_pool(step_reactant_ids, virtual_pool)
            if not matching:
                matching = self._find_matching_reactions(step_reactant_ids)
            if not matching:
                names = list(step.get("reactant_amounts", {}).keys())
                return {
                    "success": False,
                    "message": f"Step {step_num}: no reaction found for {names}.",
                }

            rxn = min(matching, key=lambda r: r.delta_G_kJ)

            # Separate catalysts from reactants
            rxn_catalyst_ids = set(rxn.catalysts)
            pure_reactant_ids = {cid: g for cid, g in step_reactant_ids.items() if cid not in rxn_catalyst_ids}
            step_catalyst_g = {cid: g for cid, g in step_reactant_ids.items() if cid in rxn_catalyst_ids}

            # Warn if non-M1 reactants are under-supplied from the virtual pool
            for cid, needed in pure_reactant_ids.items():
                chem = self._world.chemicals[cid]
                if chem.layer > 1:
                    available = virtual_pool.get(cid, 0.0)
                    if available < needed - 1e-6:
                        warnings.append(
                            f"Step {step_num}: {chem.name} needs {needed:.2f}g but only "
                            f"{available:.2f}g is available from earlier steps."
                        )

            # Accumulate M1 cost (both reactants and catalysts count as purchases)
            for cid, grams in step_reactant_ids.items():
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    m1_totals[cid] = m1_totals.get(cid, 0.0) + grams
                    total_m1_cost += chem.price_per_gram * grams

            # Simulate (no inventory side-effects)
            sim = simulate_reaction(rxn, self._world.chemicals, pure_reactant_ids, T, P, dur,
                                    catalyst_amounts_g=step_catalyst_g)
            cost = calculate_cost(rxn, self._world.chemicals, pure_reactant_ids, T, P, dur, self._world.cost_params, equipment_catalog=self._world.equipment)
            total_process_cost += cost["total_cost"]
            total_time_s += dur

            # Propagate outputs through virtual pool
            for cid, consumed in sim["consumed_g"].items():
                virtual_pool[cid] = max(0.0, virtual_pool.get(cid, 0.0) - consumed)
            for cid, produced in sim["produced_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced
            for cid, produced in sim["byproduct_g"].items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced
            # Catalysts remain in pool (not consumed)
            for cid, g in step_catalyst_g.items():
                virtual_pool[cid] = virtual_pool.get(cid, 0.0) + g

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
              - ``num_steps``, ``starting_materials``
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
                    "starting_materials": [target_compound],
                    "steps": [],
                    "message": f"{target_compound} is a base chemical available for direct purchase.",
                }],
                "message": f"{target_compound} is a base chemical available for direct purchase.",
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
                "starting_materials": m1_names,
                "steps": clean_steps,
                "message": (
                    f"Route {i}: {len(steps_fwd)} step(s) to {target_compound}. "
                    f"M1 starting materials: {', '.join(m1_names)}."
                ),
            })

        min_steps = min(r["num_steps"] for r in formatted)
        m1_for_shortest = next(r["starting_materials"] for r in formatted if r["num_steps"] == min_steps)
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
                        "starting_materials": [target_chem.name],
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
                    catalysts_in_step = [self._id_to_name(cid) for cid in rxn.catalysts]
                    route_steps.append({
                        "step": i + 1,
                        "reactants": list(step["reactant_amounts"].keys()),
                        "catalysts": catalysts_in_step,
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
                        "starting_materials": sorted(self._id_to_name(cid) for cid in m1_ids),
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
            f"{', '.join(best['route']['starting_materials'])}. "
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

    # ------------------------------------------------------------------
    # Ground-truth optimal cost (for evaluation only)
    # ------------------------------------------------------------------

    def compute_optimal_cost(
        self,
        min_medicinal_value: float = 3.0,
        max_toxicity: float = 4.0,
        min_yield_g: float = 1.0,
        max_time_seconds: float = 28800.0,
        required_phase: Optional[str] = None,
        phase_temp_C: float = 25.0,
        max_routes_per_target: int = 5,
        max_steps: int = 6,
    ) -> Dict:
        """Compute the minimum possible cost to produce min_yield_g of a qualifying compound.

        This is the ground-truth "oracle" cost for evaluation. It:
        1. Enumerates all compounds satisfying toxicity/medicinal/phase constraints
        2. Finds all synthesis chains to each
        3. For each chain, simulates with optimal conditions and computes
           the EXACT cost including actual purification costs
        4. Scales reactant amounts to meet the yield requirement
        5. Checks time budget feasibility
        6. Returns the minimum-cost feasible path
        """
        qualifying = [
            chem for chem in self._world.chemicals.values()
            if chem.medicinal_value >= min_medicinal_value
            and chem.base_toxicity <= max_toxicity
        ]

        if required_phase:
            qualifying = [
                chem for chem in qualifying
                if state_at(chem, phase_temp_C, 1.0) == required_phase
            ]

        if not qualifying:
            return {
                "success": False,
                "found": False,
                "message": "No qualifying compounds found.",
                "optimal_cost": None,
            }

        best_result = None

        for target_chem in qualifying:
            if target_chem.layer == 1:
                purchase_cost = target_chem.price_per_gram * min_yield_g
                candidate = {
                    "target": target_chem.name,
                    "medicinal_value": target_chem.medicinal_value,
                    "base_toxicity": target_chem.base_toxicity,
                    "optimal_cost": round(purchase_cost, 4),
                    "num_steps": 0,
                    "total_time_seconds": 0.0,
                    "route_detail": {"note": "Direct purchase"},
                }
                if best_result is None or candidate["optimal_cost"] < best_result["optimal_cost"]:
                    best_result = candidate
                continue

            effective_steps = max(max_steps, self._world.num_layers * 2)
            chains = self._find_reaction_chains(target_chem.id, max_routes_per_target, effective_steps)

            for chain in chains:
                result = self._compute_chain_cost(
                    chain, target_chem.id, min_yield_g, max_time_seconds
                )
                if result is None:
                    continue
                candidate = {
                    "target": target_chem.name,
                    "medicinal_value": target_chem.medicinal_value,
                    "base_toxicity": target_chem.base_toxicity,
                    "optimal_cost": result["total_cost"],
                    "num_steps": len(chain),
                    "total_time_seconds": result["total_time"],
                    "route_detail": result,
                }
                if best_result is None or candidate["optimal_cost"] < best_result["optimal_cost"]:
                    best_result = candidate

        for target_chem in qualifying:
            if target_chem.layer <= 1:
                continue
            one_pot = self._compute_one_pot_cost(
                target_chem.id, min_yield_g, max_time_seconds
            )
            if one_pot is not None:
                candidate = {
                    "target": target_chem.name,
                    "medicinal_value": target_chem.medicinal_value,
                    "base_toxicity": target_chem.base_toxicity,
                    "optimal_cost": one_pot["total_cost"],
                    "num_steps": 1,
                    "total_time_seconds": one_pot["total_time"],
                    "route_detail": one_pot,
                }
                if best_result is None or candidate["optimal_cost"] < best_result["optimal_cost"]:
                    best_result = candidate

        if best_result is None:
            return {
                "success": True,
                "found": False,
                "message": "No feasible synthesis route found within constraints.",
                "optimal_cost": None,
            }

        return {
            "success": True,
            "found": True,
            "optimal_cost": best_result["optimal_cost"],
            "target": best_result["target"],
            "medicinal_value": best_result["medicinal_value"],
            "base_toxicity": best_result["base_toxicity"],
            "num_steps": best_result["num_steps"],
            "total_time_seconds": best_result["total_time_seconds"],
            "route_detail": best_result["route_detail"],
            "message": (
                f"Optimal cost: {best_result['optimal_cost']:.2f} credits to produce "
                f"{min_yield_g}g of {best_result['target']} "
                f"(med={best_result['medicinal_value']:.2f}, tox={best_result['base_toxicity']:.2f}) "
                f"in {best_result['num_steps']} step(s), {best_result['total_time_seconds']:.0f}s."
            ),
        }

    def _compute_chain_cost(
        self,
        chain: List,
        target_id: str,
        min_yield_g: float,
        max_time_seconds: float,
    ) -> Optional[Dict]:
        """Compute exact cost for a reaction chain to produce min_yield_g of target.

        Uses binary search on M1 scaling factor to find the minimum input
        that produces at least min_yield_g of the target, then computes
        the exact cost including actual purification.

        Returns None if the chain is infeasible (e.g., exceeds time budget).
        """
        scale_lo, scale_hi = 0.1, 1000.0
        best_scale = None

        for _ in range(30):
            scale_mid = (scale_lo + scale_hi) / 2.0
            sim_result = self._simulate_chain(chain, target_id, scale_mid)
            if sim_result["target_yield_g"] >= min_yield_g:
                best_scale = scale_mid
                scale_hi = scale_mid
            else:
                scale_lo = scale_mid

        if best_scale is None:
            sim_hi = self._simulate_chain(chain, target_id, scale_hi)
            if sim_hi["target_yield_g"] >= min_yield_g:
                best_scale = scale_hi
            else:
                return None

        result = self._simulate_chain(chain, target_id, best_scale)

        if result["total_time"] > max_time_seconds:
            return result if result["total_time"] <= max_time_seconds * 1.05 else None

        return result

    def _simulate_chain(
        self,
        chain: List,
        target_id: str,
        m1_scale_g: float,
    ) -> Dict:
        """Simulate a full chain with given M1 scale and compute exact costs.

        For each step:
        - Uses optimal temperature/duration for the reaction
        - Computes process cost (energy, equipment, duration) from cost_model
        - Computes actual purification cost based on post-reaction mixture
        - Propagates products to next step via virtual pool

        Does NOT recover leftover reactants (optimal strategy: waste them).
        """
        virtual_pool: Dict[str, float] = {}
        total_purchase_cost = 0.0
        total_process_cost = 0.0
        total_purification_cost = 0.0
        total_time = 0.0
        step_details = []

        for rxn in chain:
            reactant_ids: Dict[str, float] = {}
            catalyst_g: Dict[str, float] = {}
            for cid, _ in rxn.reactants:
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    amt = m1_scale_g
                else:
                    amt = max(virtual_pool.get(cid, 0.0), 0.01)
                reactant_ids[cid] = amt
            for cid in rxn.catalysts:
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    amt = m1_scale_g * 0.1
                else:
                    amt = max(virtual_pool.get(cid, 0.0), 0.01)
                catalyst_g[cid] = amt

            T, dur = self._optimal_temp_for_reaction(rxn)
            P = 1.0

            all_amounts = dict(reactant_ids)
            for cid, g in catalyst_g.items():
                all_amounts[cid] = all_amounts.get(cid, 0.0) + g
            chain_sim = simulate_chain_reaction(
                self._world, all_amounts, T, P, dur,
                equipment="autoclave",
                catalyst_ids=set(catalyst_g.keys()),
            )
            sim = {
                "consumed_g": chain_sim.get("consumed_g", {}),
                "produced_g": chain_sim.get("produced_g", {}),
                "byproduct_g": chain_sim.get("byproduct_g", {}),
                "conversion": chain_sim.get("total_conversion", 0.0),
                "k_eff": 0.0,
            }

            for cid, amt in reactant_ids.items():
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    total_purchase_cost += chem.price_per_gram * amt
            for cid, amt in catalyst_g.items():
                chem = self._world.chemicals[cid]
                if chem.layer == 1:
                    total_purchase_cost += chem.price_per_gram * amt

            cost_info = calculate_cost(
                rxn, self._world.chemicals, reactant_ids, T, P, dur, self._world.cost_params,
                equipment_catalog=self._world.equipment,
            )
            process_cost = (
                cost_info["energy_cost"]
                + cost_info["duration_cost"]
                + cost_info["equipment_cost"]
            )
            total_process_cost += process_cost
            total_time += dur

            leftover_g: Dict[str, float] = {}
            for cid, original in reactant_ids.items():
                consumed = sim["consumed_g"].get(cid, 0.0)
                leftover = original - consumed
                if leftover > self.DETECTION_THRESHOLD_G:
                    leftover_g[cid] = leftover
            # Catalysts remain fully in the mixture
            for cid, g in catalyst_g.items():
                if g > self.DETECTION_THRESHOLD_G:
                    leftover_g[cid] = g

            observed_products = {
                cid: g for cid, g in sim["produced_g"].items()
                if g >= self.DETECTION_THRESHOLD_G
            }
            observed_byproducts = {
                cid: g for cid, g in sim["byproduct_g"].items()
                if g >= self.DETECTION_THRESHOLD_G
            }

            mixture_components = (
                len(leftover_g)
                + len(observed_products)
                + len(observed_byproducts)
            )

            step_purification = 0.0
            if mixture_components > 1:
                from .cost_model import _phase_separation_factor, _purification_cost_per_component
                from .simulator import state_at as _st
                _sp = set()
                for _cid in list(observed_products.keys()) + list(observed_byproducts.keys()) + list(leftover_g.keys()):
                    if _cid in self._world.chemicals:
                        _sp.add(_st(self._world.chemicals[_cid], T, P))
                _spf = _phase_separation_factor(_sp)
                for g in observed_products.values():
                    step_purification += _purification_cost_per_component(g, mixture_components, _spf)
                for g in observed_byproducts.values():
                    step_purification += _purification_cost_per_component(g, mixture_components, _spf)

            total_purification_cost += step_purification

            for cid, consumed in sim["consumed_g"].items():
                virtual_pool[cid] = max(0.0, virtual_pool.get(cid, 0.0) - consumed)
            for cid, produced in sim["produced_g"].items():
                if produced >= self.DETECTION_THRESHOLD_G:
                    virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced
            for cid, produced in sim["byproduct_g"].items():
                if produced >= self.DETECTION_THRESHOLD_G:
                    virtual_pool[cid] = virtual_pool.get(cid, 0.0) + produced

            step_details.append({
                "reaction_id": rxn.id,
                "reactants_g": {
                    self._id_to_name(cid): _round_sig(g)
                    for cid, g in reactant_ids.items()
                },
                "catalysts_g": {
                    self._id_to_name(cid): _round_sig(g)
                    for cid, g in catalyst_g.items()
                } if catalyst_g else {},
                "temperature_C": T,
                "duration_s": dur,
                "conversion": round(sim["conversion"], 4),
                "products_g": {
                    self._id_to_name(cid): _round_sig(g)
                    for cid, g in observed_products.items()
                },
                "purification_cost": round(step_purification, 2),
                "process_cost": round(process_cost, 2),
            })

        target_yield = virtual_pool.get(target_id, 0.0)
        total_cost = total_purchase_cost + total_process_cost + total_purification_cost

        return {
            "target_yield_g": round(target_yield, 6),
            "total_cost": round(total_cost, 4),
            "purchase_cost": round(total_purchase_cost, 4),
            "process_cost": round(total_process_cost, 4),
            "purification_cost": round(total_purification_cost, 4),
            "total_time": round(total_time, 1),
            "m1_scale_g": round(m1_scale_g, 4),
            "steps": step_details,
        }

    def _compute_one_pot_cost(
        self,
        target_id: str,
        min_yield_g: float,
        max_time_seconds: float,
    ) -> Optional[Dict]:
        """Simulate one-pot strategy: all layer-1 precursors + catalysts in a single vessel.

        Tries multiple temperatures and scales to find the cheapest one-pot
        that produces min_yield_g of the target via chain reactions.
        """
        from .cost_model import (
            calculate_cost,
            compute_purification_cost,
            _phase_separation_factor,
            _purification_cost_per_component,
        )

        chains = self._find_reaction_chains(target_id, max_routes=5, max_steps=6)
        if not chains:
            return None

        chain_configs = []
        for chain in chains:
            m1_reactants = set()
            catalysts = set()
            rxns = []
            for rxn in chain:
                for cid, _ in rxn.reactants:
                    if self._world.chemicals[cid].layer == 1:
                        m1_reactants.add(cid)
                for cat_id in rxn.catalysts:
                    if self._world.chemicals[cat_id].layer == 1:
                        catalysts.add(cat_id)
                rxns.append(rxn)
            if m1_reactants:
                chain_configs.append((m1_reactants, catalysts, rxns))

        if not chain_configs:
            return None

        best = None

        needed_solvents = set()
        for _, _, rxns in chain_configs:
            for rxn in rxns:
                reactant_ids = [cid for cid, _ in rxn.reactants]
                non_solvent_reactants = [
                    cid for cid in reactant_ids
                    if cid in self._world.chemicals and not self._world.chemicals[cid].is_solvent
                ]
                for sid, chem in self._world.chemicals.items():
                    if not chem.is_solvent:
                        continue
                    if all(sid in self._world.chemicals[r].solubility for r in non_solvent_reactants if r in self._world.chemicals):
                        needed_solvents.add(sid)
                        break

        for m1_reactants, catalysts, rxns in chain_configs:
            temps_to_try = set()
            for rxn in rxns:
                T, _ = self._optimal_temp_for_reaction(rxn)
                temps_to_try.add(T)
            temps_to_try.add(150.0)
            temps_to_try = sorted(temps_to_try)[:3]

            for T in temps_to_try:
                dur = 300.0
                if dur > max_time_seconds:
                    dur = max_time_seconds

                scale_lo, scale_hi = 0.5, 200.0
                best_scale = None
                for _ in range(12):
                    scale_mid = (scale_lo + scale_hi) / 2.0
                    amounts = {cid: scale_mid for cid in m1_reactants}
                    for cat_id in catalysts:
                        amounts[cat_id] = amounts.get(cat_id, 0) + scale_mid * 0.1
                    for sid in needed_solvents:
                        amounts[sid] = amounts.get(sid, 0) + scale_mid * 5.0

                    sim = simulate_chain_reaction(
                        self._world, amounts, T, 1.0, dur,
                        equipment="sealed_flask",
                    )
                    target_yield = sim["final_pool_g"].get(target_id, 0.0)
                    if target_yield >= min_yield_g:
                        best_scale = scale_mid
                        scale_hi = scale_mid
                    else:
                        scale_lo = scale_mid

                if best_scale is None:
                    amounts = {cid: scale_hi for cid in m1_reactants}
                    for cat_id in catalysts:
                        amounts[cat_id] = amounts.get(cat_id, 0) + scale_hi * 0.1
                    for sid in needed_solvents:
                        amounts[sid] = amounts.get(sid, 0) + scale_hi * 5.0
                    sim = simulate_chain_reaction(
                        self._world, amounts, T, 1.0, dur,
                        equipment="sealed_flask",
                    )
                    target_yield = sim["final_pool_g"].get(target_id, 0.0)
                    if target_yield >= min_yield_g:
                        best_scale = scale_hi
                    else:
                        continue

                amounts = {cid: best_scale for cid in m1_reactants}
                for cat_id in catalysts:
                    amounts[cat_id] = amounts.get(cat_id, 0) + best_scale * 0.1
                for sid in needed_solvents:
                    amounts[sid] = amounts.get(sid, 0) + best_scale * 5.0
                sim = simulate_chain_reaction(
                    self._world, amounts, T, 1.0, dur,
                    equipment="sealed_flask",
                )

                purchase_cost = sum(
                    self._world.chemicals[cid].price_per_gram * g
                    for cid, g in amounts.items()
                )

                dummy_rxn = rxns[0]
                cost_info = calculate_cost(
                    dummy_rxn, self._world.chemicals, amounts,
                    T, 1.0, dur, self._world.cost_params,
                    equipment="sealed_flask",
                    equipment_catalog=self._world.equipment,
                )
                process_cost = (
                    cost_info["energy_cost"]
                    + cost_info["duration_cost"]
                    + cost_info["equipment_cost"]
                )

                purif_cost = compute_purification_cost(
                    sim["final_pool_g"], self._world.chemicals, T, 1.0,
                )

                total_cost = purchase_cost + process_cost + purif_cost

                if best is None or total_cost < best["total_cost"]:
                    best = {
                        "total_cost": round(total_cost, 4),
                        "purchase_cost": round(purchase_cost, 4),
                        "process_cost": round(process_cost, 4),
                        "purification_cost": round(purif_cost, 4),
                        "total_time": dur,
                        "temperature_C": T,
                        "m1_scale_g": round(best_scale, 4),
                        "target_yield_g": round(sim["final_pool_g"].get(target_id, 0.0), 6),
                        "strategy": "one_pot",
                    }

        return best
