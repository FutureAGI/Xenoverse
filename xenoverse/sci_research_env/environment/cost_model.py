from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from ..world_gen.models import Chemical, Reaction, EQUIPMENT_CATALOG
from .simulator import state_at


def _phase_separation_factor(phases: Set[str]) -> float:
    if not phases:
        return 1.0
    if len(phases) == 1:
        phase = next(iter(phases))
        if phase == "solid":
            return 0.4
        elif phase == "liquid":
            return 1.0
        else:
            return 1.8
    if len(phases) == 3:
        return 3.0
    if "gas" in phases and "liquid" in phases:
        return 2.2
    if "gas" in phases and "solid" in phases:
        return 2.0
    return 0.7


def _purification_cost_per_component(
    grams: float,
    n_components: int,
    phase_factor: float,
) -> float:
    if n_components <= 1:
        return 0.0
    complexity = 1.0 + 0.5 * (n_components - 2) ** 1.3
    base_rate = 6.0
    return base_rate * complexity * phase_factor * grams ** 0.7


def compute_purification_cost(
    component_masses: Dict[str, float],
    chemicals: Dict[str, Chemical],
    temperature_C: float,
    pressure_atm: float,
    detection_threshold: float = 0.001,
) -> float:
    visible = {cid: g for cid, g in component_masses.items() if g >= detection_threshold}
    n = len(visible)
    if n <= 1:
        return 0.0
    phases: Set[str] = set()
    for cid in visible:
        if cid in chemicals:
            phases.add(state_at(chemicals[cid], temperature_C, pressure_atm))
    pf = _phase_separation_factor(phases)
    total = 0.0
    for cid, g in visible.items():
        total += _purification_cost_per_component(g, n, pf)
    return round(total, 2)


def _equipment_cost(
    equipment: Optional[str],
    duration_s: float,
    total_mass: float,
    cost_params: Dict[str, float],
    catalog: Optional[Dict[str, Dict]] = None,
) -> float:
    cat = catalog or EQUIPMENT_CATALOG
    spec = cat.get(equipment or "open_beaker", cat["open_beaker"])
    hours = duration_s / 3600.0
    base = spec["base_cost_per_hour"] * hours
    mass_factor = total_mass ** 0.6
    return base * mass_factor * spec["cost_multiplier"]


def estimate_reaction_cost(
    chemicals: Dict[str, Chemical],
    all_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
    cost_params: Dict[str, float],
    equipment: Optional[str] = None,
    equipment_catalog: Optional[Dict[str, Dict]] = None,
) -> Dict:
    raw_cost = sum(
        chemicals[cid].price_per_gram * amt
        for cid, amt in all_amounts_g.items()
        if cid in chemicals and chemicals[cid].price_per_gram is not None
    )

    total_mass = sum(all_amounts_g.values())
    n_chemicals = len(all_amounts_g)

    T_dev = abs(temperature_C - 25.0)
    if temperature_C < 25.0:
        energy_temp = cost_params["cooling_coeff"] * (T_dev / 100.0) ** cost_params["cooling_exponent"]
    else:
        energy_temp = cost_params["heating_coeff"] * (T_dev / 100.0) ** cost_params["heating_exponent"]

    if pressure_atm < 1.0:
        P_dev = 1.0 - pressure_atm
        energy_pressure = cost_params["pressure_low_coeff"] * P_dev ** cost_params["pressure_low_exp"]
    else:
        P_excess = pressure_atm - 1.0
        energy_pressure = cost_params["pressure_high_coeff"] * P_excess ** cost_params["pressure_high_exp"]

    energy_cost = (energy_temp + energy_pressure + 0.1) * total_mass

    duration_cost = cost_params["duration_coeff"] * (duration_s / 3600.0) * total_mass ** 0.5

    toxicities = [
        chemicals[cid].base_toxicity for cid in all_amounts_g if cid in chemicals
    ]
    max_toxicity = min(10.0, max(toxicities) / 2.0) if toxicities else 0.0
    toxicity_premium = 1.0 + 0.15 * max_toxicity

    equip_cost = _equipment_cost(equipment, duration_s, total_mass, cost_params, catalog=equipment_catalog)
    equip_cost *= toxicity_premium

    n_components = n_chemicals * 2
    phases = set()
    for cid in all_amounts_g:
        if cid in chemicals:
            phases.add(state_at(chemicals[cid], temperature_C, pressure_atm))
    pf = _phase_separation_factor(phases)
    purification_cost = sum(
        _purification_cost_per_component(total_mass / n_components, n_components, pf)
        for _ in range(n_components)
    )
    phase_note = ", ".join(sorted(phases)) if phases else "unknown"

    total_cost = raw_cost + energy_cost + duration_cost + equip_cost + purification_cost

    return {
        "total_cost": round(total_cost, 2),
        "raw_material_cost": round(raw_cost, 2),
        "energy_cost": round(energy_cost, 2),
        "duration_cost": round(duration_cost, 2),
        "equipment_cost": round(equip_cost, 2),
        "purification_cost_estimate": round(purification_cost, 2),
        "phases_at_conditions": phase_note,
    }


def calculate_cost(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    reactant_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
    cost_params: Dict[str, float],
    equipment: Optional[str] = None,
    equipment_catalog: Optional[Dict[str, Dict]] = None,
) -> Dict:
    raw_cost = sum(
        chemicals[cid].price_per_gram * amt
        for cid, amt in reactant_amounts_g.items()
        if cid in chemicals and chemicals[cid].price_per_gram is not None
    )

    total_mass = sum(reactant_amounts_g.values())

    T_dev = abs(temperature_C - 25.0)
    if temperature_C < 25.0:
        energy_temp = cost_params["cooling_coeff"] * (T_dev / 100.0) ** cost_params["cooling_exponent"]
    else:
        energy_temp = cost_params["heating_coeff"] * (T_dev / 100.0) ** cost_params["heating_exponent"]

    if pressure_atm < 1.0:
        P_dev = 1.0 - pressure_atm
        energy_pressure = cost_params["pressure_low_coeff"] * P_dev ** cost_params["pressure_low_exp"]
    else:
        P_excess = pressure_atm - 1.0
        energy_pressure = cost_params["pressure_high_coeff"] * P_excess ** cost_params["pressure_high_exp"]

    energy_cost = (energy_temp + energy_pressure + 0.1) * total_mass

    duration_cost = cost_params["duration_coeff"] * (duration_s / 3600.0) * total_mass ** 0.5

    reactant_toxicities = [
        chemicals[cid].base_toxicity for cid in reactant_amounts_g if cid in chemicals
    ]
    product_toxicities = [
        chemicals[pid].base_toxicity for pid, _ in reaction.products if pid in chemicals
    ]
    all_toxicities = reactant_toxicities + product_toxicities
    max_toxicity = min(10.0, max(all_toxicities) / 2.0) if all_toxicities else 0.0
    toxicity_premium = 1.0 + 0.15 * max_toxicity

    equip_cost = _equipment_cost(equipment, duration_s, total_mass, cost_params, catalog=equipment_catalog)
    equip_cost *= toxicity_premium

    n_products = len(reaction.products) + len(reaction.byproducts)
    n_reactants = len(reactant_amounts_g)
    n_components = n_products + n_reactants
    phases = set()
    for cid in reactant_amounts_g:
        if cid in chemicals:
            phases.add(state_at(chemicals[cid], temperature_C, pressure_atm))
    for pid, _ in reaction.products:
        if pid in chemicals:
            phases.add(state_at(chemicals[pid], temperature_C, pressure_atm))
    pf = _phase_separation_factor(phases)
    purification_cost = sum(
        _purification_cost_per_component(total_mass / n_components, n_components, pf)
        for _ in range(n_components)
    )

    total_cost = raw_cost + energy_cost + duration_cost + equip_cost + purification_cost

    return {
        "total_cost": round(total_cost, 2),
        "raw_material_cost": round(raw_cost, 2),
        "energy_cost": round(energy_cost, 2),
        "duration_cost": round(duration_cost, 2),
        "equipment_cost": round(equip_cost, 2),
        "purification_cost": round(purification_cost, 2),
    }
