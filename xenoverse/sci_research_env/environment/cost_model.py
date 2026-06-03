from __future__ import annotations

from typing import Dict

import numpy as np

from ..world_gen.models import Chemical, Reaction
from .simulator import state_at


def calculate_cost(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    reactant_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
    cost_params: Dict[str, float],
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

    pressure_premium = 1.0 + cost_params["equipment_pressure_coeff"] * abs(np.log(max(pressure_atm, 0.01)))
    base_equipment = cost_params["equipment_base"] * total_mass ** 0.6
    equipment_cost = base_equipment * pressure_premium * toxicity_premium

    n_products = len(reaction.products) + len(reaction.byproducts)
    phases = set(
        state_at(chemicals[pid], temperature_C, pressure_atm)
        for pid, _ in reaction.products
        if pid in chemicals
    )
    phase_complexity = len(phases)
    purification_cost = (2.0 * n_products + 3.0 * phase_complexity) * total_mass ** 0.5

    total_cost = raw_cost + energy_cost + duration_cost + equipment_cost + purification_cost

    return {
        "total_cost": round(total_cost, 2),
        "raw_material_cost": round(raw_cost, 2),
        "energy_cost": round(energy_cost, 2),
        "duration_cost": round(duration_cost, 2),
        "equipment_cost": round(equipment_cost, 2),
        "purification_cost": round(purification_cost, 2),
    }
