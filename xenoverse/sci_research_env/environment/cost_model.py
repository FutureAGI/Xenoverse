from __future__ import annotations

from typing import Dict, Optional

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
) -> Dict:
    raw_cost = sum(
        chemicals[cid].price_per_gram * amt
        for cid, amt in reactant_amounts_g.items()
        if cid in chemicals and chemicals[cid].price_per_gram is not None
    )

    total_mass = sum(reactant_amounts_g.values())

    T_excess = max(0.0, temperature_C - 25.0)
    P_excess = max(0.0, pressure_atm - 1.0)
    energy_cost = (
        0.8 * (T_excess / 100.0) ** 1.5
        + 1.5 * P_excess ** 0.7
        + 0.1
    ) * total_mass

    reactant_toxicities = [
        chemicals[cid].base_toxicity
        for cid in reactant_amounts_g
        if cid in chemicals
    ]
    product_toxicities = [
        chemicals[pid].base_toxicity
        for pid, _ in reaction.products
        if pid in chemicals
    ]
    all_toxicities = reactant_toxicities + product_toxicities
    max_toxicity = max(all_toxicities) if all_toxicities else 0.0
    max_toxicity = min(10.0, max_toxicity / 2.0)
    toxicity_premium = 1.0 + 0.15 * max_toxicity

    pressure_premium = 1.0 + 0.3 * np.log1p(pressure_atm)
    base_equipment = 5.0 * total_mass ** 0.6
    equipment_cost = base_equipment * pressure_premium * toxicity_premium

    n_products = len(reaction.products) + len(reaction.byproducts)
    phases = set(
        state_at(chemicals[pid], temperature_C, pressure_atm)
        for pid, _ in reaction.products
        if pid in chemicals
    )
    phase_complexity = len(phases)
    purification_cost = (2.0 * n_products + 3.0 * phase_complexity) * total_mass ** 0.5

    total_cost = raw_cost + energy_cost + purification_cost + equipment_cost

    return {
        "total_cost": round(total_cost, 2),
        "raw_material_cost": round(raw_cost, 2),
        "energy_cost": round(energy_cost, 2),
        "equipment_cost": round(equipment_cost, 2),
        "purification_cost": round(purification_cost, 2),
    }
