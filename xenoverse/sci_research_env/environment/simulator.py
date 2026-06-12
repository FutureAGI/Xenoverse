from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..world_gen.models import Chemical, Reaction, World, EQUIPMENT_CATALOG

R_kJ = 8.314e-3  # kJ/(mol·K)
R_J = 8.314      # J/(mol·K)

DEFAULT_STEP_SECONDS = 5.0
AMBIENT_TEMP_C = 25.0
AMBIENT_PRESSURE_ATM = 1.0


@dataclass
class VesselState:
    temperature_C: float
    pressure_atm: float
    vessel_type: str = "open"
    thermal_mode: str = "isothermal"
    heating_rate_C_per_s: float = 0.0
    volume_L: float = 1.0
    initial_temp_C: float = 25.0
    initial_pressure_atm: float = 1.0

    def __post_init__(self):
        self.initial_temp_C = self.temperature_C
        self.initial_pressure_atm = self.pressure_atm


def state_at(chem: Chemical, temp_C: float, pressure_atm: float = 1.0) -> str:
    bp_adj = chem.boiling_point + chem.clausius_C * np.log(max(0.01, pressure_atm))
    if temp_C < chem.melting_point:
        return "solid"
    elif temp_C < bp_adj:
        return "liquid"
    else:
        return "gas"


def k_eq_at_T(reaction: Reaction, T_K: float) -> float:
    T_ref = 298.0
    K_ref = np.exp(-reaction.delta_G_kJ / (R_kJ * T_ref))
    K_T = K_ref * np.exp(-reaction.delta_H_kJ / R_kJ * (1.0 / T_K - 1.0 / T_ref))
    return float(max(K_T, 1e-30))


def rate_constant_at_T(reaction: Reaction, T_K: float) -> float:
    A = 10 ** reaction.log_A_factor
    k = A * np.exp(-reaction.activation_energy_kJ / (R_kJ * T_K))
    return float(k)


def _mixture_heat_capacity(
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
) -> float:
    total_Cp = 0.0
    for cid, g in pool.items():
        if g > 1e-9 and cid in chemicals:
            total_Cp += g * chemicals[cid].heat_capacity_J_per_gK
    return max(total_Cp, 0.1)


def _count_gas_moles(
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
    temp_C: float,
    pressure_atm: float,
) -> float:
    total_mol = 0.0
    for cid, g in pool.items():
        if g > 1e-9 and cid in chemicals:
            chem = chemicals[cid]
            if state_at(chem, temp_C, pressure_atm) == "gas":
                total_mol += g / chem.molecular_weight
    return total_mol


def _process_phase_transitions(
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
    old_temp_C: float,
    new_temp_C: float,
    pressure_atm: float,
) -> float:
    heat_absorbed_J = 0.0
    for cid, g in pool.items():
        if g < 1e-9 or cid not in chemicals:
            continue
        chem = chemicals[cid]
        bp_adj = chem.boiling_point + chem.clausius_C * np.log(max(0.01, pressure_atm))
        mp = chem.melting_point

        if old_temp_C < mp <= new_temp_C:
            heat_absorbed_J += g * chem.latent_heat_fusion_J_per_g
        elif new_temp_C < mp <= old_temp_C:
            heat_absorbed_J -= g * chem.latent_heat_fusion_J_per_g

        if old_temp_C < bp_adj <= new_temp_C:
            heat_absorbed_J += g * chem.latent_heat_vaporization_J_per_g
        elif new_temp_C < bp_adj <= old_temp_C:
            heat_absorbed_J -= g * chem.latent_heat_vaporization_J_per_g

    return heat_absorbed_J


def heterogeneous_rate_factor(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    pool: Dict[str, float],
    temp_C: float,
    pressure_atm: float,
) -> float:
    phases_present = set()
    for cid, _ in reaction.reactants:
        if cid in chemicals and pool.get(cid, 0.0) > 1e-9:
            phases_present.add(state_at(chemicals[cid], temp_C, pressure_atm))

    if len(phases_present) <= 1:
        return 1.0

    factor = 1.0

    if "solid" in phases_present and "liquid" in phases_present:
        solid_mass = 0.0
        for cid, _ in reaction.reactants:
            if cid in chemicals and pool.get(cid, 0.0) > 1e-9:
                if state_at(chemicals[cid], temp_C, pressure_atm) == "solid":
                    solid_mass += pool[cid]
        contact_factor = max(0.01, solid_mass ** (2.0 / 3.0) / max(solid_mass, 0.1))
        factor *= contact_factor

    if "gas" in phases_present and "liquid" in phases_present:
        gas_mass = 0.0
        for cid, _ in reaction.reactants:
            if cid in chemicals and pool.get(cid, 0.0) > 1e-9:
                if state_at(chemicals[cid], temp_C, pressure_atm) == "gas":
                    gas_mass += pool[cid]
        gas_factor = min(1.0, pressure_atm * 0.5) * max(0.05, gas_mass ** 0.5 / max(gas_mass, 0.1))
        factor *= gas_factor

    if "solid" in phases_present and "gas" in phases_present:
        factor *= 0.05 * pressure_atm ** 0.3

    return float(np.clip(factor, 0.001, 2.0))


def phase_factor_for_reaction(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    temp_C: float,
    pressure_atm: float,
) -> float:
    PHASE_FACTORS = {"solid": 0.01, "liquid": 1.0, "gas": 8.0}
    factors = []
    for cid, _ in reaction.reactants:
        if cid in chemicals:
            phase = state_at(chemicals[cid], temp_C, pressure_atm)
            factors.append(PHASE_FACTORS[phase])
    if not factors:
        return 1.0
    log_mean = np.mean(np.log(np.array(factors, dtype=float) + 1e-30))
    return float(np.exp(log_mean))


def solve_equilibrium_extent(
    K_eq: float,
    reactant_amounts_mol: List[float],
    reactant_coeffs: List[int],
    product_amounts_mol: List[float],
    product_coeffs: List[int],
) -> float:
    max_extent = min(amt / coeff for amt, coeff in zip(reactant_amounts_mol, reactant_coeffs))
    max_extent *= 0.9999

    if max_extent <= 0:
        return 0.0

    def objective(xi: float) -> float:
        products_conc = [p + vp * xi for p, vp in zip(product_amounts_mol, product_coeffs)]
        reactants_conc = [r - vr * xi for r, vr in zip(reactant_amounts_mol, reactant_coeffs)]
        if any(c <= 0 for c in reactants_conc) or any(c < 0 for c in products_conc):
            return float("inf")
        Q = np.prod([c ** vp for c, vp in zip(products_conc, product_coeffs)])
        Q /= np.prod([c ** vr for c, vr in zip(reactants_conc, reactant_coeffs)])
        return Q - K_eq

    try:
        from scipy.optimize import brentq
        f0 = objective(0.0)
        f1 = objective(max_extent)
        if f0 * f1 < 0:
            xi_eq = float(brentq(objective, 0.0, max_extent, xtol=1e-10, maxiter=200))
        elif f0 >= 0:
            xi_eq = 0.0
        else:
            xi_eq = max_extent
    except Exception:
        xi_eq = max_extent * 0.5

    return float(np.clip(xi_eq, 0.0, max_extent))


def catalyst_acceleration(
    catalyst_amounts_g: Dict[str, float],
    reactant_total_g: float,
) -> float:
    if not catalyst_amounts_g or reactant_total_g <= 0:
        return 1.0
    total_catalyst_g = sum(catalyst_amounts_g.values())
    ratio = total_catalyst_g / reactant_total_g
    return 1.0 + 10.0 * ratio ** 0.5


def _find_common_solvent(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    pool: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
) -> Optional[str]:
    reactant_ids = [cid for cid, _ in reaction.reactants]
    solvent_reactant_ids = [
        cid for cid in reactant_ids
        if cid in chemicals and chemicals[cid].is_solvent
    ]
    non_solvent_reactants = [
        cid for cid in reactant_ids
        if cid in chemicals and not chemicals[cid].is_solvent
    ]

    if not non_solvent_reactants:
        return "__self__"

    if solvent_reactant_ids:
        for sid in solvent_reactant_ids:
            if state_at(chemicals[sid], temperature_C, pressure_atm) != "liquid":
                continue
            all_dissolve = all(
                sid in chemicals[cid].solubility
                for cid in non_solvent_reactants if cid in chemicals
            )
            if all_dissolve:
                return sid

    has_solid_or_gas = any(
        state_at(chemicals[cid], temperature_C, pressure_atm) in ("solid", "gas")
        for cid in non_solvent_reactants if cid in chemicals
    )

    solvent_candidates = [
        cid for cid, g in pool.items()
        if g > 1e-9 and cid in chemicals and chemicals[cid].is_solvent
        and cid not in reactant_ids
        and state_at(chemicals[cid], temperature_C, pressure_atm) == "liquid"
    ]

    for sid in solvent_candidates:
        all_dissolve = all(
            sid in chemicals[cid].solubility
            for cid in non_solvent_reactants if cid in chemicals
        )
        if all_dissolve:
            return sid

    if not has_solid_or_gas:
        all_liquid = all(
            state_at(chemicals[cid], temperature_C, pressure_atm) == "liquid"
            for cid in non_solvent_reactants if cid in chemicals
        )
        if all_liquid:
            return "__neat__"

    return None


def _dissolved_fraction(
    chem_id: str,
    solvent_id: str,
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
) -> float:
    if solvent_id == "__neat__":
        return 1.0
    chem = chemicals.get(chem_id)
    if chem is None or chem.is_solvent:
        return 1.0
    max_g_per_100mL = chem.solubility.get(solvent_id, 0.0)
    if max_g_per_100mL <= 0:
        return 0.0
    solvent_g = pool.get(solvent_id, 0.0)
    solvent_chem = chemicals.get(solvent_id)
    if solvent_chem is None or solvent_g < 1e-9:
        return 0.0
    density_approx = 0.9
    solvent_mL = solvent_g / density_approx
    max_dissolved_g = max_g_per_100mL * (solvent_mL / 100.0)
    chem_g = pool.get(chem_id, 0.0)
    if chem_g <= 0:
        return 0.0
    return min(1.0, max_dissolved_g / chem_g)


def _find_applicable_reactions(
    pool: Dict[str, float],
    reactions: Dict[str, Reaction],
    catalyst_ids: Optional[set] = None,
    chemicals: Optional[Dict[str, Chemical]] = None,
    temperature_C: float = 25.0,
    pressure_atm: float = 1.0,
) -> List[Reaction]:
    available = {cid for cid, g in pool.items() if g > 1e-9}
    results = []
    for rxn in reactions.values():
        rxn_reactants = {cid for cid, _ in rxn.reactants}
        rxn_cats = set(rxn.catalysts)
        needed = rxn_reactants | rxn_cats
        if needed.issubset(available):
            has_moles = True
            for cid, coeff in rxn.reactants:
                if pool.get(cid, 0.0) < 1e-9:
                    has_moles = False
                    break
            if not has_moles:
                continue
            if chemicals is not None:
                solvent = _find_common_solvent(rxn, chemicals, pool, temperature_C, pressure_atm)
                if solvent is None:
                    continue
            results.append(rxn)
    return results


def _step_single_reaction(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    pool: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    dt: float,
    solvent_id: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], float, float]:
    T_K = temperature_C + 273.15

    K_eq = k_eq_at_T(reaction, T_K)
    k_rate = rate_constant_at_T(reaction, T_K)
    pf = phase_factor_for_reaction(reaction, chemicals, temperature_C, pressure_atm)
    hetero_f = heterogeneous_rate_factor(reaction, chemicals, pool, temperature_C, pressure_atm)

    dissolution_factor = 1.0
    solvent_is_reactant = solvent_id in {cid for cid, _ in reaction.reactants} if solvent_id else False
    if solvent_id == "__neat__":
        dissolution_factor = 0.3
    elif solvent_id == "__self__":
        dissolution_factor = 1.0
    elif solvent_id and solvent_is_reactant:
        dissolution_factor = 1.0
    elif solvent_id:
        min_frac = 1.0
        for cid, _ in reaction.reactants:
            frac = _dissolved_fraction(cid, solvent_id, pool, chemicals)
            min_frac = min(min_frac, frac)
        dissolution_factor = min_frac

    rxn_reactant_ids = {cid for cid, _ in reaction.reactants}
    rxn_catalyst_ids = set(reaction.catalysts)
    reactant_g = sum(pool.get(cid, 0.0) for cid in rxn_reactant_ids)
    catalyst_g = {cid: pool.get(cid, 0.0) for cid in rxn_catalyst_ids if pool.get(cid, 0.0) > 1e-9}
    cat_factor = catalyst_acceleration(catalyst_g, reactant_g)
    k_eff = k_rate * pf * hetero_f * cat_factor * dissolution_factor

    reactant_amounts_mol = []
    reactant_coeffs = []
    for cid, coeff in reaction.reactants:
        mw = chemicals[cid].molecular_weight if cid in chemicals else 100.0
        moles = pool.get(cid, 0.0) / mw
        effective_moles = moles * dissolution_factor
        reactant_amounts_mol.append(effective_moles)
        reactant_coeffs.append(coeff)

    product_amounts_mol = []
    product_coeffs = []
    for pid, coeff in reaction.products:
        mw = chemicals[pid].molecular_weight if pid in chemicals else 100.0
        product_amounts_mol.append(pool.get(pid, 0.0) / mw)
        product_coeffs.append(coeff)

    xi_eq = solve_equilibrium_extent(
        K_eq, reactant_amounts_mol, reactant_coeffs, product_amounts_mol, product_coeffs
    )

    if xi_eq <= 1e-30:
        return {}, {}, {}, k_eff, 0.0

    xi_t = xi_eq * (1.0 - np.exp(-k_eff * dt))
    xi_t = float(np.clip(xi_t, 0.0, xi_eq))

    consumed_g: Dict[str, float] = {}
    for (cid, coeff), init_mol in zip(reaction.reactants, reactant_amounts_mol):
        mw = chemicals[cid].molecular_weight if cid in chemicals else 100.0
        consumed_mol = min(coeff * xi_t, init_mol)
        consumed_g[cid] = consumed_mol * mw

    produced_g: Dict[str, float] = {}
    for (pid, coeff) in reaction.products:
        mw = chemicals[pid].molecular_weight if pid in chemicals else 100.0
        produced_g[pid] = coeff * xi_t * mw

    byproduct_g: Dict[str, float] = {}
    for bid, bcoeff in reaction.byproducts:
        if bid in chemicals:
            mw = chemicals[bid].molecular_weight
            byproduct_g[bid] = bcoeff * xi_t * mw * 0.1

    total_consumed = sum(consumed_g.values())
    total_produced = sum(produced_g.values()) + sum(byproduct_g.values())
    if total_produced > total_consumed and total_consumed > 1e-12:
        scale = total_consumed / total_produced
        produced_g = {k: v * scale for k, v in produced_g.items()}
        byproduct_g = {k: v * scale for k, v in byproduct_g.items()}

    heat_released_J = xi_t * (-reaction.delta_H_kJ) * 1000.0

    return consumed_g, produced_g, byproduct_g, k_eff, heat_released_J


def _update_vessel_state(
    vessel: VesselState,
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
    heat_released_J: float,
    dt: float,
    initial_gas_moles: float,
    heat_transfer_coeff: float = 0.0,
    max_heat_rate_W: float = 0.0,
) -> None:
    old_temp = vessel.temperature_C

    if vessel.thermal_mode == "isothermal":
        if max_heat_rate_W > 0 and dt > 0:
            heat_rate_W = abs(heat_released_J) / dt
            if heat_rate_W > max_heat_rate_W:
                excess_J = (heat_rate_W - max_heat_rate_W) * dt
                if heat_released_J > 0:
                    excess_heat = excess_J
                else:
                    excess_heat = -excess_J
                Cp_total = _mixture_heat_capacity(pool, chemicals)
                dT = excess_heat / Cp_total
                phase_heat = _process_phase_transitions(pool, chemicals, old_temp, old_temp + dT, vessel.pressure_atm)
                vessel.temperature_C += (excess_heat - phase_heat) / Cp_total
    elif vessel.thermal_mode == "adiabatic":
        Cp_total = _mixture_heat_capacity(pool, chemicals)
        dT = heat_released_J / Cp_total
        phase_heat = _process_phase_transitions(pool, chemicals, old_temp, old_temp + dT, vessel.pressure_atm)
        net_heat = heat_released_J - phase_heat
        dT_corrected = net_heat / Cp_total
        vessel.temperature_C += dT_corrected
    elif vessel.thermal_mode == "open_air":
        Cp_total = _mixture_heat_capacity(pool, chemicals)
        heat_loss_J = heat_transfer_coeff * Cp_total * (vessel.temperature_C - AMBIENT_TEMP_C) * dt
        net_heat = heat_released_J - heat_loss_J
        tentative_dT = net_heat / Cp_total
        phase_heat = _process_phase_transitions(pool, chemicals, old_temp, old_temp + tentative_dT, vessel.pressure_atm)
        dT_corrected = (net_heat - phase_heat) / Cp_total
        vessel.temperature_C += dT_corrected
    elif vessel.thermal_mode == "heating":
        Cp_total = _mixture_heat_capacity(pool, chemicals)
        dT_heating = vessel.heating_rate_C_per_s * dt
        dT_reaction = heat_released_J / Cp_total
        total_dT = dT_heating + dT_reaction
        phase_heat = _process_phase_transitions(pool, chemicals, old_temp, old_temp + total_dT, vessel.pressure_atm)
        net_dT = (heat_released_J + vessel.heating_rate_C_per_s * dt * Cp_total - phase_heat) / Cp_total
        vessel.temperature_C += net_dT
    elif vessel.thermal_mode == "cooling":
        Cp_total = _mixture_heat_capacity(pool, chemicals)
        dT_cooling = vessel.heating_rate_C_per_s * dt
        dT_reaction = heat_released_J / Cp_total
        total_dT = dT_cooling + dT_reaction
        phase_heat = _process_phase_transitions(pool, chemicals, old_temp, old_temp + total_dT, vessel.pressure_atm)
        net_dT = (heat_released_J + vessel.heating_rate_C_per_s * dt * Cp_total - phase_heat) / Cp_total
        vessel.temperature_C += net_dT

    vessel.temperature_C = float(np.clip(vessel.temperature_C, -273.0, 5000.0))

    if vessel.vessel_type == "sealed":
        current_gas_moles = _count_gas_moles(pool, chemicals, vessel.temperature_C, vessel.pressure_atm)
        T_K_now = vessel.temperature_C + 273.15
        T_K_init = vessel.initial_temp_C + 273.15
        if initial_gas_moles > 1e-9:
            vessel.pressure_atm = vessel.initial_pressure_atm * (current_gas_moles / initial_gas_moles) * (T_K_now / T_K_init)
        elif current_gas_moles > 1e-9:
            new_P = current_gas_moles * R_J * T_K_now / (vessel.volume_L * 0.001)
            vessel.pressure_atm = new_P / 101325.0
        vessel.pressure_atm = float(np.clip(vessel.pressure_atm, 0.001, 1000.0))


def simulate_reaction(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    reactant_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
    catalyst_amounts_g: Optional[Dict[str, float]] = None,
) -> Dict:
    T_K = temperature_C + 273.15

    pool_for_solvent = dict(reactant_amounts_g)
    if catalyst_amounts_g:
        for cid, g in catalyst_amounts_g.items():
            pool_for_solvent[cid] = pool_for_solvent.get(cid, 0.0) + g
    solvent_id = _find_common_solvent(reaction, chemicals, pool_for_solvent, temperature_C, pressure_atm)

    dissolution_factor = 1.0
    solvent_is_reactant = solvent_id in {cid for cid, _ in reaction.reactants} if solvent_id else False
    if solvent_id is None:
        dissolution_factor = 0.0
    elif solvent_id == "__neat__":
        dissolution_factor = 0.3
    elif solvent_id == "__self__":
        dissolution_factor = 1.0
    elif solvent_is_reactant:
        dissolution_factor = 1.0
    else:
        min_frac = 1.0
        for cid, _ in reaction.reactants:
            frac = _dissolved_fraction(cid, solvent_id, pool_for_solvent, chemicals)
            min_frac = min(min_frac, frac)
        dissolution_factor = min_frac

    K_eq = k_eq_at_T(reaction, T_K)
    k_rate = rate_constant_at_T(reaction, T_K)
    pf = phase_factor_for_reaction(reaction, chemicals, temperature_C, pressure_atm)

    reactant_total_g = sum(reactant_amounts_g.values())
    cat_factor = catalyst_acceleration(catalyst_amounts_g or {}, reactant_total_g)
    k_eff = k_rate * pf * cat_factor * dissolution_factor

    reactant_amounts_mol = []
    reactant_coeffs = []
    for cid, coeff in reaction.reactants:
        mw = chemicals[cid].molecular_weight if cid in chemicals else 100.0
        grams = reactant_amounts_g.get(cid, 0.0)
        moles = grams / mw
        effective_moles = moles * dissolution_factor
        reactant_amounts_mol.append(effective_moles)
        reactant_coeffs.append(coeff)

    product_amounts_mol = [0.0] * len(reaction.products)
    product_coeffs = [coeff for _, coeff in reaction.products]

    xi_eq = solve_equilibrium_extent(
        K_eq, reactant_amounts_mol, reactant_coeffs, product_amounts_mol, product_coeffs
    )

    xi_t = xi_eq * (1.0 - np.exp(-k_eff * duration_s))
    xi_t = float(np.clip(xi_t, 0.0, xi_eq))

    conversion = xi_t / xi_eq if xi_eq > 1e-30 else 0.0
    conversion = float(np.clip(conversion, 0.0, 1.0))

    reached_eq = conversion > 0.95

    consumed_g: Dict[str, float] = {}
    for (cid, coeff), init_mol in zip(reaction.reactants, reactant_amounts_mol):
        mw = chemicals[cid].molecular_weight if cid in chemicals else 100.0
        consumed_mol = coeff * xi_t
        consumed_mol = min(consumed_mol, init_mol)
        consumed_g[cid] = consumed_mol * mw

    produced_g: Dict[str, float] = {}
    for (pid, coeff), init_mol in zip(reaction.products, product_amounts_mol):
        mw = chemicals[pid].molecular_weight if pid in chemicals else 100.0
        produced_mol = coeff * xi_t
        produced_g[pid] = produced_mol * mw

    byproduct_g: Dict[str, float] = {}
    for bid, bcoeff in reaction.byproducts:
        if bid in chemicals:
            mw = chemicals[bid].molecular_weight
            byproduct_g[bid] = bcoeff * xi_t * mw * 0.1

    total_consumed = sum(consumed_g.values())
    total_produced = sum(produced_g.values()) + sum(byproduct_g.values())
    if total_produced > total_consumed and total_consumed > 1e-12:
        scale = total_consumed / total_produced
        produced_g = {k: v * scale for k, v in produced_g.items()}
        byproduct_g = {k: v * scale for k, v in byproduct_g.items()}

    return {
        "xi_equilibrium": xi_eq,
        "xi_achieved": xi_t,
        "conversion": conversion,
        "reached_equilibrium": reached_eq,
        "consumed_g": consumed_g,
        "produced_g": produced_g,
        "byproduct_g": byproduct_g,
        "K_eq": K_eq,
        "k_eff": k_eff,
    }


GAS_RETENTION_RATES = {
    "open_beaker": 0.0,
    "reflux_condenser": 0.75,
}


def _apply_gas_loss(
    pool: Dict[str, float],
    chemicals: Dict[str, Chemical],
    vessel: VesselState,
    equipment: Optional[str],
) -> Dict[str, float]:
    if vessel.vessel_type == "sealed":
        return {}
    equip_name = equipment or "open_beaker"
    retention = GAS_RETENTION_RATES.get(equip_name, 0.0)
    lost: Dict[str, float] = {}
    for cid in list(pool.keys()):
        if pool.get(cid, 0) < 1e-9:
            continue
        if cid not in chemicals:
            continue
        if state_at(chemicals[cid], vessel.temperature_C, vessel.pressure_atm) == "gas":
            g = pool[cid]
            escaped = g * (1.0 - retention)
            if escaped > 1e-12:
                pool[cid] = g - escaped
                lost[cid] = escaped
    return lost


def simulate_chain_reaction(
    world: World,
    initial_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
    equipment: Optional[str] = None,
    heating_rate_C_per_s: float = 0.0,
    vessel_volume_L: float = 1.0,
    catalyst_ids: Optional[set] = None,
    step_seconds: float = DEFAULT_STEP_SECONDS,
) -> Dict:
    chemicals = world.chemicals
    reactions = world.reactions

    catalog = world.equipment
    equip_spec = catalog.get(equipment or "open_beaker", catalog["open_beaker"])
    vessel = VesselState(
        temperature_C=temperature_C,
        pressure_atm=pressure_atm,
        vessel_type=equip_spec["vessel_type"],
        thermal_mode=equip_spec["thermal_mode"],
        heating_rate_C_per_s=heating_rate_C_per_s,
        volume_L=vessel_volume_L,
    )

    if equipment and equip_spec["thermal_mode"] in ("heating", "cooling"):
        vessel.heating_rate_C_per_s = heating_rate_C_per_s

    pool: Dict[str, float] = {}
    for cid, g in initial_amounts_g.items():
        if g > 1e-9:
            pool[cid] = g

    initial_gas_moles = _count_gas_moles(pool, chemicals, temperature_C, pressure_atm)

    total_consumed: Dict[str, float] = {}
    total_produced: Dict[str, float] = {}
    total_byproduct: Dict[str, float] = {}
    total_gas_lost: Dict[str, float] = {}
    reactions_fired: Dict[str, int] = {}
    temp_history: List[Dict] = []
    reaction_log: List[Dict] = []

    num_steps = max(1, int(np.ceil(duration_s / step_seconds)))
    dt = duration_s / num_steps

    max_T = equip_spec.get("max_temp_C", 5000.0)
    min_T = equip_spec.get("min_temp_C", -273.0)
    max_P = equip_spec.get("max_pressure_atm", 1000.0)
    if vessel.temperature_C > max_T or vessel.temperature_C < min_T or vessel.pressure_atm > max_P:
        fail_reason = (
            f"Initial temperature {vessel.temperature_C:.1f}°C outside equipment range [{min_T}, {max_T}]°C"
            if vessel.temperature_C > max_T or vessel.temperature_C < min_T
            else f"Initial pressure {vessel.pressure_atm:.2f} atm exceeds equipment limit {max_P} atm"
        )
        return {
            "final_pool_g": {},
            "consumed_g": {},
            "produced_g": {},
            "byproduct_g": {},
            "net_consumed_g": {},
            "net_produced_g": {},
            "gas_lost_g": {},
            "reactions_fired": {},
            "chain_reaction": False,
            "num_steps_simulated": 0,
            "converged": False,
            "final_temperature_C": round(vessel.temperature_C, 2),
            "final_pressure_atm": round(vessel.pressure_atm, 4),
            "temperature_history": [],
            "reaction_log": [],
            "equipment": equipment or "open_beaker",
            "vessel_type": vessel.vessel_type,
            "thermal_mode": vessel.thermal_mode,
            "equipment_failure": True,
            "failure_reason": fail_reason,
        }

    converged_steps = 0

    for step_idx in range(num_steps):
        applicable = _find_applicable_reactions(
            pool, reactions, catalyst_ids,
            chemicals=chemicals, temperature_C=vessel.temperature_C, pressure_atm=vessel.pressure_atm,
        )
        if not applicable:
            converged_steps += 1
            if converged_steps >= 3:
                break
            continue

        step_consumed: Dict[str, float] = {}
        step_produced: Dict[str, float] = {}
        step_byproduct: Dict[str, float] = {}
        step_heat_J = 0.0
        step_had_progress = False

        for rxn in applicable:
            rxn_solvent = _find_common_solvent(rxn, chemicals, pool, vessel.temperature_C, vessel.pressure_atm)
            consumed, produced, byproducts, k_eff, heat_J = _step_single_reaction(
                rxn, chemicals, pool, vessel.temperature_C, vessel.pressure_atm, dt,
                solvent_id=rxn_solvent,
            )

            total_change = sum(consumed.values()) + sum(produced.values())
            if total_change < 1e-12:
                continue

            step_had_progress = True
            step_heat_J += heat_J
            reactions_fired[rxn.id] = reactions_fired.get(rxn.id, 0) + 1

            for cid, g in consumed.items():
                actual = min(g, pool.get(cid, 0.0))
                if actual > 1e-12:
                    pool[cid] = pool.get(cid, 0.0) - actual
                    step_consumed[cid] = step_consumed.get(cid, 0.0) + actual

            for cid, g in produced.items():
                if g > 1e-12:
                    pool[cid] = pool.get(cid, 0.0) + g
                    step_produced[cid] = step_produced.get(cid, 0.0) + g

            for cid, g in byproducts.items():
                if g > 1e-12:
                    pool[cid] = pool.get(cid, 0.0) + g
                    step_byproduct[cid] = step_byproduct.get(cid, 0.0) + g

        _update_vessel_state(vessel, pool, chemicals, step_heat_J, dt, initial_gas_moles,
                            heat_transfer_coeff=equip_spec.get("heat_transfer_coeff", 0.0),
                            max_heat_rate_W=equip_spec.get("max_heat_rate_W", 0.0))

        equipment_failed = False
        if vessel.temperature_C > max_T or vessel.temperature_C < min_T:
            equipment_failed = True
        if vessel.pressure_atm > max_P:
            equipment_failed = True

        if equipment_failed:
            pool.clear()
            return {
                "final_pool_g": {},
                "consumed_g": {k: v for k, v in total_consumed.items() if v > 1e-9},
                "produced_g": {},
                "byproduct_g": {},
                "net_consumed_g": {},
                "net_produced_g": {},
                "gas_lost_g": {},
                "reactions_fired": reactions_fired,
                "chain_reaction": len(reactions_fired) > 1,
                "num_steps_simulated": step_idx + 1,
                "converged": False,
                "final_temperature_C": round(vessel.temperature_C, 2),
                "final_pressure_atm": round(vessel.pressure_atm, 4),
                "temperature_history": temp_history[:30],
                "reaction_log": reaction_log[:20],
                "equipment": equipment or "open_beaker",
                "vessel_type": vessel.vessel_type,
                "thermal_mode": vessel.thermal_mode,
                "equipment_failure": True,
                "failure_reason": (
                    f"Temperature {vessel.temperature_C:.1f}°C exceeded equipment limit "
                    f"[{min_T}, {max_T}]°C"
                    if vessel.temperature_C > max_T or vessel.temperature_C < min_T
                    else f"Pressure {vessel.pressure_atm:.2f} atm exceeded equipment limit {max_P} atm"
                ),
            }

        step_gas_lost = _apply_gas_loss(pool, chemicals, vessel, equipment)
        for cid, g in step_gas_lost.items():
            total_gas_lost[cid] = total_gas_lost.get(cid, 0.0) + g

        if not step_had_progress:
            converged_steps += 1
            if converged_steps >= 3:
                break
        else:
            converged_steps = 0

        for cid, g in step_consumed.items():
            total_consumed[cid] = total_consumed.get(cid, 0.0) + g
        for cid, g in step_produced.items():
            total_produced[cid] = total_produced.get(cid, 0.0) + g
        for cid, g in step_byproduct.items():
            total_byproduct[cid] = total_byproduct.get(cid, 0.0) + g

        if step_idx < 5 or step_idx % max(1, num_steps // 20) == 0:
            temp_history.append({
                "step": step_idx,
                "time_s": round((step_idx + 1) * dt, 2),
                "temperature_C": round(vessel.temperature_C, 2),
                "pressure_atm": round(vessel.pressure_atm, 4),
            })

        if step_had_progress and (step_idx < 5 or step_idx % max(1, num_steps // 20) == 0):
            reaction_log.append({
                "step": step_idx,
                "time_s": round((step_idx + 1) * dt, 2),
                "reactions_active": len(applicable),
                "temperature_C": round(vessel.temperature_C, 2),
                "pressure_atm": round(vessel.pressure_atm, 4),
                "consumed": {k: round(v, 6) for k, v in step_consumed.items() if v > 1e-9},
                "produced": {k: round(v, 6) for k, v in step_produced.items() if v > 1e-9},
            })

    pool_clean = {cid: g for cid, g in pool.items() if g > 1e-9}

    net_consumed: Dict[str, float] = {}
    for cid, g in total_consumed.items():
        net = g - total_produced.get(cid, 0.0)
        if net > 1e-9:
            net_consumed[cid] = net

    net_produced: Dict[str, float] = {}
    for cid, g in total_produced.items():
        net = g - total_consumed.get(cid, 0.0)
        if net > 1e-9:
            net_produced[cid] = net

    chain_occurred = len(reactions_fired) > 1

    return {
        "final_pool_g": pool_clean,
        "consumed_g": {k: v for k, v in total_consumed.items() if v > 1e-9},
        "produced_g": {k: v for k, v in total_produced.items() if v > 1e-9},
        "byproduct_g": {k: v for k, v in total_byproduct.items() if v > 1e-9},
        "net_consumed_g": net_consumed,
        "net_produced_g": net_produced,
        "reactions_fired": reactions_fired,
        "chain_reaction": chain_occurred,
        "num_steps_simulated": num_steps,
        "converged": converged_steps >= 3,
        "final_temperature_C": round(vessel.temperature_C, 2),
        "final_pressure_atm": round(vessel.pressure_atm, 4),
        "temperature_history": temp_history[:30],
        "reaction_log": reaction_log[:20],
        "gas_lost_g": {k: v for k, v in total_gas_lost.items() if v > 1e-9},
        "equipment": equipment or "open_beaker",
        "vessel_type": vessel.vessel_type,
        "thermal_mode": vessel.thermal_mode,
    }
