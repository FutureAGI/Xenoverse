from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from world_gen.models import Chemical, Reaction

R_kJ = 8.314e-3  # kJ/(mol·K)

PHASE_FACTORS = {"solid": 0.01, "liquid": 1.0, "gas": 8.0}


def state_at(chem: Chemical, temp_C: float, pressure_atm: float = 1.0) -> str:
    bp_adj = chem.boiling_point + 10 * np.log(max(0.01, pressure_atm))
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


def phase_factor_for_reaction(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    temp_C: float,
    pressure_atm: float,
) -> float:
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


def simulate_reaction(
    reaction: Reaction,
    chemicals: Dict[str, Chemical],
    reactant_amounts_g: Dict[str, float],
    temperature_C: float,
    pressure_atm: float,
    duration_s: float,
) -> Dict:
    T_K = temperature_C + 273.15

    K_eq = k_eq_at_T(reaction, T_K)
    k_rate = rate_constant_at_T(reaction, T_K)
    phase_factor = phase_factor_for_reaction(reaction, chemicals, temperature_C, pressure_atm)
    k_eff = k_rate * phase_factor

    reactant_amounts_mol = []
    reactant_coeffs = []
    for cid, coeff in reaction.reactants:
        mw = chemicals[cid].molecular_weight if cid in chemicals else 100.0
        grams = reactant_amounts_g.get(cid, 0.0)
        moles = grams / mw
        reactant_amounts_mol.append(moles)
        reactant_coeffs.append(coeff)

    product_amounts_mol = [0.0] * len(reaction.products)
    product_coeffs = [coeff for _, coeff in reaction.products]

    xi_eq = solve_equilibrium_extent(
        K_eq, reactant_amounts_mol, reactant_coeffs, product_amounts_mol, product_coeffs
    )

    # Time evolution: extent(t) = xi_eq * (1 - exp(-k_eff * t))
    xi_t = xi_eq * (1.0 - np.exp(-k_eff * duration_s))
    xi_t = float(np.clip(xi_t, 0.0, xi_eq))

    conversion = xi_t / xi_eq if xi_eq > 1e-30 else 0.0
    conversion = float(np.clip(conversion, 0.0, 1.0))

    reached_eq = conversion > 0.95

    # Compute consumed reactants and produced products in grams
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
