from __future__ import annotations

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import Chemical, EQUIPMENT_CATALOG, Reaction, World
from .sampler import _OPENERS, _MIDDLES, _ENDINGS, COMPLEXITY_PRESETS

R_kJ = 8.314e-3

BACKWARD_COMPLEXITY_PRESETS = {
    "easy": {
        "num_layers": 3,
        "layer1_count": (5, 7),
        "num_solvents": (3, 4),
        "optimal_path_length": (2, 3),
        "decoy_reactions_per_layer": (3, 5),
        "filler_compounds_per_layer": (2, 4),
        "min_total_reactions": 10,
    },
    "medium": {
        "num_layers": 4,
        "layer1_count": (7, 10),
        "num_solvents": (3, 5),
        "optimal_path_length": (3, 4),
        "decoy_reactions_per_layer": (5, 8),
        "filler_compounds_per_layer": (3, 6),
        "min_total_reactions": 20,
    },
    "hard": {
        "num_layers": 5,
        "layer1_count": (10, 14),
        "num_solvents": (4, 6),
        "optimal_path_length": (4, 5),
        "decoy_reactions_per_layer": (7, 12),
        "filler_compounds_per_layer": (4, 8),
        "min_total_reactions": 35,
    },
}


class BackwardDesignSampler:
    def __init__(
        self,
        seed: int,
        complexity_level: str = "easy",
        constraints: Optional[Dict[str, Any]] = None,
    ):
        self.seed = seed
        self.complexity_level = complexity_level
        self.constraints = constraints or {}
        self._rng = np.random.RandomState(seed)
        random.seed(seed)
        self._chem_counter = 0
        self._rxn_counter = 0
        self._used_names: set = set()

        preset = BACKWARD_COMPLEXITY_PRESETS.get(complexity_level, BACKWARD_COMPLEXITY_PRESETS["easy"])
        self._preset = preset

    def _next_chem_id(self) -> str:
        self._chem_counter += 1
        return f"C{self._chem_counter:03d}"

    def _next_rxn_id(self) -> str:
        self._rxn_counter += 1
        return f"R{self._rxn_counter:03d}"

    def _generate_name(self) -> str:
        for _ in range(400):
            r = random.random()
            if r < 0.55:
                name = random.choice(_OPENERS) + random.choice(_ENDINGS)
            elif r < 0.85:
                name = random.choice(_OPENERS) + random.choice(_MIDDLES) + random.choice(_ENDINGS)
            else:
                name = random.choice(_OPENERS) + random.choice(_ENDINGS) + str(random.randint(2, 9))
            if name not in self._used_names and 6 <= len(name) <= 13:
                self._used_names.add(name)
                return name
        name = random.choice(_OPENERS) + random.choice(_ENDINGS) + str(random.randint(100, 9999))
        self._used_names.add(name)
        return name

    def _sample_physical_props(self) -> Dict[str, float]:
        mw = float(self._rng.uniform(30, 400))
        if self._rng.random() < 0.4:
            mp = float(self._rng.uniform(-100, 0))
        elif self._rng.random() < 0.5:
            mp = float(self._rng.uniform(0, 150))
        else:
            mp = float(self._rng.uniform(150, 500))
        bp_gap = float(self._rng.gamma(shape=2.0, scale=60) + 40)
        bp = mp + bp_gap
        clausius_C = float(self._rng.uniform(30, 80))
        heat_cap = float(np.clip(self._rng.lognormal(0.5, 0.5), 0.5, 10.0))
        latent_fusion = float(np.clip(self._rng.lognormal(4.5, 0.6), 20.0, 500.0))
        latent_vap = float(np.clip(self._rng.lognormal(6.0, 0.5), 100.0, 3000.0))
        return {
            "molecular_weight": round(mw, 2),
            "melting_point": round(mp, 2),
            "boiling_point": round(bp, 2),
            "clausius_C": round(clausius_C, 2),
            "heat_capacity_J_per_gK": round(heat_cap, 4),
            "latent_heat_fusion_J_per_g": round(latent_fusion, 2),
            "latent_heat_vaporization_J_per_g": round(latent_vap, 2),
        }

    def _sample_solvent(self, bp_range: Tuple[float, float] = (60, 200)) -> Chemical:
        chem_id = self._next_chem_id()
        name = self._generate_name()
        mw = float(self._rng.uniform(30, 150))
        mp = float(self._rng.uniform(-120, 5))
        bp = float(self._rng.uniform(*bp_range))
        if mp >= bp - 20:
            mp = bp - 40
        clausius_C = float(self._rng.uniform(30, 80))
        price = float(self._rng.uniform(0.01, 0.08))
        return Chemical(
            id=chem_id,
            name=name,
            layer=1,
            molecular_weight=round(mw, 2),
            melting_point=round(mp, 2),
            boiling_point=round(bp, 2),
            base_toxicity=round(float(self._rng.uniform(0.1, 1.5)), 3),
            medicinal_expected=0.0,
            medicinal_efficacy=0.0,
            price_per_gram=round(price, 4),
            heat_capacity_J_per_gK=round(float(self._rng.uniform(1.5, 4.0)), 4),
            latent_heat_fusion_J_per_g=round(float(self._rng.uniform(80, 200)), 2),
            latent_heat_vaporization_J_per_g=round(float(self._rng.uniform(300, 1200)), 2),
            clausius_C=round(clausius_C, 2),
            is_solvent=True,
        )

    def _sample_cost_params(self) -> Dict[str, float]:
        return {
            "heating_coeff": float(self._rng.uniform(0.5, 1.2)),
            "cooling_coeff": float(self._rng.uniform(0.8, 1.8)),
            "heating_exponent": float(self._rng.uniform(1.2, 1.8)),
            "cooling_exponent": float(self._rng.uniform(1.0, 1.6)),
            "pressure_high_coeff": float(self._rng.uniform(1.0, 2.5)),
            "pressure_low_coeff": float(self._rng.uniform(1.0, 2.5)),
            "pressure_high_exp": float(self._rng.uniform(0.5, 1.0)),
            "pressure_low_exp": float(self._rng.uniform(0.4, 0.8)),
            "equipment_base": float(self._rng.uniform(3.0, 8.0)),
            "equipment_pressure_coeff": float(self._rng.uniform(0.2, 0.5)),
            "duration_coeff": float(self._rng.uniform(0.02, 0.1)),
        }

    def _sample_equipment(self) -> Dict[str, Dict]:
        equipment = copy.deepcopy(EQUIPMENT_CATALOG)
        for name, spec in equipment.items():
            spec["max_capacity_g"] = round(
                spec["max_capacity_g"] * float(self._rng.uniform(0.7, 1.5)), 0
            )
            spec["max_temp_C"] = round(
                spec["max_temp_C"] * float(self._rng.uniform(0.8, 1.2)), 0
            )
            spec["min_temp_C"] = round(
                spec["min_temp_C"] * float(self._rng.uniform(0.8, 1.2)), 0
            )
            spec["max_pressure_atm"] = round(
                spec["max_pressure_atm"] * float(self._rng.uniform(0.7, 1.4)), 1
            )
            spec["base_cost_per_hour"] = round(
                spec["base_cost_per_hour"] * float(self._rng.uniform(0.6, 1.6)), 2
            )
            spec["cost_multiplier"] = round(
                spec["cost_multiplier"] * float(self._rng.uniform(0.7, 1.4)), 2
            )
        return equipment

    def _design_optimal_thermo(self, target_T_C: float) -> Tuple[float, float, float, float]:
        """Design reaction thermodynamics so optimal temperature is near target_T_C.

        Returns (delta_G_kJ, delta_H_kJ, Ea_kJ, log_A).
        The parameters are chosen so k(T_opt) ~ 0.02 s^-1 (reaching ~99% in 300s).
        K_eq is very large (delta_G strongly negative) to ensure near-complete conversion.
        """
        T_K = target_T_C + 273.15
        delta_G = float(self._rng.uniform(-120, -50))
        delta_S = float(self._rng.normal(0.0, 0.08))
        delta_H = delta_G + 298.0 * delta_S

        target_k = 0.02
        log_A = float(self._rng.uniform(9.5, 13.0))
        A = 10.0 ** log_A
        Ea = R_kJ * T_K * math.log(A / target_k)
        Ea = float(np.clip(Ea, 20.0, 160.0))

        return round(delta_G, 3), round(delta_H, 3), round(Ea, 3), round(log_A, 4)

    def _design_suboptimal_thermo(self) -> Tuple[float, float, float, float]:
        """Thermodynamics for decoy reactions — higher Ea or less favorable delta_G."""
        if self._rng.random() < 0.5:
            delta_G = float(self._rng.uniform(-10, 40))
        else:
            delta_G = float(self._rng.uniform(-50, -10))

        delta_S = float(self._rng.normal(0.0, 0.20))
        delta_H = delta_G + 298.0 * delta_S
        Ea = float(np.clip(self._rng.gamma(3.0, 60.0) + 30.0, 60.0, 300.0))
        log_A = float(np.clip(self._rng.normal(10.5, 2.5), 5.5, 16.5))
        return round(delta_G, 3), round(delta_H, 3), round(Ea, 3), round(log_A, 4)

    def sample_world(self, world_id: str) -> World:
        world = World(world_id=world_id, seed=self.seed)
        preset = self._preset

        world.cost_params = self._sample_cost_params()
        world.equipment = self._sample_equipment()

        num_layers = preset["num_layers"]
        n_solvents = int(self._rng.randint(*preset["num_solvents"]))

        bp_targets = sorted(self._rng.uniform(70, 220, size=n_solvents))
        solvents = []
        for i in range(n_solvents):
            bp_lo = max(50, bp_targets[i] - 15)
            bp_hi = bp_targets[i] + 15
            s = self._sample_solvent(bp_range=(bp_lo, bp_hi))
            solvents.append(s)
            world.chemicals[s.id] = s

        path_length = int(self._rng.randint(*preset["optimal_path_length"]))
        path_length = min(path_length, num_layers - 1)

        target_layer = min(path_length + 1, num_layers)

        max_tox = self.constraints.get("max_toxicity", 4.0)
        min_med = self.constraints.get("min_medicinal", 2.0)
        required_phase = self.constraints.get("required_phase")
        phase_temp = self.constraints.get("phase_temp_C", 25.0)

        target_props = self._sample_physical_props()
        if required_phase == "liquid":
            target_props["melting_point"] = round(float(self._rng.uniform(phase_temp - 80, phase_temp - 10)), 2)
            target_props["boiling_point"] = round(float(self._rng.uniform(phase_temp + 20, phase_temp + 150)), 2)
        elif required_phase == "solid":
            target_props["melting_point"] = round(float(self._rng.uniform(phase_temp + 5, phase_temp + 100)), 2)
            target_props["boiling_point"] = round(target_props["melting_point"] + float(self._rng.uniform(50, 200)), 2)
        else:
            target_props["melting_point"] = round(float(self._rng.uniform(-50, 50)), 2)
            target_props["boiling_point"] = round(float(self._rng.uniform(100, 300)), 2)

        target_toxicity = float(self._rng.uniform(max(0.1, max_tox - 2.5), max_tox - 0.3))
        target_med_value = float(self._rng.uniform(min_med + 0.2, min_med + 2.0))
        med_expected = float(self._rng.uniform(target_med_value * 0.8, target_med_value * 1.5))
        med_efficacy = target_med_value / med_expected

        target_chem = Chemical(
            id=self._next_chem_id(),
            name=self._generate_name(),
            layer=target_layer,
            molecular_weight=target_props["molecular_weight"],
            melting_point=target_props["melting_point"],
            boiling_point=target_props["boiling_point"],
            base_toxicity=round(target_toxicity, 3),
            medicinal_expected=round(med_expected, 3),
            medicinal_efficacy=round(med_efficacy, 4),
            clausius_C=target_props["clausius_C"],
            heat_capacity_J_per_gK=target_props["heat_capacity_J_per_gK"],
            latent_heat_fusion_J_per_g=target_props["latent_heat_fusion_J_per_g"],
            latent_heat_vaporization_J_per_g=target_props["latent_heat_vaporization_J_per_g"],
        )
        world.chemicals[target_chem.id] = target_chem

        chemicals_by_layer: Dict[int, List[Chemical]] = {1: list(solvents)}

        n_l1 = int(self._rng.randint(*preset["layer1_count"]))
        l1_non_solvents: List[Chemical] = []
        for _ in range(n_l1):
            props = self._sample_physical_props()
            props["melting_point"] = round(float(self._rng.uniform(-60, 40)), 2)
            props["boiling_point"] = round(float(self._rng.uniform(
                max(props["melting_point"] + 60, 120), 350
            )), 2)
            toxicity = float(self._rng.uniform(0, 8))
            k = 1
            med_exp = float(self._rng.beta(k, 6 - k) * 5)
            med_eff = float(self._rng.beta(0.3, 3.0))
            price = float(self._rng.lognormal(mean=1.5, sigma=0.8))
            chem = Chemical(
                id=self._next_chem_id(),
                name=self._generate_name(),
                layer=1,
                molecular_weight=props["molecular_weight"],
                melting_point=props["melting_point"],
                boiling_point=props["boiling_point"],
                base_toxicity=round(toxicity, 3),
                medicinal_expected=round(med_exp, 3),
                medicinal_efficacy=round(med_eff, 4),
                price_per_gram=round(price, 4),
                clausius_C=props["clausius_C"],
                heat_capacity_J_per_gK=props["heat_capacity_J_per_gK"],
                latent_heat_fusion_J_per_g=props["latent_heat_fusion_J_per_g"],
                latent_heat_vaporization_J_per_g=props["latent_heat_vaporization_J_per_g"],
            )
            l1_non_solvents.append(chem)
            world.chemicals[chem.id] = chem

        chemicals_by_layer[1] = list(solvents) + l1_non_solvents

        for layer in range(2, num_layers + 1):
            chemicals_by_layer.setdefault(layer, [])

        chemicals_by_layer[target_layer].append(target_chem)

        optimal_path_chems: List[Chemical] = [target_chem]
        optimal_path_reactions: List[Reaction] = []
        optimal_intermediates: List[Chemical] = []

        optimal_temps = []
        for step in range(path_length):
            base_T = float(self._rng.uniform(60, 200))
            optimal_temps.append(base_T)

        solvent_assignment: List[str] = []
        available_solvents = list(solvents)
        for step in range(path_length):
            T_step = optimal_temps[step]
            valid_solvents = [
                s for s in available_solvents
                if s.melting_point < T_step < s.boiling_point + s.clausius_C * math.log(max(0.01, 1.0))
            ]
            if not valid_solvents:
                valid_solvents = sorted(available_solvents, key=lambda s: abs(s.boiling_point - T_step - 20))[:2]
                if valid_solvents:
                    chosen = valid_solvents[0]
                    if chosen.boiling_point <= T_step:
                        optimal_temps[step] = chosen.boiling_point - 10
                else:
                    chosen = available_solvents[0]
            else:
                if len(solvent_assignment) > 0:
                    prev_sid = solvent_assignment[-1]
                    diff_solvents = [s for s in valid_solvents if s.id != prev_sid]
                    if diff_solvents:
                        valid_solvents = diff_solvents
                chosen = random.choice(valid_solvents)
            solvent_assignment.append(chosen.id)

        for step_idx in range(path_length - 1, -1, -1):
            current_product = optimal_path_chems[0]
            product_layer = current_product.layer
            precursor_layer = product_layer - 1

            if precursor_layer == 1:
                available_precursors = [c for c in l1_non_solvents]
            else:
                available_precursors = [c for c in optimal_intermediates if c.layer == precursor_layer]
                if len(available_precursors) < 2:
                    for _ in range(2 - len(available_precursors)):
                        props = self._sample_physical_props()
                        props["melting_point"] = min(props["melting_point"], 40.0)
                        props["boiling_point"] = max(props["boiling_point"], 120.0)
                        chem = Chemical(
                            id=self._next_chem_id(),
                            name=self._generate_name(),
                            layer=precursor_layer,
                            molecular_weight=props["molecular_weight"],
                            melting_point=props["melting_point"],
                            boiling_point=props["boiling_point"],
                            base_toxicity=round(float(self._rng.uniform(0, 8)), 3),
                            medicinal_expected=round(float(self._rng.beta(min(precursor_layer, 4), 5) * 6), 3),
                            medicinal_efficacy=round(float(self._rng.beta(0.4, 2.5)), 4),
                            clausius_C=props["clausius_C"],
                            heat_capacity_J_per_gK=props["heat_capacity_J_per_gK"],
                            latent_heat_fusion_J_per_g=props["latent_heat_fusion_J_per_g"],
                            latent_heat_vaporization_J_per_g=props["latent_heat_vaporization_J_per_g"],
                        )
                        chemicals_by_layer.setdefault(precursor_layer, []).append(chem)
                        world.chemicals[chem.id] = chem
                        available_precursors.append(chem)
                        optimal_intermediates.append(chem)

            also_from_l1 = [c for c in l1_non_solvents if c.id not in {p.id for p in available_precursors}]

            must_have_prev = True
            if precursor_layer >= 2:
                n_from_prev = int(self._rng.randint(1, min(3, len(available_precursors) + 1)))
                selected_prev = random.sample(available_precursors, min(n_from_prev, len(available_precursors)))
                n_from_l1 = int(self._rng.randint(1, min(3, len(also_from_l1) + 1)))
                selected_l1 = random.sample(also_from_l1, min(n_from_l1, len(also_from_l1)))
                selected_precursors = selected_prev + selected_l1
            else:
                n_reactants = int(self._rng.randint(2, min(4, len(available_precursors) + 1)))
                selected_precursors = random.sample(available_precursors, n_reactants)

            seen_ids = set()
            unique_precursors = []
            for c in selected_precursors:
                if c.id not in seen_ids:
                    seen_ids.add(c.id)
                    unique_precursors.append(c)
            selected_precursors = unique_precursors
            reactants = [(c.id, int(self._rng.randint(1, 2))) for c in selected_precursors]

            step_solvent_id = solvent_assignment[step_idx] if step_idx < len(solvent_assignment) else solvents[0].id
            if step_solvent_id not in seen_ids:
                reactants.append((step_solvent_id, 1))

            catalyst_pool = [c for c in l1_non_solvents if c.id not in {cid for cid, _ in reactants}]
            catalysts = []
            if catalyst_pool and self._rng.random() < 0.6:
                cat = random.choice(catalyst_pool)
                catalysts = [cat.id]

            max_reactant_coeff = max(coeff for _, coeff in reactants)
            product_coeff = int(self._rng.randint(max_reactant_coeff, max_reactant_coeff + 2))
            products = [(current_product.id, product_coeff)]
            byproducts = []
            if self._rng.random() < 0.3:
                bp_pool = [c for c in chemicals_by_layer.get(1, []) if c.id not in {cid for cid, _ in reactants} and c.id != current_product.id]
                if bp_pool:
                    byproducts = [(random.choice(bp_pool).id, 1)]

            T_opt = optimal_temps[step_idx]
            min_reactant_bp = min(
                world.chemicals[cid].boiling_point for cid, _ in reactants
            )
            max_reactant_mp = max(
                world.chemicals[cid].melting_point for cid, _ in reactants
            )
            if T_opt > min_reactant_bp - 10.0:
                T_opt = min_reactant_bp - 10.0
            if T_opt < max_reactant_mp + 5.0:
                T_opt = max_reactant_mp + 5.0
            T_opt = max(30.0, min(T_opt, 400.0))
            optimal_temps[step_idx] = T_opt
            delta_G, delta_H, Ea, log_A = self._design_optimal_thermo(T_opt)

            reactant_mass_per_mol = sum(
                world.chemicals[cid].molecular_weight * coeff
                for cid, coeff in reactants
            )
            product_mass_per_mol = current_product.molecular_weight * product_coeff
            byproduct_mass_per_mol = sum(
                world.chemicals[bid].molecular_weight * bcoeff * 0.1
                for bid, bcoeff in byproducts
            )
            target_product_mass = reactant_mass_per_mol - byproduct_mass_per_mol
            if target_product_mass > 0 and product_coeff > 0:
                current_product.molecular_weight = round(target_product_mass / product_coeff, 2)

            rxn = Reaction(
                id=self._next_rxn_id(),
                reactants=reactants,
                catalysts=catalysts,
                products=products,
                byproducts=byproducts,
                delta_G_kJ=delta_G,
                delta_H_kJ=delta_H,
                activation_energy_kJ=Ea,
                log_A_factor=log_A,
            )
            optimal_path_reactions.insert(0, rxn)
            world.reactions[rxn.id] = rxn

            for c in selected_precursors:
                if c.layer >= 2 and c not in optimal_path_chems:
                    optimal_path_chems.insert(0, c)
                    if c not in optimal_intermediates:
                        optimal_intermediates.append(c)

        solvent_ids = [s.id for s in solvents]
        for step_idx, rxn in enumerate(optimal_path_reactions):
            sid = solvent_assignment[step_idx] if step_idx < len(solvent_assignment) else random.choice(solvent_ids)
            for cid, _ in rxn.reactants:
                chem = world.chemicals[cid]
                if chem.is_solvent:
                    continue
                chem.solubility[sid] = round(float(self._rng.uniform(25, 70)), 2)
                for other_sid in solvent_ids:
                    if other_sid != sid and other_sid not in chem.solubility:
                        if self._rng.random() < 0.4:
                            chem.solubility[other_sid] = round(float(self._rng.uniform(1, 8)), 2)

        for layer in range(2, num_layers + 1):
            n_filler = int(self._rng.randint(*preset["filler_compounds_per_layer"]))
            existing = len(chemicals_by_layer.get(layer, []))
            for _ in range(max(0, n_filler - existing)):
                props = self._sample_physical_props()
                toxicity = float(self._rng.uniform(0, 10))
                k = min(layer, 5)
                med_exp = float(self._rng.beta(k, 6 - k) * 8)
                med_eff = float(self._rng.beta(0.4, 2.5))
                if (med_exp * med_eff) >= min_med and toxicity < max_tox:
                    if self._rng.random() < 0.7:
                        toxicity = max_tox + float(self._rng.uniform(0.2, 2.0))
                    else:
                        med_eff *= 0.3

                chem = Chemical(
                    id=self._next_chem_id(),
                    name=self._generate_name(),
                    layer=layer,
                    molecular_weight=props["molecular_weight"],
                    melting_point=props["melting_point"],
                    boiling_point=props["boiling_point"],
                    base_toxicity=round(toxicity, 3),
                    medicinal_expected=round(med_exp, 3),
                    medicinal_efficacy=round(med_eff, 4),
                    clausius_C=props["clausius_C"],
                    heat_capacity_J_per_gK=props["heat_capacity_J_per_gK"],
                    latent_heat_fusion_J_per_g=props["latent_heat_fusion_J_per_g"],
                    latent_heat_vaporization_J_per_g=props["latent_heat_vaporization_J_per_g"],
                )
                chemicals_by_layer.setdefault(layer, []).append(chem)
                world.chemicals[chem.id] = chem

        for layer in range(2, num_layers + 1):
            layer_chems = chemicals_by_layer.get(layer, [])
            optimal_product_ids = {cid for rxn in optimal_path_reactions for cid, _ in rxn.products}
            uncovered = [c for c in layer_chems if c.id not in optimal_product_ids]

            for chem in uncovered:
                prev_layer_chems = chemicals_by_layer.get(layer - 1, [])
                if not prev_layer_chems:
                    continue
                all_prev_chems = []
                for l in range(1, layer):
                    all_prev_chems.extend(chemicals_by_layer.get(l, []))
                if len(all_prev_chems) < 2:
                    continue
                mandatory = random.choice(prev_layer_chems)
                other_pool = [c for c in all_prev_chems if c.id != mandatory.id]
                n_extra = int(self._rng.randint(1, min(3, len(other_pool) + 1)))
                extra = random.sample(other_pool, min(n_extra, len(other_pool)))
                reactant_chems = [mandatory] + extra
                reactants = [(c.id, int(self._rng.randint(1, 3))) for c in reactant_chems]
                products = [(chem.id, int(self._rng.randint(1, 2)))]
                delta_G, delta_H, Ea, log_A = self._design_suboptimal_thermo()
                rxn = Reaction(
                    id=self._next_rxn_id(),
                    reactants=reactants,
                    catalysts=[],
                    products=products,
                    byproducts=[],
                    delta_G_kJ=delta_G,
                    delta_H_kJ=delta_H,
                    activation_energy_kJ=Ea,
                    log_A_factor=log_A,
                )
                world.reactions[rxn.id] = rxn

        for layer in range(2, num_layers + 1):
            n_decoy = int(self._rng.randint(*preset["decoy_reactions_per_layer"]))
            for _ in range(n_decoy):
                self._add_decoy_reaction(world, chemicals_by_layer, layer)

        min_rxns = preset["min_total_reactions"]
        while len(world.reactions) < min_rxns:
            layer = int(self._rng.randint(2, num_layers + 1))
            self._add_decoy_reaction(world, chemicals_by_layer, layer)

        self._assign_remaining_solubility(world, solvents, chemicals_by_layer)

        world._solvents = solvents
        world._optimal_path_reactions = [r.id for r in optimal_path_reactions]
        world._target_id = target_chem.id

        return world

    def _add_decoy_reaction(
        self,
        world: World,
        chemicals_by_layer: Dict[int, List[Chemical]],
        target_layer: int,
    ) -> None:
        prev_chems = []
        for l in range(1, target_layer):
            prev_chems.extend(chemicals_by_layer.get(l, []))
        target_chems = chemicals_by_layer.get(target_layer, [])
        if len(prev_chems) < 2 or not target_chems:
            return

        prev_layer_chems = chemicals_by_layer.get(target_layer - 1, [])
        if not prev_layer_chems:
            return

        mandatory = random.choice(prev_layer_chems)
        other_pool = [c for c in prev_chems if c.id != mandatory.id]
        n_extra = int(self._rng.randint(1, min(4, len(other_pool) + 1)))
        extra_chems = random.sample(other_pool, min(n_extra, len(other_pool)))
        reactant_chems = [mandatory] + extra_chems
        reactants = [(c.id, int(self._rng.randint(1, 4))) for c in reactant_chems]

        n_p = int(self._rng.randint(1, min(3, len(target_chems) + 1)))
        product_chems = random.sample(target_chems, n_p)
        products = [(c.id, int(self._rng.randint(1, 3))) for c in product_chems]

        byproducts = []
        if self._rng.random() < 0.3:
            bp_pool = [c for c in prev_chems if c.id not in {cid for cid, _ in reactants + products}]
            if bp_pool:
                byproducts = [(random.choice(bp_pool).id, int(self._rng.randint(1, 2)))]

        catalyst_pool = [c for c in chemicals_by_layer.get(1, []) if c.id not in {cid for cid, _ in reactants}]
        catalysts = []
        if catalyst_pool and self._rng.random() < 0.3:
            catalysts = [random.choice(catalyst_pool).id]

        delta_G, delta_H, Ea, log_A = self._design_suboptimal_thermo()
        rxn = Reaction(
            id=self._next_rxn_id(),
            reactants=reactants,
            catalysts=catalysts,
            products=products,
            byproducts=byproducts,
            delta_G_kJ=delta_G,
            delta_H_kJ=delta_H,
            activation_energy_kJ=Ea,
            log_A_factor=log_A,
        )
        world.reactions[rxn.id] = rxn

    def _assign_remaining_solubility(
        self,
        world: World,
        solvents: List[Chemical],
        chemicals_by_layer: Dict[int, List[Chemical]],
    ) -> None:
        solvent_ids = [s.id for s in solvents]
        for chem in world.chemicals.values():
            if chem.is_solvent:
                continue
            if chem.solubility:
                for sid in solvent_ids:
                    if sid not in chem.solubility and self._rng.random() < 0.3:
                        chem.solubility[sid] = round(float(self._rng.uniform(0.5, 5.0)), 2)
            else:
                n_sol = int(self._rng.randint(1, min(3, len(solvent_ids)) + 1))
                chosen = random.sample(solvent_ids, n_sol)
                for sid in chosen:
                    chem.solubility[sid] = round(float(self._rng.lognormal(2.0, 0.8)), 2)
                    chem.solubility[sid] = round(np.clip(chem.solubility[sid], 1.0, 60.0), 2)

        for rxn in world.reactions.values():
            reactant_ids = [cid for cid, _ in rxn.reactants]
            non_solvent_reactants = [
                cid for cid in reactant_ids
                if cid in world.chemicals and not world.chemicals[cid].is_solvent
            ]
            if not non_solvent_reactants:
                continue
            common = set(solvent_ids)
            for cid in non_solvent_reactants:
                common &= set(world.chemicals[cid].solubility.keys())
            if not common:
                fallback = random.choice(solvent_ids)
                for cid in non_solvent_reactants:
                    if fallback not in world.chemicals[cid].solubility:
                        world.chemicals[cid].solubility[fallback] = round(
                            float(self._rng.uniform(2.0, 15.0)), 2
                        )

    def sample_unsolvable_world(self, world_id: str) -> World:
        world = World(world_id=world_id, seed=self.seed)
        preset = self._preset

        world.cost_params = self._sample_cost_params()
        world.equipment = self._sample_equipment()

        num_layers = preset["num_layers"]
        n_solvents = int(self._rng.randint(*preset["num_solvents"]))

        bp_targets = sorted(self._rng.uniform(70, 220, size=n_solvents))
        solvents = []
        for i in range(n_solvents):
            bp_lo = max(50, bp_targets[i] - 15)
            bp_hi = bp_targets[i] + 15
            s = self._sample_solvent(bp_range=(bp_lo, bp_hi))
            solvents.append(s)
            world.chemicals[s.id] = s

        max_tox = self.constraints.get("max_toxicity", 4.0)
        min_med = self.constraints.get("min_medicinal", 2.0)

        chemicals_by_layer: Dict[int, List[Chemical]] = {1: list(solvents)}

        n_l1 = int(self._rng.randint(*preset["layer1_count"]))
        l1_non_solvents: List[Chemical] = []
        for _ in range(n_l1):
            props = self._sample_physical_props()
            props["melting_point"] = round(float(self._rng.uniform(-60, 40)), 2)
            props["boiling_point"] = round(float(self._rng.uniform(
                max(props["melting_point"] + 60, 120), 350
            )), 2)
            toxicity = float(self._rng.uniform(0, 8))
            med_exp = float(self._rng.beta(1, 5) * 3)
            med_eff = float(self._rng.beta(0.3, 3.0))
            price = float(self._rng.lognormal(mean=1.5, sigma=0.8))
            chem = Chemical(
                id=self._next_chem_id(),
                name=self._generate_name(),
                layer=1,
                molecular_weight=props["molecular_weight"],
                melting_point=props["melting_point"],
                boiling_point=props["boiling_point"],
                base_toxicity=round(toxicity, 3),
                medicinal_expected=round(med_exp, 3),
                medicinal_efficacy=round(med_eff, 4),
                price_per_gram=round(price, 4),
                clausius_C=props["clausius_C"],
                heat_capacity_J_per_gK=props["heat_capacity_J_per_gK"],
                latent_heat_fusion_J_per_g=props["latent_heat_fusion_J_per_g"],
                latent_heat_vaporization_J_per_g=props["latent_heat_vaporization_J_per_g"],
            )
            l1_non_solvents.append(chem)
            world.chemicals[chem.id] = chem

        chemicals_by_layer[1] = list(solvents) + l1_non_solvents

        for layer in range(2, num_layers + 1):
            chemicals_by_layer.setdefault(layer, [])
            n_filler = int(self._rng.randint(*preset["filler_compounds_per_layer"]))
            for _ in range(n_filler):
                props = self._sample_physical_props()
                if self._rng.random() < 0.5:
                    toxicity = max_tox + float(self._rng.uniform(0.5, 3.0))
                    med_exp = float(self._rng.beta(2, 3) * 6)
                    med_eff = float(self._rng.uniform(0.3, 0.8))
                else:
                    toxicity = float(self._rng.uniform(0.5, max_tox - 0.3))
                    med_exp = float(self._rng.uniform(0.1, min_med * 0.4))
                    med_eff = float(self._rng.uniform(0.1, 0.5))

                chem = Chemical(
                    id=self._next_chem_id(),
                    name=self._generate_name(),
                    layer=layer,
                    molecular_weight=props["molecular_weight"],
                    melting_point=props["melting_point"],
                    boiling_point=props["boiling_point"],
                    base_toxicity=round(toxicity, 3),
                    medicinal_expected=round(med_exp, 3),
                    medicinal_efficacy=round(med_eff, 4),
                    clausius_C=props["clausius_C"],
                    heat_capacity_J_per_gK=props["heat_capacity_J_per_gK"],
                    latent_heat_fusion_J_per_g=props["latent_heat_fusion_J_per_g"],
                    latent_heat_vaporization_J_per_g=props["latent_heat_vaporization_J_per_g"],
                )
                chemicals_by_layer[layer].append(chem)
                world.chemicals[chem.id] = chem

        for layer in range(2, num_layers + 1):
            layer_chems = chemicals_by_layer.get(layer, [])
            for chem in layer_chems:
                prev_chems = []
                for l in range(1, layer):
                    prev_chems.extend(chemicals_by_layer.get(l, []))
                if len(prev_chems) < 2:
                    continue
                n_reactants = int(self._rng.randint(2, min(4, len(prev_chems) + 1)))
                reactant_chems = random.sample(prev_chems, n_reactants)
                reactants = [(c.id, int(self._rng.randint(1, 3))) for c in reactant_chems]
                products = [(chem.id, int(self._rng.randint(1, 2)))]

                delta_G, delta_H, Ea, log_A = self._design_suboptimal_thermo()
                if self._rng.random() < 0.4:
                    delta_G = float(self._rng.uniform(-80, -30))
                    Ea = float(self._rng.uniform(40, 100))
                    log_A = float(self._rng.uniform(9.0, 12.0))

                rxn = Reaction(
                    id=self._next_rxn_id(),
                    reactants=reactants,
                    catalysts=[],
                    products=products,
                    byproducts=[],
                    delta_G_kJ=delta_G,
                    delta_H_kJ=delta_H,
                    activation_energy_kJ=Ea,
                    log_A_factor=log_A,
                )
                world.reactions[rxn.id] = rxn

        n_decoy = int(self._rng.randint(3, 8))
        for _ in range(n_decoy):
            layer = int(self._rng.randint(2, num_layers + 1))
            self._add_decoy_reaction(world, chemicals_by_layer, layer)

        self._assign_remaining_solubility(world, solvents, chemicals_by_layer)

        world._solvents = solvents
        world._optimal_path_reactions = []
        world._target_id = None

        return world
