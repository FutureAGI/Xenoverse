from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from .models import Chemical, Reaction, World

# Name-generation vocabulary — fully invented syllables with no IUPAC / real-compound overlap.
# Three pools are combined in different patterns to maximise diversity:
#   2-part : OPENER + ENDING
#   3-part : OPENER + MIDDLE + ENDING
#   2-part with numeric tag : OPENER + ENDING + digit

_OPENERS = [
    # plosive / sharp
    "Grix", "Vrax", "Drix", "Blax", "Klom", "Trax", "Brix", "Plox",
    "Snex", "Fleck", "Blex", "Brex", "Drex", "Grex", "Klux", "Plex",
    # fricative / sibilant
    "Skav", "Sliv", "Sneth", "Sklor", "Skelf", "Slorn", "Skroth", "Snelv",
    "Zhiv", "Zilf", "Zwil", "Zeph", "Zorn", "Zovk", "Zlev", "Zreth",
    # voiced stops and blends
    "Drem", "Drev", "Doven", "Dwelv", "Brath", "Bleth", "Breth",
    "Grath", "Glev", "Greth", "Glovn", "Droven",
    # nasals and laterals
    "Menth", "Melv", "Morx", "Norx", "Neph", "Nelv", "Wulm", "Weth",
    "Lrev", "Lurv", "Lethk",
    # th / ph clusters
    "Thrix", "Thorm", "Thriv", "Torsh", "Torv", "Torph",
    "Phrox", "Phreth", "Treph", "Phlov",
    # labiovelar and unusual clusters
    "Quav", "Queth", "Quilv", "Vrex", "Vreth", "Vrith", "Vrelk", "Vrox",
    "Gwern", "Gwilv", "Gnex",
    # affricate-like openers
    "Forn", "Flem", "Fleck", "Frevl", "Klav", "Kelv", "Klorv", "Knelv",
    # short exotic syllables
    "Reth", "Strev", "Ulph", "Auvr", "Ceth", "Exk", "Yorn",
    "Vlex", "Grev", "Glex", "Trix", "Splov",
]

_MIDDLES = [
    "ar", "iv", "en", "ax", "eth", "ov", "em", "ath", "ox", "av",
    "ek", "or", "il", "ev", "ith", "ur", "al", "uf", "im", "esh",
    "elv", "orv", "irx", "arv", "ulm", "enk", "orn", "elth", "umv", "ixt",
    "ovr", "ark", "elf", "umb", "iph", "orx", "elph", "olvr",
]

_ENDINGS = [
    # crisp / short
    "yx", "ox", "ex", "ax", "ix", "ux",
    # medium voiced
    "elph", "orth", "aven", "oxen", "ixen", "evon", "ivon",
    "ulven", "olvex", "irvex", "ython", "axen", "urven",
    "ithon", "avex", "arxen", "irven", "erath", "alven", "irvon",
    # flowing / multisyllabic
    "ovex", "urath", "ixon", "orith", "iveth", "ormal",
    "exon", "olveth", "avorn", "ixeth", "urvex", "olvon",
    "exeth", "iroth", "avon", "elvorn", "ixorn", "ureth",
    "ovorn", "ithven", "ornex",
    # complex / alien-sounding
    "orthex", "ivorn", "axeth", "ombex", "irveth", "ulfon",
    "exorn", "uveth", "orlex", "ambex", "orchon", "elveth",
    "irnox", "axorn", "alvex", "ombrex", "olvorn", "ixelph",
]

COMPLEXITY_PRESETS = {
    "easy": {
        "layer1_min": 4,
        "layer1_max": 6,
        "last_layer_min": 2,
        "last_layer_max": 3,
        "num_layers_choices": [3],
        "extra_reactions_bonus": 2,
    },
    "medium": {
        "layer1_min": 6,
        "layer1_max": 10,
        "last_layer_min": 2,
        "last_layer_max": 5,
        "num_layers_choices": [3, 4],
        "extra_reactions_bonus": 3,
    },
    "hard": {
        "layer1_min": 8,
        "layer1_max": 14,
        "last_layer_min": 3,
        "last_layer_max": 7,
        "num_layers_choices": [4, 5, 6],
        "extra_reactions_bonus": 5,
    },
}

_DEFAULT_NUM_LAYERS_CHOICES = [3, 4, 5]
_DEFAULT_EXTRA_REACTIONS_BONUS = 3


class WorldSampler:
    def __init__(
        self,
        seed: int,
        complexity_level: Optional[str] = None,
        layer1_min: Optional[int] = None,
        layer1_max: Optional[int] = None,
        last_layer_min: Optional[int] = None,
        last_layer_max: Optional[int] = None,
    ):
        self.seed = seed

        if complexity_level is not None:
            if complexity_level not in COMPLEXITY_PRESETS:
                raise ValueError(f"Unknown complexity_level: {complexity_level}. Choose from: {list(COMPLEXITY_PRESETS.keys())}")
            preset = COMPLEXITY_PRESETS[complexity_level]
            self._num_layers_choices = preset["num_layers_choices"]
            self._extra_reactions_bonus = preset["extra_reactions_bonus"]
        else:
            preset = None
            self._num_layers_choices = _DEFAULT_NUM_LAYERS_CHOICES
            self._extra_reactions_bonus = _DEFAULT_EXTRA_REACTIONS_BONUS

        def _resolve(explicit, preset_key, default):
            if explicit is not None:
                return explicit
            if preset is not None:
                return preset[preset_key]
            return default

        self.layer1_min = max(1, _resolve(layer1_min, "layer1_min", 6))
        self.layer1_max = max(self.layer1_min, _resolve(layer1_max, "layer1_max", 10))
        self.last_layer_min = max(1, _resolve(last_layer_min, "last_layer_min", 2))
        self.last_layer_max = max(self.last_layer_min, _resolve(last_layer_max, "last_layer_max", 5))

        random.seed(seed)
        np.random.seed(seed)
        self._chem_counter = 0
        self._rxn_counter = 0
        self._used_names: set = set()

    def _sample_layer_sizes(self, num_layers: int) -> List[int]:
        """Return monotonically non-increasing compound counts per layer (pyramid).

        Layer 1  : [layer1_min, layer1_max]
        Last layer: [last_layer_min, min(prev, last_layer_max)]
        Middle    : [last_layer_min, prev]
        """
        sizes: List[int] = []
        prev = self.layer1_max  # upper bound before layer 1 is decided
        for l in range(1, num_layers + 1):
            is_last = (l == num_layers)
            if l == 1:
                lo, hi = self.layer1_min, self.layer1_max
            elif is_last:
                lo = self.last_layer_min
                hi = min(prev, self.last_layer_max)
                hi = max(hi, lo)  # guard: if prev < last_layer_min, clamp
            else:
                lo = self.last_layer_min  # ensure remaining layers can stay ≥ last_layer_min
                hi = prev
                lo = min(lo, hi)
            n = random.randint(lo, hi)
            sizes.append(n)
            prev = n
        return sizes

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
                # 2-part: opener + ending
                name = random.choice(_OPENERS) + random.choice(_ENDINGS)
            elif r < 0.85:
                # 3-part: opener + middle + ending
                name = random.choice(_OPENERS) + random.choice(_MIDDLES) + random.choice(_ENDINGS)
            else:
                # 2-part with a single trailing digit
                name = random.choice(_OPENERS) + random.choice(_ENDINGS) + str(random.randint(2, 9))
            if name not in self._used_names and 6 <= len(name) <= 13:
                self._used_names.add(name)
                return name
        # Fallback: guaranteed-unique by appending a large number
        name = random.choice(_OPENERS) + random.choice(_ENDINGS) + str(random.randint(100, 9999))
        self._used_names.add(name)
        return name

    def _sample_chemical(self, layer: int) -> Chemical:
        chem_id = self._next_chem_id()
        name = self._generate_name()
        mw = float(np.random.uniform(15, 450))
        mp = float(np.clip(np.random.normal(layer * 20, 150), -200, 3000))
        bp = float(mp + np.random.gamma(shape=2, scale=80))
        toxicity = float(np.random.uniform(0, 10))
        k = min(layer, 5)
        med_expected = float(np.random.beta(k, 6 - k) * 10)
        med_efficacy = float(np.random.beta(0.4, 2.5))
        price = float(np.random.lognormal(mean=1.5, sigma=0.8)) if layer == 1 else None
        heat_cap = float(np.random.lognormal(mean=0.5, sigma=0.5))
        heat_cap = round(np.clip(heat_cap, 0.5, 10.0), 4)
        latent_fusion = float(np.random.lognormal(mean=4.5, sigma=0.6))
        latent_fusion = round(np.clip(latent_fusion, 20.0, 500.0), 2)
        latent_vap = float(np.random.lognormal(mean=6.0, sigma=0.5))
        latent_vap = round(np.clip(latent_vap, 100.0, 3000.0), 2)
        return Chemical(
            id=chem_id,
            name=name,
            layer=layer,
            molecular_weight=round(mw, 2),
            melting_point=round(mp, 2),
            boiling_point=round(bp, 2),
            base_toxicity=round(toxicity, 3),
            medicinal_expected=round(med_expected, 3),
            medicinal_efficacy=round(med_efficacy, 4),
            price_per_gram=round(price, 4) if price is not None else None,
            heat_capacity_J_per_gK=heat_cap,
            latent_heat_fusion_J_per_g=latent_fusion,
            latent_heat_vaporization_J_per_g=latent_vap,
        )

    def _sample_thermo_favorable(self) -> Tuple[float, float, float, float]:
        """Thermodynamic parameters for *mandatory* reactions.

        Biased toward thermodynamic feasibility so that the guaranteed synthesis
        path to every compound is practically achievable:
          ΔG  — 80 % from Normal(-45, 40) (exergonic), 20 % Normal(15, 25) (mildly endergonic)
          Ea  — same broad Gamma as _sample_thermo but capped at 200 kJ/mol
        """
        if np.random.random() < 0.80:
            delta_G = float(np.random.normal(-45.0, 40.0))
        else:
            delta_G = float(np.random.normal(15.0, 25.0))
        delta_S = float(np.random.normal(0.0, 0.20))
        delta_H = delta_G + 298.0 * delta_S
        Ea    = float(np.clip(np.random.gamma(2.5, 45.0) + 8.0, 5.0, 200.0))
        log_A = float(np.clip(np.random.normal(10.5, 2.5), 5.5, 16.5))
        return round(delta_G, 3), round(delta_H, 3), round(Ea, 3), round(log_A, 4)

    def _sample_thermo(self) -> Tuple[float, float, float, float]:
        """Sample (delta_G_kJ, delta_H_kJ, Ea_kJ, log_A) for one reaction.

        Ranges are informed by real-world chemistry:
          ΔG   Normal(-30, 80)         → p5≈-162, p95≈+102  kJ/mol
          ΔS   Normal(0, 0.20)         → TΔS@298K ≈ ±119 kJ/mol (2σ)
          Ea   Gamma(2.5, 45)+8 ∈[5,300]  → right-skewed, peak ~50, tail to 300
          logA Normal(10.5, 2.5) ∈[5.5,16.5]  → centred on 10^10.5 s⁻¹
        """
        delta_G = float(np.random.normal(-30.0, 80.0))
        delta_S = float(np.random.normal(0.0, 0.20))   # kJ/(mol·K)
        delta_H = delta_G + 298.0 * delta_S
        Ea      = float(np.clip(np.random.gamma(2.5, 45.0) + 8.0, 5.0, 300.0))
        log_A   = float(np.clip(np.random.normal(10.5, 2.5), 5.5, 16.5))
        return round(delta_G, 3), round(delta_H, 3), round(Ea, 3), round(log_A, 4)

    def _lower_layer_coproducts(
        self,
        target_layer: int,
        reactants: List[Tuple[str, int]],
        existing_products: List[Tuple[str, int]],
        chemicals_by_layer: Dict[int, List[Chemical]],
        all_chemicals: Dict[str, Chemical],
        probability: float = 0.35,
    ) -> List[Tuple[str, int]]:
        """With `probability`, add 1 co-product from a layer below target_layer.

        Layer-constraint rule: a product of layer L requires at least one reactant
        of layer L-1.  Layer-1 products are always safe (no constraint).
        For layer 2+ co-products we only add them when the corresponding layer is
        already represented among the reactants.
        """
        if random.random() > probability:
            return []
        excluded = {cid for cid, _ in existing_products} | {cid for cid, _ in reactants}
        reactant_layers = {
            all_chemicals[cid].layer for cid, _ in reactants if cid in all_chemicals
        }
        pool = []
        for lay in range(1, target_layer):
            for c in chemicals_by_layer.get(lay, []):
                if c.id in excluded:
                    continue
                # Layer-1 products have no predecessor constraint
                if c.layer == 1 or (c.layer - 1) in reactant_layers:
                    pool.append(c)
        if not pool:
            return []
        c = random.choice(pool)
        return [(c.id, random.randint(1, 2))]

    def _sample_reaction(
        self,
        target_layer: int,
        chemicals_by_layer: Dict[int, List[Chemical]],
        all_chemicals: Dict[str, Chemical],
    ) -> Optional[Reaction]:
        if target_layer < 2:
            return None

        prev_layer_chems = chemicals_by_layer.get(target_layer - 1, [])
        target_layer_chems = chemicals_by_layer.get(target_layer, [])

        if not prev_layer_chems or not target_layer_chems:
            return None

        n_reactants = random.randint(2, 4)

        # Must include at least one from previous layer
        mandatory = random.choice(prev_layer_chems)
        reactant_pool = list(prev_layer_chems)

        # Can also draw from earlier layers
        for lay in range(1, target_layer - 1):
            reactant_pool.extend(chemicals_by_layer.get(lay, []))

        # Build reactant list
        reactants_chems = [mandatory]
        pool_without_mandatory = [c for c in reactant_pool if c.id != mandatory.id]
        extra = min(n_reactants - 1, len(pool_without_mandatory))
        if extra > 0:
            reactants_chems.extend(random.sample(pool_without_mandatory, extra))

        reactants = [(c.id, random.randint(1, 4)) for c in reactants_chems]

        # Catalysts: pick from lower layers only (< target_layer) so they are purchasable/available
        n_cats = random.randint(1, 2)
        catalyst_pool = []
        for lay in range(1, target_layer):
            catalyst_pool.extend(chemicals_by_layer.get(lay, []))
        catalyst_pool = [c for c in catalyst_pool if c.id not in {cid for cid, _ in reactants}]
        if catalyst_pool:
            cats = random.sample(catalyst_pool, min(n_cats, len(catalyst_pool)))
            catalysts = [c.id for c in cats]
        else:
            catalysts = []

        # Products: 1-3 from target layer, plus optional co-products from lower layers
        n_products = random.randint(1, min(3, len(target_layer_chems)))
        product_chems = random.sample(target_layer_chems, n_products)
        products = [(c.id, random.randint(1, 3)) for c in product_chems]
        products += self._lower_layer_coproducts(
            target_layer, reactants, products, chemicals_by_layer, all_chemicals
        )

        # Byproducts: 0-2 from any layer <= target
        byproduct_chems_pool = []
        for lay in range(1, target_layer + 1):
            byproduct_chems_pool.extend(chemicals_by_layer.get(lay, []))
        byproduct_chems_pool = [c for c in byproduct_chems_pool if c.id not in {cid for cid, _ in products}]
        n_by = random.randint(0, min(2, len(byproduct_chems_pool)))
        if n_by > 0:
            by_chems = random.sample(byproduct_chems_pool, n_by)
            byproducts = [(c.id, random.randint(1, 2)) for c in by_chems]
        else:
            byproducts = []

        # Extra (non-mandatory) reactions use the full-range distribution for diversity
        delta_G, delta_H, Ea, log_A = self._sample_thermo()
        rxn_id = self._next_rxn_id()
        return Reaction(
            id=rxn_id,
            reactants=reactants,
            catalysts=catalysts,
            products=products,
            byproducts=byproducts,
            delta_G_kJ=delta_G,
            delta_H_kJ=delta_H,
            activation_energy_kJ=Ea,
            log_A_factor=log_A,
        )

    def _sample_equipment(self) -> Dict[str, Dict]:
        from .models import EQUIPMENT_CATALOG
        import copy
        equipment = copy.deepcopy(EQUIPMENT_CATALOG)

        for name, spec in equipment.items():
            spec["max_capacity_g"] = round(
                spec["max_capacity_g"] * float(np.random.uniform(0.7, 1.5)), 0
            )
            spec["max_temp_C"] = round(
                spec["max_temp_C"] * float(np.random.uniform(0.8, 1.2)), 0
            )
            spec["min_temp_C"] = round(
                spec["min_temp_C"] * float(np.random.uniform(0.8, 1.2)), 0
            )
            spec["max_pressure_atm"] = round(
                spec["max_pressure_atm"] * float(np.random.uniform(0.7, 1.4)), 1
            )
            spec["base_cost_per_hour"] = round(
                spec["base_cost_per_hour"] * float(np.random.uniform(0.6, 1.6)), 2
            )
            spec["cost_multiplier"] = round(
                spec["cost_multiplier"] * float(np.random.uniform(0.7, 1.4)), 2
            )

        return equipment

    def _sample_solvent(self) -> Chemical:
        chem_id = self._next_chem_id()
        name = self._generate_name()
        mw = float(np.random.uniform(30, 120))
        mp = float(np.random.uniform(-120, 10))
        bp = float(np.random.uniform(50, 200))
        if mp > 10:
            mp = 10.0
        if bp < 50:
            bp = 50.0
        price = float(np.random.uniform(0.01, 0.05))
        return Chemical(
            id=chem_id,
            name=name,
            layer=1,
            molecular_weight=round(mw, 2),
            melting_point=round(mp, 2),
            boiling_point=round(bp, 2),
            base_toxicity=round(float(np.random.uniform(0.1, 1.5)), 3),
            medicinal_expected=0.0,
            medicinal_efficacy=0.0,
            price_per_gram=round(price, 4),
            heat_capacity_J_per_gK=round(float(np.random.uniform(1.5, 4.0)), 4),
            latent_heat_fusion_J_per_g=round(float(np.random.uniform(80, 200)), 2),
            latent_heat_vaporization_J_per_g=round(float(np.random.uniform(300, 1200)), 2),
            is_solvent=True,
        )

    def _assign_solubility(
        self,
        chemicals_by_layer: Dict[int, List[Chemical]],
        solvents: List[Chemical],
    ) -> None:
        solvent_ids = [s.id for s in solvents]
        for layer_chems in chemicals_by_layer.values():
            for chem in layer_chems:
                if chem.is_solvent:
                    continue
                n_solvents = random.randint(1, min(3, len(solvent_ids)))
                chosen = random.sample(solvent_ids, n_solvents)
                for sid in chosen:
                    max_conc = float(np.random.lognormal(mean=2.5, sigma=0.8))
                    max_conc = round(np.clip(max_conc, 1.0, 80.0), 2)
                    chem.solubility[sid] = max_conc

    def _assign_reaction_solvents(self, world: World, solvents: List[Chemical]) -> None:
        solvent_ids = [s.id for s in solvents]
        for rxn in world.reactions.values():
            reactant_ids = [cid for cid, _ in rxn.reactants]
            common_solvents = set(solvent_ids)
            for cid in reactant_ids:
                if cid in world.chemicals and not world.chemicals[cid].is_solvent:
                    chem_solvents = set(world.chemicals[cid].solubility.keys())
                    common_solvents &= chem_solvents
            if not common_solvents:
                fallback_solvent = random.choice(solvent_ids)
                for cid in reactant_ids:
                    if cid in world.chemicals and not world.chemicals[cid].is_solvent:
                        chem = world.chemicals[cid]
                        if fallback_solvent not in chem.solubility:
                            max_conc = round(float(np.random.uniform(2.0, 20.0)), 2)
                            chem.solubility[fallback_solvent] = max_conc
            for pid, _ in rxn.products:
                if pid in world.chemicals and not world.chemicals[pid].is_solvent:
                    prod_chem = world.chemicals[pid]
                    reactant_solvents = set(solvent_ids)
                    for cid in reactant_ids:
                        if cid in world.chemicals and not world.chemicals[cid].is_solvent:
                            reactant_solvents &= set(world.chemicals[cid].solubility.keys())
                    for sid in reactant_solvents:
                        if sid not in prod_chem.solubility:
                            max_conc = round(float(np.random.uniform(2.0, 30.0)), 2)
                            prod_chem.solubility[sid] = max_conc

    def sample_world(self, world_id: str) -> World:
        world = World(world_id=world_id, seed=self.seed)

        num_layers = random.choice(self._num_layers_choices)
        layer_sizes = self._sample_layer_sizes(num_layers)
        chemicals_by_layer: Dict[int, List[Chemical]] = {}

        n_solvents = random.randint(2, 4)
        solvents = [self._sample_solvent() for _ in range(n_solvents)]
        chemicals_by_layer[1] = list(solvents)
        for s in solvents:
            world.chemicals[s.id] = s

        for layer in range(1, num_layers + 1):
            n = layer_sizes[layer - 1]
            chems = [self._sample_chemical(layer) for _ in range(n)]
            if layer == 1:
                chemicals_by_layer[1].extend(chems)
            else:
                chemicals_by_layer[layer] = chems
            for c in chems:
                world.chemicals[c.id] = c

        self._assign_solubility(chemicals_by_layer, solvents)
        world._solvents = solvents

        # Generate reactions for each non-M1 layer
        # Track which non-M1 chemicals have been covered by reactions
        covered: set = set()

        for layer in range(2, num_layers + 1):
            chems_in_layer = chemicals_by_layer[layer]
            for chem in chems_in_layer:
                rxn = None
                for _attempt in range(5):
                    rxn = self._sample_reaction_for_product(
                        chem, layer, chemicals_by_layer, world.chemicals
                    )
                    if rxn is not None:
                        break
                if rxn is None:
                    rxn = self._sample_fallback_reaction(
                        chem, layer, chemicals_by_layer, world.chemicals
                    )
                if rxn is not None:
                    world.reactions[rxn.id] = rxn
                    for pid, _ in rxn.products:
                        covered.add(pid)

            # Add extra reactions: at least as many as compounds in this layer
            n_layer = len(chems_in_layer)
            n_extra = random.randint(n_layer, n_layer + self._extra_reactions_bonus)
            for _ in range(n_extra):
                rxn = self._sample_reaction(layer, chemicals_by_layer, world.chemicals)
                if rxn is not None:
                    world.reactions[rxn.id] = rxn
                    for pid, _ in rxn.products:
                        covered.add(pid)

        self._ensure_reachability(world, chemicals_by_layer, covered)
        self._assign_reaction_solvents(world, solvents)

        world.cost_params = {
            "heating_coeff": np.random.uniform(0.5, 1.2),
            "cooling_coeff": np.random.uniform(0.8, 1.8),
            "heating_exponent": np.random.uniform(1.2, 1.8),
            "cooling_exponent": np.random.uniform(1.0, 1.6),
            "pressure_high_coeff": np.random.uniform(1.0, 2.5),
            "pressure_low_coeff": np.random.uniform(1.0, 2.5),
            "pressure_high_exp": np.random.uniform(0.5, 1.0),
            "pressure_low_exp": np.random.uniform(0.4, 0.8),
            "equipment_base": np.random.uniform(3.0, 8.0),
            "equipment_pressure_coeff": np.random.uniform(0.2, 0.5),
            "duration_coeff": np.random.uniform(0.02, 0.1),
        }

        world.equipment = self._sample_equipment()

        return world

    def _ensure_reachability(
        self,
        world,
        chemicals_by_layer: Dict[int, List[Chemical]],
        covered: set,
    ):
        producible: set = set()
        for lay in chemicals_by_layer.get(1, []):
            producible.add(lay.id)

        num_layers = max(chemicals_by_layer.keys()) if chemicals_by_layer else 1
        for layer in range(2, num_layers + 1):
            for rxn in world.reactions.values():
                reactant_ids = {cid for cid, _ in rxn.reactants}
                if reactant_ids <= producible:
                    for pid, _ in rxn.products:
                        producible.add(pid)

            for chem in chemicals_by_layer.get(layer, []):
                if chem.id not in producible:
                    rxn = self._sample_fallback_reaction(
                        chem, layer, chemicals_by_layer, world.chemicals
                    )
                    if rxn is not None:
                        world.reactions[rxn.id] = rxn
                        for pid, _ in rxn.products:
                            producible.add(pid)
                            covered.add(pid)

    def _sample_fallback_reaction(
        self,
        target_chem: Chemical,
        target_layer: int,
        chemicals_by_layer: Dict[int, List[Chemical]],
        all_chemicals: Dict[str, Chemical],
    ) -> Optional[Reaction]:
        all_lower = []
        for lay in range(1, target_layer):
            all_lower.extend(chemicals_by_layer.get(lay, []))
        if len(all_lower) < 2:
            return None
        n_reactants = min(3, len(all_lower))
        reactants_chems = random.sample(all_lower, n_reactants)
        reactants = [(c.id, random.randint(1, 3)) for c in reactants_chems]
        products = [(target_chem.id, random.randint(1, 2))]
        delta_G, delta_H, Ea, log_A = self._sample_thermo_favorable()
        rxn_id = self._next_rxn_id()
        return Reaction(
            id=rxn_id,
            reactants=reactants,
            catalysts=[],
            products=products,
            byproducts=[],
            delta_G_kJ=delta_G,
            delta_H_kJ=delta_H,
            activation_energy_kJ=Ea,
            log_A_factor=log_A,
        )

    def _sample_reaction_for_product(
        self,
        target_chem: Chemical,
        target_layer: int,
        chemicals_by_layer: Dict[int, List[Chemical]],
        all_chemicals: Dict[str, Chemical],
    ) -> Optional[Reaction]:
        if target_layer < 2:
            return None

        prev_layer_chems = chemicals_by_layer.get(target_layer - 1, [])
        if not prev_layer_chems:
            return None

        n_reactants = random.randint(2, 4)
        mandatory = random.choice(prev_layer_chems)
        reactant_pool = list(prev_layer_chems)
        for lay in range(1, target_layer - 1):
            reactant_pool.extend(chemicals_by_layer.get(lay, []))

        reactants_chems = [mandatory]
        pool_without_mandatory = [c for c in reactant_pool if c.id != mandatory.id]
        extra = min(n_reactants - 1, len(pool_without_mandatory))
        if extra > 0:
            reactants_chems.extend(random.sample(pool_without_mandatory, extra))

        reactants = [(c.id, random.randint(1, 4)) for c in reactants_chems]

        # Catalysts: pick from lower layers only (< target_layer) so they are purchasable/available
        n_cats = random.randint(1, 2)
        catalyst_pool = []
        for lay in range(1, target_layer):
            catalyst_pool.extend(chemicals_by_layer.get(lay, []))
        catalyst_pool = [c for c in catalyst_pool if c.id not in {cid for cid, _ in reactants}]
        if catalyst_pool:
            cats = random.sample(catalyst_pool, min(n_cats, len(catalyst_pool)))
            catalysts = [c.id for c in cats]
        else:
            catalysts = []

        # Target chem must be in products
        target_coeff = random.randint(1, 3)
        products = [(target_chem.id, target_coeff)]

        # Possibly add more products from same layer
        layer_chems = [c for c in chemicals_by_layer.get(target_layer, []) if c.id != target_chem.id]
        n_extra_products = random.randint(0, min(2, len(layer_chems)))
        if n_extra_products > 0:
            extra_prods = random.sample(layer_chems, n_extra_products)
            products.extend([(c.id, random.randint(1, 3)) for c in extra_prods])

        # Possibly add a co-product from a lower layer (lower probability for mandatory reactions)
        products += self._lower_layer_coproducts(
            target_layer, reactants, products, chemicals_by_layer, all_chemicals, probability=0.25
        )

        byproduct_pool: List[Chemical] = []
        for lay in range(1, target_layer + 1):
            byproduct_pool.extend(chemicals_by_layer.get(lay, []))
        byproduct_pool = [c for c in byproduct_pool if c.id not in {cid for cid, _ in products}]
        n_by = random.randint(0, min(2, len(byproduct_pool)))
        if n_by > 0:
            by_chems = random.sample(byproduct_pool, n_by)
            byproducts = [(c.id, random.randint(1, 2)) for c in by_chems]
        else:
            byproducts = []

        # Mandatory reactions use feasibility-biased distribution
        delta_G, delta_H, Ea, log_A = self._sample_thermo_favorable()
        rxn_id = self._next_rxn_id()
        return Reaction(
            id=rxn_id,
            reactants=reactants,
            catalysts=catalysts,
            products=products,
            byproducts=byproducts,
            delta_G_kJ=delta_G,
            delta_H_kJ=delta_H,
            activation_energy_kJ=Ea,
            log_A_factor=log_A,
        )
