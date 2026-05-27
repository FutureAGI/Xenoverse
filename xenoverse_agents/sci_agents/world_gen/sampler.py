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

class WorldSampler:
    def __init__(
        self,
        seed: int,
        layer1_min: int = 6,
        layer1_max: int = 10,
        last_layer_min: int = 2,
        last_layer_max: int = 5,
    ):
        self.seed = seed
        self.layer1_min = max(1, layer1_min)
        self.layer1_max = max(self.layer1_min, layer1_max)
        self.last_layer_min = max(1, last_layer_min)
        self.last_layer_max = max(self.last_layer_min, last_layer_max)
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

    def sample_world(self, world_id: str) -> World:
        world = World(world_id=world_id, seed=self.seed)

        num_layers = random.choice([3, 4, 5])
        layer_sizes = self._sample_layer_sizes(num_layers)
        chemicals_by_layer: Dict[int, List[Chemical]] = {}

        for layer in range(1, num_layers + 1):
            n = layer_sizes[layer - 1]
            chems = [self._sample_chemical(layer) for _ in range(n)]
            chemicals_by_layer[layer] = chems
            for c in chems:
                world.chemicals[c.id] = c

        # Generate reactions for each non-M1 layer
        # Track which non-M1 chemicals have been covered by reactions
        covered: set = set()

        for layer in range(2, num_layers + 1):
            chems_in_layer = chemicals_by_layer[layer]
            # Each chemical in this layer needs at least one reaction producing it
            for chem in chems_in_layer:
                # Force at least one reaction producing this chem
                rxn = self._sample_reaction_for_product(
                    chem, layer, chemicals_by_layer, world.chemicals
                )
                if rxn is not None:
                    world.reactions[rxn.id] = rxn
                    for pid, _ in rxn.products:
                        covered.add(pid)

            # Add extra reactions: at least as many as compounds in this layer
            n_layer = len(chems_in_layer)
            n_extra = random.randint(n_layer, n_layer + 3)
            for _ in range(n_extra):
                rxn = self._sample_reaction(layer, chemicals_by_layer, world.chemicals)
                if rxn is not None:
                    world.reactions[rxn.id] = rxn
                    for pid, _ in rxn.products:
                        covered.add(pid)

        return world

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
