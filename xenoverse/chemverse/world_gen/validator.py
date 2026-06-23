from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import Chemical, Reaction, World

_R = 8.314e-3  # kJ/(mol·K)
_T_MAX_K = 873.15  # 600 °C — practical synthesis ceiling


def _k_eq(rxn: Reaction, T_K: float) -> float:
    K_ref = math.exp(-rxn.delta_G_kJ / (_R * 298.0))
    return K_ref * math.exp(-rxn.delta_H_kJ / _R * (1.0 / T_K - 1.0 / 298.0))


def _reaction_feasible(rxn: Reaction, T_K: float = _T_MAX_K, min_K: float = 1e-4) -> bool:
    """Return True if K_eq >= min_K at any temperature up to T_K."""
    return any(_k_eq(rxn, T) >= min_K for T in (298.0, 500.0, T_K))


class WorldValidator:
    def __init__(self, max_toxicity: Optional[float] = None, min_medicinal: float = 2.0):
        self._max_toxicity = max_toxicity if max_toxicity is not None else 4.0
        self._min_medicinal = min_medicinal

    def validate(self, world: World) -> Tuple[bool, str]:
        ok, msg = self._check_qualifying_compound(world)
        if not ok:
            return False, msg

        ok, msg = self._check_layer_constraints(world)
        if not ok:
            return False, msg

        ok, msg = self._check_reachability(world)
        if not ok:
            return False, msg

        ok, msg = self._check_qualifying_route_feasible(world)
        if not ok:
            return False, msg

        return True, "valid"

    def _qualifying_compounds(self, world: World) -> List[Chemical]:
        return [
            c for c in world.chemicals.values()
            if c.medicinal_value >= self._min_medicinal and c.base_toxicity < self._max_toxicity
        ]

    def _check_qualifying_compound(self, world: World) -> Tuple[bool, str]:
        if self._qualifying_compounds(world):
            return True, ""
        return False, (
            f"No compound with medicinal_value >= {self._min_medicinal} "
            f"and toxicity < {self._max_toxicity}"
        )

    def _check_layer_constraints(self, world: World) -> Tuple[bool, str]:
        chemicals = world.chemicals
        for rxn in world.reactions.values():
            reactant_layers = [chemicals[cid].layer for cid, _ in rxn.reactants if cid in chemicals]
            if not reactant_layers:
                continue
            for pid, _ in rxn.products:
                if pid not in chemicals:
                    continue
                product_layer = chemicals[pid].layer
                if product_layer > 1:
                    required_layer = product_layer - 1
                    if not any(chemicals[cid].layer == required_layer for cid, _ in rxn.reactants if cid in chemicals):
                        return False, f"Reaction {rxn.id}: product {pid} (layer {product_layer}) has no reactant from layer {required_layer}"
        return True, ""

    def _check_reachability(self, world: World) -> Tuple[bool, str]:
        produced_by: dict = {cid: [] for cid in world.chemicals}
        for rxn in world.reactions.values():
            for pid, _ in rxn.products:
                if pid in produced_by:
                    produced_by[pid].append(rxn.id)

        for chem in world.chemicals.values():
            if chem.layer == 1:
                continue
            if not produced_by[chem.id]:
                return False, f"Chemical {chem.id} ({chem.name}, layer {chem.layer}) is not produced by any reaction"
        return True, ""

    def _check_qualifying_route_feasible(self, world: World) -> Tuple[bool, str]:
        """Verify that at least one qualifying compound has a synthesis route
        where every step has K_eq >= 1e-4 at some temperature <= 600 C."""
        qualifying = self._qualifying_compounds(world)
        if not qualifying:
            return False, "No qualifying compounds exist"

        produces: Dict[str, List[Reaction]] = {}
        for rxn in world.reactions.values():
            for pid, _ in rxn.products:
                produces.setdefault(pid, []).append(rxn)

        def has_feasible_path(target_id: str, visited: frozenset, depth: int) -> bool:
            chem = world.chemicals[target_id]
            if chem.layer == 1:
                return True
            if depth > world.num_layers:
                return False
            for rxn in produces.get(target_id, []):
                if not _reaction_feasible(rxn):
                    continue
                if all(
                    has_feasible_path(cid, visited | {target_id}, depth + 1)
                    for cid, _ in rxn.reactants
                    if cid not in visited and world.chemicals[cid].layer > 1
                ):
                    return True
            return False

        for chem in qualifying:
            if chem.layer == 1 or has_feasible_path(chem.id, frozenset(), 0):
                return True, ""

        return False, "No qualifying compound has a thermodynamically feasible synthesis route"
