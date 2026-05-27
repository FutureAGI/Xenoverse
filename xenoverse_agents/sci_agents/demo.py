#!/usr/bin/env python3
"""
Demo: sample a fresh world each run and print key environmental information.
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from collections import Counter

from world_gen.sampler import WorldSampler
from world_gen.validator import WorldValidator
from environment.api import ChemistryEnvironment


def sample_valid_world(max_attempts: int = 200):
    seed = (int(time.time() * 1000) ^ random.getrandbits(20)) & 0xFFFFFFFF
    validator = WorldValidator()
    for i in range(max_attempts):
        s = seed + i
        world = WorldSampler(seed=s).sample_world(f"world_{s}")
        valid, reason = validator.validate(world)
        if valid:
            return world, s
    raise RuntimeError("Could not generate a valid world.")


def sep(title: str = "", width: int = 64) -> None:
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─' * pad} {title} {'─' * (width - pad - len(title) - 2)}")
    else:
        print("─" * width)


def main() -> None:
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║         PARALLEL CHEMISTRY WORLD — ENVIRONMENT SAMPLER  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    print("Sampling a valid world...")
    world, seed = sample_valid_world()

    sep("WORLD OVERVIEW")
    print(f"  World ID  : {world.world_id}")
    print(f"  Seed      : {seed}")
    print(f"  Layers    : {world.num_layers}")
    print(f"  Chemicals : {len(world.chemicals)}")
    print(f"  Reactions : {len(world.reactions)}")

    # --- Layer breakdown ---
    sep("CHEMICALS BY LAYER")
    for layer in range(1, world.num_layers + 1):
        chems = [c for c in world.chemicals.values() if c.layer == layer]
        label = "purchasable" if layer == 1 else f"layer {layer}"
        print(f"\n  [{label}] {len(chems)} compounds")
        for c in sorted(chems, key=lambda x: -x.medicinal_value):
            med = f"med={c.medicinal_value:.3f}"
            tox = f"tox={c.base_toxicity:.3f}"
            mw  = f"MW={c.molecular_weight:.1f}"
            price = f"  {c.price_per_gram:.3f} cr/g" if c.price_per_gram is not None else ""
            flag = " ★" if c.medicinal_value > 4.0 and c.base_toxicity < 4.0 else ""
            print(f"    {c.id}  {c.name:<16s}  {med}  {tox}  {mw}{price}{flag}")

    # --- Reaction network summary ---
    sep("REACTION NETWORK")
    dg_vals  = [r.delta_G_kJ for r in world.reactions.values()]
    ea_vals  = [r.activation_energy_kJ for r in world.reactions.values()]
    loga_vals = [r.log_A_factor for r in world.reactions.values()]
    n_rxns = len(dg_vals)
    exergonic = sum(1 for g in dg_vals if g < 0)
    print(f"  Total reactions : {n_rxns}")
    print(f"  Exergonic (ΔG<0): {exergonic}/{n_rxns} ({exergonic/n_rxns:.0%})")
    print(f"  ΔG  range  : {min(dg_vals):.1f} … {max(dg_vals):.1f} kJ/mol"
          f"  (median {sorted(dg_vals)[n_rxns//2]:.1f})")
    print(f"  Ea  range  : {min(ea_vals):.1f} … {max(ea_vals):.1f} kJ/mol"
          f"  (median {sorted(ea_vals)[n_rxns//2]:.1f})")
    print(f"  logA range : {min(loga_vals):.2f} … {max(loga_vals):.2f}")

    # Reaction count per target layer
    products_layer: Counter = Counter()
    for rxn in world.reactions.values():
        for pid, _ in rxn.products:
            if pid in world.chemicals:
                products_layer[world.chemicals[pid].layer] += 1
    print(f"  Reactions by product layer: " +
          "  ".join(f"L{l}:{c}" for l, c in sorted(products_layer.items())))

    # --- Medicinal candidates ---
    sep("MEDICINAL CANDIDATES  (med > 4.0, tox < 4.0  ★)")
    candidates = [c for c in world.chemicals.values()
                  if c.medicinal_value > 4.0 and c.base_toxicity < 4.0]
    if candidates:
        for c in sorted(candidates, key=lambda x: -x.medicinal_value):
            print(f"  {c.id}  {c.name:<16s}  layer={c.layer}"
                  f"  med={c.medicinal_value:.3f}  tox={c.base_toxicity:.3f}")
    else:
        print("  (none — validation should have caught this)")

    # --- Environment API ---
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(world.to_dict(), f, indent=2)
        world_path = f.name

    try:
        env = ChemistryEnvironment(world_path)

        sep("PURCHASABLE CHEMICALS  (M1)")
        purchasable = env.list_purchasable()
        for name, info in sorted(purchasable.items(), key=lambda x: x[1]["price_per_gram"]):
            print(f"  {name:<18s}  {info['price_per_gram']:.3f} cr/g"
                  f"  [{info['state_at_room_temp']}]")

        # Buy all M1 so catalysts are available
        for cname in purchasable:
            env.purchase(cname, 50.0)

        sep("CHEAPEST MEDICINAL PATHWAY  (med ≥ 3.0, tox ≤ 4.0)")
        r = env.find_cheapest_medicinal_pathway(
            min_medicinal_value=3.0, max_toxicity=4.0, per_m1_g=10.0
        )
        print(f"  Qualifying compounds : {r['num_qualifying_compounds']}")
        print(f"  Routes evaluated     : {r['num_evaluated_routes']}")
        print(f"  {r['message']}")

        if r["found"]:
            b = r["best_pathway"]
            ps = b["pathway_summary"]
            target_chem = next(c for c in world.chemicals.values() if c.name == b["target"])
            print(f"\n  Target     : {b['target']}"
                  f"  (layer {target_chem.layer})")
            print(f"  Medicinal  : {b['medicinal_value']:.3f}   Toxicity: {b['base_toxicity']:.3f}")
            print(f"  Yield      : {ps['target_yield_g']:.4f} g"
                  f"   Cost: {ps['total_cost']:.2f} cr"
                  f"   Cost/g: {ps['cost_per_gram_target']:.2f} cr/g")
            print(f"  Cost/med   : {b['cost_per_medicinal_unit']:.2f} cr"
                  f"   Steps: {b['route']['num_steps']}"
                  f"   Efficiency: {ps['efficiency_rating']} ({ps['mass_efficiency']:.1%})")
            print(f"  M1 inputs  : {', '.join(b['route']['m1_starting_materials'])}")
            print()
            for step in b["route"]["steps"]:
                cats = ", ".join(step["catalysts"]) or "—"
                rxns = " + ".join(step["reactants"])
                print(f"    Step {step['step']}: {rxns}")
                print(f"           {step['temperature_C']:.0f}°C  {step['pressure_atm']:.1f} atm"
                      f"  {step['duration_seconds']:.0f}s  cat=[{cats}]")
                print(f"           {step['conditions_hint']}")

            if len(r["all_candidates"]) > 1:
                print(f"\n  All {len(r['all_candidates'])} candidates:")
                for cand in r["all_candidates"]:
                    marker = " ← best" if cand is b else ""
                    print(f"    {cand['target']:<18s}"
                          f"  med={cand['medicinal_value']:.2f}"
                          f"  tox={cand['base_toxicity']:.1f}"
                          f"  cost/med={cand['cost_per_medicinal_unit']:.2f}"
                          f"  steps={cand['route']['num_steps']}{marker}")

            if b.get("warnings"):
                print()
                for w in b["warnings"]:
                    print(f"  [!] {w}")

        elif r["num_qualifying_compounds"] == 0:
            # Fallback: relax constraints
            print("\n  Relaxing to med ≥ 1.5, tox ≤ 7.0 ...")
            r2 = env.find_cheapest_medicinal_pathway(
                min_medicinal_value=1.5, max_toxicity=7.0, per_m1_g=10.0
            )
            print(f"  {r2['message']}")
            if r2["found"]:
                b2 = r2["best_pathway"]
                print(f"  Best: {b2['target']}"
                      f"  med={b2['medicinal_value']:.2f}"
                      f"  tox={b2['base_toxicity']:.1f}"
                      f"  cost/med={b2['cost_per_medicinal_unit']:.2f} cr")

    finally:
        os.unlink(world_path)

    sep()
    print("  Done.\n")


if __name__ == "__main__":
    main()
