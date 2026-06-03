#!/usr/bin/env python3
"""Generate and manage parallel chemistry worlds."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

try:
    from .world_gen.sampler import WorldSampler, COMPLEXITY_PRESETS
    from .world_gen.validator import WorldValidator
except ImportError:
    from world_gen.sampler import WorldSampler, COMPLEXITY_PRESETS
    from world_gen.validator import WorldValidator

DEFAULT_WORLDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worlds")


def list_worlds(directory: str = None) -> List[Dict[str, Any]]:
    """Scan the worlds directory and return metadata for each saved world."""
    directory = directory or DEFAULT_WORLDS_DIR
    if not os.path.isdir(directory):
        return []
    results = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(directory, fname)
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        meta = data.get("metadata", {})
        results.append({
            "world_id": data.get("world_id", fname[:-5]),
            "seed": meta.get("seed"),
            "num_layers": meta.get("num_layers"),
            "num_chemicals": meta.get("num_chemicals"),
            "num_reactions": meta.get("num_reactions"),
            "file_path": path,
        })
    return results


def generate_worlds(
    n: int,
    output_dir: str = None,
    base_seed: int = 0,
    complexity_level: str = None,
    layer1_min: int = None,
    layer1_max: int = None,
    last_layer_min: int = None,
    last_layer_max: int = None,
) -> None:
    output_dir = output_dir or DEFAULT_WORLDS_DIR
    os.makedirs(output_dir, exist_ok=True)
    validator = WorldValidator()

    generated = 0
    seed = base_seed
    max_attempts_per_world = 50

    while generated < n:
        world_id = f"world_{generated + 1:03d}"
        success = False

        for trial in range(max_attempts_per_world):
            current_seed = seed + trial
            sampler = WorldSampler(
                seed=current_seed,
                complexity_level=complexity_level,
                layer1_min=layer1_min,
                layer1_max=layer1_max,
                last_layer_min=last_layer_min,
                last_layer_max=last_layer_max,
            )
            world = sampler.sample_world(world_id)
            valid, reason = validator.validate(world)
            if valid:
                world.seed = current_seed
                path = os.path.join(output_dir, f"{world_id}.json")
                world.save(path)
                print(
                    f"[{generated + 1}/{n}] {world_id} saved to {path} "
                    f"(seed={current_seed}, chemicals={len(world.chemicals)}, reactions={len(world.reactions)}, layers={world.num_layers})"
                )
                seed = current_seed + 1
                success = True
                break
            else:
                seed += 1

        if not success:
            print(
                f"Warning: Could not generate valid world {world_id} after {max_attempts_per_world} attempts. Skipping.",
                file=sys.stderr,
            )
            seed += max_attempts_per_world

        generated += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and manage parallel chemistry worlds.")
    parser.add_argument("--n", type=int, default=10, help="Number of worlds to generate (default: 10)")
    parser.add_argument("--output-dir", type=str, default=None, help=f"Output directory (default: {DEFAULT_WORLDS_DIR})")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0)")
    parser.add_argument("--complexity", type=str, choices=list(COMPLEXITY_PRESETS.keys()), default=None,
                        help="Complexity level preset (overrides layer defaults; explicit layer params still override)")
    parser.add_argument("--layer1-min", type=int, default=None, help="Min compounds in layer 1")
    parser.add_argument("--layer1-max", type=int, default=None, help="Max compounds in layer 1")
    parser.add_argument("--last-layer-min", type=int, default=None, help="Min compounds in last layer")
    parser.add_argument("--last-layer-max", type=int, default=None, help="Max compounds in last layer")
    parser.add_argument("--list", action="store_true", help="List all saved worlds and exit")
    args = parser.parse_args()

    if args.list:
        worlds = list_worlds(args.output_dir)
        if not worlds:
            print("No worlds found.")
        else:
            print(f"{'world_id':<15} {'seed':<8} {'layers':<8} {'chemicals':<12} {'reactions':<12} path")
            print("-" * 80)
            for w in worlds:
                print(
                    f"{w['world_id']:<15} {w['seed']:<8} {w['num_layers']:<8} "
                    f"{w['num_chemicals']:<12} {w['num_reactions']:<12} {w['file_path']}"
                )
            print(f"\nTotal: {len(worlds)} worlds")
        return

    generate_worlds(
        n=args.n,
        output_dir=args.output_dir,
        base_seed=args.seed,
        complexity_level=args.complexity,
        layer1_min=args.layer1_min,
        layer1_max=args.layer1_max,
        last_layer_min=args.last_layer_min,
        last_layer_max=args.last_layer_max,
    )
    print(f"\nDone. Generated worlds saved to: {args.output_dir or DEFAULT_WORLDS_DIR}")


if __name__ == "__main__":
    main()
