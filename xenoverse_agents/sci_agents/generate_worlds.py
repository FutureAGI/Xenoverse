#!/usr/bin/env python3
"""Generate parallel chemistry worlds and save them to JSON files."""

from __future__ import annotations

import argparse
import os
import sys

from world_gen.sampler import WorldSampler
from world_gen.validator import WorldValidator


def generate_worlds(
    n: int,
    output_dir: str,
    base_seed: int = 0,
    layer1_min: int = 6,
    layer1_max: int = 10,
    last_layer_min: int = 2,
    last_layer_max: int = 5,
) -> None:
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
    parser = argparse.ArgumentParser(description="Generate parallel chemistry worlds.")
    parser.add_argument("--n", type=int, default=10, help="Number of worlds to generate (default: 10)")
    parser.add_argument("--output-dir", type=str, default="worlds/", help="Output directory (default: worlds/)")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed (default: 0)")
    parser.add_argument("--layer1-min", type=int, default=6, help="Min compounds in layer 1 (default: 6)")
    parser.add_argument("--layer1-max", type=int, default=10, help="Max compounds in layer 1 (default: 10)")
    parser.add_argument("--last-layer-min", type=int, default=2, help="Min compounds in last layer (default: 2)")
    parser.add_argument("--last-layer-max", type=int, default=5, help="Max compounds in last layer (default: 5)")
    args = parser.parse_args()

    generate_worlds(
        n=args.n,
        output_dir=args.output_dir,
        base_seed=args.seed,
        layer1_min=args.layer1_min,
        layer1_max=args.layer1_max,
        last_layer_min=args.last_layer_min,
        last_layer_max=args.last_layer_max,
    )
    print(f"\nDone. Generated worlds saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
