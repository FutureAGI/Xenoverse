#!/usr/bin/env python
"""Visual smoke test for the randomized MuJoCo Humanoid environment."""

import argparse
import os

import gymnasium as gym

import xenoverse.metacontrol  # noqa: F401 - registers the environment
from xenoverse.metacontrol import sample_humanoid


def run_visual_test(
    steps=2000,
    noise_scale=1.0,
    action_scale=0.25,
    seed=0,
):
    """Render a random humanoid and reset it whenever it falls."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if not 0.0 <= action_scale <= 1.0:
        raise ValueError("action_scale must be in [0, 1]")

    env = None
    task = None
    try:
        env = gym.make("random-humanoid-v0", render_mode="human")
        task = sample_humanoid(noise_scale=noise_scale)
        env.unwrapped.set_task(task)
        env.action_space.seed(seed)
        env.reset(seed=seed)
        episode = 1

        print(f"Rendering task: {task}")
        print("Close the MuJoCo window or press Ctrl+C to stop.")
        for step in range(1, steps + 1):
            action = env.action_space.sample() * action_scale
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                episode += 1
                env.reset()
                print(f"Reset after fall: episode={episode}, step={step}")
    except KeyboardInterrupt:
        print("Visual test stopped by user.")
    finally:
        if env is not None:
            env.close()
        if task is not None and os.path.exists(task):
            os.unlink(task)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a procedurally randomized Humanoid task."
    )
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=0.25,
        help="Scale sampled actions by a value in [0, 1].",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visual_test(
        steps=args.steps,
        noise_scale=args.noise_scale,
        action_scale=args.action_scale,
        seed=args.seed,
    )
