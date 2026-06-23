#!/usr/bin/env python3
"""Interactive Chemverse demo: prints task_description and function specs as JSON."""

from __future__ import annotations

import json
import random
import time

try:
    from .environment import SciResearchBackend
except ImportError:
    from environment.backend import SciResearchBackend


def sample_environment(max_attempts: int = 200):
    seed = (int(time.time() * 1000) ^ random.getrandbits(20)) & 0xFFFFFFFF
    backend = SciResearchBackend()
    session = backend.handle_request(
        {
            "action": "sample_environment",
            "sampler_kwargs": {"seed": seed, "max_attempts": max_attempts},
        }
    )
    return backend, session


def parse_user_input(raw: str) -> dict | None:
    raw = raw.strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    parts = raw.split(None, 1)
    name = parts[0]
    arguments = {}
    if len(parts) > 1:
        try:
            arguments = json.loads(parts[1])
        except json.JSONDecodeError:
            print("[Warning] Could not parse arguments as JSON, sending empty arguments.")
    return {"name": name, "arguments": arguments}


def main() -> None:
    print("Sampling environment...")
    backend, session = sample_environment()
    session_id = session["session_id"]

    system_info = {
        "session_id": session_id,
        "task_description": session["task_description"],
        "tool_prompt": session["tool_prompt"],
        "function_tools": session["observation"]["function_tools"],
    }
    print(json.dumps(system_info, indent=2, ensure_ascii=False))
    print("\n--- Interactive Mode ---")
    print("Input format: {\"name\": \"<tool>\", \"arguments\": {...}}")
    print("Or shorthand: <tool_name> <json_arguments>")
    print("Commands: quit | tools | task")
    print()

    while True:
        try:
            raw = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            break
        if raw.lower() == "tools":
            print(json.dumps(session["observation"]["function_tools"], indent=2, ensure_ascii=False))
            continue
        if raw.lower() == "task":
            print(json.dumps(session["task_description"], indent=2, ensure_ascii=False))
            continue

        function_call = parse_user_input(raw)
        if function_call is None:
            continue

        response = backend.handle_request(
            {
                "action": "dispatch_function_call",
                "session_id": session_id,
                "function_call": function_call,
            }
        )
        print(json.dumps(response, indent=2, ensure_ascii=False))
        print()

    backend.handle_request({"action": "close_session", "session_id": session_id})
    print("Session closed.")


if __name__ == "__main__":
    main()
