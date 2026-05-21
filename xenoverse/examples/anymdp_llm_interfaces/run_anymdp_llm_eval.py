import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from anymdp_branch.evaluator import EvalConfig, evaluate_tasks
from anymdp_branch.llm_clients import build_client


def _parse_sizes(raw: str) -> List[int]:
    vals = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(int(item))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser("AnyMDP LLM evaluator")
    parser.add_argument("--tasks-dir", type=str, required=True, help="Folder containing *.pkl task files")
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["claude", "gpt", "deepseek", "gemini", "local"],
    )
    parser.add_argument("--model", type=str, default=None, help="Model name override")
    parser.add_argument("--task-type", type=str, default="all", choices=["all", "mdp", "pomdp"])
    parser.add_argument("--state-sizes", type=str, default="1,4,8,16", help="Comma-separated state sizes")
    parser.add_argument("--action-size", type=int, default=5, help="Expected action count in tasks")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps for each task")
    parser.add_argument("--max-tasks", type=int, default=None, help="Optional cap for number of tasks to run")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--api-base", type=str, default=None, help="Optional custom API base URL")
    parser.add_argument("--output-dir", type=str, default="anymdp_branch/runs")
    parser.add_argument("--action-only", action="store_true", help="Force model output to contain only action JSON")
    parser.add_argument(
        "--thinking-mode",
        action="store_true",
        help="Model may use reasoning; prompt tells it not to think on every step",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.provider}_{ts}"

    client = build_client(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        api_base=args.api_base,
    )
    resolved_model = args.model or getattr(getattr(client, "config", None), "model", args.provider)
    cfg = EvalConfig(
        tasks_dir=Path(args.tasks_dir),
        output_dir=run_dir,
        model_name=resolved_model,
        task_type=args.task_type,
        state_sizes=_parse_sizes(args.state_sizes),
        action_size=args.action_size,
        max_steps=args.max_steps,
        max_tasks=args.max_tasks,
        action_only=bool(args.action_only),
        thinking_mode=bool(args.thinking_mode),
    )
    result = evaluate_tasks(client=client, cfg=cfg)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
