import json
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from anymdp_branch.llm_clients import BaseLLMClient, TokenUsage
from anymdp_branch.prompting import (
    build_step_prompt,
    build_system_prompt,
    build_task_prompt,
    parse_action,
    task_matches_filters,
)


@dataclass
class EvalConfig:
    tasks_dir: Path
    output_dir: Path
    model_name: str
    task_type: str
    state_sizes: List[int]
    action_size: int
    max_steps: Optional[int] = None
    history_window: int = 5
    max_tasks: Optional[int] = None
    action_only: bool = False
    thinking_mode: bool = False


def _safe_obs(obs: Any) -> Any:
    if hasattr(obs, "tolist"):
        try:
            return obs.tolist()
        except Exception:
            pass
    if isinstance(obs, (int, float, str, bool, list, dict)):
        return obs
    if hasattr(obs, "item"):
        try:
            return obs.item()
        except Exception:
            pass
    return obs


def _load_task_file(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def _safe_name(raw: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join([ch if ch in allowed else "_" for ch in raw]).strip("_") or "unknown"


def _task_family(task: Dict[str, Any], task_file: Path) -> str:
    name = task_file.name.lower()
    if "multitoken" in name:
        return "multitoken_anypomdp"
    if "anypomdp" in name:
        return "pomdp"
    if "anymdp" in name:
        return "anymdp"
    task_type = str(task.get("task_type", "MDP")).upper()
    return "pomdp" if task_type == "POMDP" else "anymdp"


def _task_state_count(task: Dict[str, Any]) -> int:
    state_mapping = task.get("state_mapping")
    if state_mapping is not None:
        return int(len(state_mapping))
    return int(task.get("ns", 0))


def collect_task_files(root: Path) -> List[Path]:
    files = sorted([*root.rglob("*.pkl"), *root.rglob("*.pickle")])
    return files


def evaluate_tasks(client: BaseLLMClient, cfg: EvalConfig) -> Dict[str, Any]:
    import gymnasium as gym
    import xenoverse.anymdp  # noqa: F401

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    task_logs_root = cfg.output_dir / "task_logs"
    summary_path = cfg.output_dir / "tasks_summary.json"

    task_files = collect_task_files(cfg.tasks_dir)
    selected_tasks: List[Path] = []
    for p in task_files:
        try:
            task = _load_task_file(p)
        except Exception:
            continue
        if task_matches_filters(task, cfg.task_type, cfg.state_sizes, cfg.action_size):
            selected_tasks.append(p)
    if cfg.max_tasks is not None:
        selected_tasks = selected_tasks[: cfg.max_tasks]

    env = gym.make("anymdp-v0", max_steps=cfg.max_steps or 5000)
    client.reset_usage()

    all_task_results: List[Dict[str, Any]] = []
    step_counter = 0
    for task_idx, task_file in enumerate(selected_tasks):
        task = _load_task_file(task_file)
        if cfg.max_steps is not None:
            task["max_steps"] = int(cfg.max_steps)

        env.unwrapped.set_task(task)
        obs, info = env.reset()
        na = int(env.unwrapped.na)
        task_max_steps = int(env.unwrapped.max_steps)

        family = _task_family(task, task_file)
        state_count = _task_state_count(task)
        model_dir = _safe_name(cfg.model_name)
        log_dir = task_logs_root / model_dir / family / f"{state_count}state"
        log_dir.mkdir(parents=True, exist_ok=True)
        task_log_path = log_dir / f"{task_file.stem}.jsonl"

        system_prompt = build_system_prompt(
            action_only=cfg.action_only,
            thinking_mode=cfg.thinking_mode,
        )
        task_prompt = build_task_prompt(
            task,
            task_name=task_file.name,
            max_steps=task_max_steps,
            action_only=cfg.action_only,
            thinking_mode=cfg.thinking_mode,
        )
        init_reply = client.reset_task(system_prompt=system_prompt, task_prompt=task_prompt)
        task_usage = TokenUsage()
        task_usage.add(client.last_usage)

        done = False
        truncated = False
        step_in_task = 0
        episode_count = 1
        reset_count = 0
        total_reward = 0.0
        total_gt_reward = 0.0
        prev_action: Optional[int] = None
        prev_reward: Optional[float] = None
        prev_terminated = False
        prev_truncated = False
        history: List[Dict[str, Any]] = []

        with task_log_path.open("w", encoding="utf-8") as logf:
            init_record = {
                "run_step": step_counter,
                "task_index": task_idx,
                "task_file": str(task_file),
                "task_log_file": str(task_log_path),
                "model": cfg.model_name,
                "task_family": family,
                "state_count": state_count,
                "event": "task_init",
                "llm_reply": init_reply,
                "token_usage": client.last_usage.to_dict(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            logf.write(json.dumps(init_record, ensure_ascii=True) + "\n")
            step_counter += 1

            while step_in_task < task_max_steps:
                step_prompt = build_step_prompt(
                    step_idx=step_in_task,
                    observation=_safe_obs(obs),
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                    prev_terminated=prev_terminated,
                    prev_truncated=prev_truncated,
                    recent_history=history[-cfg.history_window :],
                    na=na,
                    task_max_steps=task_max_steps,
                )
                llm_reply = client.choose_action(step_prompt)
                task_usage.add(client.last_usage)
                action, raw_reply = parse_action(llm_reply, na=na)
                next_obs, reward, done, truncated, step_info = env.step(action)

                reward_gt = float(step_info.get("reward_gt", reward))
                rec = {
                    "run_step": step_counter,
                    "task_index": task_idx,
                    "task_file": str(task_file),
                    "step": step_in_task,
                    "episode_in_task": episode_count,
                    "observation": _safe_obs(obs),
                    "action": int(action),
                    "reward": float(reward),
                    "reward_gt": reward_gt,
                    "terminated": bool(done),
                    "truncated": bool(truncated),
                    "llm_reply": raw_reply,
                    "token_usage": client.last_usage.to_dict(),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                logf.write(json.dumps(rec, ensure_ascii=True) + "\n")
                logf.flush()

                history.append(
                    {
                        "step": step_in_task,
                        "observation": _safe_obs(obs),
                        "action": int(action),
                        "reward": float(reward),
                    }
                )
                total_reward += float(reward)
                total_gt_reward += float(reward_gt)
                prev_action = int(action)
                prev_reward = float(reward)
                prev_terminated = bool(done)
                prev_truncated = bool(truncated)
                obs = next_obs
                step_in_task += 1
                step_counter += 1

                if step_in_task >= task_max_steps:
                    break
                if done or truncated:
                    reset_reason = "terminated" if done else "truncated"
                    reset_count += 1
                    episode_count += 1
                    obs, info = env.reset()
                    done = False
                    truncated = False
                    reset_record = {
                        "run_step": step_counter,
                        "task_index": task_idx,
                        "task_file": str(task_file),
                        "step": step_in_task,
                        "episode_in_task": episode_count,
                        "event": "env_reset_within_task",
                        "reason": reset_reason,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                    logf.write(json.dumps(reset_record, ensure_ascii=True) + "\n")
                    step_counter += 1

        mean_reward = total_reward / max(1, step_in_task)
        task_result = {
            "task_index": task_idx,
            "task_file": str(task_file),
            "task_log_file": str(task_log_path),
            "model": cfg.model_name,
            "task_family": family,
            "task_type": str(task.get("task_type", "MDP")),
            "ns": int(task.get("ns", 0)),
            "real_ns": _task_state_count(task),
            "na": int(task.get("na", 0)),
            "steps": step_in_task,
            "episodes_in_task": episode_count,
            "env_resets_within_task": reset_count,
            "total_reward": total_reward,
            "mean_reward": mean_reward,
            "total_reward_gt": total_gt_reward,
            "mean_reward_gt": total_gt_reward / max(1, step_in_task),
            "terminated": bool(done),
            "truncated": bool(truncated),
            "token_usage": task_usage.to_dict(),
        }
        all_task_results.append(task_result)

    run_usage = client.usage_summary()
    aggregate = {
        "evaluated_tasks": len(all_task_results),
        "task_files_found": len(task_files),
        "task_files_selected": len(selected_tasks),
        "overall_mean_reward": float(
            (sum(x["mean_reward"] for x in all_task_results) / len(all_task_results)) if all_task_results else 0.0
        ),
        "overall_mean_reward_gt": float(
            (sum(x["mean_reward_gt"] for x in all_task_results) / len(all_task_results)) if all_task_results else 0.0
        ),
        "tasks": all_task_results,
        "token_usage": run_usage,
        "task_logs_root": str(task_logs_root),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=True, indent=2)
    return aggregate
