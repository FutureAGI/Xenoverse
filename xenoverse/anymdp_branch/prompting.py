import json
import re
from typing import Any, Dict, List, Optional, Tuple


def _recent_actions(recent_history: List[Dict[str, Any]], k: int = 8) -> List[int]:
    actions: List[int] = []
    for item in recent_history[-k:]:
        if "action" in item:
            try:
                actions.append(int(item["action"]))
            except Exception:
                continue
    return actions


def _looks_like_index_cycle(actions: List[int], na: int) -> bool:
    if na <= 1 or len(actions) < na:
        return False
    tail = actions[-na:]
    for start in range(na):
        expected = [int((start + i) % na) for i in range(na)]
        if tail == expected:
            return True
    return False


def _phase_guidance(step_idx: int, task_max_steps: int) -> str:
    if task_max_steps <= 0:
        task_max_steps = 1
    ratio = float(step_idx) / float(task_max_steps)
    if ratio < 0.2:
        return (
            "Phase: exploration (0%-20%). Prioritize information gain. "
            "Test uncertain actions, but do NOT mechanically rotate action indices.\n"
        )
    if ratio < 0.5:
        return (
            "Phase: transition (20%-50%). Blend exploration and exploitation. "
            "Keep high-value actions more often, with occasional probes of uncertain actions.\n"
        )
    return (
        "Phase: policy control (50%-100%). Prioritize actions with best observed reward trends. "
        "Only explore when evidence is weak or performance drops.\n"
    )


def _real_state_size(task: Dict[str, Any]) -> int:
    state_mapping = task.get("state_mapping")
    if state_mapping is None:
        return int(task.get("ns", 0))
    return int(len(state_mapping))


def build_system_prompt(action_only: bool = False, thinking_mode: bool = False) -> str:
    thinking_rule = ""
    if thinking_mode:
        thinking_rule = (
            "- If you use internal reasoning/thinking, do NOT write long reasoning on every step. "
            "Think briefly when needed, then output the action JSON immediately.\n"
        )

    if action_only:
        return (
            "You are a reinforcement-learning control agent.\n"
            "You interact with an external environment step by step.\n"
            "At each step, you MUST output a single JSON object only, with this schema:\n"
            '{"action": <int>}\n'
            "Rules:\n"
            "- action must be an integer in [0, action_space-1].\n"
            "- Do not include markdown/code fences.\n"
            "- Optimize long-term mean reward.\n"
            f"{thinking_rule}"
        )
    return (
        "You are a reinforcement-learning control agent.\n"
        "You interact with an external environment step by step.\n"
        "At each step, you MUST output a single JSON object only, with this schema:\n"
        '{"action": <int>, "reason": "<short reason>"}\n'
        "Rules:\n"
        "- action must be an integer in [0, action_space-1].\n"
        "- Keep reason concise.\n"
        "- Do not include markdown/code fences.\n"
        "- Optimize long-term mean reward.\n"
        f"{thinking_rule}"
    )


def build_task_prompt(
    task: Dict[str, Any],
    task_name: str,
    max_steps: int,
    action_only: bool = False,
    thinking_mode: bool = False,
) -> str:
    task_type = task.get("task_type", "MDP")
    ns = int(task.get("ns", _real_state_size(task)))
    na = int(task.get("na", 0))
    no = int(task.get("no", ns))
    real_ns = _real_state_size(task)
    s0 = list(map(int, task.get("s_0", [])))
    se = list(map(int, task.get("s_e", [])))

    confirm_line = (
        'Reply with {"ready": true} to confirm understanding.'
        if action_only
        else 'Reply with {"ready": true, "reason": "ready"} to confirm understanding.'
    )

    return (
        f"Task: {task_name}\n"
        f"Environment type: {task_type}\n"
        f"Declared state space (ns): {ns}\n"
        f"Effective active states: {real_ns}\n"
        f"Observation space size (no): {no}\n"
        f"Action space size (na): {na}\n"
        f"Start states (s_0): {s0}\n"
        f"Terminal states (s_e): {se}\n"
        f"Episode max steps: {max_steps}\n\n"
        "Interaction protocol:\n"
        "1) You receive observation and reward from last step.\n"
        "2) You output JSON with one action.\n"
        "3) Environment returns next observation and reward.\n\n"
        "Goal: maximize the AVERAGE (mean) reward per step over the entire episode.\n"
        "Important: terminal states are NOT goals.\n"
        "Exploration and control policy:\n"
        "- Stage A (0%-20% steps): explore efficiently to reduce uncertainty.\n"
        "- Stage B (20%-50% steps): gradually shift from exploration to reward-driven control.\n"
        "- Stage C (50%-100% steps): mostly exploit best-known actions; only small targeted exploration.\n"
        "- Never use mechanical action index loops (0,1,2,3,4,...) as an exploration strategy.\n\n"
        + (
            "Thinking mode:\n"
            "- Reasoning/thinking is enabled, but you do NOT need to think on every step.\n"
            "- Use short reasoning only when the situation is unclear; otherwise output "
            "the action JSON directly.\n\n"
            if thinking_mode
            else ""
        )
        + f"{confirm_line}"
    )


def build_step_prompt(
    step_idx: int,
    observation: Any,
    prev_action: Optional[int],
    prev_reward: Optional[float],
    prev_terminated: bool,
    prev_truncated: bool,
    recent_history: List[Dict[str, Any]],
    na: int,
    task_max_steps: int,
) -> str:
    obs_json = json.dumps(observation)
    prev_action_json = json.dumps(prev_action)
    prev_reward_text = "null" if prev_reward is None else f"{prev_reward:.6f}"
    actions = _recent_actions(recent_history, k=8)
    actions_json = json.dumps(actions, ensure_ascii=True)
    prev_transition_note = ""
    if prev_terminated or prev_truncated:
        prev_transition_note = (
            f"Previous step triggered episode end: terminated={str(prev_terminated).lower()}, "
            f"truncated={str(prev_truncated).lower()}\n"
        )
    anti_cycle_hint = ""
    if _looks_like_index_cycle(actions, na):
        anti_cycle_hint = (
            "Warning: recent actions look like a mechanical index cycle. Break this pattern now "
            "and choose by expected reward for the current observation.\n"
        )
    phase_note = _phase_guidance(step_idx=step_idx, task_max_steps=task_max_steps)
    return (
        f"Step: {step_idx}\n"
        f"Previous action: {prev_action_json}\n"
        f"Previous reward: {prev_reward_text}\n"
        f"{prev_transition_note}"
        f"Current observation: {obs_json}\n"
        f"Recent actions (oldest->newest): {actions_json}\n"
        f"Action space size: {na}\n"
        f"{phase_note}"
        f"{anti_cycle_hint}"
        "Choose the next action now. Output strict JSON only."
    )


def parse_action(response_text: str, na: int) -> Tuple[int, str]:
    raw = response_text.strip()

    # Try pure JSON first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict) and "action" in obj:
            act = int(obj["action"])
            act = max(0, min(na - 1, act))
            return act, raw
    except Exception:
        pass

    # Try finding a JSON object inside text.
    match = re.search(r"\{[\s\S]*\}", raw)
    if match is not None:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and "action" in obj:
                act = int(obj["action"])
                act = max(0, min(na - 1, act))
                return act, raw
        except Exception:
            pass

    # Fallback: first integer in response.
    num = re.search(r"-?\d+", raw)
    if num is not None:
        act = int(num.group(0))
        act = max(0, min(na - 1, act))
        return act, raw

    # Final fallback.
    return 0, raw


def task_matches_filters(
    task: Dict[str, Any],
    task_type_filter: str,
    target_state_sizes: List[int],
    target_actions: int,
) -> bool:
    task_type = str(task.get("task_type", "MDP")).upper()
    real_ns = _real_state_size(task)
    na = int(task.get("na", -1))

    if task_type_filter.lower() == "mdp" and task_type != "MDP":
        return False
    if task_type_filter.lower() == "pomdp" and task_type != "POMDP":
        return False
    if target_state_sizes and real_ns not in target_state_sizes:
        return False
    if target_actions > 0 and na != target_actions:
        return False
    return True
