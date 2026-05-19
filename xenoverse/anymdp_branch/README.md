# AnyMDP LLM Eval Branch

This directory provides a standalone AnyMDP/AnyPOMDP LLM evaluation pipeline with these goals:

- Call paid model APIs: Claude / GPT / DeepSeek / Gemini.
- Support local OpenAI-compatible endpoints (e.g. LM Studio).
- Keep session context within a single task (cache/session); do not restart from scratch every step.
- Reset the session after each task to avoid cross-task leakage.
- Batch-load task folders and filter by task type and scale.
- Save full step-level trajectories (observation / action / reward / raw model reply) for analysis.

## Layout

- `run_anymdp_llm_eval.py`: CLI entry point
- `evaluator.py`: evaluation loop, environment interaction, logging
- `llm_clients.py`: multi-provider API adapters
- `prompting.py`: task/step prompt construction and action parsing

## Supported models and API keys

- Claude: `ANTHROPIC_API_KEY`
- GPT: `OPENAI_API_KEY`
- DeepSeek: `DEEPSEEK_API_KEY`
- Gemini: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)

Override default models with `--model`. Example defaults:

- Claude: `claude-sonnet-4-5`
- GPT: `gpt-5`
- DeepSeek: `deepseek-v4`
- Gemini: `gemini-2.5-pro`

> Note: available model names vary by account. If you get a model-not-found error, pass `--model` with a name your account supports.

## Task filters

- `--task-type all|mdp|pomdp`
- `--state-sizes 1,4,8,16` (filter by effective state count, i.e. `len(state_mapping)`)
- `--action-size 5`

## Run examples

### 1) GPT: batch run over a task folder

```bash
python -m anymdp_branch.run_anymdp_llm_eval \
  --tasks-dir /path/to/tasks \
  --provider gpt \
  --task-type all \
  --state-sizes 1,4,8,16 \
  --action-size 5 \
  --max-steps 200 \
  --output-dir anymdp_branch/runs
```

### 0) Shell scripts (one per model)

```bash
bash anymdp_branch/scripts/run_claude.sh /path/to/tasks --task-type all --state-sizes 1,4,8,16 --action-size 5
bash anymdp_branch/scripts/run_gpt.sh /path/to/tasks --task-type all --state-sizes 1,4,8,16 --action-size 5
bash anymdp_branch/scripts/run_deepseek.sh /path/to/tasks --task-type all --state-sizes 1,4,8,16 --action-size 5
bash anymdp_branch/scripts/run_gemini.sh /path/to/tasks --task-type all --state-sizes 1,4,8,16 --action-size 5
```

### Run all four models sequentially

```bash
bash anymdp_branch/scripts/run_all_models.sh /path/to/tasks --task-type all --state-sizes 1,4,8,16 --action-size 5
```

### 4) Local LM Studio (`nvidia/nemotron-3-super`)

After starting the local server in LM Studio (e.g. `http://127.0.0.1:1234`):

```bash
bash anymdp_branch/scripts/run_local.sh /path/to/tasks \
  --task-type all \
  --state-sizes 1,4,8,16 \
  --action-size 5 \
  --action-only
```

Optional environment variables:

```bash
export LOCAL_API_BASE="http://127.0.0.1:1234"
export LOCAL_MODEL="nvidia/nemotron-3-super"
```

### 2) POMDP only (Claude)

```bash
python -m anymdp_branch.run_anymdp_llm_eval \
  --tasks-dir /path/to/tasks \
  --provider claude \
  --task-type pomdp \
  --state-sizes 1,4,8,16 \
  --action-size 5 \
  --max-tasks 20
```

### 3) DeepSeek / Gemini with a custom model

```bash
python -m anymdp_branch.run_anymdp_llm_eval \
  --tasks-dir /path/to/tasks \
  --provider deepseek \
  --model deepseek-v4 \
  --max-steps 150
```

## Outputs

Each run writes under `--output-dir/<provider>_<timestamp>/`:

- `steps.jsonl`: full per-step trajectory log
- `tasks_summary.json`: per-task stats and run-level aggregates (includes `mean_reward`)

`steps.jsonl` fields per line:

- Task id: `task_index`, `task_file`
- Environment: `step`, `observation`, `action`, `reward`, `reward_gt`, `terminated`, `truncated`
- Model: `llm_reply`
- Time: `timestamp`

## Cache / session strategy

- Within one task: keep conversation state (`messages` / `previous_response_id`).
- On task change: call `reset_task()` to clear context.
- Claude: prompt caching headers; GPT: Responses API `previous_response_id` chain; DeepSeek/Gemini: local message history.
- Local `provider=local`: task-scoped cache via OpenAI-compatible `chat/completions` message history.

## Lower token usage (`--action-only`)

With `--action-only`, the model is prompted to return only:

```json
{"action": 3}
```

This reduces output tokens and usually lowers cost. Example:

```bash
bash anymdp_branch/scripts/run_gpt.sh /path/to/tasks --action-only --max-steps 200
```
