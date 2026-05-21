#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <tasks_dir> [extra evaluator args...]"
  echo "Example: $0 ./anymdp_tasks --task-type all --state-sizes 1,4,8,16 --action-size 5 --action-only"
  exit 1
fi

TASKS_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Fill your keys here, or export them before running.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-...}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-...}"
export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-...}"
export GEMINI_API_KEY="${GEMINI_API_KEY:-...}"
if [ "${ANTHROPIC_API_KEY}" = "..." ] || [ "${OPENAI_API_KEY}" = "..." ] || [ "${DEEPSEEK_API_KEY}" = "..." ] || [ "${GEMINI_API_KEY}" = "..." ]; then
  echo "Please set all API keys before running all providers."
  exit 1
fi

echo "[run_all_models] Running Claude..."
"${SCRIPT_DIR}/run_claude.sh" "${TASKS_DIR}" "$@"

echo "[run_all_models] Running GPT..."
"${SCRIPT_DIR}/run_gpt.sh" "${TASKS_DIR}" "$@"

echo "[run_all_models] Running DeepSeek..."
"${SCRIPT_DIR}/run_deepseek.sh" "${TASKS_DIR}" "$@"

echo "[run_all_models] Running Gemini..."
"${SCRIPT_DIR}/run_gemini.sh" "${TASKS_DIR}" "$@"

echo "[run_all_models] Completed all providers."
