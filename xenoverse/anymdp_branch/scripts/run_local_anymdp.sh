#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

MODEL="${LOCAL_MODEL:-nvidia/nemotron-3-super}"
API_BASE="${LOCAL_API_BASE:-http://127.0.0.1:1234}"
TASKS_DIR="${LOCAL_TASKS_DIR:-anymdp_branch/task_files/anymdp}"
LOCAL_TEMPERATURE="${LOCAL_TEMPERATURE:-0.7}"
LOCAL_MAX_HISTORY_MESSAGES="${LOCAL_MAX_HISTORY_MESSAGES:-40}"

export LOCAL_MAX_HISTORY_MESSAGES

THINKING_ARGS=()
if [ "${LOCAL_THINKING_MODE:-1}" = "1" ]; then
  THINKING_ARGS=(--thinking-mode)
fi

python3 -m anymdp_branch.run_anymdp_llm_eval \
  --tasks-dir "${TASKS_DIR}" \
  --provider local \
  --model "${MODEL}" \
  --api-base "${API_BASE}" \
  --task-type mdp \
  --state-sizes 8 \
  --action-size 5 \
  --action-only \
  --temperature "${LOCAL_TEMPERATURE}" \
  --max-steps 2000 \
  --max-tokens 256 \
  "${THINKING_ARGS[@]}" \
  "$@"
