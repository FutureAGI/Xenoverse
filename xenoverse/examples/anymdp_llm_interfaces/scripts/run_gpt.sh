#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <tasks_dir> [extra evaluator args...]"
  exit 1
fi

TASKS_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT_DIR}"

# Fill your key here, or export OPENAI_API_KEY before running.
export OPENAI_API_KEY="${OPENAI_API_KEY:-...}"
if [ "${OPENAI_API_KEY}" = "..." ] || [ -z "${OPENAI_API_KEY}" ]; then
  echo "Please set OPENAI_API_KEY in this script or environment."
  exit 1
fi

MODEL="${GPT_MODEL:-gpt-5}"

python3 -m anymdp_branch.run_anymdp_llm_eval \
  --tasks-dir "${TASKS_DIR}" \
  --provider gpt \
  --model "${MODEL}" \
  "$@"
