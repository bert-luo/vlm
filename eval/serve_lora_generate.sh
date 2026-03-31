#!/usr/bin/env bash
# Start vLLM (optionally with a PEFT LoRA), then run eval/generate.py against it.
#
# Usage:
#   ./eval/serve_lora_generate.sh <model_or_adapter> [prompt words...] [-- extra args for generate.py]
#
# The first argument is either:
#   - A LoRA adapter dir (contains adapter_config.json) under train/output/ or an abs path
#   - A plain HF model name or path (served without LoRA)
#
# Examples:
#   # LoRA adapter
#   ./eval/serve_lora_generate.sh lora-qwen3.5-0.8b
#   ./eval/serve_lora_generate.sh /home/cowork0330b/vlm/train/output/lora-qwen3.5-0.8b "cozy coffee shop"
#   # Plain model (no LoRA)
#   ./eval/serve_lora_generate.sh Qwen/Qwen3-0.6B "high school study room"
#   ./eval/serve_lora_generate.sh Qwen/Qwen3-0.6B -- --output /tmp/out.png
#
# Environment:
#   VLLM_PORT       (default 8000)
#   VLLM_HOST       bind address (default 127.0.0.1)
#   LORA_API_NAME   name used in OpenAI "model" field (default lora; LoRA mode only)
#   EXTRA_VLLM_ARGS extra CLI args passed to vLLM (quoted string)

set -euo pipefail

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"
OUTPUT_ROOT="${REPO_ROOT}/train/output"

usage() {
  echo "Usage: $0 <model_or_adapter> [prompt...] [-- generate.py args...]" >&2
  exit 1
}

[[ "${1:-}" ]] || usage
MODEL_INPUT="$1"
shift || true

VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
LORA_API_NAME="${LORA_API_NAME:-lora}"

USE_LORA=0
if [[ "${MODEL_INPUT}" == /* ]]; then
  CANDIDATE="${MODEL_INPUT}"
else
  CANDIDATE="${OUTPUT_ROOT}/${MODEL_INPUT}"
fi

if [[ -d "${CANDIDATE}" && -f "${CANDIDATE}/adapter_config.json" ]]; then
  USE_LORA=1
  ADAPTER="${CANDIDATE}"
fi

PROMPT_PARTS=()
GEN_EXTRA=()
mode=prompt
for arg in "$@"; do
  if [[ "${arg}" == "--" ]]; then
    mode=gen
    continue
  fi
  if [[ "${mode}" == "prompt" ]]; then
    PROMPT_PARTS+=("${arg}")
  else
    GEN_EXTRA+=("${arg}")
  fi
done

if [[ ${#PROMPT_PARTS[@]} -eq 0 ]]; then
  PROMPT="high school student study room"
else
  PROMPT="${PROMPT_PARTS[*]}"
fi

if [[ "${USE_LORA}" -eq 1 ]]; then
  BASE_MODEL="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "${ADAPTER}/adapter_config.json")"
  LORA_RANK="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['r'])" "${ADAPTER}/adapter_config.json")"
else
  BASE_MODEL="${MODEL_INPUT}"
fi

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "[serve_lora_generate] Stopping vLLM (pid ${VLLM_PID})..."
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ENDPOINT="http://${VLLM_HOST}:${VLLM_PORT}/v1"

echo "[serve_lora_generate] Base model: ${BASE_MODEL}"
if [[ "${USE_LORA}" -eq 1 ]]; then
  echo "[serve_lora_generate] LoRA path:  ${ADAPTER}"
  echo "[serve_lora_generate] API model:  ${LORA_API_NAME}"
else
  echo "[serve_lora_generate] Mode:       plain (no LoRA)"
fi
echo "[serve_lora_generate] Endpoint:   ${ENDPOINT}"

# Run all uv commands from the eval dir so they use eval's venv / pyproject.toml
cd "${EVAL_DIR}"

VLLM_CMD=(uv run vllm serve "${BASE_MODEL}")

set -- "${VLLM_CMD[@]}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --enforce-eager

if [[ "${USE_LORA}" -eq 1 ]]; then
  set -- "$@" \
    --enable-lora \
    --max-lora-rank "${LORA_RANK}" \
    --lora-modules "${LORA_API_NAME}=${ADAPTER}"
fi

if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_VLLM_ARGS})
  set -- "$@" "${EXTRA_ARR[@]}"
fi

echo "[serve_lora_generate] Starting vLLM: $*"
"$@" &
VLLM_PID=$!

echo "[serve_lora_generate] Waiting for ${ENDPOINT} ..."
ok=0
for _ in $(seq 1 300); do
  if curl -sf "${ENDPOINT}/models" >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 1
done
if [[ "${ok}" -ne 1 ]]; then
  echo "ERROR: vLLM did not become ready on port ${VLLM_PORT} (timeout 300s)." >&2
  exit 1
fi

if [[ "${USE_LORA}" -eq 1 ]]; then
  GEN_MODEL="${LORA_API_NAME}"
else
  GEN_MODEL="${BASE_MODEL}"
fi

echo "[serve_lora_generate] Running generate.py with prompt: ${PROMPT}"
uv run python generate.py "${PROMPT}" \
  --endpoint "${ENDPOINT}" \
  --model "${GEN_MODEL}" \
  "${GEN_EXTRA[@]}"
