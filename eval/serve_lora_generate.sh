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
# Port assignment:
#   By default the port is auto-derived from CUDA_VISIBLE_DEVICES so parallel
#   runs on different GPUs never collide:  port = 8000 + first_gpu_id
#   Override with VLLM_PORT if needed.
#
# Reasoning:
#   Qwen3/3.5 models emit <think>...</think> reasoning tokens.  vLLM's
#   --enable-reasoning --reasoning-parser qwen3 flags separate these from the
#   actual content so generate.py only sees the code in .message.content.
#
# Environment:
#   VLLM_PORT       (default: auto from CUDA_VISIBLE_DEVICES, fallback 8000)
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

# Auto-derive port from CUDA_VISIBLE_DEVICES to avoid collisions in parallel runs.
if [[ -z "${VLLM_PORT:-}" ]]; then
  _first_gpu="${CUDA_VISIBLE_DEVICES%%,*}"
  if [[ "${_first_gpu}" =~ ^[0-9]+$ ]]; then
    VLLM_PORT=$(( 8000 + _first_gpu ))
  else
    VLLM_PORT=8000
  fi
fi
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
  BASE_MODEL="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "${ADAPTER}/adapter_config.json")"
  LORA_RANK="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['r'])" "${ADAPTER}/adapter_config.json")"
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
  --enforce-eager \
  --reasoning-parser qwen3

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

LOG_DIR="${REPO_ROOT}/eval/logs"

echo "[serve_lora_generate] Running generate.py with prompt: ${PROMPT}"
echo "[serve_lora_generate] Logs will be saved to: ${LOG_DIR}"
GEN_EXIT=0
uv run python generate.py "${PROMPT}" \
  --endpoint "${ENDPOINT}" \
  --model "${GEN_MODEL}" \
  --log-dir "${LOG_DIR}" \
  "${GEN_EXTRA[@]}" || GEN_EXIT=$?

echo ""
echo "============================================================"
echo "[serve_lora_generate] SUMMARY"
echo "  Model:    ${GEN_MODEL}"
if [[ "${USE_LORA}" -eq 1 ]]; then
  echo "  LoRA:     ${ADAPTER}"
fi
echo "  Prompt:   ${PROMPT}"
echo "  Exit:     ${GEN_EXIT}"
echo "  Logs:     ${LOG_DIR}"
if ls "${LOG_DIR}"/*.py >/dev/null 2>&1; then
  echo "  Saved responses:"
  for f in "${LOG_DIR}"/*.py; do
    echo "    - $(basename "$f")"
  done
fi
echo "============================================================"
exit "${GEN_EXIT}"
