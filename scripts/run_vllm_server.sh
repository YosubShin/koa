#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server on the current node.
#
# Usage:
#   scripts/run_vllm_server.sh --model Qwen/Qwen2-7B-Instruct [--port 8000] [--tp 1] [--gpu-mem-util 0.90]
#
# Notes:
# - Exposes OpenAI API at: http://127.0.0.1:${PORT}/v1
# - Set VLLM_API_BASE accordingly for clients.
# - For multi-GPU, adjust tensor parallel size (--tensor-parallel-size).
# - For Qwen3-VL, pass the appropriate model repo/path and ensure GPU has enough memory.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${KOA_VLLM_VENV:-}" ]]; then
  if [[ -d "${KOA_VLLM_VENV}" ]]; then
    echo "[vllm-server] Activating KOA_VLLM_VENV: ${KOA_VLLM_VENV}"
    # shellcheck disable=SC1090
    source "${KOA_VLLM_VENV}/bin/activate"
  else
    echo "[vllm-server] KOA_VLLM_VENV is set but not a directory: ${KOA_VLLM_VENV}" >&2
  fi
fi
PORT=8000
TP=1
GPU_MEM_UTIL=0.90
MODEL=""
MAX_MODEL_LEN=""
MAX_BATCH_TOKENS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2;;
    --tp|--tensor-parallel-size)
      TP="$2"; shift 2;;
    --gpu-mem-util)
      GPU_MEM_UTIL="$2"; shift 2;;
    --model)
      MODEL="$2"; shift 2;;
    --max-model-len)
      MAX_MODEL_LEN="$2"; shift 2;;
    --max-batch-tokens)
      MAX_BATCH_TOKENS="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 2;;
  esac
done

if [[ -z "${MODEL}" ]]; then
  echo "Error: --model is required (HF repo id or local path)" >&2
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python not found. Activate your venv first (e.g., source ${PROJECT_ROOT}/.venv/bin/activate)" >&2
  exit 1
fi

echo "[vllm-server] Starting vLLM for model: ${MODEL}"
echo "[vllm-server] Port: ${PORT} | TP: ${TP} | GPU mem util: ${GPU_MEM_UTIL}"
echo "[vllm-server] API base: http://127.0.0.1:${PORT}"

ARGS=(
  "--model" "${MODEL}"
  "--host" "127.0.0.1"
  "--port" "${PORT}"
  "--tensor-parallel-size" "${TP}"
  "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
  "--served-model-name" "${MODEL}"
)

if [[ -n "${MAX_MODEL_LEN}" ]]; then
  ARGS+=("--max-model-len" "${MAX_MODEL_LEN}")
fi
if [[ -n "${MAX_BATCH_TOKENS}" ]]; then
  ARGS+=("--max-num-seqs" "1" "--max-num-batched-tokens" "${MAX_BATCH_TOKENS}")
fi

exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"