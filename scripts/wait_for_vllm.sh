#!/usr/bin/env bash
# Wait for a vLLM OpenAI-compatible server to report the expected model id.
#
# Usage:
#   scripts/wait_for_vllm.sh --api-base http://127.0.0.1:8000 --model <MODEL_ID> [--retries 300] [--sleep 2]

set -euo pipefail

API_BASE=""
EXPECT_MODEL=""
RETRIES=60
SLEEP_SECS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-base)
      API_BASE="$2"; shift 2;;
    --model)
      EXPECT_MODEL="$2"; shift 2;;
    --retries)
      RETRIES="$2"; shift 2;;
    --sleep)
      SLEEP_SECS="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 2;;
  esac
done

if [[ -z "${API_BASE}" || -z "${EXPECT_MODEL}" ]]; then
  echo "Usage: wait_for_vllm.sh --api-base <URL> --model <MODEL_ID> [--retries N] [--sleep S]" >&2
  exit 2
fi

echo "[wait-for-vllm] Waiting for ${EXPECT_MODEL} at ${API_BASE} ..."

for ((i=0; i<RETRIES; i++)); do
  if curl -fsS "${API_BASE}/v1/models" | grep -Eq "\"id\"[[:space:]]*:[[:space:]]*\"${EXPECT_MODEL}\""; then
    echo "[wait-for-vllm] Ready: ${EXPECT_MODEL}"
    exit 0
  fi
  sleep "${SLEEP_SECS}"
done

echo "[wait-for-vllm] Timeout waiting for ${EXPECT_MODEL} at ${API_BASE}" >&2
exit 1


