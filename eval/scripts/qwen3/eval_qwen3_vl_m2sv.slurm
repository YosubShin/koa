#!/bin/bash
#SBATCH --job-name=eval-qwen3-vl-m2sv
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:{best_gpu}:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/lustre/koa/scratch/%u/koa-ml/eval/results/%j/job.log
#SBATCH --error=/mnt/lustre/koa/scratch/%u/koa-ml/eval/results/%j/job.log

set -euo pipefail

# Error handling: log failures and cleanup
cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "======================================"
        echo "Job FAILED with exit code: $exit_code"
        echo "Failed at: $(date)"
        echo "======================================"
        echo "ERROR: Job failed with exit code $exit_code at $(date)" >> "$RESULTS_DIR/error.log" 2>/dev/null || true
    fi
}
trap cleanup EXIT ERR

REMOTE_CODE_ROOT="${KOA_ML_CODE_ROOT:-${KOA_ML_WORKDIR:-$HOME/koa-ml}}"
REMOTE_DATA_ROOT="${KOA_ML_DATA_ROOT:-/mnt/lustre/koa/scratch/$USER/koa-ml}"

if [ ! -d "$REMOTE_CODE_ROOT" ]; then
  echo "Remote code directory not found at $REMOTE_CODE_ROOT" >&2
  exit 1
fi

mkdir -p "$REMOTE_DATA_ROOT/eval/results"
cd "$REMOTE_CODE_ROOT"

# Load environment variables (HF_TOKEN, WANDB_API_KEY, etc.)
if [ -f "$REMOTE_CODE_ROOT/.env" ]; then
  echo "Loading environment from .env file..."
  set -a  # automatically export all variables
  source "$REMOTE_CODE_ROOT/.env"
  set +a
else
  echo "Warning: No .env file found. HuggingFace and W&B tokens may not be available."
  echo "To sync your tokens, run: koa-ml auth --sync"
fi

# Create job-specific results directory
RESULTS_DIR="$REMOTE_DATA_ROOT/eval/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"
export KOA_RESULTS_DIR="$RESULTS_DIR"

# Copy version control files to job directory
echo "Copying version control files..."
SLURM_SCRIPT_PATH="$REMOTE_CODE_ROOT/eval/scripts/qwen3/eval_qwen3_vl_m2sv.slurm"
PYTHON_SCRIPT_PATH="$REMOTE_CODE_ROOT/eval/qwen3_vl_eval.py"
CONFIG_PATH="$REMOTE_CODE_ROOT/eval/configs/qwen3_vl_m2sv.yaml"

cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy SLURM script"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy Python script"
cp "$CONFIG_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy config file"
echo "Version control files copied to: $RESULTS_DIR"
echo ""

echo "======================================"
echo "Evaluating Qwen3-VL on M2SV"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load required modules (override with KOA_PYTHON_MODULE / KOA_CUDA_MODULE)
if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
  if [ -n "${KOA_PYTHON_MODULE:-}" ]; then
    module load "${KOA_PYTHON_MODULE}"
  else
    module load lang/Python/3.11.5-GCCcore-13.2.0
  fi
  if [ -n "${KOA_CUDA_MODULE:-}" ]; then
    module load "${KOA_CUDA_MODULE}"
  fi
fi

# Activate virtual environment
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_TELEMETRY=1
VENV_PATH="${KOA_ML_VENV:-$REMOTE_CODE_ROOT/.venv}"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

echo "==== GPU Info ===="
nvidia-smi
echo ""

echo "==== Python Environment ===="
which python
python --version
echo ""

CONFIG_PATH="eval/configs/qwen3_vl_m2sv.yaml"

# Detect backend preference: env takes precedence, else read from YAML config
BACKEND="${KOA_EVAL_BACKEND:-${KOA_EVALUATOR_BACKEND:-}}"
if [ -z "${BACKEND}" ]; then
  BACKEND=$(python scripts/yaml_get.py "${CONFIG_PATH}" inference.backend 2>/dev/null || true)
  BACKEND="${BACKEND,,}"
fi

## Resolve model name: env override > YAML > default
MODEL="${KOA_EVAL_MODEL:-${VLLM_MODEL:-}}"
if [ -z "${MODEL}" ]; then
  MODEL=$(python scripts/yaml_get.py "${CONFIG_PATH}" model.model_name 2>/dev/null || true)
fi
if [ -z "${MODEL}" ]; then
  MODEL="Qwen/Qwen3-VL-30B-A3B-Instruct"
fi

STARTED_VLLM=0

if [ "${BACKEND}" = "vllm" ]; then
  if [ -z "${VLLM_API_BASE:-}" ]; then
    echo "==== Starting vLLM server (background) ===="
    PORT=${VLLM_PORT:-$((8000 + SLURM_JOB_ID % 1000))}
    export KOA_VLLM_VENV="${KOA_VLLM_VENV:-$HOME/koa-ml/.venv-vllm}"
    scripts/run_vllm_server.sh --model "${MODEL}" --port "${PORT}" --tp ${VLLM_TP:-1} --gpu-mem-util ${VLLM_GPU_MEM_UTIL:-0.92} &
    VLLM_PID=$!
    STARTED_VLLM=1

    scripts/wait_for_vllm.sh --api-base "http://127.0.0.1:${PORT}" --model "${MODEL}" --retries ${VLLM_WAIT_RETRIES:-60} --sleep ${VLLM_WAIT_SLEEP:-10} || {
      echo "vLLM did not load model ${MODEL} in time." >&2
      kill ${VLLM_PID} || true
      exit 1
    }
    export VLLM_API_BASE="http://127.0.0.1:${PORT}"
  fi
  export KOA_EVAL_BACKEND=vllm
  export VLLM_MODEL="${MODEL}"
else
  echo "Skipping vLLM startup; backend='${BACKEND:-hf}'"
fi

echo "==== Starting Evaluation ===="
python eval/qwen3_vl_eval.py \
  --config "${CONFIG_PATH}" || STATUS=$?

if [ ${STARTED_VLLM} -eq 1 ]; then
  echo "Stopping vLLM server (PID ${VLLM_PID})..."
  kill ${VLLM_PID} || true
  wait ${VLLM_PID} || true
fi
exit ${STATUS:-0}

echo ""
echo "======================================"
echo "Evaluation complete!"
echo "Ended: $(date)"
echo "======================================"
