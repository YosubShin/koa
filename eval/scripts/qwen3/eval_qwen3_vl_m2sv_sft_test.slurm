#!/bin/bash
#SBATCH --job-name=eval-qwen3vl-test
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/koa-ml/eval/results/%j/job.log
#SBATCH --error=/home/%u/koa-ml/eval/results/%j/job.log

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

REMOTE_WORKDIR="${KOA_ML_WORKDIR:-$HOME/koa-ml}"
if [ ! -d "$REMOTE_WORKDIR" ]; then
  echo "Remote workdir not found at $REMOTE_WORKDIR" >&2
  exit 1
fi
cd "$REMOTE_WORKDIR"

# Load environment variables (HF_TOKEN, WANDB_API_KEY, etc.)
if [ -f "$REMOTE_WORKDIR/.env" ]; then
  echo "Loading environment from .env file..."
  set -a  # automatically export all variables
  source "$REMOTE_WORKDIR/.env"
  set +a
else
  echo "Warning: No .env file found. HuggingFace and W&B tokens may not be available."
  echo "To sync your tokens, run: koa-ml auth --sync"
fi

# Create job-specific results directory
RESULTS_DIR="$REMOTE_WORKDIR/eval/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy version control files to job directory
echo "Copying version control files..."
SLURM_SCRIPT_PATH="$REMOTE_WORKDIR/eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_test.slurm"
PYTHON_SCRIPT_PATH="$REMOTE_WORKDIR/eval/qwen3_vl_eval.py"
CONFIG_PATH="$REMOTE_WORKDIR/eval/configs/qwen3_vl_m2sv_sft_test.yaml"

cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy SLURM script"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy Python script"
cp "$CONFIG_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy config file"
echo "Version control files copied to: $RESULTS_DIR"
echo ""

echo "======================================"
echo "Evaluating Qwen3-VL on M2SV-SFT TEST Split"
echo "Checking generalization to unseen data"
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
VENV_PATH="${KOA_ML_VENV:-$HOME/koa-ml/.venv}"
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

echo "==== Starting Evaluation (TEST Split) ===="
echo "NOTE: This tests generalization - accuracy may be lower than train split"
python eval/qwen3_vl_eval.py \
  --config eval/configs/qwen3_vl_m2sv_sft_test.yaml \
  --output-dir "$RESULTS_DIR"

echo ""
echo "======================================"
echo "Evaluation complete!"
echo "Ended: $(date)"
echo "======================================"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Check accuracy in: $RESULTS_DIR/summary.json"
echo ""
echo "Compare train vs test accuracy to assess model convergence:"
echo "  - High train accuracy, low test = overfitting"
echo "  - Both improving together = good convergence"
echo "  - Both low = need more training or better hyperparameters"
