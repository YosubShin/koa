#!/bin/bash
#SBATCH --job-name=train-qwen3vl-4b-m2sv
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=/mnt/lustre/koa/scratch/%u/koa-ml/train/results/%j/job.log
#SBATCH --error=/mnt/lustre/koa/scratch/%u/koa-ml/train/results/%j/job.log

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

mkdir -p "$REMOTE_DATA_ROOT/train/results"
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
RESULTS_DIR="$REMOTE_DATA_ROOT/train/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy version control files to job directory
echo "Copying version control files..."
SLURM_SCRIPT_PATH="$REMOTE_CODE_ROOT/train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm"
PYTHON_SCRIPT_PATH="$REMOTE_CODE_ROOT/train/qwen3_vl_train.py"
CONFIG_PATH="$REMOTE_CODE_ROOT/configs/recipes/qwen3_vl/4b_lora_m2sv.yaml"

cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy SLURM script"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy Python script"
cp "$CONFIG_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy config file"
echo "Version control files copied to: $RESULTS_DIR"
echo ""

echo "======================================"
echo "Training Qwen3-VL 4B on M2SV-SFT"
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

echo "==== Starting Training ===="
python train/qwen3_vl_train.py \
  --config configs/recipes/qwen3_vl/4b_lora_m2sv.yaml \
  --output_dir "$RESULTS_DIR"

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for training metrics"
echo "  2. Evaluate on train split: koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_train.slurm"
echo "  3. Evaluate on test split: koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_test.slurm"
echo "  4. Compare train vs test accuracy to assess convergence and generalization"
