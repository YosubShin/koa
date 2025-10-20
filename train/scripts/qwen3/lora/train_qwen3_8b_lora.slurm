#!/bin/bash
#SBATCH --job-name=train-qwen3-8b-lora
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/%u/koa-ml/train/results/%j/job.log
#SBATCH --error=/home/%u/koa-ml/train/results/%j/job.log

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
RESULTS_DIR="$REMOTE_WORKDIR/train/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy version control files to job directory
echo "Copying version control files..."
SLURM_SCRIPT_PATH="$REMOTE_WORKDIR/train/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm"
PYTHON_SCRIPT_PATH="$REMOTE_WORKDIR/train/train.py"
CONFIG_PATH="$REMOTE_WORKDIR/configs/recipes/qwen3/8b/lora.yaml"

cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy SLURM script"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy Python script"
cp "$CONFIG_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy config file"
echo "Version control files copied to: $RESULTS_DIR"
echo ""

echo "======================================"
echo "Training Qwen3 8B with LoRA"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

# Load modules
# module load python/3.11
# module load cuda/12.1

# Activate virtual environment
VENV_PATH="${KOA_ML_VENV:-$HOME/koa-ml/.venv}"
if [ ! -d "$VENV_PATH" ]; then
  echo "Virtualenv not found at $VENV_PATH" >&2
  exit 1
fi
source "$VENV_PATH/bin/activate"

# GPU info
echo "==== GPU Info ===="
nvidia-smi
echo ""

# Training
echo "==== Starting Training ===="
python train/train.py \
    --config configs/recipes/qwen3/8b/lora.yaml \
    --output_dir "$RESULTS_DIR"

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
