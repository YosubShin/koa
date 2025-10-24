#!/bin/bash
#SBATCH --job-name=sft-qwen3vl-4b-m2sv
#SBATCH --partition=kill-shared
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
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
SLURM_SCRIPT_PATH="$REMOTE_CODE_ROOT/train/scripts/sft_qwen3_4b/sft_qwen3_4b.slurm"
PYTHON_SCRIPT_PATH="$REMOTE_CODE_ROOT/train/scripts/sft_qwen3_4b/sft_qwen3_4b.py"

cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy SLURM script"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/" 2>/dev/null || echo "Warning: Could not copy Python script"
echo "Version control files copied to: $RESULTS_DIR"
echo ""

echo "======================================"
echo "Qwen3-VL 4B Fine-tuning on M2SV-SFT"
echo "Official Qwen3-VL Training Paradigm"
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
echo "Using official Qwen3-VL training script structure"
echo ""

# Training configuration following official Qwen3-VL paradigm
# https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune

torchrun --nproc_per_node=1 \
  train/scripts/sft_qwen3_4b/sft_qwen3_4b.py \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --dataset_name yosubshin/m2sv-sft-11k \
  --dataset_split train \
  --use_dataset_validation_split True \
  --output_dir "$RESULTS_DIR" \
  --tune_mm_llm True \
  --tune_mm_mlp False \
  --tune_mm_vision False \
  --max_pixels $((576 * 32 * 32)) \
  --min_pixels $((16 * 32 * 32)) \
  --model_max_length 4096 \
  --lora_enable True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.0 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj" \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_steps 10 \
  --eval_strategy steps \
  --eval_steps 273 \
  --save_steps 273 \
  --save_strategy steps \
  --save_total_limit 3 \
  --load_best_model_at_end \
  --metric_for_best_model eval_loss \
  --gradient_checkpointing True \
  --optim adamw_torch \
  --report_to wandb \
  --run_name "qwen3-vl-4b-m2sv-lora-${SLURM_JOB_ID}" \
  --remove_unused_columns False \
  --dataloader_pin_memory False

echo ""
echo "======================================"
echo "Training complete!"
echo "Ended: $(date)"
echo "======================================"
echo ""
echo "Next steps:"
echo "  1. Check W&B dashboard for training metrics"
echo "  2. Evaluate on train split: koa-ml submit eval/scripts/sft_qwen3_4b/eval_train.slurm"
echo "  3. Evaluate on test split: koa-ml submit eval/scripts/sft_qwen3_4b/eval_test.slurm"
echo "  4. Compare train vs test accuracy to assess convergence and generalization"
echo ""
