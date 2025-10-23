# Training Script & SLURM Best Practices

Best practices learned from Qwen3-VL training jobs to avoid warnings and ensure clean, professional logs.

## Table of Contents
1. [Python Training Script Best Practices](#python-training-script-best-practices)
2. [SLURM Script Best Practices](#slurm-script-best-practices)
3. [Configuration File Best Practices](#configuration-file-best-practices)
4. [Pre-Flight Checklist](#pre-flight-checklist)

---

## Python Training Script Best Practices

### 1. **Suppress Expected Warnings**

When using LoRA with gradient checkpointing, certain warnings are expected and harmless. Suppress them to keep logs clean:

```python
import warnings

# Suppress gradient checkpointing warnings with frozen layers (expected with LoRA)
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

# Suppress padding warnings (if handling manually)
warnings.filterwarnings("ignore", message="`max_length` is ignored when `padding`")

# Suppress pydantic warnings from transformers internals
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic._internal._generate_schema")
```

**Why**: With LoRA, most model parameters are frozen, so gradient checkpointing warnings about `requires_grad=False` are expected.

---

### 2. **Use Modern Transformers API Parameters**

**Old (deprecated)**:
```python
model_kwargs = {
    'torch_dtype': torch.bfloat16,  # Deprecated!
}
```

**New (correct)**:
```python
model_kwargs = {
    'dtype': torch.bfloat16,  # Use 'dtype' instead
}
```

**Why**: `torch_dtype` is deprecated in newer transformers versions and causes warnings.

---

### 3. **Explicitly Handle Gradient Checkpointing + Cache**

When using gradient checkpointing, explicitly disable KV cache to avoid warnings:

```python
# Load model
model = Qwen3VLForConditionalGeneration.from_pretrained(**model_kwargs)

# Disable cache if gradient checkpointing is enabled
if config['training'].get('gradient_checkpointing', False):
    model.config.use_cache = False
```

**Why**: `use_cache=True` is incompatible with gradient checkpointing. Setting it explicitly avoids the warning.

---

### 4. **Proper Padding and Truncation**

When processing inputs with a processor/tokenizer:

**Incomplete**:
```python
processed = processor(
    text=text,
    images=images,
    padding=True,
    max_length=2048
)
```

**Complete**:
```python
processed = processor(
    text=text,
    images=images,
    padding=True,
    truncation=True,  # Add this!
    max_length=2048
)
```

**Why**: Without `truncation=True`, the `max_length` parameter is ignored when `padding=True`, causing warnings.

---

### 5. **Environment Variables for HuggingFace**

Always set these environment variables early in your script:

```python
import os

# Disable HF Transfer (can be unreliable on HPC clusters)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")

# Disable telemetry on shared clusters
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
```

**Why**: Prevents connectivity issues and unwanted telemetry on HPC systems.

---

### 6. **PIL Image Size Limits**

For vision models processing large images:

```python
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
```

**Why**: Prevents PIL from rejecting large images that might be in your dataset.

---

## SLURM Script Best Practices

### 1. **Robust Error Handling**

Always include error handling and cleanup:

```bash
set -euo pipefail  # Exit on error, undefined vars, pipe failures

cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "======================================"
        echo "Job FAILED with exit code: $exit_code"
        echo "Failed at: $(date)"
        echo "======================================"
    fi
}
trap cleanup EXIT ERR
```

**Why**: Ensures you know if/when jobs fail and provides clear error messages.

---

### 2. **Environment Variable Loading**

Load `.env` file for API tokens:

```bash
# Load environment variables (HF_TOKEN, WANDB_API_KEY, etc.)
if [ -f "$REMOTE_CODE_ROOT/.env" ]; then
  echo "Loading environment from .env file..."
  set -a  # automatically export all variables
  source "$REMOTE_CODE_ROOT/.env"
  set +a
else
  echo "Warning: No .env file found."
  echo "To sync your tokens, run: koa-ml auth --sync"
fi
```

**Why**: Keeps sensitive tokens out of version control while making them available to jobs.

---

### 3. **Version Control for Reproducibility**

Copy all relevant files to the job results directory:

```bash
RESULTS_DIR="$REMOTE_DATA_ROOT/train/results/${SLURM_JOB_ID}"
mkdir -p "$RESULTS_DIR"

# Copy scripts and configs
cp "$SLURM_SCRIPT_PATH" "$RESULTS_DIR/"
cp "$PYTHON_SCRIPT_PATH" "$RESULTS_DIR/"
cp "$CONFIG_PATH" "$RESULTS_DIR/"
```

**Why**: Enables reproducibility - you can always see exactly what code/config was used for each job.

---

### 4. **Resource Allocation Guidelines**

Based on Qwen3-VL 4B with QLoRA training:

```bash
#SBATCH --gres=gpu:1           # 1 GPU sufficient for 4B model with QLoRA
#SBATCH --mem=64G              # 48-64GB for safety
#SBATCH --cpus-per-task=8      # 8 CPUs for data loading
#SBATCH --time=48:00:00        # Generous time for full training
```

**Memory estimates** (with 4-bit quantization + LoRA):
- 4B model: ~6-7GB GPU memory
- 7B model: ~10-12GB GPU memory
- 13B model: ~18-22GB GPU memory

**Time estimates** (at ~5-9s/step):
- 1000 steps: ~1.5-2.5 hours
- 5000 steps: ~7-13 hours
- 10000 steps: ~14-25 hours

---

### 5. **Informative Logging**

Include clear section markers and useful information:

```bash
echo "======================================"
echo "Training Qwen3-VL 4B on M2SV-SFT"
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo ""

echo "==== GPU Info ===="
nvidia-smi
echo ""

echo "==== Python Environment ===="
which python
python --version
echo ""
```

**Why**: Makes debugging easier and provides context when reviewing old job logs.

---

### 6. **Memory Optimization Settings**

For CUDA memory management:

```bash
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Why**: Helps prevent memory fragmentation during long training runs.

---

## Configuration File Best Practices

### 1. **Test Configuration vs Production Configuration**

**Test run** (quick validation):
```yaml
data:
  train_dataset:
    limit: 10  # Small subset for testing

training:
  max_steps: 100
  save_steps: 50
  report_to: "none"  # Disable W&B for quick tests
```

**Production run** (full training):
```yaml
data:
  train_dataset:
    # No limit - use full dataset

training:
  max_steps: 5000
  save_steps: 500
  eval_steps: 500
  report_to: "wandb"  # Enable experiment tracking
```

---

### 2. **QLoRA Configuration**

For memory-efficient training:

```yaml
model:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true
  use_cache: false  # Required for gradient checkpointing

training:
  gradient_checkpointing: true  # Reduces memory usage
  bf16: true  # Use bfloat16 for H200/A100 GPUs
```

---

### 3. **LoRA Target Modules**

For Qwen3-VL (and most transformer models):

```yaml
peft:
  type: "lora"
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  lora_target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
  bias: "none"
```

**Expected trainable %**: ~0.2-0.5% of total parameters

---

### 4. **Batch Size and Gradient Accumulation**

For limited GPU memory:

```yaml
training:
  per_device_train_batch_size: 1  # Fit in memory
  gradient_accumulation_steps: 16  # Effective batch size = 16
```

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` × `num_gpus`

---

## Pre-Flight Checklist

Before submitting any training job, verify:

### Local Environment
- [ ] Changes committed or documented
- [ ] Config file updated (test vs production)
- [ ] `.env` file contains required tokens (HF_TOKEN, WANDB_API_KEY)

### Cluster Sync
- [ ] Run `koa-ml refresh` to sync code
- [ ] Run `koa-ml storage link` to recreate symlinks
- [ ] Verify with `koa-ml check` that cluster is accessible

### SLURM Script
- [ ] Job name is descriptive
- [ ] Resource allocation appropriate (GPU, memory, time)
- [ ] Correct partition/queue specified
- [ ] Paths to scripts/configs are correct
- [ ] Error handling and logging in place

### Training Config
- [ ] Dataset limit removed (or intentionally set for testing)
- [ ] `max_steps` appropriate for full/test run
- [ ] W&B logging enabled/disabled as desired
- [ ] Output directory correctly specified
- [ ] Save/eval steps reasonable for run length

### Post-Submission
- [ ] Note job ID for future reference
- [ ] Check W&B dashboard starts logging
- [ ] Monitor first few steps for errors
- [ ] Verify checkpoints are being saved

---

## Common Issues and Solutions

### Issue: "No space left on device"
**Solution**: Check you're writing to scratch storage, not home directory:
```bash
RESULTS_DIR="/mnt/lustre/koa/scratch/$USER/koa-ml/train/results/${SLURM_JOB_ID}"
```

### Issue: "HuggingFace token not found"
**Solution**: Sync your `.env` file:
```bash
koa-ml auth --sync
```

### Issue: "W&B not logging"
**Solution**:
1. Check `report_to: "wandb"` in config
2. Verify `WANDB_API_KEY` in `.env`
3. Ensure `WANDB_PROJECT` and `WANDB_ENTITY` are set

### Issue: "Symlinks broken after refresh"
**Solution**: Recreate symlinks:
```bash
koa-ml storage link
```

### Issue: "Job killed for exceeding time limit"
**Solution**: Increase time in SLURM script or reduce `max_steps` in config

### Issue: "CUDA out of memory"
**Solution**:
1. Reduce `per_device_train_batch_size`
2. Enable gradient checkpointing
3. Use 4-bit quantization
4. Reduce `model_max_length`

---

## Summary: Golden Rules

1. **Always test first**: Run with `limit: 10` and `max_steps: 100` before full training
2. **Suppress expected warnings**: Keep logs clean and readable
3. **Use modern APIs**: Avoid deprecated parameters (`torch_dtype` → `dtype`)
4. **Track everything**: Enable W&B, copy configs to results, use version control
5. **Fail fast and loud**: Use `set -euo pipefail` and error handlers
6. **Be generous with resources**: Overestimate time/memory to avoid job kills
7. **Sync before submit**: Always `koa-ml refresh` and `koa-ml storage link`

---

*Last updated: 2025-10-22*
*Based on: Qwen3-VL 4B training on M2SV-SFT dataset (Job ID: 8721426)*
