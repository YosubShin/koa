# Fine-Tuning Models on KOA

This directory contains configurations and scripts for fine-tuning language models on the KOA HPC cluster.

> **Storage reminder**: Run `koa-ml storage setup --link` so checkpoints and
> logs land on `/mnt/lustre/koa/scratch/<user>/koa-ml` with symlinks in
> `~/koa-ml/train/results`. If you prefer explicit paths, set
> `export KOA_ML_DATA_ROOT=/mnt/lustre/koa/scratch/$USER/koa-ml` when working on
> KOA.

## Quick Start

### 1. Install Dependencies

```bash
# On your local machine
pip install -e ".[ml]"

# Or install only training dependencies
pip install -e ".[train]"
```

### 2. Test Locally (Optional)

```bash
# Quick test with Qwen3 0.6B (small model, trains in minutes)
python train/train.py --config configs/recipes/qwen3/0.6b/lora.yaml
```

### 3. Submit to KOA

```bash
# Quick test job (30 minutes) - Qwen3 0.6B
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Qwen3 8B LoRA training (12 hours)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm

# Qwen3 14B with QLoRA (memory efficient)
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_14b_qlora.slurm
```

## Configuration Files

All configs are in `configs/recipes/qwen3/`:

### Qwen3 0.6B LoRA (Quickstart)
- **File**: [configs/recipes/qwen3/0.6b/lora.yaml](../configs/recipes/qwen3/0.6b/lora.yaml)
- **Purpose**: Fast testing, validates your pipeline works
- **Time**: ~30 minutes
- **Memory**: ~8GB GPU
- **Use case**: Development, debugging, quick iterations

### Qwen3 4B LoRA
- **File**: [configs/recipes/qwen3/4b/lora.yaml](../configs/recipes/qwen3/4b/lora.yaml)
- **Purpose**: Standard LoRA fine-tuning, mid-size model
- **Time**: 4-8 hours (depends on dataset)
- **Memory**: ~16GB GPU
- **Use case**: Production fine-tuning, good balance of size/performance

### Qwen3 8B LoRA
- **File**: [configs/recipes/qwen3/8b/lora.yaml](../configs/recipes/qwen3/8b/lora.yaml)
- **Purpose**: Standard LoRA fine-tuning, larger model
- **Time**: 8-12 hours
- **Memory**: ~24GB GPU
- **Use case**: Production fine-tuning, adapter-based deployment

### Qwen3 8B QLoRA
- **File**: [configs/recipes/qwen3/8b/qlora.yaml](../configs/recipes/qwen3/8b/qlora.yaml)
- **Purpose**: Memory-efficient fine-tuning with 4-bit quantization
- **Time**: 8-12 hours
- **Memory**: ~12GB GPU
- **Use case**: Limited GPU memory, experimentation

### Qwen3 14B QLoRA
- **File**: [configs/recipes/qwen3/14b/qlora.yaml](../configs/recipes/qwen3/14b/qlora.yaml)
- **Purpose**: Large model with memory-efficient QLoRA
- **Time**: 12-16 hours
- **Memory**: ~16GB GPU
- **Use case**: Largest Qwen3 model, maximum performance

## Understanding the Config Files

Each config has 4 main sections:

### 1. Model Section
```yaml
model:
  model_name: "Qwen/Qwen3-8B"                     # HuggingFace model ID
  model_max_length: 8192                          # Context length
  dtype: "bfloat16"                               # Precision
  attn_implementation: "flash_attention_2"        # Attention type
```

### 2. Data Section
```yaml
data:
  train_dataset:
    dataset_name: "yahma/alpaca-cleaned"  # HuggingFace dataset
    split: "train"                        # Which split to use
  target_column: "text"                   # Column with training text
  template: "qwen"                        # Prompt template
```

### 3. Training Section
```yaml
training:
  trainer_type: "sft"                      # Supervised Fine-Tuning
  per_device_train_batch_size: 2           # Batch size per GPU
  gradient_accumulation_steps: 32          # Effective batch: 2 * 32 = 64
  learning_rate: 3.0e-04                   # Learning rate
  lr_scheduler_type: "cosine"              # LR schedule
  warmup_ratio: 0.03                       # Warmup proportion
  num_train_epochs: 3                      # How many epochs
  output_dir: "./train/results/local/my_model"  # Where to save
```

### 4. PEFT Section (LoRA/QLoRA only)
```yaml
peft:
  type: "lora"                             # LoRA type
  lora_r: 8                                # LoRA rank (higher = more params)
  lora_alpha: 16                           # LoRA alpha (scaling)
  lora_target_modules:                     # Which layers to adapt
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

## Customizing Training

### Change Dataset

Edit the config file:
```yaml
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"  # Any HuggingFace dataset
    split: "train"
```

### Adjust Training Duration

Quick test:
```yaml
training:
  max_steps: 100  # Train for 100 steps only
```

Full training:
```yaml
training:
  num_train_epochs: 3  # Train for 3 epochs
```

### Change LoRA Settings

More parameters (better but slower):
```yaml
peft:
  lora_r: 16      # Double the rank
  lora_alpha: 32  # Keep alpha = 2 * r
```

Fewer parameters (faster but less expressive):
```yaml
peft:
  lora_r: 4
  lora_alpha: 8
```

### Enable Weights & Biases Logging

In config file:
```yaml
training:
  report_to: "wandb"  # Enable W&B logging
```

Or via CLI:
```bash
python train/train.py --config configs/recipes/qwen3/8b/lora.yaml --wandb
```

## CLI Options

```bash
# Basic usage
python train/train.py --config <config_file>

# Override output directory
python train/train.py --config <config_file> --output_dir ./my_output

# Quick test (100 steps only)
python train/train.py --config <config_file> --max_steps 100

# Enable W&B logging
python train/train.py --config <config_file> --wandb
```

## Monitoring Jobs on KOA

```bash
# Check job status
koa-ml jobs

# View job output and results
# SSH to KOA and check:
# - $KOA_ML_DATA_ROOT/train/results/{job_id}/job.log for logs
# - $KOA_ML_DATA_ROOT/train/results/{job_id}/ for model checkpoints and outputs
```

## Output Structure

After training, your results will be in `$KOA_ML_DATA_ROOT/train/results/{job_id}/`
(`~/koa-ml/train/results/{job_id}` if you created the symlink):

```
train/results/{job_id}/
|-- job.log                   # SLURM job log
|-- adapter_config.json       # LoRA config
|-- adapter_model.safetensors # LoRA weights
|-- tokenizer_config.json     # Tokenizer
|-- training_args.bin         # Training metadata
`-- logs/                     # TensorBoard logs
```

## Using Your Fine-Tuned Model

### Load with transformers + PEFT

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype="bfloat16",
    device_map="auto"
)

# Load LoRA adapter (replace 123456 with your job ID)
model = PeftModel.from_pretrained(base_model, "./train/results/123456")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./train/results/123456")

# Generate
inputs = tokenizer("Your prompt here", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0]))
```

## Tips for Success

1. **Start small**: Always test with Qwen3 0.6B first to validate your pipeline
2. **Check memory**: Monitor GPU memory with `nvidia-smi` in job outputs
3. **Use QLoRA**: If you hit OOM errors, switch to QLoRA config
4. **Save often**: Set `save_steps` to checkpoint frequently
5. **Monitor logs**: Check `$KOA_ML_DATA_ROOT/train/results/{job_id}/job.log` for errors

## Troubleshooting

### Out of Memory (OOM)

Try these in order:
1. Switch to QLoRA config
2. Reduce `per_device_train_batch_size`
3. Increase `gradient_accumulation_steps`
4. Reduce `model_max_length`

### Slow Training

1. Ensure `flash_attention_2` is enabled
2. Increase batch size if memory allows
3. Use gradient checkpointing
4. Check if wandb logging is slowing things down

### Dataset Errors

Make sure your dataset has the right columns:
- Alpaca format: `instruction`, `input`, `output`
- Or a single `text` column with pre-formatted prompts

## Next Steps

- Try different datasets from HuggingFace
- Experiment with LoRA ranks (r=4, 8, 16, 32)
- Adjust learning rates and schedules
- Evaluate your models with the `eval/` tools
