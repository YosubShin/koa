# Machine Learning on KOA: Fine-Tuning & Evaluation Guide

This guide covers the complete workflow for training and evaluating language models on the KOA HPC cluster. The system is inspired by [oumi](https://github.com/oumi-ai/oumi), [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and [Tinker Cookbook](https://github.com/ThinkingMachinesLab/tinker-cookbook).

## System Overview

```
koa-ml/
|- configs/               # Centralized configuration system
|  |- datasets/           # Dataset templates
|  `- recipes/            # Training recipes (Qwen3 family)
|- eval/                  # Evaluation utilities
|  |- configs/            # Benchmark definitions
|  |- results/            # Evaluation outputs (per job ID)
|  |- scripts/            # SLURM helpers for evaluation
|  |- evaluate.py         # Evaluation entry point
|  `- README.md           # Evaluation guide
|- scripts/               # Developer utilities (validation, comparison, setup)
|- train/                  # Training utilities
|  |- results/            # Training outputs (per job ID)
|  |- scripts/            # SLURM helpers for tuning
|  |- train.py            # Training entry point
|  `- README.md           # Training guide
|- src/koa_ml/            # CLI implementation
`- tests/                 # Automated test suite
```

## Philosophy

### Config-Driven Workflow

All training and evaluation is controlled by YAML config files. This means:
- **Reproducible**: Share a config, reproduce the exact experiment
- **Version-controlled**: Commit configs to git
- **Educational**: Configs document all hyperparameters clearly
- **Flexible**: Override any setting via CLI

### Three Fine-Tuning Approaches

1. **LoRA** (Low-Rank Adaptation)
   - Trains only 0.1% of parameters
   - Fast, memory-efficient
   - Easy to merge or swap adapters

2. **QLoRA** (Quantized LoRA)
   - LoRA + 4-bit quantization
   - Ultra low memory (50% less than LoRA)
   - Slight accuracy trade-off

3. **Full Fine-Tuning**
   - Trains all parameters
   - Maximum flexibility
   - Requires significant GPU memory

## Quick Start

### 1. Installation

```bash
# Clone and install
cd /path/to/koa-ml
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[ml]"
```

### 2. Test the Pipeline Locally

```bash
# Quick training test (10 minutes on GPU)
python train/train.py --config configs/recipes/qwen3/0.6b/lora.yaml --max_steps 10

# Quick evaluation test (5 minutes on GPU)
python eval/evaluate.py --config eval/configs/qwen3_quickstart.yaml --limit 10
```

### 3. Submit to KOA

```bash
# Fine-tune on KOA
koa-ml submit train/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm

# Monitor job
koa-ml jobs

# Evaluate on KOA
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm
```

## Complete Workflow Example

### Scenario: Fine-tune Qwen3 8B on custom data, then evaluate

#### Step 1: Prepare Your Dataset

Option A - Use HuggingFace dataset:
```yaml
# Edit configs/recipes/qwen3/8b/lora.yaml
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"
    split: "train"
```

Option B - Upload to HuggingFace Hub:
```bash
huggingface-cli login
python - <<'PY'
import json
from datasets import Dataset

with open("data.jsonl") as f:
    records = [json.loads(line) for line in f]

Dataset.from_list(records).push_to_hub("your-username/your-dataset")
PY
```

#### Step 2: Configure Training

```yaml
# configs/recipes/qwen3/8b/lora.yaml
model:
  model_name: "Qwen/Qwen3-8B"

data:
  train_dataset:
    dataset_name: "your-username/your-dataset"

training:
  num_train_epochs: 3
  learning_rate: 3.0e-04
  output_dir: "./train/results/local/my_qwen3_8b_lora"

peft:
  lora_r: 8
  lora_alpha: 16
```

#### Step 3: Submit Training Job

```bash
# Submit to KOA
koa-ml submit train/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm

# Monitor progress
koa-ml jobs

# SSH to KOA to check detailed logs
ssh koa.its.hawaii.edu
tail -f ~/koa-ml/train/results/<JOB_ID>/job.out
```

#### Step 4: Evaluate Base Model

While training runs, evaluate the base model:

```bash
# Edit eval/configs/benchmarks/mmlu.yaml to use base model
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm
```

#### Step 5: Evaluate Fine-Tuned Model

After training completes:

```bash
# Create custom evaluation config
cp eval/configs/benchmarks/mmlu.yaml eval/configs/my_model_eval.yaml

# Edit to point to your checkpoint
# model:
#   model_name: "./train/results/<JOB_ID>"

# Submit evaluation
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm
```

#### Step 6: Compare Results

```bash
# Download baseline and trained results (replace JOB IDs)
rsync -avz koa.its.hawaii.edu:~/koa-ml/eval/results/<BASE_JOB>/ ./local_results/base_job/
rsync -avz koa.its.hawaii.edu:~/koa-ml/eval/results/<FINETUNED_JOB>/ ./local_results/finetuned_job/

# Generate a comparison report
python scripts/compare_results.py \
  --baseline local_results/base_job \
  --checkpoint local_results/finetuned_job \
  --output local_results/comparison.md

cat local_results/comparison.md
```

## Understanding Configs

### Training Config Anatomy

```yaml
# Model: What model to load
model:
  model_name: "Qwen/Qwen3-8B"  # HF Hub ID or local path
  model_max_length: 8192                          # Context window
  dtype: "bfloat16"                               # Use bf16 for A100/A30
  attn_implementation: "flash_attention_2"        # 2-3x faster

# Data: What to train on
data:
  train_dataset:
    dataset_name: "yahma/alpaca-cleaned"          # HF dataset ID
    split: "train"                                 # Which split
  target_column: "text"                           # Text column name
  template: "qwen"                                # Prompt format

# Training: How to train
training:
  trainer_type: "sft"                             # Supervised training
  per_device_train_batch_size: 2                  # Batch size per GPU
  gradient_accumulation_steps: 32                 # Effective batch: 2*32=64
  learning_rate: 3.0e-04                          # LR (higher for LoRA)
  lr_scheduler_type: "cosine"                     # Cosine schedule
  warmup_ratio: 0.03                              # 3% warmup
  num_train_epochs: 3                             # Train for 3 epochs
  save_steps: 50                                  # Checkpoint every 50 steps
  output_dir: "./train/results/local/my_model"     # Save location

# PEFT: Optional LoRA config
peft:
  type: "lora"
  lora_r: 8                                       # Rank (4, 8, 16, 32)
  lora_alpha: 16                                  # Scaling (usually 2*r)
  lora_target_modules:                            # Which layers
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
```

### Evaluation Config Anatomy

```yaml
# Model: What to evaluate
model:
  model_name: "Qwen/Qwen3-8B"  # Base or trained
  model_max_length: 2048
  dtype: "bfloat16"

# Generation: How to generate
generation:
  per_device_batch_size: 4                        # Batch size
  temperature: 0.0                                # Greedy (deterministic)
  max_new_tokens: 512

# Tasks: What benchmarks
tasks:
  - backend: "lm_harness"
    task: "mmlu"                                  # Benchmark name
    num_fewshot: 5                                # Few-shot examples
    output_path: "./eval/results/local/mmlu"
```

## Available Models

### Pre-trained Models (HuggingFace)

**Small (for testing)**:
- `HuggingFaceTB/SmolLM2-135M-Instruct` - 135M params
- `microsoft/phi-2` - 2.7B params

**Medium**:
- `meta-llama/Llama-3.2-3B-Instruct` - 3B params
- `Qwen/Qwen2.5-7B-Instruct` - 7B params

**Large**:
- `Qwen/Qwen3-8B` - 8B params
- `meta-llama/Llama-3.1-70B-Instruct` - 70B params

### Your Fine-Tuned Models

After training, reference them by path:
```yaml
model:
  model_name: "./train/results/<JOB_ID>"
```

## Available Benchmarks

### Knowledge & Reasoning
- **MMLU**: 57 academic subjects (STEM, humanities, social sciences)
- **ARC**: Science exam questions (easy and challenge)
- **HellaSwag**: Commonsense reasoning
- **WinoGrande**: Pronoun resolution

### Math
- **GSM8K**: Grade school math word problems
- **MATH**: High school competition math

### Truthfulness
- **TruthfulQA**: Avoiding common misconceptions

### Code
- **HumanEval**: Python code generation
- **MBPP**: Python programming problems

### Comprehensive Suites
- **BBH** (Big-Bench Hard): 23 challenging tasks
- **GPQA**: Graduate-level science questions

See [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks) for the full list.

## Tips & Best Practices

### Memory Management

**If you hit OOM (Out of Memory)**:

1. Use QLoRA instead of LoRA
2. Reduce batch size: `per_device_train_batch_size: 1`
3. Increase gradient accumulation: `gradient_accumulation_steps: 64`
4. Reduce context: `model_max_length: 4096`
5. Enable gradient checkpointing (already default)

**Memory requirements** (approximate):

| Model | LoRA | QLoRA | Full FT |
|-------|------|-------|---------|
| SmolLM 135M | 2GB | 1GB | 4GB |
| Llama 3B | 8GB | 4GB | 16GB |
| Qwen3 8B | 24GB | 12GB | 40GB |
| Llama 70B | 80GB | 40GB | 280GB |

### Training Tips

1. **Always start with SmolLM**: Validate pipeline before expensive runs
2. **Use warmup**: Prevents early training instability
3. **Cosine schedule**: Generally works better than linear
4. **Save often**: KOA jobs can be killed/preempted
5. **Monitor logs**: Check for NaN losses or divergence

### LoRA Hyperparameters

**Rank (r)**:
- `r=4`: Very parameter-efficient, may underfit
- `r=8`: Good default
- `r=16`: More expressive, still efficient
- `r=32`: High capacity, approaching full training cost

**Target modules**:
- Minimum: `["q_proj", "v_proj"]` (attention only)
- Standard: `["q_proj", "v_proj", "k_proj", "o_proj"]` (all attention)
- Maximum: Add `["gate_proj", "up_proj", "down_proj"]` (attention + FFN)

**Learning rate**:
- LoRA: `2e-4` to `5e-4` (higher than full FT)
- Full FT: `5e-6` to `2e-5` (much lower)

### Evaluation Best Practices

1. **Use standard few-shot counts**: Enables comparison with papers
2. **Temperature 0.0**: For deterministic, reproducible results
3. **Evaluate before and after**: Show improvement from training
4. **Multiple benchmarks**: Don't rely on a single metric
5. **Check for overfitting**: Eval on held-out test sets

## Common Workflows

### 1. Quick Experimentation

```bash
# Fast iterations with small model
python train/train.py \
  --config configs/recipes/qwen3/0.6b/lora.yaml \
  --max_steps 100

python eval/evaluate.py \
  --model ./output/smollm_lora \
  --tasks mmlu_computer_science \
  --limit 50
```

### 2. Ablation Study (Compare LoRA Ranks)

```bash
# Train with different ranks
for rank in 4 8 16 32; do
  python train/train.py \
    --config configs/recipes/qwen3/8b/lora.yaml \
    --output_dir ./output/qwen3_8b_lora_r${rank}
  # (Edit config to set lora_r: ${rank})
done

# Evaluate all
for rank in 4 8 16 32; do
  python eval/evaluate.py \
    --model ./output/qwen3_8b_lora_r${rank} \
    --tasks mmlu
done
```

### 3. Domain Adaptation

```yaml
# Use domain-specific dataset
data:
  train_dataset:
    dataset_name: "medmcqa"  # Medical domain

# Evaluate on domain benchmarks
tasks:
  - task: "medqa"
  - task: "pubmedqa"
```

### 4. Multi-Stage Training

```bash
# Stage 1: General instruction following
python train/train.py --config configs/stage1_general.yaml

# Stage 2: Domain-specific training
python train/train.py \
  --config configs/stage2_domain.yaml \
  --base_model ./output/stage1_model
```

## Troubleshooting

### Training Issues

**Loss is NaN**:
- Reduce learning rate by 10x
- Check dataset for invalid tokens
- Enable gradient clipping

**Loss not decreasing**:
- Increase learning rate
- Check if warmup is too long
- Verify data formatting

**Training is slow**:
- Enable flash attention
- Increase batch size
- Check if data loading is bottleneck

### Evaluation Issues

**Unexpected low scores**:
- Check if model format matches task (instruct vs base)
- Verify few-shot prompts are correct
- Try temperature 0.0 for consistency

**OOM during evaluation**:
- Reduce batch size
- Reduce max_new_tokens
- Use smaller context window

### KOA-Specific Issues

**Job killed unexpectedly**:
- Check time limit (extend if needed)
- Check memory limit
- Look for OOM in logs

**Can't access model checkpoint**:
- Ensure output_dir is in your KOA home or scratch
- Check disk quota: `quota -s`

## Going Further

### Advanced Features to Explore

1. **Multi-GPU training**: Edit SLURM script to request multiple GPUs
2. **DeepSpeed**: Enable in config for large models
3. **Custom datasets**: Create your own on HuggingFace Hub
4. **Weights & Biases**: Track experiments with W&B
5. **Prompt templates**: Customize for your use case

### Recommended Reading

- [Oumi Documentation](https://oumi.ai/docs) - Training best practices
- [PEFT Documentation](https://huggingface.co/docs/peft) - LoRA details
- [TRL Documentation](https://huggingface.co/docs/trl) - SFT, RLHF, DPO
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) - Benchmark details

### Integration Ideas

- **MLflow**: Track experiments systematically
- **Ray Tune**: Hyperparameter optimization
- **Gradio**: Build interactive demos
- **vLLM**: Fast inference serving

## Getting Help

- **Documentation**: Check [train/README.md](train/README.md) and [eval/README.md](eval/README.md)
- **KOA Support**: uh-hpc-help@lists.hawaii.edu
- **HuggingFace Forum**: https://discuss.huggingface.co/
- **GitHub Issues**: Open an issue in this repo

Happy training!
