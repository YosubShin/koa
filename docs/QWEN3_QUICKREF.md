# Qwen3 Quick Reference

One-page reference for working with Qwen3 models on KOA.

## Available Models

| Model | Command | Time | Memory |
|-------|---------|------|--------|
| Qwen3-0.6B | `koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm` | 30 min | 4GB |
| Qwen3-4B | `koa-ml submit train/scripts/qwen3/lora/train_qwen3_4b_lora.slurm` | 8 hrs | 12GB |
| Qwen3-8B (LoRA) | `koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm` | 12 hrs | 24GB |
| Qwen3-8B (QLoRA) | `koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm` | 12 hrs | 12GB |
| Qwen3-14B (QLoRA) | `koa-ml submit train/scripts/qwen3/qlora/train_qwen3_14b_qlora.slurm` | 16 hrs | 20GB |

## Quick Commands

```bash
# Quick test (0.6B model)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Production training (8B with QLoRA)
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm

# Evaluate base model
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Evaluate your trained model
python eval/evaluate.py \
  --model ./output/qwen3_8b_lora \
  --tasks mmlu,gsm8k,hellaswag

# Check job status
koa-ml jobs

# Cancel job
koa-ml cancel <job_id>
```

## Config Files

### Training
- [configs/recipes/qwen3/0.6b/lora.yaml](../configs/recipes/qwen3/0.6b/lora.yaml)
- [configs/recipes/qwen3/4b/lora.yaml](../configs/recipes/qwen3/4b/lora.yaml)
- [configs/recipes/qwen3/8b/lora.yaml](../configs/recipes/qwen3/8b/lora.yaml)
- [configs/recipes/qwen3/8b/qlora.yaml](../configs/recipes/qwen3/8b/qlora.yaml)
- [configs/recipes/qwen3/14b/qlora.yaml](../configs/recipes/qwen3/14b/qlora.yaml)

### Evaluation
- [eval/configs/qwen3_quickstart.yaml](eval/configs/qwen3_quickstart.yaml)
- [eval/configs/qwen3_8b_full_eval.yaml](eval/configs/qwen3_8b_full_eval.yaml)

## Common Customizations

### Use Your Own Dataset

Edit config file:
```yaml
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"
```

### Adjust LoRA Rank

```yaml
peft:
  lora_r: 32      # Higher = more parameters (default: 16)
  lora_alpha: 64  # Keep alpha = 2 * r
```

### Enable W&B Logging

```yaml
training:
  report_to: "wandb"
```

### Extend Context Length

```yaml
model:
  model_max_length: 65536  # Up to 131K supported
```

### Change Learning Rate

```yaml
training:
  learning_rate: 3.0e-04  # Default: 2.0e-04
```

## Memory Requirements

| Model | LoRA | QLoRA | GPU Recommendation |
|-------|------|-------|-------------------|
| 0.6B | 4GB | 2GB | Any GPU |
| 4B | 12GB | 6GB | RTX A4000+ |
| 8B | 24GB | 12GB | A30 (LoRA) / RTX A4000 (QLoRA) |
| 14B | 40GB | 20GB | A30 (QLoRA) |

## Qwen3 Features

- **Context**: 32K tokens (native), 131K (with YaRN)
- **Languages**: 100+ languages supported
- **Thinking Mode**: Automatic reasoning mode for complex tasks
- **License**: Apache 2.0 (commercial use allowed)

## Benchmarks (Qwen3-8B Base)

| Task | Score |
|------|-------|
| MMLU | ~72% |
| GSM8K | ~84% |
| HumanEval | ~67% |
| MATH | ~54% |

## Troubleshooting

**Out of Memory**: Switch to QLoRA
```bash
# Instead of LoRA
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm
```

**Slow Training**: Check flash attention is enabled
```yaml
attn_implementation: "flash_attention_2"
```

**Model Not Found**: Update transformers
```bash
pip install --upgrade transformers
```

## Full Documentation

See [QWEN3_GUIDE.md](QWEN3_GUIDE.md) for complete details.
