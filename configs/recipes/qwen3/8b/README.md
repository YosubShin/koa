# Qwen3 8B Recipes

Production-ready model with excellent performance-to-cost ratio.

## Available Recipes

### [lora.yaml](lora.yaml) - LoRA Fine-Tuning (Recommended)

**Resource Requirements:**
- GPU Memory: ~24GB
- Training Time: 8-12 hours (on Alpaca ~52K examples)
- Recommended GPU: A100 40GB, RTX A6000

**Expected Performance:**
- Base MMLU: ~68-72%
- After training: 70-75% (task-dependent)
- GSM8K: ~75-80%

**Configuration:**
```yaml
Model: Qwen/Qwen3-8B
LoRA Rank: 8
LoRA Alpha: 16
Batch Size: 2
Gradient Accumulation: 32
Learning Rate: 3e-4
```

**When to Use:**
- ✅ Production deployments
- ✅ Strong performance needed
- ✅ Reasonable inference costs
- ✅ General-purpose applications

---

### [qlora.yaml](qlora.yaml) - QLoRA Fine-Tuning (Memory-Efficient)

**Resource Requirements:**
- GPU Memory: ~12GB (50% reduction!)
- Training Time: 10-14 hours (slightly slower)
- Recommended GPU: RTX 3090, RTX 4090, A100

**Expected Performance:**
- Same as LoRA after training
- Slightly slower training speed (~15-20%)

**Configuration:**
```yaml
Model: Qwen/Qwen3-8B
LoRA Rank: 8
LoRA Alpha: 16
Quantization: 4-bit
Batch Size: 2
Gradient Accumulation: 32
Learning Rate: 3e-4
```

**When to Use:**
- ✅ Limited GPU memory
- ✅ Cost optimization
- ✅ Experimentation on consumer GPUs
- ❌ Time-critical training (use LoRA instead)

## Comparison: LoRA vs QLoRA

| Metric | LoRA | QLoRA |
|--------|------|-------|
| GPU Memory | ~24GB | ~12GB |
| Training Speed | 1.0x | 0.8-0.85x |
| Final Quality | Excellent | Excellent |
| Setup Complexity | Simple | Simple |

## Usage

### LoRA Training
```bash
# Submit to KOA
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm

# Or run directly
python train/train.py --config configs/recipes/qwen3/8b/lora.yaml
```

### QLoRA Training
```bash
# Submit to KOA
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm

# Or run directly
python train/train.py --config configs/recipes/qwen3/8b/qlora.yaml
```

## Tips

1. **Choose LoRA if**: You have adequate GPU memory (24GB+) and want faster training
2. **Choose QLoRA if**: GPU memory is limited or you want to train on consumer hardware
3. **Batch size**: Adjust based on your GPU - reduce if OOM occurs
4. **Learning rate**: 3e-4 is a good starting point, experiment with 1e-4 to 5e-4
5. **Evaluation**: Run evaluation every 500 steps to monitor progress

## Expected Results

After training on Alpaca-cleaned for 3 epochs:

- Instruction following: Significantly improved
- Task generalization: Good
- Reasoning: Strong
- Factual accuracy: Maintained from base model

## Troubleshooting

**Out of Memory (OOM)**:
- Switch from LoRA to QLoRA recipe
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to maintain effective batch size

**Slow Training**:
- Ensure `attn_implementation: "flash_attention_2"` is enabled
- Check GPU utilization with `nvidia-smi`
- Consider reducing `model_max_length` if processing very long sequences

**Poor Results**:
- Train for more epochs (3-5 recommended)
- Try different learning rates (1e-4, 3e-4, 5e-4)
- Check dataset quality and format
- Validate with `python scripts/validate_config.py`
