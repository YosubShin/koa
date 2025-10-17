# Qwen3 14B Recipes

Largest Qwen3 model - maximum performance for demanding applications.

## Available Recipes

### [qlora.yaml](qlora.yaml) - QLoRA Fine-Tuning (Only Option)

**Resource Requirements:**
- GPU Memory: ~16GB (with QLoRA)
- Training Time: 12-16 hours (on Alpaca ~52K examples)
- Recommended GPU: A100 40GB/80GB

**Expected Performance:**
- Base MMLU: ~75-78%
- After fine-tuning: 76-80% (task-dependent)
- Best-in-class for Qwen3 family

**Configuration:**
```yaml
Model: Qwen/Qwen3-14B
LoRA Rank: 8
LoRA Alpha: 16
Quantization: 4-bit (required)
Batch Size: 2
Gradient Accumulation: 32
Learning Rate: 3e-4
```

**When to Use:**
- ✅ Maximum performance needed
- ✅ Complex reasoning tasks
- ✅ Research applications
- ✅ When 8B isn't sufficient
- ❌ Time-sensitive training
- ❌ Limited computational budget

## Why QLoRA Only?

The 14B model is too large for standard LoRA on typical GPUs:
- Full LoRA requires 40GB+ GPU memory
- QLoRA enables training on more accessible hardware
- Minimal quality difference vs full LoRA

## Usage

```bash
# Submit to KOA
koa-ml submit tune/scripts/qwen3/qlora/tune_qwen3_14b_qlora.slurm

# Or run directly
python tune/train.py --config configs/recipes/qwen3/14b/qlora.yaml
```

## Tips

1. **Patience required**: This is a slow but powerful model
2. **Evaluation strategy**: Evaluate less frequently (every 1000 steps) to save time
3. **Dataset size**: Full benefits visible with larger datasets (50K+ examples)
4. **Production considerations**: Inference is slower - ensure latency requirements are met

## Expected Results

After training on Alpaca-cleaned:

- Complex reasoning: Excellent
- Instruction following: Near-perfect
- Long-form generation: Superior
- Factual accuracy: Best in Qwen3 family

## Comparison with 8B

| Metric | 8B | 14B |
|--------|-----|-----|
| Training Time | 8-12h | 12-16h |
| GPU Memory (QLoRA) | 12GB | 16GB |
| MMLU Performance | 68-72% | 75-78% |
| Inference Speed | Fast | Moderate |
| Use Case | General | Specialized/High-perf |

## When to Choose 14B Over 8B

Choose 14B if:
- Complex reasoning is critical
- You have adequate time budget
- Performance > speed/cost
- Working on research problems

Stick with 8B if:
- Good performance is sufficient
- Faster iteration needed
- Cost/latency sensitive
- General-purpose application

## Troubleshooting

**OOM with 16GB GPU**:
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps` to 64
- Reduce `model_max_length` to 4096 or 2048

**Very slow training**:
- Expected - 14B is compute-intensive
- Check GPU utilization is high (>80%)
- Consider using gradient checkpointing (may reduce memory at cost of speed)

**Diminishing returns**:
- 14B may not significantly outperform 8B on simple tasks
- Reserve for tasks where the extra capacity matters
