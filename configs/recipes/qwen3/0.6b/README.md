# Qwen3 0.6B Recipes

Small, fast model perfect for experimentation and quick iterations.

## Available Recipes

### [lora.yaml](lora.yaml) - LoRA Fine-Tuning (Recommended for Testing)

**Resource Requirements:**
- GPU Memory: ~8GB
- Training Time: 30-60 minutes (on Alpaca ~52K examples)
- Recommended GPU: Any modern GPU

**Expected Performance:**
- Base MMLU: ~45-50%
- After training: Improved task-specific performance

**Configuration:**
```yaml
Model: Qwen/Qwen3-0.6B
LoRA Rank: 8
LoRA Alpha: 16
Batch Size: 2
Gradient Accumulation: 32
Learning Rate: 3e-4
```

**When to Use:**
- ✅ Testing your pipeline
- ✅ Quick experiments
- ✅ Limited GPU memory
- ✅ Dataset exploration
- ❌ Production deployments (use 4B+ for production)

**Usage:**
```bash
# Test locally
python train/train.py --config configs/recipes/qwen3/0.6b/lora.yaml --max_steps 100

# Submit to KOA
koa-ml submit train/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm
```

## Tips

1. **Quick validation**: Use `--max_steps 10` for pipeline testing
2. **Small dataset**: Start with 1K examples for rapid iteration
3. **Debugging**: This model loads fast, perfect for debugging issues
4. **Cost-effective**: Ideal for students and researchers with limited resources

## Typical Timeline

- Setup & data loading: 2-5 minutes
- Training (full Alpaca): 30-60 minutes
- Evaluation: 5-10 minutes

**Total: ~1 hour for complete experiment**
