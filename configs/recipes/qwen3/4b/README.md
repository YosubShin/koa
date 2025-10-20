# Qwen3 4B Recipes

Mid-size model offering a great balance between performance and resource requirements.

## Available Recipes

### [lora.yaml](lora.yaml) - LoRA Fine-Tuning

**Resource Requirements:**
- GPU Memory: ~16GB
- Training Time: 4-8 hours (on Alpaca ~52K examples)
- Recommended GPU: RTX A4000, RTX A5000, A100

**Expected Performance:**
- Base MMLU: ~60-65%
- After training: 63-68% (task-dependent)
- Sweet spot between 0.6B and 8B

**Configuration:**
```yaml
Model: Qwen/Qwen3-4B
LoRA Rank: 8
LoRA Alpha: 16
Batch Size: 2
Gradient Accumulation: 32
Learning Rate: 3e-4
```

**When to Use:**
- ✅ Good performance with moderate resources
- ✅ Faster iteration than 8B
- ✅ Better than 0.6B for production
- ✅ Cost-conscious deployments

## Usage

```bash
# Submit to KOA
koa-ml submit train/scripts/qwen3/lora/tune_qwen3_4b_lora.slurm

# Or run directly
python train/train.py --config configs/recipes/qwen3/4b/lora.yaml
```

## Tips

1. **Middle ground**: Perfect when 8B is too slow/expensive but 0.6B isn't enough
2. **Fast iteration**: Trains 2x faster than 8B, good for experimentation
3. **Production viable**: Suitable for production with performance requirements
4. **Inference**: Fast enough for real-time applications

## Typical Timeline

- Setup & data loading: 3-5 minutes
- Training (full Alpaca): 4-8 hours
- Evaluation: 10-15 minutes

**Total: ~5-9 hours for complete experiment**
