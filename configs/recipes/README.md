# Training Recipes

Pre-configured, tested recipes for fine-tuning models on KOA HPC.

## Quick Start

**Choose your model size based on your needs:**

| Model | Best For | GPU Memory | Training Time | Quality |
|-------|----------|------------|---------------|---------|
| [0.6B](qwen3/0.6b/) | Testing, experimentation | 8GB | 30-60 min | Good |
| [4B](qwen3/4b/) | Balanced performance | 16GB | 4-8 hours | Very Good |
| [8B](qwen3/8b/) | Production (recommended) | 24GB (12GB QLoRA) | 8-12 hours | Excellent |
| [14B](qwen3/14b/) | Maximum performance | 16GB (QLoRA only) | 12-16 hours | Best |

## Directory Structure

```
configs/recipes/
|-- qwen3/                 # Qwen3 model family
|   |-- 0.6b/
|   |   |-- lora.yaml      # LoRA recipe
|   |   `-- README.md      # Performance specs and tips
|   |-- 4b/
|   |   |-- lora.yaml
|   |   `-- README.md
|   |-- 8b/
|   |   |-- lora.yaml      # Standard LoRA
|   |   |-- qlora.yaml     # Memory-efficient QLoRA
|   |   `-- README.md
|   `-- 14b/
|       |-- qlora.yaml     # QLoRA (only option for 14B)
|       `-- README.md
```

## Usage

### Method 1: Use Recipe Directly

```bash
# Submit to KOA with SLURM script
koa-ml submit tune/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm

# Or run directly
python tune/train.py --config configs/recipes/qwen3/8b/lora.yaml
```

### Method 2: Copy and Customize

```bash
# Copy a recipe
cp configs/recipes/qwen3/8b/lora.yaml my_custom_recipe.yaml

# Edit your custom recipe
# - Change dataset
# - Adjust hyperparameters
# - Modify output directory

# Run your custom recipe
python tune/train.py --config my_custom_recipe.yaml
```

### Method 3: Override on Command Line

```bash
python tune/train.py \
  --config configs/recipes/qwen3/8b/lora.yaml \
  --output_dir ./my_experiment \
  --max_steps 1000
```

## Customization Guide

### Change the Dataset

Edit the `data` section in any recipe:

```yaml
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"
    split: "train"
  target_column: "text"
  template: "alpaca"
```

Or reference a dataset config:

```yaml
data: !include ../../datasets/dolly.yaml
```

See [configs/datasets/](../datasets/) for pre-configured datasets.

### Adjust Training Duration

**Train for specific number of steps:**
```yaml
training:
  max_steps: 1000  # Stop after 1000 steps
```

**Train for specific number of epochs:**
```yaml
training:
  num_train_epochs: 3  # Train for 3 full passes
```

### Change Hyperparameters

Common adjustments:

```yaml
training:
  learning_rate: 3.0e-04           # Try 1e-4 to 5e-4
  per_device_train_batch_size: 2   # Reduce if OOM
  gradient_accumulation_steps: 32  # Increase to maintain effective batch size
  warmup_ratio: 0.03               # 3% warmup typically good
```

### Modify LoRA Settings

```yaml
peft:
  lora_r: 8          # Rank: higher = more parameters (4, 8, 16, 32)
  lora_alpha: 16     # Typically 2 * lora_r
  lora_dropout: 0.1  # Regularization
```

## Choosing a Recipe

Use the following guidelines when selecting a recipe:

- Just testing the pipeline? Start with the 0.6B LoRA recipe for the quickest validation run.
- Working with less than 16 GB of GPU memory? Use the 8B QLoRA recipe (runs comfortably in about 12 GB).
- Targeting production-quality adapters with access to 24 GB GPUs? Choose the 8B LoRA recipe.
- Seeking maximum quality on hardware with at least 16 GB GPUs? Train the 14B QLoRA recipe.
- Looking for a balance between cost and accuracy? The 4B LoRA recipe is a solid middle ground.

## Best Practices

### Before Training

1. **Validate your config:**
   ```bash
   python scripts/validate_config.py configs/recipes/qwen3/8b/lora.yaml
   ```

2. **Test with small dataset:**
   ```bash
   python tune/train.py \
     --config configs/recipes/qwen3/8b/lora.yaml \
     --max_steps 10
   ```

3. **Check GPU availability:**
   ```bash
   nvidia-smi
   ```

### During Training

1. **Monitor progress:**
   - Check `tune/results/{job_id}/job.out` for logs
   - Watch GPU utilization
   - Track loss curve

2. **Validate periodically:**
   - Run evaluation every 500-1000 steps
   - Check for overfitting

3. **Save checkpoints:**
   ```yaml
   training:
     save_steps: 500
     save_total_limit: 3  # Keep only last 3 checkpoints
   ```

### After Training

1. **Evaluate on benchmarks:**
   ```bash
   python eval/evaluate.py \
     --model tune/results/123456 \
     --tasks mmlu,gsm8k,hellaswag
   ```

2. **Compare with baseline:**
   ```bash
   python scripts/compare_results.py \
     --baseline eval/results/baseline \
     --checkpoint eval/results/finetuned
   ```

3. **Document your results:**
   - Save training config
   - Record final metrics
   - Note any issues/observations

## Troubleshooting

### Out of Memory (OOM)

1. Switch from LoRA to QLoRA recipe
2. Reduce `per_device_train_batch_size`
3. Increase `gradient_accumulation_steps`
4. Reduce `model_max_length`

### Slow Training

1. Ensure `flash_attention_2` is enabled
2. Check GPU utilization (should be >80%)
3. Consider larger batch size if memory allows
4. Disable wandb if enabled

### Poor Results

1. Train for more epochs (3-5)
2. Try different learning rates
3. Check dataset quality
4. Increase LoRA rank (8 -> 16 -> 32)
5. Use larger model size

### Config Errors

1. Run validation:
   ```bash
   python scripts/validate_config.py your_config.yaml
   ```

2. Check YAML syntax
3. Verify dataset exists
4. Ensure required fields are present

## Additional Resources

- [Dataset Configurations](../datasets/README.md)
- [Validation Guide](../../scripts/validate_config.py)
- [Results Comparison](../../scripts/compare_results.py)
- [Main Documentation](../../README.md)

## Getting Help

1. Check model-specific README in each directory
2. Validate your config with `scripts/validate_config.py`
3. Review error logs in `tune/results/{job_id}/error.log`
4. Open an issue with your config and error details

## Tips for Success

- **Start small:** Always test with 0.6B or `--max_steps 10` first
- **Validate configs:** Use the validation script before submitting jobs
- **Monitor resources:** Check GPU memory and utilization
- **Save checkpoints:** Don't lose progress from interrupted jobs
- **Compare results:** Use the comparison script to track improvements
- **Document experiments:** Keep notes on what works and what doesn't

Happy fine-tuning!
