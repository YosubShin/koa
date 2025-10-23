# W&B Logging for Evaluation Runs

## Overview

Evaluation runs now automatically log to Weights & Biases (W&B) to track:
- **Real-time metrics** during evaluation (accuracy, progress)
- **Final results** (total accuracy, correct/incorrect counts)
- **Sample predictions** (first 50 examples with ground truth vs prediction)
- **Evaluation artifacts** (summary JSON files)

## Configuration

Add a `wandb` section to your eval config YAML:

```yaml
wandb:
  enabled: true  # Set to false to disable
  project: "koa-ml-eval"  # Optional: defaults to env WANDB_PROJECT
  entity: "your-username"  # Optional: defaults to env WANDB_ENTITY
  run_name: "eval-my-model"  # Optional: auto-generated if not provided
  tags:
    - "my-experiment"
    - "m2sv"
  notes: "Testing the fine-tuned model on training data"
```

## Environment Variables

W&B settings can also be controlled via environment variables (already set in your .env):

- `WANDB_PROJECT` - Default project name
- `WANDB_ENTITY` - Your W&B username/team
- `WANDB_API_KEY` - Authentication token

## What Gets Logged

### 1. Run Configuration
- Model path
- Dataset name and split
- Generation parameters (temperature, max_tokens)
- Training job ID (extracted from model path)
- Evaluation job ID (SLURM job ID)

### 2. Real-Time Metrics (every 10 examples)
- `accuracy` - Current accuracy percentage
- `correct` - Number of correct predictions
- `processed` - Number of examples processed
- `progress` - Fraction of dataset completed

### 3. Final Results
- `final_accuracy` - Overall accuracy
- `total_samples` - Total examples evaluated
- `correct_predictions` - Total correct
- `incorrect_predictions` - Total incorrect

### 4. Sample Predictions Table
A table with the first 50 predictions showing:
- Example ID
- Question (truncated)
- Ground truth answer
- Model prediction
- Correctness (True/False)
- Raw model response (truncated)

### 5. Artifacts
- `summary.json` file saved as W&B artifact for reproducibility

## Linking Eval to Training

The eval script automatically:
1. Extracts the training job ID from your model path
2. Logs it as `training_job_id` in the config
3. Uses it to generate the run name (e.g., `eval-train-8722545`)

This makes it easy to:
- Find all evals for a specific training run
- Compare eval results across different checkpoints
- Track the full pipeline: training → evaluation → results

## Viewing Results

After submitting an eval job, check:

1. **Console output** for the W&B run URL:
   ```
   W&B run initialized: https://wandb.ai/your-username/koa-ml-eval/runs/abc123
   ```

2. **W&B dashboard** at https://wandb.ai to see:
   - Live accuracy updates during evaluation
   - Final results and comparison charts
   - Sample predictions in table format
   - Full run configuration and logs

## Disabling W&B

To disable W&B logging:

**Option 1: Config file**
```yaml
wandb:
  enabled: false
```

**Option 2: Remove wandb section**
Just don't include the `wandb:` section in your config.

**Option 3: Uninstall wandb**
If wandb isn't installed, logging is automatically skipped.

## Example Workflow

1. **Train a model** (job 8722507)
   - W&B logs training metrics to `koa-ml-experiments` project

2. **Evaluate on train split** (job 8722545)
   - W&B logs eval metrics to `koa-ml-eval` project
   - Run name: `eval-train-8722545`
   - Tags: `["sft-qwen3-4b", "m2sv", "memorization-check"]`
   - Linked to training job via `training_job_id: 8722507`

3. **Evaluate on test split** (job 8722550)
   - W&B logs eval metrics to `koa-ml-eval` project
   - Run name: `eval-test-8722550`
   - Tags: `["sft-qwen3-4b", "m2sv", "generalization-check"]`
   - Linked to training job via `training_job_id: 8722507`

4. **Compare results in W&B**
   - Filter by tag: `sft-qwen3-4b`
   - Compare train vs test accuracy
   - View prediction samples
   - Track accuracy over time

## Best Practices

1. **Use descriptive tags** - Makes filtering easier
   - Model name: `sft-qwen3-4b`, `qwen3-8b-lora`
   - Dataset: `m2sv`, `alpaca`, `mmlu`
   - Purpose: `memorization-check`, `generalization-check`, `ablation`

2. **Add notes** - Explain what you're testing
   ```yaml
   notes: "Testing the effect of higher LoRA rank (r=32) on M2SV accuracy"
   ```

3. **Link to training runs** - Use consistent naming
   - Training: `train-qwen3-4b-8722507`
   - Eval: `eval-train-8722545` (references training job ID)

4. **Monitor live** - Check W&B URL during long evals
   - Catch issues early (0% accuracy, errors)
   - Estimate completion time
   - Compare with baseline in real-time

## Troubleshooting

**W&B not logging**
- Check `.env` has `WANDB_API_KEY`
- Verify `koa-ml auth --sync` was run
- Look for "W&B run initialized" in job logs

**Import errors**
- W&B is installed in the KOA .venv by default
- If missing: `pip install wandb` on compute node

**Rate limiting**
- W&B has rate limits on free tier
- Logs every 10 examples to avoid hitting limits
- Consider upgrading for high-volume evaluations
