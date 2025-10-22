# M2SV Vision-Language Model Experiment

This guide walks through training and evaluating Qwen3-VL on the M2SV-SFT dataset to assess model convergence and memorization.

> **Storage reminder**: Run `koa-ml storage setup --link` so checkpoints and
> evaluation outputs live under `/mnt/lustre/koa/scratch/<user>/koa-ml` with
> symlinks in your `~/koa-ml` checkout. When referencing paths below, you can
> substitute `$KOA_ML_DATA_ROOT` for `/mnt/lustre/koa/scratch/$USER/koa-ml`.

## Experiment Overview

**Goal**: Fine-tune Qwen3-VL-4B on M2SV-SFT (street-view to map matching) and evaluate on both train and test splits to understand if the model is:
1. **Converging**: Loss decreasing, weights adjusting
2. **Memorizing**: High accuracy on training data
3. **Generalizing**: Reasonable accuracy on test data

## Step-by-Step Workflow

### Step 1: Sync Latest Code to KOA

```bash
# Make sure all new files are synced
koa-ml refresh
```

### Step 2: Submit Training Job

```bash
# Submit the training job (~4-6 hours)
koa-ml submit train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm

# Check status
koa-ml jobs
```

**What's happening:**
- Model: Qwen3-VL-4B-Instruct
- Method: LoRA (efficient, ~20-24GB GPU memory)
- Dataset: M2SV-SFT training split
- Duration: ~3000 steps (~4-6 hours)
- Output: Results saved to `$KOA_ML_DATA_ROOT/train/results/<job_id>/`
- Tracking: Real-time metrics in W&B

### Step 3: Monitor Training

Visit your Weights & Biases dashboard:
- URL: https://wandb.ai/your-username/koa-ml-experiments
- Look for run: `tune-qwen3vl-4b-m2sv`

**Key metrics to watch:**
- **Loss curve**: Should decrease steadily
- **Learning rate**: Following cosine schedule
- **GPU memory**: Should stay under 24GB
- **Throughput**: Steps per second

### Step 4: Update Evaluation Configs

Once training completes, note the job ID (e.g., `123456`).

Update both evaluation configs to point to your trained model:

**Edit `eval/configs/qwen3_vl_m2sv_sft_train.yaml`:**
```yaml
model:
  model_name: "/mnt/lustre/koa/scratch/your_username/koa-ml/train/results/123456"  # Your job ID
  dtype: "float16"
  device_map: "auto"
```

**Edit `eval/configs/qwen3_vl_m2sv_sft_test.yaml`:**
```yaml
model:
  model_name: "/mnt/lustre/koa/scratch/your_username/koa-ml/train/results/123456"  # Your job ID
  dtype: "float16"
  device_map: "auto"
```

### Step 5: Sync Updated Configs

```bash
koa-ml refresh
```

### Step 6: Evaluate on Training Split

```bash
# Evaluate on the training data (check memorization)
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_train.slurm
```

**Expected result**: High accuracy (70-90%+) if model is learning/memorizing

### Step 7: Evaluate on Test Split

```bash
# Evaluate on held-out test data (check generalization)
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_test.slurm
```

**Expected result**: Lower than train, but should still show improvement over baseline

### Step 8: Analyze Results

Check the results in each job directory:

```
eval/results/<train_job_id>/summary.json
eval/results/<test_job_id>/summary.json
```

Example summary.json:
```json
{
  "model": "/mnt/lustre/koa/scratch/username/koa-ml/train/results/123456",
  "dataset": "yosubshin/m2sv-sft",
  "split": "train",
  "total_samples": 1000,
  "correct": 850,
  "accuracy": 0.85,
  "timestamp": "2025-01-19T12:34:56"
}
```

## Interpreting Results

### Scenario 1: Good Convergence ✅
- **Train accuracy**: 80-95%
- **Test accuracy**: 60-75%
- **Interpretation**: Model is learning patterns, not just memorizing
- **Action**: Continue training or deploy model

### Scenario 2: Overfitting ⚠️
- **Train accuracy**: 90%+
- **Test accuracy**: <40%
- **Interpretation**: Model memorizing training data, poor generalization
- **Action**: Increase dropout, add regularization, or get more data

### Scenario 3: Underfitting ⚠️
- **Train accuracy**: <60%
- **Test accuracy**: <50%
- **Interpretation**: Model not learning effectively
- **Action**: Train longer, increase model capacity, or adjust learning rate

### Scenario 4: Not Learning ❌
- **Train accuracy**: ~25% (random guess for 4 options)
- **Test accuracy**: ~25%
- **Interpretation**: Model weights not adjusting properly
- **Action**: Check training logs, verify data format, check loss is decreasing

## Quick Commands Reference

```bash
# 1. Sync code
koa-ml refresh

# 2. Start training
koa-ml submit train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm

# 3. Monitor jobs
koa-ml jobs

# 4. After training, update configs with checkpoint path
# (Edit eval/configs/qwen3_vl_m2sv_sft_*.yaml files)

# 5. Sync updated configs
koa-ml refresh

# 6. Evaluate on train
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_train.slurm

# 7. Evaluate on test
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_test.slurm

# 8. Check results
# Look in eval/results/<job_id>/summary.json
```

## Files Created for This Experiment

### Training
- `train/qwen3_vl_train.py` - VLM training script
- `configs/recipes/qwen3_vl/4b_lora_m2sv.yaml` - Training config
- `train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm` - Training SLURM script

### Evaluation
- `eval/configs/qwen3_vl_m2sv_sft_train.yaml` - Eval config for train split
- `eval/configs/qwen3_vl_m2sv_sft_test.yaml` - Eval config for test split
- `eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_train.slurm` - Eval SLURM for train
- `eval/scripts/qwen3/eval_qwen3_vl_m2sv_sft_test.slurm` - Eval SLURM for test

## Troubleshooting

### Training Job Fails with OOM (Out of Memory)

**Solution**: Enable QLoRA in the config:

Edit `configs/recipes/qwen3_vl/4b_lora_m2sv.yaml`:
```yaml
model:
  load_in_4bit: true  # Enable QLoRA
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
```

### Evaluation Can't Find Model

**Problem**: Config points to wrong path

**Solution**: Check the training job ID and update config:
```yaml
model:
  model_name: "/mnt/lustre/koa/scratch/mburiek/koa-ml/train/results/YOUR_JOB_ID"
```

### Dataset Not Loading

**Problem**: HF_TOKEN not set

**Solution**:
```bash
koa-ml auth --check
# If tokens not set:
koa-ml auth --sync
```

## Next Steps

After completing this experiment:

1. **If results are good**: Try training on larger model (8B) or longer training
2. **If results are poor**: Adjust hyperparameters or try different approach
3. **Compare with baseline**: Run evaluation on the base model (before training) for comparison

---

**Questions or issues?** Check job logs in `$KOA_ML_DATA_ROOT/train/results/<job_id>/job.log` and `$KOA_ML_DATA_ROOT/eval/results/<job_id>/job.log`
