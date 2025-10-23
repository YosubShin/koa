# Storage Cleanup Guide for KOA-ML

This guide explains how to manage and clean up your storage on the KOA cluster.

## Quick Start

### 1. Check Your Current Storage Usage

```bash
ssh koa
du -sh /mnt/lustre/koa/scratch/$USER/koa-ml
```

Or more detailed:
```bash
du -h --max-depth=2 /mnt/lustre/koa/scratch/$USER/koa-ml
```

### 2. Run Cleanup Script (Dry Run First!)

Always do a dry run first to see what would be deleted:

```bash
ssh koa
bash ~/koa-ml/scripts/cleanup_storage.sh --dry-run
```

### 3. Actually Delete Files

Once you're happy with what will be deleted:

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh
```

## Common Cleanup Scenarios

### Keep Only the Latest 5 Training Jobs

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 5
```

### Delete Training Results Older Than 7 Days

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --older-than 7
```

### Clean Only Training Results (Keep Evaluations)

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --train-only
```

### Clean Only Evaluation Results (Keep Training)

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --eval-only
```

### Delete Everything (Nuclear Option)

**WARNING: This deletes ALL your results!**

```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --all
```

## Manual Cleanup Commands

If you prefer manual control, here are some useful commands:

### List All Training Jobs by Date

```bash
ls -lht /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/
```

### List All Evaluation Jobs by Date

```bash
ls -lht /mnt/lustre/koa/scratch/$USER/koa-ml/eval/results/
```

### Delete a Specific Job by ID

```bash
# Delete training job 8722396
rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/8722396

# Delete evaluation job 8722500
rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/eval/results/8722500
```

### Delete All Training Results

```bash
rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/*
```

### Delete All Evaluation Results

```bash
rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/eval/results/*
```

### Delete Everything

```bash
rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/*
```

## Understanding Storage Structure

Your KOA storage is organized like this:

```
/mnt/lustre/koa/scratch/$USER/koa-ml/
├── train/
│   └── results/
│       ├── 8722361/          # Job ID from training run
│       ├── 8722396/          # Each contains:
│       └── 8722427/          #   - Model checkpoint
│                             #   - Training logs
│                             #   - Config files
│
└── eval/
    └── results/
        ├── 8722500/          # Job ID from evaluation run
        └── 8722501/          # Each contains:
                              #   - Predictions CSV
                              #   - Summary JSON
                              #   - Eval logs
```

## What Takes Up Space

### Training Checkpoints (Large! ~9GB each)

Each training job saves:
- LoRA adapter weights (~45MB for LoRA, but full model ~9GB if saved)
- Optimizer states (~200MB+)
- Training logs and configs (~10MB)

**Storage Impact:** ~500MB to 9GB per training job

### Evaluation Results (Small! ~10MB each)

Each evaluation job saves:
- Predictions CSV (~5MB)
- Summary JSON (~1KB)
- Logs and configs (~5MB)

**Storage Impact:** ~10MB per evaluation job

## Best Practices

### 1. Regular Cleanup Schedule

Set up a reminder to clean up weekly:
```bash
# Keep only last 3 training runs, last 5 evaluations
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 3 --train-only
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 5 --eval-only
```

### 2. Before Starting New Experiments

Clean up old failed/test runs:
```bash
bash ~/koa-ml/scripts/cleanup_storage.sh --older-than 14
```

### 3. Archive Important Checkpoints

Before cleanup, copy important checkpoints to your local machine:

```bash
# From your local machine
scp -r koa:/mnt/lustre/koa/scratch/mburiek/koa-ml/train/results/8722396 \
    ~/koa-ml-checkpoints/qwen3-4b-m2sv-final/
```

Or use rsync for better performance:
```bash
rsync -avz --progress \
    koa:/mnt/lustre/koa/scratch/mburiek/koa-ml/train/results/8722396/ \
    ~/koa-ml-checkpoints/qwen3-4b-m2sv-final/
```

### 4. Upload to Hugging Face Hub

Save your best models to Hugging Face instead of keeping them on KOA:

```bash
# On KOA cluster
python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()

# Upload checkpoint to your HF account
api.upload_folder(
    folder_path="/mnt/lustre/koa/scratch/mburiek/koa-ml/train/results/8722396",
    repo_id="your-username/qwen3-4b-m2sv",
    repo_type="model",
)
EOF
```

## Troubleshooting

### Issue: "Permission denied" when deleting

**Solution:** Make sure you own the files:
```bash
ls -la /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/
```

### Issue: "Disk quota exceeded"

**Solution:** You've run out of space. Delete immediately:
```bash
# Emergency cleanup - delete all but latest job
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 1
```

### Issue: Script says "No such file or directory"

**Solution:** The cleanup script expects the standard directory structure. Check:
```bash
ls -la /mnt/lustre/koa/scratch/$USER/koa-ml/
```

## Storage Quotas

Check your quota usage on KOA:
```bash
ssh koa
quota -s
```

Or for detailed Lustre filesystem info:
```bash
lfs quota -u $USER /mnt/lustre
```

## Example Workflow

Here's a typical cleanup workflow:

```bash
# 1. SSH to KOA
ssh koa

# 2. Check current usage
du -sh ~/koa-ml

# 3. See what cleanup would do (dry run)
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 2 --dry-run

# 4. Actually perform cleanup
bash ~/koa-ml/scripts/cleanup_storage.sh --keep-latest 2

# 5. Verify new usage
du -sh ~/koa-ml
```

## Related Scripts

- `cleanup_storage.sh` - Main cleanup script
- `setup_koa_env.sh` - Initial environment setup
- Training scripts create results in: `train/results/`
- Evaluation scripts create results in: `eval/results/`
