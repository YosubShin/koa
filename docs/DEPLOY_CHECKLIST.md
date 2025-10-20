# KOA Deployment Checklist

Quick reference for deploying koa-ml to KOA cluster.

## Quick Start Commands

### 1. Local Pre-Flight Check

```bash
# Validate all configurations
python scripts/validate_config.py --all

# Verify directory structure
ls -la configs/recipes/qwen3/*/
ls -la train/scripts/qwen3/
```

### 2. Sync to KOA

```bash
# Replace USERNAME with your KOA username
export KOA_USER="USERNAME"

# Sync repository (first time or full update)
rsync -avz --exclude='.git' --exclude='.venv' --exclude='*.pyc' \
  ./ ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/
```

### 3. Setup on KOA (First Time Only)

```bash
# SSH to KOA
ssh ${KOA_USER}@koa.cs.uoregon.edu

# Setup environment
cd ~/koa-ml
module load python/3.11
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[ml]"

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### 4. Submit Test Job

```bash
# On KOA
cd ~/koa-ml

# Submit quickstart (30 min)
sbatch train/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm

# Check status
squeue -u $USER

# Monitor output (replace JOBID)
tail -f train/results/JOBID/job.out
```

## Verification Checklist

Before submitting production jobs:

- [ ] `python scripts/validate_config.py --all` passes
- [ ] All configs in `configs/recipes/qwen3/` exist
- [ ] SLURM scripts reference `configs/recipes/` (not old paths)
- [ ] `.env` file configured with `HF_TOKEN`
- [ ] Virtual environment works on KOA
- [ ] Quickstart job completes successfully
- [ ] Results appear in `train/results/JOBID/`

## Available Jobs

### Fine-Tuning

```bash
# Quick test (30 min, 8GB GPU)
sbatch train/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm

# Production 4B (4-8 hrs, 16GB GPU)
sbatch train/scripts/qwen3/lora/tune_qwen3_4b_lora.slurm

# Production 8B LoRA (8-12 hrs, 24GB GPU)
sbatch train/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm

# Production 8B QLoRA (8-12 hrs, 12GB GPU - memory efficient)
sbatch train/scripts/qwen3/qlora/tune_qwen3_8b_qlora.slurm

# Large 14B (12-16 hrs, 16GB GPU)
sbatch train/scripts/qwen3/qlora/tune_qwen3_14b_qlora.slurm
```

### Evaluation

```bash
# Quick eval test
sbatch eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Full benchmark suite
sbatch eval/scripts/qwen3/eval_qwen3_8b_full.slurm

# Vision-language eval
sbatch eval/scripts/qwen3/eval_qwen3_vl_m2sv.slurm
```

## Common Tasks

### Update Repository on KOA

```bash
# From local machine
rsync -avz --exclude='.git' --exclude='.venv' \
  --delete \
  ./ ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/
```

**Warning:** `--delete` removes files on KOA that aren't local. Remove this flag to preserve KOA results.

### Download Results

```bash
# From local machine
# Download all results
rsync -avz ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/train/results/ ./train/results/
rsync -avz ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/eval/results/ ./eval/results/

# Download specific job
rsync -avz ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/train/results/JOBID/ ./train/results/JOBID/
```

### Compare Results

```bash
# After downloading results
python scripts/compare_results.py \
  --baseline eval/results/BASELINE_JOBID \
  --checkpoint eval/results/FINETUNED_JOBID \
  --output comparison.md

cat comparison.md
```

## Troubleshooting

### Job Fails Immediately

```bash
# Check logs
cat train/results/JOBID/error.log
cat train/results/JOBID/job.out

# Common fixes:
# 1. Check HF_TOKEN is set
# 2. Verify venv path: echo $KOA_ML_VENV
# 3. Check config exists: ls configs/recipes/qwen3/8b/lora.yaml
```

### Out of Memory

```bash
# Switch to QLoRA
sbatch train/scripts/qwen3/qlora/tune_qwen3_8b_qlora.slurm

# Or edit config:
nano configs/recipes/qwen3/8b/lora.yaml
# Reduce: per_device_train_batch_size
# Increase: gradient_accumulation_steps
```

### Config Not Found

```bash
# Verify SLURM script uses new paths
grep "config" train/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm
# Should show: configs/recipes/qwen3/8b/lora.yaml
# NOT: train/configs/models/qwen3_8b_lora.yaml

# Re-sync if needed
rsync -avz ./ ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/
```

## Monitoring

### Check Job Status

```bash
# All your jobs
squeue -u $USER

# Specific job
scontrol show job JOBID

# Recent completed jobs
sacct -u $USER --format=JobID,JobName,State,Elapsed
```

### Monitor Running Job

```bash
# Watch output
tail -f train/results/JOBID/job.out

# Check GPU usage
grep "GPU memory" train/results/JOBID/job.out
```

### Cancel Job

```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER
```

## Success Indicators

Your job succeeded if:

1.  SLURM exit code 0
2.  `adapter_model.safetensors` exists in results
3.  No `error.log` or it's empty
4.  "Training complete!" in job.out
5.  Training metrics printed (time, throughput, GPU mem)

Check with:

```bash
ls -lh train/results/JOBID/adapter_model.safetensors
grep "Training complete" train/results/JOBID/job.out
grep "Peak GPU memory" train/results/JOBID/job.out
```

## Documentation

- **Full Testing Guide**: [TESTING.md](TESTING.md)
- **Enhancements Summary**: [ENHANCEMENTS.md](ENHANCEMENTS.md)
- **Recipe Guide**: [configs/recipes/README.md](configs/recipes/README.md)
- **Tune Guide**: [train/README.md](train/README.md)
- **Eval Guide**: [eval/README.md](eval/README.md)
- **Main README**: [README.md](README.md)

## Workflow Summary

```
Local -> Validate -> Sync to KOA -> Setup (first time) -> Submit Job -> Monitor -> Download Results -> Compare
```

1. **Validate**: `python scripts/validate_config.py --all`
2. **Sync**: `rsync -avz ./ ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/`
3. **Submit**: `sbatch train/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm`
4. **Monitor**: `tail -f train/results/JOBID/job.out`
5. **Download**: `rsync -avz ${KOA_USER}@koa.cs.uoregon.edu:~/koa-ml/train/results/JOBID/ ./train/results/JOBID/`
6. **Compare**: `python scripts/compare_results.py ...`

---

**Ready to deploy?** Start with the quickstart job to verify everything works!
