# Authentication Setup Guide

This guide explains how to configure HuggingFace and Weights & Biases (W&B) authentication for your koa-ml jobs.

## Overview

To access gated models on HuggingFace and track experiments with W&B, you need to configure API tokens. The `koa-ml` toolkit provides a secure way to manage these credentials.

## Prerequisites

1. **HuggingFace Token** (optional, but recommended for gated models):
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with "Read" permissions
   - Copy the token (starts with `hf_...`)

2. **Weights & Biases API Key** (optional, for experiment tracking):
   - Go to https://wandb.ai/authorize
   - Copy your API key

## Setup Steps

### Step 1: Create Local .env File

In your local `koa-ml` directory, create a `.env` file from the template:

```bash
cp .env.example .env
```

### Step 2: Add Your Tokens

Edit the `.env` file and add your tokens:

```bash
# Required for accessing gated HuggingFace models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Required for W&B experiment tracking
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx

# Optional: Configure W&B project/entity
WANDB_PROJECT=koa-ml-experiments
WANDB_ENTITY=your-username
```

**Important:** The `.env` file is gitignored and will NOT be committed to version control.

### Step 3: Sync Tokens to KOA

Use the `koa-ml auth` command to securely sync your tokens to the remote KOA server:

```bash
koa-ml auth --sync
```

This command will:
- Copy your local `.env` file to `~/koa-ml/.env` on KOA
- Set proper file permissions (600 - readable only by you)
- The tokens will be automatically loaded by all SLURM jobs

### Step 4: Verify Setup

Check that your tokens are configured correctly:

```bash
koa-ml auth --check
```

Example output:
```
Checking authentication configuration on KOA...
✓ .env file found at /home/mburiek/koa-ml/.env

Checking for tokens:
  HF_TOKEN: ✓ Set
  WANDB_API_KEY: ✓ Set
  WANDB_PROJECT: ✓ Set
  WANDB_ENTITY: ✓ Set
```

## Using W&B in Training Jobs

### Enable W&B Logging

To enable W&B logging for a training job, you have two options:

**Option 1: Use the --wandb flag**
```bash
koa-ml submit train/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm
# Then modify your Python call to include --wandb
```

**Option 2: Update your config file**

Edit your recipe config (e.g., `configs/recipes/qwen3/8b/lora.yaml`):

```yaml
training:
  output_dir: "./output/qwen3_8b_lora"
  report_to: "wandb"  # Change from "none" to "wandb"
  # ... other training params
```

### View Your Experiments

Once a job is running with W&B enabled:
1. Go to https://wandb.ai
2. Navigate to your project (e.g., `koa-ml-experiments`)
3. View real-time metrics, charts, and logs

## Security Best Practices

1. **Never commit `.env` files** - They contain sensitive credentials
2. **Use read-only tokens** - Your HF token only needs read permissions
3. **Rotate tokens periodically** - Update tokens every few months
4. **Check file permissions** - The `.env` file on KOA should be mode 600:
   ```bash
   ssh koa.its.hawaii.edu "ls -l ~/koa-ml/.env"
   # Should show: -rw------- (600)
   ```

## Troubleshooting

### Token Not Loading in Jobs

If your jobs can't access HuggingFace or W&B:

1. Verify the .env file exists on KOA:
   ```bash
   koa-ml auth --check
   ```

2. Re-sync your tokens:
   ```bash
   koa-ml auth --sync
   ```

3. Check the job log for the "Loading environment" message:
   ```
   Loading environment from .env file...
   ```

### HuggingFace Authentication Errors

If you see errors like "Repository not found" or "Access denied":

1. Verify your token has the correct permissions
2. For gated models (like Llama), accept the license on HuggingFace first
3. Ensure your token is correctly set:
   ```bash
   koa-ml auth --check
   ```

### W&B Login Errors

If W&B can't authenticate:

1. Verify your API key is correct (check https://wandb.ai/authorize)
2. Re-sync your tokens: `koa-ml auth --sync`
3. Ensure `WANDB_API_KEY` is set in your `.env` file

## Advanced: Manual Setup

If you prefer not to use `koa-ml auth`, you can manually create the `.env` file on KOA:

```bash
# SSH to KOA
ssh koa.its.hawaii.edu

# Navigate to your koa-ml directory
cd ~/koa-ml

# Create .env file
cat > .env << 'EOF'
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=koa-ml-experiments
WANDB_ENTITY=your-username
EOF

# Set proper permissions
chmod 600 .env
```

## Reference

### Available Environment Variables

All variables from `.env.example`:

- **`HF_TOKEN`** - HuggingFace API token
- **`WANDB_API_KEY`** - Weights & Biases API key
- **`WANDB_PROJECT`** - W&B project name
- **`WANDB_ENTITY`** - W&B username/team name
- **`KOA_ML_WORKDIR`** - Remote working directory (default: `$HOME/koa-ml`)
- **`KOA_ML_VENV`** - Virtual environment path (default: `$HOME/koa-ml/.venv`)
- **`HF_HUB_DISABLE_TELEMETRY`** - Disable HF telemetry (recommended: `1`)
- **`PYTORCH_CUDA_ALLOC_CONF`** - PyTorch CUDA memory settings

### koa-ml auth Command Reference

```bash
# Sync local .env to KOA
koa-ml auth --sync

# Check auth status on KOA
koa-ml auth --check

# Use a different .env file
koa-ml auth --sync --env-file /path/to/custom.env
```

## Next Steps

- [Quickstart Guide](QUICKSTART.md) - Getting started with koa-ml
- [Qwen3 Guide](QWEN3_GUIDE.md) - Training Qwen3 models
- [ML Guide](ML_GUIDE.md) - Comprehensive ML training guide
