# koa-ml

Complete toolkit for training and evaluating language models on the University of Hawaii's KOA HPC cluster. Submit jobs from your local machine, track experiments with Weights & Biases, and manage everything with a single CLI.

---

## Table of Contents

- [Features](#features)
- [Complete Setup Guide](#complete-setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step 1: SSH Key Setup](#step-1-ssh-key-setup)
  - [Step 2: SSH Connection Persistence](#step-2-ssh-connection-persistence-controlmaster)
  - [Step 3: Local Environment Setup](#step-3-local-environment-setup)
  - [Step 4: Remote KOA Environment Setup](#step-4-remote-koa-environment-setup)
  - [Step 5: Authentication Tokens (HuggingFace & W&B)](#step-5-authentication-tokens-huggingface--wb)
  - [Step 6: Verify Everything Works](#step-6-verify-everything-works)
- [Quick Start (After Setup)](#quick-start-after-setup)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Getting Help](#getting-help)

---

## Features

### Core Capabilities
- **Job Management**: Submit, monitor, and cancel jobs on KOA from your local machine
- **Fine-Tuning**: LoRA, QLoRA, and full training with pre-configured recipes
- **Evaluation**: Standard benchmarks (MMLU, GSM8K, HellaSwag, etc.) with multi-format output
- **Experiment Tracking**: Automatic Weights & Biases integration for all training runs
- **Version Control**: Every job automatically saves the exact code, config, and SLURM script used
- **HuggingFace Integration**: Seamless model and dataset loading with gated model support

### Developer Experience
- **One-Command Job Submission**: `koa-ml submit <script>` handles everything
- **SSH Connection Persistence**: Authenticate once, use for 60 minutes without re-entering credentials
- **Config Validation**: Validate YAML configs before submission
- **Results Comparison**: Compare model performance before/after training
- **Performance Metrics**: Automatic tracking of training throughput, GPU memory, and timing
- **Comprehensive Error Handling**: Detailed logs and graceful failure handling

---

## Complete Setup Guide

Follow these steps **in order** to get fully set up. This is a one-time setup process.

### Prerequisites

- **KOA Account**: You must have access to `koa.its.hawaii.edu`
- **Local Machine**: macOS or Linux with Python 3.9+
- **HuggingFace Account** (optional): For downloading models - https://huggingface.co/join
- **Weights & Biases Account** (optional): For experiment tracking - https://wandb.ai/signup

---

### Step 1: SSH Key Setup

Set up SSH key-based authentication to KOA (required for the toolkit to work).

#### Generate SSH Key (if you don't have one)

```bash
# Check if you already have an SSH key
ls -la ~/.ssh/id_*.pub

# If not, generate one (press Enter to accept defaults)
ssh-keygen -t ed25519 -C "your_email@hawaii.edu"
```

#### Copy SSH Key to KOA

```bash
# Copy your public key to KOA
ssh-copy-id your_username@koa.its.hawaii.edu

# You'll need to authenticate with your password + Duo 2FA
```

#### Verify SSH Key Works

```bash
# This should log you in WITHOUT asking for a password (only Duo)
ssh your_username@koa.its.hawaii.edu
```

---

### Step 2: SSH Connection Persistence (ControlMaster)

Configure SSH to keep connections alive for 60 minutes, so you don't have to re-authenticate with Duo for every `koa-ml` command.

#### Create SSH Config

Edit or create `~/.ssh/config`:

```bash
nano ~/.ssh/config
```

Add the following (replace `your_username` with your KOA username):

```
Host koa koa.its.hawaii.edu
    HostName koa.its.hawaii.edu
    User your_username
    IdentityFile ~/.ssh/id_ed25519

    # Connection persistence - reuse connection for 60 minutes
    ControlMaster auto
    ControlPath ~/.ssh/control-%r@%h:%p
    ControlPersist 60m

    # Keep connection alive
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

#### Test Connection Persistence

```bash
# First connection - you'll need to authenticate with Duo
ssh koa.its.hawaii.edu "hostname"

# Second command - should work immediately without Duo!
ssh koa.its.hawaii.edu "date"
```

If the second command works instantly, you're all set! The connection will persist for 60 minutes.

---

### Step 3: Local Environment Setup

Set up the koa-ml CLI on your local machine.

```bash
# Clone the repository
cd ~/Documents/GitHub  # or wherever you keep your code
git clone <your-repo-url> koa-ml
cd koa-ml

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install koa-ml CLI
pip install --upgrade pip setuptools wheel
pip install -e .
```

#### Configure KOA Connection

```bash
# Create koa-ml config directory
mkdir -p ~/.config/koa-ml

# Copy and edit the config template
cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
nano ~/.config/koa-ml/config.yaml
```

Update the config with your information:

```yaml
user: "your_koa_username"
host: koa.its.hawaii.edu
identity_file: ~/.ssh/id_ed25519
remote_workdir: /home/your_koa_username/koa-ml
remote_data_dir: /mnt/lustre/koa/scratch/your_koa_username/koa-ml
```

#### Verify CLI Works

```bash
# Test connection
koa-ml check

# You should see output like:
# Hostname: koa.its.hawaii.edu
# SLURM cluster status: ...
```

#### Initialize Remote Storage

```bash
# Create scratch directories and optional symlinks under ~/koa-ml
koa-ml storage setup --link
```

This command ensures your code checkout lives in `/home/.../koa-ml`, while large
artifacts (checkpoints, logs, evaluation outputs) are written to
`/mnt/lustre/koa/scratch/<user>/koa-ml`. The `--link` flag creates convenient
symlinks so `~/koa-ml/train/results` and `~/koa-ml/eval/results` point at the
scratch locations.

---

### Step 4: Remote KOA Environment Setup

Set up the Python environment on KOA where your jobs will actually run.

#### Sync Code to KOA

```bash
# From your local machine, sync the repository to KOA
koa-ml refresh

# This uploads all code to ~/koa-ml on KOA (excluding .venv, .git, etc.)
```

#### Create Python Environment on KOA Compute Node

**Important**: You must create the `.venv` on a **compute node** (not the login node) because it needs to compile packages for the GPU environment.

```bash
# SSH to KOA
ssh koa.its.hawaii.edu

# Request an interactive compute node session (4 CPUs, 16GB RAM, 1 hour)
srun -p gpu-sandbox --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 --pty /bin/bash

# Once you're on a compute node, navigate to the repository
cd ~/koa-ml

# Run the environment setup script (this takes ~15-20 minutes)
source scripts/setup_koa_env.sh
```

The script will:
- Load Python 3.11 and CUDA modules
- Create a fresh `.venv` in `~/koa-ml/.venv`
- Install PyTorch with CUDA 12.1 support
- Install all ML dependencies (transformers, peft, trl, bitsandbytes, flash-attn, etc.)
- Install koa-ml in editable mode

**Note**: If `flash-attn` fails to compile, the script will automatically retry without it (it's optional).

#### Exit Compute Node

```bash
# Once setup is complete, exit the compute node
exit

# You're back on the login node
exit

# Now you're back on your local machine
```

---

### Step 5: Authentication Tokens (HuggingFace & W&B)

Configure API tokens for downloading models and tracking experiments.

#### Get Your HuggingFace Token

1. Go to https://huggingface.co/settings/tokens
2. Click "Create new token"
3. **Token type**: Select **"Read"** (not Fine-grained or Write)
4. **Token name**: `koa-ml`
5. **Permissions**: Check only these two:
   - ✅ Read access to contents of all repos under your personal namespace
   - ✅ Read access to contents of all public gated repos you can access
6. Click "Create token" and copy it (starts with `hf_...`)

#### Get Your Weights & Biases API Key

1. Go to https://wandb.ai/authorize
2. Copy your API key

#### Create Local .env File

```bash
# From your local koa-ml directory
cd ~/Documents/GitHub/koa-ml

# Copy the template
cp .env.example .env

# Edit the .env file
nano .env
```

Add your tokens:

```bash
# HuggingFace token for downloading models
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# Weights & Biases for experiment tracking
WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxx
WANDB_PROJECT=koa-ml-experiments
WANDB_ENTITY=your-wandb-username
```

**Important**: The `.env` file is gitignored and will NOT be committed.

#### Sync Tokens to KOA

```bash
# Upload your .env file to KOA securely
koa-ml auth --sync

# Verify tokens are configured
koa-ml auth --check
```

You should see:

```
✓ .env file found at /home/your_username/koa-ml/.env

Checking for tokens:
  HF_TOKEN: ✓ Set
  WANDB_API_KEY: ✓ Set
  WANDB_PROJECT: ✓ Set
  WANDB_ENTITY: ✓ Set
```

---

### Step 6: Verify Everything Works

Let's test the complete setup with a quick training job!

```bash
# Submit a quick test training job (runs for ~30 minutes)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Check job status
koa-ml jobs
```

You should see your job in the queue:

```
JOBID|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)
123456|train-qwen3-0.6b|RUNNING|1:23|0:30:00|1|gpu-node-01
```

#### View Live Results in Weights & Biases

1. Go to https://wandb.ai
2. Navigate to your project (e.g., `koa-ml-experiments`)
3. You should see your training run with live metrics!

**Congratulations!** Your complete setup is done! 🎉

---

## Quick Start (After Setup)

Once you've completed the setup above, here's how to use koa-ml:

### Submit a Training Job

```bash
# Quick test - Qwen3 0.6B LoRA (30 min)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Production - Qwen3 8B LoRA (12 hours)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm

# Memory-efficient - Qwen3 14B QLoRA
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_14b_qlora.slurm
```

### Submit an Evaluation Job

```bash
# Quick evaluation
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Full evaluation
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm

# Vision-language evaluation (Qwen3-VL)
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv.slurm
```

### Monitor Jobs

```bash
# List your running/pending jobs
koa-ml jobs

# Cancel a job
koa-ml cancel <job_id>
```

### Inspect Results

```bash
# List recent training runs on scratch
koa-ml results list --kind train

# Download an evaluation result locally
koa-ml results pull 123456 --kind eval --dest ./artifacts/eval-123456
```

### Update Code on KOA

```bash
# After making local changes, sync to KOA
koa-ml refresh

# This preserves your .venv and .env on KOA
```

---

## Documentation

### Getting Started
- **[docs/AUTH_SETUP.md](docs/AUTH_SETUP.md)** - Authentication and token management
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute guide to training & evaluation
- **[docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)** - One-page Qwen3 reference

### Detailed Guides
- **[docs/QWEN3_GUIDE.md](docs/QWEN3_GUIDE.md)** - Complete Qwen3 training guide
- **[docs/ML_GUIDE.md](docs/ML_GUIDE.md)** - Comprehensive ML workflow
- **[train/README.md](train/README.md)** - Training deep dive
- **[eval/README.md](eval/README.md)** - Evaluation deep dive

### Development
- **[docs/TESTING.md](docs/TESTING.md)** - Testing instructions
- **[docs/DEPLOY_CHECKLIST.md](docs/DEPLOY_CHECKLIST.md)** - Deployment checklist

---

## CLI Reference

### Core Commands

```bash
# Job Management
koa-ml submit <script.slurm>              # Submit a SLURM job
koa-ml jobs                                # List your jobs
koa-ml cancel <job_id>                    # Cancel a job
koa-ml check                              # Health check

# Code Sync
koa-ml refresh                            # Sync local code to KOA

# Authentication
koa-ml auth --sync                        # Sync .env tokens to KOA
koa-ml auth --check                       # Verify token configuration
```

### Advanced Options

```bash
# Submit with custom resources
koa-ml submit job.slurm --partition gpu --gpus 2 --mem 64G --time 24:00:00

# Use custom config
koa-ml --config /path/to/config.yaml check

# Sync with custom excludes
koa-ml refresh --exclude "*.pyc" --exclude "__pycache__"
```

---

## Project Structure

```
koa-ml/
├── configs/                    # Configuration system
│   ├── datasets/              # Reusable dataset configs
│   └── recipes/               # Pre-configured training recipes
│       └── qwen3/             # Qwen3 model family
│           ├── 0.6b/lora.yaml # 0.6B LoRA (quick test)
│           ├── 4b/lora.yaml   # 4B LoRA
│           ├── 8b/lora.yaml   # 8B LoRA
│           ├── 8b/qlora.yaml  # 8B QLoRA (memory-efficient)
│           └── 14b/qlora.yaml # 14B QLoRA
│
├── docs/                       # Documentation
│   ├── AUTH_SETUP.md          # Authentication guide
│   ├── QUICKSTART.md          # Quick start guide
│   ├── QWEN3_GUIDE.md         # Qwen3 guide
│   └── ML_GUIDE.md            # ML workflow guide
│
├── eval/                       # Evaluation system
│   ├── configs/               # Evaluation configs
│   ├── results/               # Symlink to scratch results (via `koa-ml storage link`)
│   │   └── <job_id>/          # Each job gets its own directory
│   │       ├── predictions.csv
│   │       ├── summary.json
│   │       ├── job.log        # Combined stdout/stderr
│   │       ├── eval_*.slurm   # SLURM script used
│   │       ├── *.py           # Python script used
│   │       └── *.yaml         # Config used
│   ├── scripts/qwen3/         # SLURM job scripts
│   ├── evaluate.py            # Standard benchmark eval
│   └── qwen3_vl_eval.py      # Vision-language eval
│
├── train/                       # Training system
│   ├── results/               # Symlink to scratch results (via `koa-ml storage link`)
│   │   └── <job_id>/          # Each job gets its own directory
│   │       ├── checkpoint-*/  # Model checkpoints
│   │       ├── logs/          # Training logs
│   │       ├── job.log        # Combined stdout/stderr
│   │       ├── train_*.slurm  # SLURM script used
│   │       ├── train.py       # Python script used
│   │       └── *.yaml         # Config used
│   ├── scripts/qwen3/         # SLURM job scripts
│   │   ├── lora/              # LoRA scripts
│   │   └── qlora/             # QLoRA scripts
│   └── train.py               # Training script
│
├── scripts/                    # Utilities
│   ├── setup_koa_env.sh       # KOA environment setup
│   ├── compare_results.py     # Compare model performance
│   └── validate_config.py     # Validate configs
│
├── src/koa_ml/                 # CLI implementation
│   ├── __main__.py            # Main CLI entry point
│   ├── config.py              # Config management
│   ├── ssh.py                 # SSH/rsync operations
│   └── slurm.py               # SLURM operations
│
├── .env.example                # Environment template
├── .koa-config.example.yaml    # KOA config template
└── README.md                   # This file
```

---

## Key Features Explained

### Automatic Version Control

Every job automatically saves the exact code used:

```
/mnt/lustre/koa/scratch/<user>/koa-ml/eval/results/123456/
├── job.log                          # All output (stdout + stderr)
├── predictions.csv                  # Results
├── summary.json                     # Metrics
├── eval_qwen3_vl_m2sv.slurm        # SLURM script used
├── qwen3_vl_eval.py                # Python script used
└── qwen3_vl_m2sv.yaml              # Config used
```

You can always reproduce any run by looking at these files!

### Unified Logging

Jobs no longer split logs into `.err` and `.out`. Everything goes to a single `job.log` file for easier debugging.

### Weights & Biases Integration

All training jobs automatically log to W&B:
- Real-time loss curves
- Learning rate schedules
- GPU memory usage
- Training throughput
- Evaluation metrics

View at: https://wandb.ai/your-username/koa-ml-experiments

### SSH Connection Persistence

The ControlMaster configuration means:
- Authenticate once with Duo
- Run unlimited `koa-ml` commands for 60 minutes
- No re-authentication needed!

---

## KOA Quick Reference

### Cluster Info
- **Login node**: `koa.its.hawaii.edu` (SSH/MFA required)
- **Web portal**: https://koa.its.hawaii.edu
- **Storage**:
  - `/home/<user>` - 50 GB, backed up daily
  - `/mnt/lustre/koa/scratch/<user>` - Large storage, 90-day purge

### Common Partitions
- `gpu-sandbox` - 4 hours, for testing
- `gpu` - Up to 3 days, production
- `kill-shared` - Preemptible, unlimited time
- `shared` - 3 days, CPU only

### Useful Commands
```bash
# Check cluster status
sinfo

# Check your jobs
squeue -u $USER

# Interactive session
srun -p gpu-sandbox --gres=gpu:1 --mem=16G --time=1:00:00 --pty /bin/bash

# Cancel all your jobs
scancel -u $USER
```

---

## Troubleshooting

### Connection Issues

**Problem**: `koa-ml check` fails with connection error

**Solution**:
1. Verify SSH key: `ssh koa.its.hawaii.edu`
2. Check ControlMaster: `cat ~/.ssh/config`
3. Check config: `cat ~/.config/koa-ml/config.yaml`

### Authentication Issues

**Problem**: Jobs can't download models from HuggingFace

**Solution**:
```bash
# Re-sync tokens
koa-ml auth --sync

# Verify
koa-ml auth --check
```

### W&B Not Logging

**Problem**: Jobs run but don't appear in W&B

**Solution**:
1. Check token is set: `koa-ml auth --check`
2. Verify config has `report_to: "wandb"` in the training section
3. Check job log for W&B initialization messages

### Virtual Environment Issues

**Problem**: Jobs fail with "module not found" errors

**Solution**:
```bash
# SSH to KOA
ssh koa.its.hawaii.edu

# Request compute node
srun -p gpu-sandbox --gres=gpu:1 --mem=16G --time=1:00:00 --pty /bin/bash

# Rebuild environment
cd ~/koa-ml
rm -rf .venv
source scripts/setup_koa_env.sh
```

---

## Getting Help

- **Documentation**: See [docs/](docs/) directory
- **Quick Reference**: [docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)
- **KOA Support**: uh-hpc-help@lists.hawaii.edu (include job ID and error logs)
- **Issues**: Open an issue in this repository

---

## Acknowledgments

Inspired by:
- [oumi-ai/oumi](https://github.com/oumi-ai/oumi) - Training framework design
- [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) - Evaluation patterns
- [ThinkingMachinesLab/tinker-cookbook](https://github.com/ThinkingMachinesLab/tinker-cookbook) - Recipe structure

---

## License

Apache 2.0 - See [LICENSE](LICENSE)
