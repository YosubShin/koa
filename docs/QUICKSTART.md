# KOA-ML Quickstart

This guide walks through the minimum set of steps needed to install the `koa-ml` CLI, prepare the KOA cluster environment, launch supervised fine-tuning jobs, evaluate checkpoints, and keep storage tidy. It distills the project documentation into a single reference with direct, reproducible commands.

## Overview
- Submit and monitor SLURM jobs on KOA from your local workstation
- Fine-tune Qwen3-VL-4B with the M2SV-SFT dataset using battle-tested scripts
- Evaluate checkpoints with automated logging to Weights & Biases
- Manage remote storage so scratch space never becomes the bottleneck

## Prerequisites
- KOA account with Duo MFA access (`koa.its.hawaii.edu`)
- macOS or Linux workstation with Python 3.9+, Git, and SSH
- SSH key-based access to KOA (`ssh koa.its.hawaii.edu` without a password prompt)
- Optional but recommended: Hugging Face account (for gated models) and W&B account for experiment tracking

## 1. Install Locally
```bash
# Clone and create a virtual environment
git clone <repo-url> koa-ml
cd koa-ml
python3 -m venv .venv
source .venv/bin/activate

# Install CLI and ML extras
pip install --upgrade pip setuptools wheel
pip install -e ".[ml]"
```

If you use multiple projects, consider `direnv` or `uv` to manage virtual environments automatically.

## 2. Configure the CLI
```bash
mkdir -p ~/.config/koa-ml
cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
```

Edit `~/.config/koa-ml/config.yaml` with your details:

```yaml
user: "your_koa_username"
host: "koa.its.hawaii.edu"
identity_file: "~/.ssh/id_ed25519"
remote_workdir: "/home/your_koa_username/koa-ml"
remote_data_dir: "/mnt/lustre/koa/scratch/your_koa_username/koa-ml"
```

Then create the remote directory layout and validate connectivity:

```bash
koa-ml storage setup --link   # Creates ~/koa-ml + scratch symlinks
koa-ml check                  # Verifies SSH, rsync, and SLURM access
```

## 3. Prepare the KOA Environment
1. Sync your current code: `koa-ml refresh`
2. SSH to KOA and start an interactive GPU session (adjust partition/time as needed):
   ```bash
   ssh koa.its.hawaii.edu
   srun -p gpu-sandbox --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=02:00:00 --pty /bin/bash
   ```
3. Inside the session, install the project environment:
   ```bash
   cd ~/koa-ml
   source scripts/setup_koa_env.sh
   ```
   The script loads KOA modules, recreates `.venv`, installs PyTorch with CUDA 12.1 wheels, and installs `koa-ml` with the `[ml]` extras. Re-run it whenever PyTorch or Transformers needs to be refreshed.
4. Exit the session once the environment is built.

## 4. Manage Authentication Tokens
```bash
cp .env.example .env
# Edit .env and add:
#   HF_TOKEN=<your huggingface token>
#   WANDB_API_KEY=<your wandb api key>

koa-ml auth --sync   # Uploads .env to KOA
koa-ml auth --check  # Confirms tokens are available on the cluster
```

Job scripts automatically source `.env` before launching Python so credentials are available to Hugging Face and W&B without hard-coding secrets.

## 5. Everyday CLI Reference
| Command | Purpose |
|---------|---------|
| `koa-ml refresh` | Rsync the current directory to KOA (respects default excludes) |
| `koa-ml submit <script.slurm>` | Submit a SLURM batch job |
| `koa-ml jobs` | Show running and pending jobs for your KOA account |
| `koa-ml cancel <job_id>` | Cancel a specific job |
| `koa-ml storage link` | Recreate symlinks from `~/koa-ml` to scratch results |
| `koa-ml results list --kind train` | List recent training jobs and their IDs |
| `koa-ml results pull <id> --kind eval` | Download an eval result directory locally |
| `koa-ml auth --sync / --check` | Manage Hugging Face and W&B tokens |

Use `--config` to point at alternative configuration files if you maintain multiple KOA profiles.

## 6. Train Qwen3-VL-4B on M2SV

### Choose Your Assets
| Component | Path | Notes |
|-----------|------|-------|
| SLURM script | `train/scripts/sft_qwen3_4b/sft_qwen3_4b.slurm` | 1× GPU, 48h walltime, 64G RAM |
| Python entry point | `train/scripts/sft_qwen3_4b/sft_qwen3_4b.py` | LoRA fine-tuning script with image/text processors preconfigured |
| YAML recipe | `configs/recipes/sft_qwen3_4b/sft_qwen3_4b.yaml` | Defines dataset, optimizer, LoRA rank, and logging defaults |

The SLURM script copies itself, the Python script, and the active config into the results folder to guarantee reproducibility.

### Launch Training
```bash
koa-ml refresh
koa-ml submit train/scripts/sft_qwen3_4b/sft_qwen3_4b.slurm
```

Monitor progress:
```bash
koa-ml jobs
ssh koa.its.hawaii.edu "tail -f /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/<job_id>/job.log"
```

Weights & Biases logging starts automatically (see the job log for the run URL). Expect ~4–6 hours on a single A100/L40 GPU for 3 epochs with gradient accumulation.

### Customize the Run
- Edit `configs/recipes/sft_qwen3_4b/sft_qwen3_4b.yaml` to change LoRA rank, learning rate, or logging cadence.
- Switch datasets by updating `data.train_dataset.dataset_name`.
- For quick smoke tests, lower `max_steps`, raise `logging_steps`, and disable W&B with `report_to: "none"`.

### Best Practices Before Submitting
- `koa-ml refresh` right before submission so scripts/configs match what runs remotely.
- Keep SSH ControlMaster enabled (see `~/.ssh/config`) to avoid repeated Duo prompts.
- Gradient checkpointing plus LoRA is the default; the Python script already disables `use_cache`, suppresses known transformer warnings, and sets `Image.MAX_IMAGE_PIXELS = None`.
- SLURM scripts export `HF_HUB_DISABLE_HF_TRANSFER=1` and `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`; retain those lines if you clone the pattern.
- Pre-flight checklist:
  - Virtualenv exists at `~/koa-ml/.venv` on KOA
  - `.env` contains valid `HF_TOKEN` and `WANDB_API_KEY`
  - Desired config file is committed or otherwise backed up
  - Scratch storage has at least 20 GB free (`du -sh /mnt/lustre/koa/scratch/$USER/koa-ml`)

## 7. Evaluate Checkpoints

Evaluation scripts live under `eval/scripts/sft_qwen3_4b/` and reuse the same environment.

1. Update the model path in the desired config:
   ```yaml
   # eval/configs/sft_qwen3_4b/eval_train.yaml
   model:
     model_name: "/mnt/lustre/koa/scratch/<user>/koa-ml/train/results/<train_job_id>"
     dtype: "float16"
     device_map: "auto"
   ```
2. Sync changes: `koa-ml refresh`
3. Submit an evaluation job:
   ```bash
   koa-ml submit eval/scripts/sft_qwen3_4b/eval_base.slurm    # Baseline (pretrained model)
   koa-ml submit eval/scripts/sft_qwen3_4b/eval_train.slurm   # Memorization check
   koa-ml submit eval/scripts/sft_qwen3_4b/eval_test.slurm    # Generalization check
   ```

During evaluation you will see a W&B URL in the log plus streaming accuracy updates every few batches. Results are written to:

```
/mnt/lustre/koa/scratch/<user>/koa-ml/eval/results/<eval_job_id>/
├── job.log
├── summary.json
├── predictions.csv
├── eval_train.yaml (or eval_test.yaml)
└── sft_qwen3_4b.py
```

`summary.json` exposes overall accuracy, per-task metrics, and metadata so you can compare checkpoints programmatically.

## 8. Manage Results and Storage

Training and evaluation outputs can accumulate quickly. Use the provided cleanup helpers to stay under KOA scratch quotas.

```bash
# Inspect usage
ssh koa "du -sh /mnt/lustre/koa/scratch/$USER/koa-ml"

# Dry run: see what would be deleted
ssh koa "bash ~/koa-ml/scripts/cleanup_storage.sh --dry-run"

# Keep the five newest training runs, delete older ones
ssh koa "bash ~/koa-ml/scripts/cleanup_storage.sh --train-only --keep-latest 5"

# Delete everything (no undo!)
ssh koa "bash ~/koa-ml/scripts/cleanup_storage.sh --all"
```

Manual removal is always available:
```bash
ssh koa "rm -rf /mnt/lustre/koa/scratch/$USER/koa-ml/train/results/<job_id>"
```

To archive locally:
```bash
rsync -avz --progress \
  koa:/mnt/lustre/koa/scratch/$USER/koa-ml/train/results/<job_id>/ \
  ~/koa-ml-checkpoints/<job_id>/
```

## 9. Troubleshooting

**SSH or rsync issues**
- Confirm `ssh koa.its.hawaii.edu hostname` succeeds without a password prompt
- Ensure `ControlMaster` options exist in `~/.ssh/config` for connection reuse

**Authentication failures when downloading models**
- Re-run `koa-ml auth --sync` and `koa-ml auth --check`
- Verify `.env` contains fresh `HF_TOKEN` and the token has access to gated models

**Out-of-memory or unstable training**
- Reduce `per_device_train_batch_size` or shorten `model_max_length`
- Keep LoRA rank at 16 for 4B models; higher ranks require more GPU memory
- Confirm `load_in_4bit: true` is present when running on 24 GB GPUs

**W&B run missing**
- Check for a W&B URL in `job.log`; if absent, verify `WANDB_API_KEY` is synced
- Set `WANDB_MODE=offline` inside `.env` if you deliberately want to disable uploads

**Broken Python environment on KOA**
- Request a new interactive node (`srun ...`)
- Remove the remote `.venv` and rerun `source scripts/setup_koa_env.sh`

**Need to inspect specific predictions**
- Download results: `koa-ml results pull <eval_job_id> --kind eval`
- Open `predictions.csv` locally for side-by-side comparisons

## 10. Getting Help
- Repository issues: open a ticket with logs and the SLURM job ID
- KOA support: uh-hpc-help@lists.hawaii.edu (include job ID, hostname, and error snippet)
- W&B: https://wandb.ai/support
- Hugging Face access: https://huggingface.co/support

Re-run this quickstart whenever you onboard a new teammate or rebuild your environment. It captures the canonical workflow for KOA ML experimentation in one place.
