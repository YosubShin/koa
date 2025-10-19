# koa-ml

---

Utilities for managing KOA HPC GPU jobs and fine-tuning/evaluating language models from your local workstation.

## Features

### Core Capabilities
- **Job Management**: Submit, monitor, and cancel jobs on KOA
- **Fine-Tuning**: LoRA, QLoRA, and full fine-tuning with config files
- **Evaluation**: Standard benchmarks (MMLU, GSM8K, HellaSwag, etc.) with multi-format output (JSON/CSV/TSV)
- **SLURM Integration**: Production-ready job templates with error handling
- **HuggingFace Integration**: Seamless model and dataset loading

### Developer Tools
- **Config Validation**: Validate YAML configs before submission (`scripts/validate_config.py`)
- **Results Comparison**: Compare model performance before/after fine-tuning (`scripts/compare_results.py`)
- **Performance Metrics**: Automatic tracking of training throughput, GPU memory, and timing
- **Error Handling**: Comprehensive error logging and graceful failure handling

### Organization
- **Recipe System**: Pre-configured, tested recipes for each model size
- **Dataset Configs**: Reusable dataset configurations
- **Results by Job ID**: All outputs organized by SLURM job ID for easy tracking
- **Environment Templates**: `.env.example` for easy configuration

## Quick Start

### 1. Setup KOA Connection

```bash
# Create Python environment
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .

# Configure KOA credentials
cp .koa-config.example.yaml ~/.config/koa-ml/config.yaml
# Edit config.yaml with your KOA username, host, and SSH key

# Verify connectivity
koa-ml check
```

### KOA Environment Setup

When preparing the environment directly on KOA, start an interactive session if needed (e.g. `srun -p gpu-sandbox --gres=gpu:1 --mem=8G -t 0-00:30 --pty /bin/bash`) and run:

```bash
cd ~/koa-ml
source scripts/setup_koa_env.sh              # defaults to Python 3.11.5 + cuda/12.1
# Optional overrides:
#   PYTHON_MODULE=lang/Python/3.10.8-GCCcore-12.2.0 source scripts/setup_koa_env.sh
#   CUDA_MODULE=cuda/12.4 source scripts/setup_koa_env.sh
```

The script automatically loads a KOA Python module (3.11 by default with fallbacks), attempts to load `system/CUDA/12.2.0`, recreates `.venv`, installs PyTorch (CUDA 12.x build), and installs the project with `.[ml]` extras.

If `flash-attn` fails to build (missing `nvcc`/CUDA headers), the script retries automatically without that dependency. To skip it entirely, run `INSTALL_FLASH_ATTN=0 source scripts/setup_koa_env.sh`.

### 2. Install ML Dependencies (Optional)

```bash
# For fine-tuning and evaluation
pip install -e ".[ml]"

# Or install separately:
# pip install -e ".[train]"  # Just training
# pip install -e ".[eval]"   # Just evaluation

# Note: GPU-only extras (`bitsandbytes`, `flash-attn`) are available on Linux.
# On macOS these optional packages are skipped automatically.
```

### 3. Try It Out

```bash
# Fine-tune a Qwen3 model (requires ML dependencies)
koa-ml submit tune/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm

# Evaluate a Qwen3 model (requires ML dependencies)
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Check job status
koa-ml jobs
```

## Documentation

### Getting Started
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - 5-minute guide to fine-tuning & evaluation
- **[docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)** - One-page Qwen3 reference

### Detailed Guides
- **[docs/QWEN3_GUIDE.md](docs/QWEN3_GUIDE.md)** - Complete Qwen3 guide
- **[docs/ML_GUIDE.md](docs/ML_GUIDE.md)** - Complete ML workflow documentation
- **[tune/README.md](tune/README.md)** - Fine-tuning details
- **[eval/README.md](eval/README.md)** - Evaluation details

### Development
- **[docs/TESTING.md](docs/TESTING.md)** - Testing instructions

## CLI usage

After installation you can run the CLI as `koa-ml <command>` (the `koa_ml` alias also works if you prefer underscores).

- `koa-ml check` - run a lightweight KOA health check (`hostname`, `sinfo`)
- `koa-ml submit tune/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm --partition kill-shared --gpus 1` - copy the job script to KOA, submit it with `sbatch`, and print the job id
- `koa-ml jobs` - list your active jobs using `squeue`
- `koa-ml cancel <job_id>` - cancel an active job (`scancel`)
- `koa-ml refresh` - rsync the current directory to the remote workdir (defaults to `~/koa-ml`)

Each command accepts `--config` if you want to point at a custom configuration path.
By default, submissions target the `kill-shared` partition; use `--partition` to override.

## Examples

### Fine-Tuning (Qwen3 Models)

```bash
# Quick test with Qwen3 0.6B LoRA (30 min)
koa-ml submit tune/scripts/qwen3/lora/tune_qwen3_0.6b_quickstart.slurm

# Qwen3 8B with LoRA (12 hours)
koa-ml submit tune/scripts/qwen3/lora/tune_qwen3_8b_lora.slurm

# Memory-efficient QLoRA for Qwen3 14B
koa-ml submit tune/scripts/qwen3/qlora/tune_qwen3_14b_qlora.slurm
```

### Evaluation (Qwen3 Models)

```bash
# Quick test with Qwen3 0.6B
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Full evaluation of Qwen3 8B
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm

# Evaluate your fine-tuned model
python eval/evaluate.py \
  --model tune/results/123456 \
  --tasks mmlu,gsm8k,hellaswag

# Multimodal M2SV evaluation (Qwen3-VL)
koa-ml submit eval/scripts/qwen3/eval_qwen3_vl_m2sv.slurm
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run unit tests
pytest tests/ -m "not integration" -v
```

See [docs/TESTING.md](docs/TESTING.md) for details.

## Developer Utilities

### Config Validation

Validate your configuration files before submitting jobs:

```bash
# Validate a single config
python scripts/validate_config.py configs/recipes/qwen3/8b/lora.yaml

# Validate all configs
python scripts/validate_config.py --all
```

### Results Comparison

Compare model performance before and after fine-tuning:

```bash
python scripts/compare_results.py \
  --baseline eval/results/baseline_job_id \
  --checkpoint eval/results/finetuned_job_id \
  --output comparison_report.md
```

### Multi-Format Evaluation Output

Save evaluation results in multiple formats:

```bash
# JSON only (default)
python eval/evaluate.py --config eval/configs/qwen3_8b_full_eval.yaml

# CSV format
python eval/evaluate.py --config eval/configs/qwen3_8b_full_eval.yaml --output_format csv

# Multiple formats
python eval/evaluate.py --config eval/configs/qwen3_8b_full_eval.yaml --output_format json,csv,tsv
```

### Environment Configuration

Copy and customize the environment template:

```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Project Structure

```
koa-ml/
|- configs/               # Centralized configuration system
|  |- datasets/           # Reusable dataset configurations
|  `- recipes/            # Pre-configured training recipes (Qwen3 family)
|- docs/                  # Documentation
|  |- QUICKSTART.md       # 5-minute start guide
|  |- QWEN3_QUICKREF.md   # Qwen3 quick reference
|  |- QWEN3_GUIDE.md      # Complete Qwen3 guide
|  |- ML_GUIDE.md         # ML workflow guide
|  `- TESTING.md          # Testing guide
|- eval/                  # Evaluation system
|  |- configs/            # Evaluation configs and benchmarks
|  |- results/            # Evaluation outputs organized by job ID
|  |- scripts/            # SLURM job scripts for evaluation
|  |- evaluate.py         # Evaluation script with multiple output formats
|  |- qwen3_vl_eval.py    # Vision-language evaluation
|  `- README.md           # Evaluation guide
|- scripts/               # Developer utilities
|  |- compare_results.py  # Compare model performance
|  |- setup_koa_env.sh    # KOA environment setup
|  `- validate_config.py  # Validate YAML configs
|- tune/                  # Fine-tuning system
|  |- results/            # Training outputs organized by job ID
|  |- scripts/            # SLURM job scripts for tuning
|  |- train.py            # Training script with performance metrics
|  `- README.md           # Fine-tuning guide
|- src/koa_ml/            # Job management CLI
|- tests/                 # Automated test suite
|- .env.example           # Environment configuration template
`- .gitignore             # Ignore patterns for generated artifacts
```

## KOA quick reference

- Login node: `koa.its.hawaii.edu` (SSH/MFA required); Open OnDemand is available at https://koa.its.hawaii.edu
- Storage: `/home/<user>` (50 GB, snapshotted daily) and `/mnt/lustre/koa/scratch/<user>` (90-day purge); copy results off scratch promptly
- Common partitions: `gpu` (up to 3 days), `gpu-sandbox` (4 h, unique job slot), `sandbox` (CPU, 4 h), `shared` (3 d), `shared-long` (7 d)
- Slurm commands of interest: `sinfo`, `squeue -u <user>`, `sbatch`, `srun`, `scancel`
- Interactive test session: `srun -I30 -p sandbox -N 1 -c 1 --mem=6G -t 0-01:00:00 --pty /bin/bash`
- Support: uh-hpc-help@lists.hawaii.edu (include job id, script path, error/output files when reporting issues)

## Getting Help

- **Documentation**: See guides in [docs/](docs/)
- **Quick Reference**: [docs/QWEN3_QUICKREF.md](docs/QWEN3_QUICKREF.md)
- **KOA Support**: uh-hpc-help@lists.hawaii.edu
- **Issues**: Open an issue in this repository

## Acknowledgments

ML features inspired by:
- [oumi-ai/oumi](https://github.com/oumi-ai/oumi) - Training framework
- [open-compass/VLMEvalKit](https://github.com/open-compass/VLMEvalKit) - Evaluation patterns
- [ThinkingMachinesLab/tinker-cookbook](https://github.com/ThinkingMachinesLab/tinker-cookbook) - Recipe structure
# Notes for KOA jobs
- Job scripts automatically activate the KOA virtualenv at `$HOME/koa-ml/.venv`. Override with `KOA_ML_VENV=/path/to/venv koa-ml submit ...` if you keep the env elsewhere.
- Jobs default to loading `lang/Python/3.11.5-GCCcore-13.2.0` (and no CUDA module). Override with `KOA_PYTHON_MODULE=` / `KOA_CUDA_MODULE=`.
- Scripts `cd` into `$HOME/koa-ml` automatically; change with `KOA_ML_WORKDIR=/path/on/koa`.
