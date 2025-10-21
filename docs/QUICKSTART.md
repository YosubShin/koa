# Quick Start: Fine-Tuning & Evaluation on KOA

Get started with training and evaluating models on KOA in 5 minutes.

## Setup (One Time)

```bash
# 1. Install dependencies
source .venv/bin/activate
pip install -e ".[ml]"
```

> Setting up on KOA? Start an interactive session (`srun -p gpu-sandbox --gres=gpu:1 --mem=8G -t 0-00:30 --pty /bin/bash`), then run
> `source scripts/setup_koa_env.sh` inside the repo. The script tries the standard KOA Python module (3.11.5) with fallbacks, attempts to load `system/CUDA/12.2.0`, and will automatically retry `pip install` without flash-attn if nvcc isn't available. Override modules if needed (e.g.
> `PYTHON_MODULE=lang/Python/3.10.8-GCCcore-12.2.0 CUDA_MODULE= source scripts/setup_koa_env.sh`).
> The SLURM jobs expect the venv at `$HOME/koa-ml/.venv` (set `KOA_ML_VENV` to override), load the selected Python module (`KOA_PYTHON_MODULE`), and execute from `$HOME/koa-ml` unless `KOA_ML_WORKDIR` is provided.

This installs:
- PyTorch, Transformers, Accelerate (core ML)
- PEFT, TRL (training)
- lm-eval (benchmarking)
- bitsandbytes, flash-attn (optimizations; Linux only extras and skipped on macOS)

## Test Locally (Optional)

```bash
# Quick training test (requires GPU)
python train/train.py \
  --config configs/recipes/qwen3/0.6b/lora.yaml \
  --max_steps 10

# Quick evaluation test (requires GPU)
python eval/evaluate.py \
  --config eval/configs/qwen3_quickstart.yaml \
  --limit 10
```

## Run on KOA

### Fine-Tuning

```bash
# Quick test (30 min)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Qwen3 8B LoRA (12 hours)
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm

# Qwen3 8B QLoRA - memory efficient (12 hours)
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm
```

### Evaluation

```bash
# Quick test (30 min)
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm

# Full MMLU benchmark (2 hours)
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm
```

### Monitor Jobs

```bash
# Check job status
koa-ml jobs

# Cancel a job
koa-ml cancel <job_id>
```

## What Gets Created

After training:
```
train/results/<job_id>/
|-- adapter_model.safetensors  # LoRA weights
|-- adapter_config.json        # LoRA config
|-- tokenizer_config.json      # Tokenizer metadata
|-- training_args.bin          # Trainer state
`-- job.log                    # SLURM job log
```

After evaluation:
```
eval/results/<job_id>/
|-- mmlu_results.json          # Benchmark scores
|-- mmlu_results.csv           # Optional CSV export
`-- job.log                    # SLURM job log
```

Job logs:
```
train/results/<job_id>/job.log      # Training logs
eval/results/<job_id>/job.log      # Evaluation logs
```

## Next Steps

### Customize Training

Edit config files in [configs/recipes/qwen3/](../configs/recipes/qwen3/):

```yaml
# Change dataset
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"

# Adjust training
training:
  learning_rate: 2.0e-04
  num_train_epochs: 3

# Tweak LoRA
peft:
  lora_r: 16      # Higher = more parameters
  lora_alpha: 32
```

### Evaluate Your Model

```bash
# Option 1: Edit config file
# Edit eval/configs/qwen3_8b_full_eval.yaml
# Change model_name to "./train/results/<job_id>"

# Option 2: Use CLI
python eval/evaluate.py \
  --model ./train/results/123456 \
  --tasks mmlu,gsm8k,hellaswag
```

### Try Different Models

SmolLM (testing):
```yaml
model_name: "HuggingFaceTB/SmolLM2-135M-Instruct"
```

Llama 3.1 (production):
```yaml
model_name: "meta-llama/Llama-3.1-8B-Instruct"
```

Qwen (alternative):
```yaml
model_name: "Qwen/Qwen2.5-7B-Instruct"
```

## Available Configs

### Training Recipes ([configs/recipes/qwen3/](../configs/recipes/qwen3/))
- `0.6b/lora.yaml` - Quick testing (30-60 minutes)
- `4b/lora.yaml` - Balanced LoRA training
- `8b/lora.yaml` - Production LoRA training
- `8b/qlora.yaml` - Memory-efficient QLoRA
- `14b/qlora.yaml` - Large-model QLoRA

### Evaluation Configs ([eval/configs/](../eval/configs/))
- `qwen3_quickstart.yaml` - Quick verification
- `qwen3_8b_full_eval.yaml` - Comprehensive benchmark suite
- `qwen3_vl_m2sv.yaml` - Vision-language evaluation
- `benchmarks/mmlu.yaml` - MMLU baseline
- `benchmarks/gsm8k.yaml` - Math reasoning
- `benchmarks/hellaswag.yaml` - Commonsense reasoning

## Detailed Guides

- [ML_GUIDE.md](ML_GUIDE.md) - Complete workflow guide
- [train/README.md](train/README.md) - Training details
- [eval/README.md](eval/README.md) - Evaluation details

## Common Commands

```bash
# Training
python train/train.py --config configs/recipes/qwen3/8b/lora.yaml
python train/train.py --config <config> --output_dir ./train/results/local/my_run
python train/train.py --config <config> --max_steps 100  # Quick test

# Evaluation
python eval/evaluate.py --config eval/configs/qwen3_8b_full_eval.yaml
python eval/evaluate.py --model ./train/results/123456 --tasks mmlu,gsm8k
python eval/evaluate.py --config <config> --limit 10  # Quick test

# KOA job management
koa-ml submit <job.slurm>
koa-ml jobs
koa-ml cancel <job_id>
koa-ml check
```

## Tips

1. **Start small**: Test with SmolLM before expensive runs
2. **Use QLoRA**: If you hit memory issues
3. **Check logs**: SSH to KOA and inspect `train/results/{job_id}/job.log`
4. **Save configs**: Commit your configs to git for reproducibility
5. **Monitor training**: Look for steady loss decrease in logs

## Troubleshooting

**Out of memory**: Switch to QLoRA or reduce batch size

**Job killed**: Increase time limit in SLURM script

**Dataset error**: Check dataset format matches expected columns

**Slow training**: Ensure flash attention is enabled

For more help, see the detailed guides or open an issue!
