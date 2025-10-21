# Automatic GPU Selection

## Overview

The `koa-ml` CLI now automatically selects the best available GPU when submitting jobs! This feature scans the KOA cluster in real-time and requests the highest-performing available GPU.

## How It Works

1. **Before job submission**, `koa-ml` queries `sinfo` to check available GPUs on the target partition
2. **Ranks GPUs** by performance (H200 > H100 > A100 > A30 > V100 > RTX 2080 Ti)
3. **Automatically adds** `--gres=gpu:{best_gpu}:1` to your sbatch command
4. **Falls back** to RTX 2080 Ti if detection fails

## GPU Priority Ranking

| GPU           | Priority Score | Typical Use Case                  |
|---------------|----------------|-----------------------------------|
| H200 / H200 NVL | 110          | Latest-generation, fastest training |
| H100          | 100            | Large models, high throughput      |
| A100          | 90             | Large-scale training               |
| A30           | 80             | Efficient training                 |
| V100          | 70             | Legacy workloads                   |
| RTX 2080 Ti   | 50             | Small-medium models                |

## Usage

### Default Behavior (Auto-Selection Enabled)

```bash
# Automatically selects best available GPU
koa-ml submit train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm

# Output:
# Auto-selected GPU: h200 (priority: 110, 1 nodes available)
# Submitted KOA job 8720999
```

### Disable Auto-Selection

If you want to use the GPU specified in your SLURM script or via `--gpus` flag:

```bash
# Use GPU from SLURM script #SBATCH --gres=gpu:...
koa-ml submit --no-auto-gpu your_script.slurm

# Or specify GPU manually
koa-ml submit --gpus 1 your_script.slurm
```

### Check Available GPUs

```bash
# View cluster status
koa-ml check

# Output shows GPU availability:
# == sinfo ==
# PARTITION AVAIL  TIME  NODES  GRES        MEMORY
# kill-shared  up    infinite  5  gpu:h100:1  500000
# kill-shared  up    infinite  10 gpu:a100:1  250000
```

## Benefits

✅ **Always grab the best hardware** - H200/H100s when available, fallback to A100/V100
✅ **Zero configuration** - Works automatically with existing scripts
✅ **Backward compatible** - Existing workflows continue working
✅ **Flexible override** - Use `--no-auto-gpu` when needed

## Examples

### Training Job (Auto)
```bash
# Grabs best available GPU automatically
koa-ml submit train/scripts/qwen3_vl/train_qwen3_vl_4b_m2sv.slurm
```

### Evaluation Job (Specific GPU)
```bash
# Force RTX 2080 Ti for consistency with training
koa-ml submit --no-auto-gpu eval/scripts/eval_model.slurm
```

### Multi-GPU Job (Manual)
```bash
# Auto-selection disabled when --gpus specified
koa-ml submit --gpus 4 train/scripts/distributed_training.slurm
```

## Troubleshooting

### "No GPU info available, using fallback: rtx2080ti"

This happens if:
- SSH connection failed
- `sinfo` returned unexpected format
- No GPUs are idle/mixed state

**Action:** Job still submits with RTX 2080 Ti (safe fallback)

### "Auto-selected GPU: rtx2080ti"

All higher-tier GPUs (H200, H100, A100, V100) are busy.

**Action:** Job runs on RTX 2080 Ti (still works, just slower)

### Want to force a specific GPU?

Use `--no-auto-gpu` flag:
```bash
koa-ml submit --no-auto-gpu your_script.slurm
```

## Technical Details

### Detection Logic

```python
# Queries SLURM:
sinfo -p kill-shared --Format=nodehost,gres:30,statecompact --noheader

# Parses output:
# node01  gpu:h200:1       idle
# node02  gpu:h100:1       mix
# node03  gpu:a100:1       mix
# node04  gpu:rtx2080ti:1  alloc  # Not counted (fully allocated)

# Returns: {"h200": 1, "h100": 1, "a100": 1}
# Selects: h200 (highest priority with available nodes)
```

### Backward Compatibility

The feature is **opt-out by design**:

- ✅ Existing scripts with `#SBATCH --gres=gpu:...` → Use `--no-auto-gpu`
- ✅ Scripts without GPU spec → Auto-selection enabled
- ✅ CLI `--gpus N` flag → Auto-selection disabled
- ✅ SLURM scripts still work without modification

## Configuration

No configuration needed! The feature works out-of-the-box.

To modify GPU priority ranking, edit `src/koa_ml/slurm.py`:

```python
GPU_PRIORITY = {
    "h200": 110,
    "nvidiah200nvl": 110,
    "h100": 100,
    "nvidiah100": 100,
    "a100": 90,
    "nvidiaa100": 90,
    "a30": 80,
    "nvidiaa30": 80,
    "v100": 70,
    "nvidiav100": 70,
    "rtx2080ti": 50,
}
```
