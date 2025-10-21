# Qwen3 Fine-Tuning & Evaluation Guide

This guide is specifically for working with **Qwen3** models - the latest generation from Alibaba Cloud's Qwen team.

## Why Qwen3?

- **Thinking Mode**: Seamlessly switches between complex reasoning and efficient dialogue
- **Multilingual**: Supports 100+ languages and dialects
- **Long Context**: 32K tokens natively, up to 131K with YaRN
- **Strong Performance**: Excellent at math, coding, and logical reasoning
- **Apache 2.0 License**: Free for commercial use

## Available Qwen3 Models

| Model | Size | Memory (LoRA) | Memory (QLoRA) | Best For |
|-------|------|---------------|----------------|----------|
| Qwen3-0.6B | 600M | 4GB | 2GB | Quick testing, development |
| Qwen3-4B | 4B | 12GB | 6GB | Balanced capability/efficiency |
| Qwen3-8B | 8B | 24GB | 12GB | Production use, flagship model |
| Qwen3-14B | 14B | 40GB | 20GB | Maximum capability |
| Qwen3-32B | 32B | 80GB | 40GB | Top-tier performance |

## Quick Start

### 1. Quick Test (0.6B Model - 30 minutes)

```bash
# Submit training job
koa-ml submit train/scripts/qwen3/lora/train_qwen3_0.6b_quickstart.slurm

# Monitor
koa-ml jobs

# Evaluate when done
koa-ml submit eval/scripts/qwen3/eval_qwen3_quickstart.slurm
```

### 2. Production Training (8B Model)

**LoRA (requires 24GB GPU like A30)**:
```bash
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm
```

**QLoRA (requires 12GB GPU like RTX A4000)**:
```bash
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm
```

### 3. Evaluate Your Model

```bash
# Quick check
python eval/evaluate.py \
  --model ./output/qwen3_8b_lora \
  --tasks mmlu_computer_science \
  --limit 50

# Full evaluation
koa-ml submit eval/scripts/qwen3/eval_qwen3_8b_full.slurm
```

## Available Configs

### Training Configs

All located in [configs/recipes/qwen3/](../configs/recipes/qwen3/):

- **[0.6b/lora.yaml](../configs/recipes/qwen3/0.6b/lora.yaml)** - Quick testing
- **[4b/lora.yaml](../configs/recipes/qwen3/4b/lora.yaml)** - Balanced model
- **[8b/lora.yaml](../configs/recipes/qwen3/8b/lora.yaml)** - Flagship LoRA
- **[8b/qlora.yaml](../configs/recipes/qwen3/8b/qlora.yaml)** - Memory-efficient
- **[14b/qlora.yaml](../configs/recipes/qwen3/14b/qlora.yaml)** - Large model

### Evaluation Configs

All located in [eval/configs/](eval/configs/):

- **[qwen3_quickstart.yaml](eval/configs/qwen3_quickstart.yaml)** - Quick test
- **[qwen3_8b_full_eval.yaml](eval/configs/qwen3_8b_full_eval.yaml)** - Comprehensive

## Qwen3 Special Features

### 1. Thinking Mode

Qwen3 can automatically activate "thinking mode" for complex reasoning:

**Config Settings**:
```yaml
generation:
  temperature: 0.6  # Recommended for thinking mode
  top_p: 0.95
  top_k: 20
```

**When to use**:
- Mathematical reasoning
- Complex coding problems
- Multi-step logical reasoning
- Planning and strategy

### 2. Long Context Support

Qwen3 supports up to 131K tokens with YaRN:

```yaml
model:
  model_max_length: 32768  # Native
  # Or extend:
  # model_max_length: 131072  # With YaRN
```

**Use cases**:
- Long document analysis
- Book-length context
- Multi-turn conversations
- Code repositories

### 3. Multilingual Capabilities

Qwen3 excels at 100+ languages. Great for:
- Translation tasks
- Multilingual customer support
- Cross-lingual reasoning

## Customization Examples

### Change Dataset

Edit the config file:

```yaml
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"
    split: "train"
```

### Adjust LoRA Rank

For more expressive adapters:

```yaml
peft:
  lora_r: 32      # Higher rank (default is 16 for 8B)
  lora_alpha: 64  # Keep alpha = 2 * r
```

For faster, smaller adapters:

```yaml
peft:
  lora_r: 8
  lora_alpha: 16
```

### Enable W&B Logging

```yaml
training:
  report_to: "wandb"
```

Or via CLI:
```bash
python train/train.py --config configs/recipes/qwen3/8b/lora.yaml --wandb
```

### Adjust Context Length

For longer sequences:

```yaml
model:
  model_max_length: 65536  # 64K context

training:
  per_device_train_batch_size: 1  # May need to reduce batch size
  gradient_accumulation_steps: 64  # Compensate with more accumulation
```

## Performance Tips

### 1. GPU Selection on KOA

**For Qwen3-0.6B/4B**:
- Any GPU works (RTX 5000, RTX A4000, A30)
- Use LoRA for best speed

**For Qwen3-8B**:
- **LoRA**: A30 (24GB) or better
- **QLoRA**: RTX A4000 (16GB) or better

**For Qwen3-14B**:
- **QLoRA only**: A30 (24GB) or better
- Or use multi-GPU setup

### 2. Speed Optimization

Enable flash attention (already in configs):
```yaml
model:
  attn_implementation: "flash_attention_2"
```

Use fused optimizer:
```yaml
training:
  optim: "adamw_torch_fused"  # Faster than regular adamw
```

### 3. Memory Optimization

If you hit OOM:

1. Switch from LoRA to QLoRA
2. Reduce batch size:
   ```yaml
   training:
     per_device_train_batch_size: 1
   ```
3. Increase gradient accumulation:
   ```yaml
   training:
     gradient_accumulation_steps: 64
   ```
4. Reduce context length:
   ```yaml
   model:
     model_max_length: 16384  # Half of default
   ```

## Expected Results

### Qwen3-8B Benchmarks (Base Model)

Based on official benchmarks:

| Benchmark | Score |
|-----------|-------|
| MMLU (5-shot) | ~72% |
| GSM8K (8-shot) | ~84% |
| HumanEval | ~67% |
| MATH | ~54% |
| BBH | ~74% |

Your trained models should maintain or improve these scores, especially on your target domain.

## Common Workflows

### 1. Domain Adaptation

```bash
# Fine-tune on domain data
python train/train.py \
  --config configs/recipes/qwen3/8b/lora.yaml \
  # (Edit config to use your domain dataset)

# Evaluate on domain benchmarks
python eval/evaluate.py \
  --model ./output/qwen3_8b_domain \
  --tasks <your_domain_tasks>
```

### 2. Instruction Following

```bash
# Use instruction dataset
# data:
#   train_dataset:
#     dataset_name: "tatsu-lab/alpaca"

# Evaluate on instruction-following benchmarks
python eval/evaluate.py \
  --model ./output/qwen3_8b_instruct \
  --tasks mmlu,bbh,hellaswag
```

### 3. Code Generation

```bash
# Fine-tune on code
# data:
#   train_dataset:
#     dataset_name: "bigcode/the-stack-dedup"

# Evaluate on code benchmarks
python eval/evaluate.py \
  --model ./output/qwen3_8b_code \
  --tasks humaneval,mbpp
```

### 4. Comparison Study

```bash
# Train multiple configs
for config in configs/recipes/qwen3/8b/lora.yaml configs/recipes/qwen3/8b/qlora.yaml; do
  python train/train.py --config "${config}"
done

# Evaluate all
for model in ./output/qwen3_8b_*; do
  python eval/evaluate.py --model $model --tasks mmlu,gsm8k
done
```

## Using Your Fine-Tuned Qwen3 Model

### In Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    dtype="bfloat16",
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./output/qwen3_8b_lora")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output/qwen3_8b_lora")

# Generate
prompt = "Explain quantum computing:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### For Thinking Mode

```python
# Enable thinking mode with appropriate generation config
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    temperature=0.6,
    top_p=0.95,
    top_k=20
)
```

## Troubleshooting

### "Out of memory" Error

**Solution**: Switch to QLoRA
```bash
# Instead of:
koa-ml submit train/scripts/qwen3/lora/train_qwen3_8b_lora.slurm

# Use:
koa-ml submit train/scripts/qwen3/qlora/train_qwen3_8b_qlora.slurm
```

### Slow Training

**Check**: Is flash attention enabled?
```yaml
model:
  attn_implementation: "flash_attention_2"
```

**Verify**: In job logs, look for "Using flash attention"

### Model Not Found

**Issue**: Qwen3 requires transformers >= 4.51.0

**Solution**:
```bash
pip install --upgrade transformers
```

### Evaluation Scores Lower Than Expected

**Possible causes**:
1. Model not in "thinking mode" (try temperature=0.6)
2. Wrong prompt format (ensure using Qwen chat template)
3. Training not converged (check loss curves)

## Next Steps

1. **Start small**: Test with Qwen3-0.6B first
2. **Scale up**: Move to 4B or 8B for production
3. **Experiment**: Try different LoRA ranks and learning rates
4. **Evaluate**: Compare before/after training
5. **Deploy**: Use vLLM or SGLang for inference

## Resources

- **Official Blog**: https://qwenlm.github.io/blog/qwen3/
- **HuggingFace Models**: https://huggingface.co/Qwen
- **Documentation**: https://huggingface.co/docs/transformers/model_doc/qwen3
- **GitHub**: https://github.com/QwenLM/Qwen

Happy training with Qwen3!
