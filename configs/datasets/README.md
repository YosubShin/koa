# Dataset Configurations

This directory contains reusable dataset configurations for training models.

## Available Datasets

### [alpaca_cleaned.yaml](alpaca_cleaned.yaml)
- **Size**: ~52K examples
- **Quality**: High (human-curated)
- **License**: CC-BY-NC-4.0
- **Use**: General instruction following

### [dolly.yaml](dolly.yaml)
- **Size**: ~15K examples
- **Quality**: Very High (employee-generated)
- **License**: CC-BY-SA-3.0 (commercial-friendly!)
- **Use**: High-quality instruction following

### [custom_template.yaml](custom_template.yaml)
- Template for creating your own dataset config

## Usage

### Option 1: Reference in model config

```yaml
# In tune/configs/models/qwen3_8b_lora.yaml
data: !include ../../datasets/alpaca_cleaned.yaml
```

### Option 2: Direct copy and modify

```bash
cp configs/datasets/custom_template.yaml configs/datasets/my_dataset.yaml
# Edit my_dataset.yaml with your dataset details
```

### Option 3: Inline in model config

```yaml
# In your model config
data:
  train_dataset:
    dataset_name: "your-username/your-dataset"
    split: "train"
  target_column: "text"
  template: "alpaca"
```

## Creating a New Dataset Config

1. Copy `custom_template.yaml`
2. Fill in your dataset details
3. Test with a small run:
   ```bash
   python tune/train.py --config your_config.yaml --max_steps 10
   ```
4. Validate:
   ```bash
   python scripts/validate_config.py your_config.yaml
   ```

## Dataset Format Requirements

Your dataset should have one of:

### Pre-formatted Text
```json
{"text": "### Instruction:\\nWhat is the capital of France?\\n\\n### Response:\\nParis"}
```

### Instruction-Response Format
```json
{
  "instruction": "What is the capital of France?",
  "input": "",
  "output": "Paris"
}
```

### Q&A Format
```json
{
  "question": "What is the capital of France?",
  "answer": "Paris"
}
```

## Tips

- Start with small datasets (1K-10K examples) for testing
- Use high-quality datasets for better results
- Check dataset license before commercial use
- Validate your config before submitting jobs
