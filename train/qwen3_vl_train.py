#!/usr/bin/env python3
"""
Training script for Qwen3-VL models with LoRA support.
Adapted for vision-language models that process both images and text.

Usage:
    python train/qwen3_vl_train.py --config configs/recipes/qwen3_vl/4b_lora_m2sv.yaml
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

Image.MAX_IMAGE_PIXELS = None

# Disable HF Transfer fallback (not reliable on KOA cluster)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_processor(config: dict):
    """Load Qwen3-VL model and processor with optional quantization."""
    model_name = config['model']['model_name']

    print(f"Loading model: {model_name}")

    # Processor (handles both images and text)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Quantization config for QLoRA
    quantization_config = None
    if config['model'].get('load_in_4bit'):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config['model'].get('bnb_4bit_quant_type', 'nf4'),
            bnb_4bit_compute_dtype=getattr(torch, config['model'].get('bnb_4bit_compute_dtype', 'bfloat16')),
            bnb_4bit_use_double_quant=config['model'].get('bnb_4bit_use_double_quant', True),
        )

    # Model loading arguments
    model_kwargs = {
        'pretrained_model_name_or_path': model_name,
        'torch_dtype': getattr(torch, config['model'].get('dtype', 'bfloat16')),
        'device_map': 'auto',
        'trust_remote_code': True,
    }

    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config

    # Load model
    model = Qwen3VLForConditionalGeneration.from_pretrained(**model_kwargs)

    # Prepare for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    return model, processor


def setup_peft(model, config: dict):
    """Apply LoRA to the model."""
    if 'peft' not in config:
        return model

    peft_config = config['peft']

    lora_config = LoraConfig(
        r=peft_config.get('lora_r', 16),
        lora_alpha=peft_config.get('lora_alpha', 32),
        lora_dropout=peft_config.get('lora_dropout', 0.05),
        target_modules=peft_config.get('lora_target_modules'),
        bias=peft_config.get('bias', 'none'),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_data(config: dict):
    """Load M2SV-SFT dataset for vision-language training."""
    data_config = config['data']['train_dataset']

    # Load from HuggingFace Hub
    dataset = load_dataset(
        data_config['dataset_name'],
        split=data_config.get('split', 'train'),
    )

    # Apply limit if specified
    if data_config.get('limit'):
        dataset = dataset.select(range(min(data_config['limit'], len(dataset))))

    print(f"Loaded {len(dataset)} training examples")
    print(f"Dataset columns: {dataset.column_names}")

    return dataset


def format_m2sv_prompt(example: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format M2SV-SFT examples into Qwen3-VL chat format.

    M2SV-SFT has:
    - question: str (the task description)
    - image_sv: PIL.Image (street view)
    - image_map: PIL.Image (overhead map with arrows)
    - options: List[str] (A, B, C, D options)
    - answer: str (correct answer letter)
    """
    question = example.get('question', '')
    answer = example.get('answer', '')

    # Build the conversation format that Qwen3-VL expects
    system_prompt = (
        "You will be given two images: (1) a north-up overhead map "
        "with arrows labeled A, B, C, ... and (2) a street-view photo.\n"
        "Rules:\n"
        "- The camera location is the same for all options: the center of the intersection.\n"
        "- Each letter corresponds to facing outward from that center along the arrow of that label.\n"
        "- The small circles near labels are markers only; they are not camera locations.\n"
        "- The map and photo may be captured years apart. Ignore transient objects (cars, people).\n"
        "Think step by step to compare the street-view with the map (buildings, angles, lanes, landmarks).\n"
        "On the final line, output only: Final answer: \\boxed{X} where X is a single letter (A, B, C, ...)."
    )

    user_prompt = f"{system_prompt}\n\n{question}"
    assistant_response = f"Final answer: \\boxed{{{answer}}}"

    # Build messages in Qwen3-VL chat format
    image_content = []

    # Add street view image
    if 'image_sv' in example and example['image_sv'] is not None:
        image_content.append({"type": "image", "image": example['image_sv']})

    # Add map image
    if 'image_map' in example and example['image_map'] is not None:
        image_content.append({"type": "image", "image": example['image_map']})

    # Add text prompt
    image_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "user", "content": image_content},
        {"role": "assistant", "content": assistant_response}
    ]

    return {"messages": messages}


class VLMDataCollator:
    """Data collator for vision-language models."""

    def __init__(self, processor, max_length: int = 2048):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of vision-language examples.
        """
        batch_messages = [f['messages'] for f in features]

        # Process with the processor's chat template
        texts = []
        for messages in batch_messages:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)

        # Tokenize the batch
        batch = self.processor(
            text=texts,
            images=[msg['content'] for msg in batch_messages],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()

        return batch


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Qwen3-VL models')
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', help='Override output directory')
    parser.add_argument('--max_steps', type=int, help='Override max training steps')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps

    # Load model and processor
    print("\nLoading Qwen3-VL model and processor...")
    model, processor = load_model_and_processor(config)

    # Apply LoRA if configured
    if 'peft' in config:
        print("\nApplying LoRA...")
        model = setup_peft(model, config)

    # Load dataset
    print("\nLoading M2SV-SFT dataset...")
    dataset = load_data(config)

    # Format dataset for VLM training
    print("\nFormatting dataset...")
    dataset = dataset.map(
        format_m2sv_prompt,
        remove_columns=[col for col in dataset.column_names if col not in ['messages']],
        desc="Formatting examples"
    )

    # Training arguments
    training_config = config['training']
    output_dir = training_config['output_dir']

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 2),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 8),
        learning_rate=training_config.get('learning_rate', 2e-4),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        max_steps=training_config.get('max_steps', 3000),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config.get('save_total_limit', 2),
        bf16=training_config.get('bf16', True),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', 'adamw_torch'),
        report_to=training_config.get('report_to', 'wandb'),
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Data collator
    data_collator = VLMDataCollator(processor, max_length=config['model'].get('model_max_length', 2048))

    # Trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    start_time = time.time()

    trainer.train()

    training_time = time.time() - start_time

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Print performance metrics
    print("\n" + "="*50)
    print("Training Performance Metrics")
    print("="*50)

    # Training time
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    print(f"Training time: {hours}h {minutes}m {seconds}s ({training_time:.2f}s total)")

    # Get metrics from trainer
    if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
        # Calculate throughput
        total_steps = trainer.state.global_step
        if total_steps > 0 and training_time > 0:
            steps_per_second = total_steps / training_time
            print(f"Training throughput: {steps_per_second:.2f} steps/sec")

        # Find final loss
        final_loss = None
        for log_entry in reversed(trainer.state.log_history):
            if 'loss' in log_entry:
                final_loss = log_entry['loss']
                break

        if final_loss is not None:
            print(f"Final training loss: {final_loss:.4f}")

    # GPU memory stats (if available)
    if torch.cuda.is_available():
        max_memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # Convert to GB
        print(f"Peak GPU memory: {max_memory_allocated:.2f} GB")

        # Reset peak stats for next run
        torch.cuda.reset_peak_memory_stats()

    print("="*50)

    print(f"\n✓ Training complete! Model saved to: {output_dir}")
    print(f"\nTo evaluate the fine-tuned model:")
    print(f"  1. Update eval config to point to: {output_dir}")
    print(f"  2. Run eval on train split (check memorization)")
    print(f"  3. Run eval on test split (check generalization)")


if __name__ == '__main__':
    main()
