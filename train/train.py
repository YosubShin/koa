#!/usr/bin/env python3
"""
Fine-tuning script for language models with LoRA/QLoRA/Full support.
Based on best practices from oumi-ai/oumi, using HuggingFace transformers + PEFT + TRL.

Usage:
    python train/train.py --config train/configs/models/smollm_135m_lora.yaml
    python train/train.py --config train/configs/models/llama_8b_qlora.yaml --output_dir ./my_output
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model_and_tokenizer(config: dict):
    """Load model and tokenizer with optional quantization."""
    model_name = config['model']['model_name']

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        'dtype': getattr(torch, config['model'].get('dtype', 'bfloat16')),
        'device_map': 'auto',
        'trust_remote_code': True,
    }

    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config

    # Attention implementation
    if config['model'].get('attn_implementation'):
        model_kwargs['attn_implementation'] = config['model']['attn_implementation']

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Prepare for k-bit training if using quantization
    if quantization_config:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer


def setup_peft(model, config: dict):
    """Apply PEFT (LoRA/QLoRA) to model if configured."""
    if 'peft' not in config:
        return model

    peft_config = config['peft']

    lora_config = LoraConfig(
        r=peft_config.get('lora_r', 8),
        lora_alpha=peft_config.get('lora_alpha', 16),
        lora_dropout=peft_config.get('lora_dropout', 0.05),
        target_modules=peft_config.get('lora_target_modules'),
        bias=peft_config.get('bias', 'none'),
        task_type=peft_config.get('task_type', 'CAUSAL_LM'),
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_data(config: dict, tokenizer):
    """Load and prepare dataset."""
    data_config = config['data']['train_dataset']

    # Load from HuggingFace Hub
    dataset = load_dataset(
        data_config['dataset_name'],
        split=data_config.get('split', 'train'),
    )

    print(f"Loaded {len(dataset)} training examples")
    print(f"Dataset columns: {dataset.column_names}")

    return dataset


def format_instruction(example):
    """Format Alpaca-style instruction prompts."""
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return {'text': prompt}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune language models')
    parser.add_argument('-c', '--config', required=True, help='Path to config YAML file')
    parser.add_argument('--output_dir', help='Override output directory')
    parser.add_argument('--max_steps', type=int, help='Override max training steps')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    if args.max_steps:
        config['training']['max_steps'] = args.max_steps
    if args.wandb:
        config['training']['report_to'] = 'wandb'

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    # Apply PEFT if configured
    if 'peft' in config:
        print("\nApplying PEFT (LoRA/QLoRA)...")
        model = setup_peft(model, config)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_data(config, tokenizer)

    # Format dataset for Alpaca-style prompts
    if 'instruction' in dataset.column_names:
        dataset = dataset.map(format_instruction, remove_columns=dataset.column_names)

    # Training arguments
    training_config = config['training']
    output_dir = training_config['output_dir']

    sft_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 4),
        learning_rate=training_config.get('learning_rate', 2e-4),
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
        warmup_ratio=training_config.get('warmup_ratio', 0.03),
        max_steps=training_config.get('max_steps', -1),
        num_train_epochs=training_config.get('num_train_epochs', 1),
        logging_steps=training_config.get('logging_steps', 10),
        save_steps=training_config.get('save_steps', 100),
        save_total_limit=training_config.get('save_total_limit', 2),
        bf16=training_config.get('bf16', True),
        gradient_checkpointing=training_config.get('gradient_checkpointing', True),
        optim=training_config.get('optim', 'adamw_torch'),
        report_to=training_config.get('report_to', 'none'),
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,
        dataset_text_field='text',
        max_length=config['model'].get('model_max_length', 2048),
        packing=False,
    )

    # Trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    import time
    start_time = time.time()

    trainer.train()

    training_time = time.time() - start_time

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

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


if __name__ == '__main__':
    main()
