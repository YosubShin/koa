#!/usr/bin/env python3
"""
Qwen3-VL 4B Fine-tuning Script for M2SV-SFT Dataset
Adapted for koa-ml repository structure

Usage:
    torchrun --nproc_per_node=1 train/scripts/sft_qwen3_4b/sft_qwen3_4b.py \
        --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
        --dataset_name yosubshin/m2sv-sft \
        --output_dir /path/to/output \
        --bf16 True \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence
import argparse
import copy
import json
import logging
import os
import pathlib
import warnings

import torch
import transformers
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from PIL import Image

# Suppress warnings
warnings.filterwarnings(
    "ignore", message="None of the inputs have requires_grad=True")
warnings.filterwarnings("ignore", category=UserWarning,
                        module="pydantic._internal._generate_schema")

Image.MAX_IMAGE_PIXELS = None

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def print_model_architecture():
    """Print ASCII visualization of Qwen3-VL-4B architecture."""
    print("\n" + "="*100)
    print("QWEN3-VL-4B MODEL ARCHITECTURE")
    print("="*100)
    print("""
┌────────────────────────────────────────────────────────────────────┐
│                      QWEN3-VL-4B MODEL                             │
│                  Total: 4.45 Billion Parameters                    │
└────────────────────────────────────────────────────────────────────┘
                                 │
           ┌─────────────────────┴─────────────────────┐
           │                                           │
           ▼                                           ▼
┌──────────────────────┐                   ┌──────────────────────┐
│  VISION ENCODER      │                   │  LANGUAGE MODEL      │
│  (Vision Tower)      │                   │  (Qwen3 4B LLM)      │
│                      │                   │                      │
│ • Processes images   │                   │ • Text understanding │
│ • 32x32 patches      │                   │ • Text generation    │
│ • Extracts visual    │                   │ • 4B parameters      │
│   features           │                   │ • 28 layers          │
│                      │                   │ • Attention heads    │
│ FROZEN               │                   │                      │
└──────────┬───────────┘                   └──────────┬───────────┘
           │                                          │
           │            ┌──────────────┐              │
           └───────────►│ MLP          │◄─────────────┘
                        │ PROJECTOR    │
                        │ (Connector)  │
                        │              │
                        │ • Maps vision│
                        │   to LLM     │
                        │   space      │
                        │ FROZEN       │
                        └──────────────┘
""")


def print_lora_config(training_args, trainable_params, total_params):
    """Print LoRA configuration visualization."""
    trainable_pct = (trainable_params / total_params) * 100

    print("\n" + "="*70)
    print("🔧 LORA FINE-TUNING CONFIGURATION")
    print("="*70)
    print(f"""
Total Parameters:        {total_params:,}  ({total_params/1e9:.2f}B)
Trainable Parameters:    {trainable_params:,}  ({trainable_params/1e6:.1f}M)
Trainable Percentage:    {trainable_pct:.2f}%

┌────────────────────────────────────────────────────────────────────┐
│                   ATTENTION LAYERS (LoRA Adapters)                 │
│                                                                    │
│  For each of the 28 transformer layers:                            │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │  Query Projection (q_proj)       TRAINED                   │    │
│  │    • LoRA rank: {training_args.lora_r:<2}                  │    │
│  │    • LoRA alpha: {training_args.lora_alpha:<2}             │    │
│  │    • Dropout: {training_args.lora_dropout}                 │    │
│  │                                                            │    │
│  │  Key Projection (k_proj)         TRAINED                   │    │
│  │  Value Projection (v_proj)       TRAINED                   │    │
│  │  Output Projection (o_proj)      TRAINED                   │    │
│  │                                                            │    │
│  │  [Base weights remain frozen]                              │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
│  Feed-Forward Layers (gate/up/down_proj): FROZEN                   │
│  Vision Encoder: FROZEN                                            │
│  MLP Projector: FROZEN                                             │
└────────────────────────────────────────────────────────────────────┘
""")


def print_training_flow(data_args, training_args):
    """Print training data flow visualization."""
    print("\n" + "="*70)
    print("📊 TRAINING FLOW")
    print("="*70)
    print(f"""
INPUT: 2 Images (Street View + Map) + Question
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 1: IMAGE PROCESSING                                           │
│                                                                    │
│  Street View Image                                                 │
│       │                                                            │
│       ├─► Resize to fit min/max pixels                              │
│       │   Min: {data_args.min_pixels:,} ({int((data_args.min_pixels/(32*32)))} tokens)│
│       │   Max: {data_args.max_pixels:,} ({int((data_args.max_pixels/(32*32)))} tokens)│
│       │                                                            │
│       └─► Split into 32×32 patches                                 │
│           Each patch = 1 token                                     │
│                                                                    │
│  Map Image                                                         │
│       │                                                            │
│                                                                    │
│  Vision Encoder (FROZEN)                                           │
│       └─► Extract features: [~1152 visual tokens]                  │
└────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 2: MULTIMODAL FUSION                                          │
│                                                                    │
│  MLP Projector (FROZEN)                                            │
│       └─► Map visual features to LLM embedding space               │
│                                                                    │
│  Combined Sequence:                                                │
│  [1152 image tokens] + [text tokens from question]                 │
│  Total: ~1200-1500 tokens (max: {training_args.model_max_length})  │
└────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│ STEP 3: LANGUAGE MODEL PROCESSING (LoRA Training)                  │
│                                                                    │
│  Layer 1-28 (each layer):                                          │
│                                                                    │
│        ┌──────────────────────────────────────────┐                │
│        │ Self-Attention (with LoRA)               │                │
│        │   q_proj: frozen + LoRA adapter          │                │
│        │   k_proj: frozen + LoRA adapter          │                │
│        │   v_proj: frozen + LoRA adapter          │                │
│        │   o_proj: frozen + LoRA adapter          │                │
│        └────────────────┬─────────────────────────┘                │
│                         │                                          │
│        ┌────────────────▼─────────────────────────┐                │
│        │ Feed-Forward (FROZEN)                    │                │
│        └────────────────┬─────────────────────────┘                │
│                         │                                          │
│                         ▼                                          │
│        ┌────────────────────────────┐                              │
│        │ Output: Next token         │                              │
│        │ "Final answer: \\boxed{{X}}"                              │
│        └────────────────────────────┘                              │
└────────────────────────────────────────────────────────────────────┘
""")


def print_training_config(training_args, dataset_size):
    """Print training configuration details."""
    effective_batch = training_args.per_device_train_batch_size * \
        training_args.gradient_accumulation_steps
    steps_per_epoch = dataset_size // effective_batch
    total_steps = steps_per_epoch * training_args.num_train_epochs

    print("\n" + "="*70)
    print("⚙️  TRAINING CONFIGURATION")
    print("="*70)
    print(f"""
┌────────────────────────────────────────────────────────────────────┐
│ Dataset                                                            │
├────────────────────────────────────────────────────────────────────┤
│ Total examples:      {dataset_size:<6}                             │
│ Epochs:              {training_args.num_train_epochs:<6}           │
│ Steps per epoch:     ~{steps_per_epoch:<5}                         │
│ Total steps:         {total_steps:<6}                              │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Batch Processing (Gradient Accumulation)                           │
├────────────────────────────────────────────────────────────────────┤
│ Physical batch:      {training_args.per_device_train_batch_size} example at a time │
│ Accumulation steps:  {training_args.gradient_accumulation_steps:<6}│
│ Effective batch:     {effective_batch:<6}                          │
│ Update frequency:    Every {training_args.gradient_accumulation_steps} forward passes│
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Learning Rate Schedule (Cosine)                                    │
├────────────────────────────────────────────────────────────────────┤
│ Initial LR:          {training_args.learning_rate:.2e}             │
│ Warmup ratio:        {training_args.warmup_ratio:.1%}              │
│ Warmup steps:        ~{int(total_steps * training_args.warmup_ratio):<5}│
│ Schedule:            {training_args.lr_scheduler_type:<15}         │
│ Weight decay:        {training_args.weight_decay}                  │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Optimization                                                       │
├────────────────────────────────────────────────────────────────────┤
│ Optimizer:           {training_args.optim:<15}                     │
│ Precision:           {'BF16' if training_args.bf16 else 'FP16':<15}│
│ Gradient clipping:   {'Enabled' if training_args.max_grad_norm else 'Disabled':<15}│
│ Grad checkpointing:  {'Enabled (non-reentrant)' if training_args.gradient_checkpointing else 'Disabled':<27}│
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│ Logging & Checkpointing                                            │
├────────────────────────────────────────────────────────────────────┤
│ Logging steps:       {training_args.logging_steps:<6}              │
│ Save steps:          {training_args.save_steps:<6}                 │
│ Save total limit:    {training_args.save_total_limit:<6}           │
│ Report to:           {training_args.report_to[0] if training_args.report_to else 'none':<15}│
└────────────────────────────────────────────────────────────────────┘
""")


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: Optional[str] = field(
        default="Qwen/Qwen3-VL-4B-Instruct",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tune_mm_llm: bool = field(default=True, metadata={
                              "help": "Whether to tune the language model"})
    tune_mm_mlp: bool = field(default=False, metadata={
                              "help": "Whether to tune the MLP projector"})
    tune_mm_vision: bool = field(default=False, metadata={
                                 "help": "Whether to tune the vision encoder"})


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_name: Optional[str] = field(
        default="yosubshin/m2sv-sft-11k",
        metadata={"help": "HuggingFace dataset name"}
    )
    dataset_split: str = field(default="train", metadata={
                               "help": "Dataset split to use for training"})
    data_limit: Optional[int] = field(
        default=None, metadata={"help": "Limit number of training examples"})

    # Validation configuration
    use_dataset_validation_split: bool = field(
        default=True, metadata={"help": "Use pre-existing validation split from dataset (if available)"})
    validation_split_percentage: float = field(
        default=0.1, metadata={"help": "Percentage of training data to use for validation if no validation split exists (0.0-1.0)"})
    validation_seed: int = field(
        default=42, metadata={"help": "Random seed for train/validation split"})

    # Image resolution (Qwen3-VL uses 32*32 token blocks)
    max_pixels: int = field(
        default=576*32*32, metadata={"help": "Maximum image pixels"})
    min_pixels: int = field(
        default=16*32*32, metadata={"help": "Minimum image pixels"})

    # Video parameters (not used for M2SV but included for compatibility)
    video_max_frames: int = field(
        default=8, metadata={"help": "Maximum video frames"})
    video_min_frames: int = field(
        default=4, metadata={"help": "Minimum video frames"})
    video_max_pixels: int = field(
        default=1024*32*32, metadata={"help": "Maximum video pixels"})
    video_min_pixels: int = field(
        default=256*32*32, metadata={"help": "Minimum video pixels"})
    video_fps: float = field(default=2.0, metadata={"help": "Video FPS"})


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extended training arguments with LoRA support."""
    # Model configuration
    model_max_length: int = field(default=4096, metadata={
                                  "help": "Maximum sequence length"})

    # Learning rates for different components
    mm_projector_lr: Optional[float] = field(
        default=None, metadata={"help": "Learning rate for projector"})
    vision_tower_lr: Optional[float] = field(
        default=None, metadata={"help": "Learning rate for vision tower"})

    # LoRA configuration
    lora_enable: bool = field(default=True, metadata={
                              "help": "Enable LoRA fine-tuning"})
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj",
        metadata={"help": "Comma-separated list of target modules"}
    )


def rank0_print(*args):
    """Print only on rank 0 in distributed training."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)


class M2SVDataset(Dataset):
    """Dataset for M2SV-SFT data in Qwen3-VL format."""

    def __init__(self, dataset_name_or_data, split: Optional[str] = None, processor=None, limit: Optional[int] = None):
        super().__init__()
        self.processor = processor

        # Support both dataset name (str) and pre-loaded dataset object
        if isinstance(dataset_name_or_data, str):
            # Load dataset from HuggingFace
            rank0_print(f"Loading dataset: {dataset_name_or_data} (split: {split})")
            self.dataset = load_dataset(dataset_name_or_data, split=split)
        else:
            # Use pre-loaded dataset (for train/val splits)
            self.dataset = dataset_name_or_data

        if limit:
            self.dataset = self.dataset.select(
                range(min(limit, len(self.dataset))))

        if isinstance(dataset_name_or_data, str):
            rank0_print(f"Loaded {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        # Extract fields
        question = example.get('question', '')
        answer = example.get('answer', '')
        image_sv = example['image_sv']
        image_map = example['image_map']

        # Build system prompt
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

        # Build messages in Qwen3-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_sv},
                    {"type": "image", "image": image_map},
                    {"type": "text", "text": user_prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_response}]
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Process with images
        images = [image_sv, image_map]

        return {
            "text": text,
            "images": images,
            "messages": messages
        }


class DataCollatorForM2SV:
    """Collator for M2SV dataset with Qwen3-VL processor."""

    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of instances."""
        batch_texts = []
        batch_images = []

        for instance in instances:
            batch_texts.append(instance["text"])
            batch_images.append(instance["images"])

        # Process batch
        batch = self.processor(
            text=batch_texts,
            images=batch_images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )

        # Create labels (same as input_ids for causal LM)
        batch["labels"] = batch["input_ids"].clone()

        # Mask padding tokens in labels
        if self.processor.tokenizer.pad_token_id is not None:
            batch["labels"][batch["labels"] ==
                            self.processor.tokenizer.pad_token_id] = -100

        return batch


def set_trainable_params(model, model_args: ModelArguments):
    """Configure which model components are trainable."""
    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze based on configuration
    if model_args.tune_mm_llm:
        rank0_print("Enabling training for language model")
        if hasattr(model, 'model'):
            for param in model.model.parameters():
                param.requires_grad = True

    if model_args.tune_mm_mlp:
        rank0_print("Enabling training for MLP projector")
        # Qwen3-VL projector location may vary
        if hasattr(model, 'visual') and hasattr(model.visual, 'merger'):
            for param in model.visual.merger.parameters():
                param.requires_grad = True

    if model_args.tune_mm_vision:
        rank0_print("Enabling training for vision tower")
        if hasattr(model, 'visual'):
            for param in model.visual.parameters():
                param.requires_grad = True


def train():
    """Main training function."""
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    rank0_print("="*100)
    rank0_print("Qwen3-VL 4B Fine-tuning on M2SV-SFT")
    rank0_print("="*100)
    rank0_print(f"Model: {model_args.model_name_or_path}")
    rank0_print(f"Dataset: {data_args.dataset_name}")
    rank0_print(f"Output: {training_args.output_dir}")
    rank0_print("="*100)

    # Print model architecture visualization
    print_model_architecture()

    # Load processor
    rank0_print("\nLoading processor...")
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        min_pixels=data_args.min_pixels,
        max_pixels=data_args.max_pixels,
    )

    # Load model
    rank0_print("\nLoading model...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Disable cache for gradient checkpointing
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    # Set trainable parameters
    if not training_args.lora_enable:
        set_trainable_params(model, model_args)

    # Apply LoRA if enabled
    if training_args.lora_enable:
        rank0_print("\nApplying LoRA...")

        # Enable input require grads for LoRA with gradient checkpointing
        # This is critical for vision-language models to avoid "does not require grad" error
        model.enable_input_require_grads()
        rank0_print("Input require grads enabled")

        # Enable gradient checkpointing before applying LoRA
        if training_args.gradient_checkpointing:
            # Use non-reentrant mode for better compatibility with frozen layers
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            rank0_print("Gradient checkpointing enabled (use_reentrant=False)")

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            target_modules=training_args.lora_target_modules.split(","),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Print detailed LoRA configuration visualization
        trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print_lora_config(training_args, trainable_params, total_params)

        # Print training flow visualization
        print_training_flow(data_args, training_args)

    # Load dataset
    rank0_print("\nLoading dataset...")

    # Check if dataset has pre-existing validation split
    if data_args.use_dataset_validation_split:
        try:
            rank0_print(f"Attempting to load pre-existing train and validation splits from {data_args.dataset_name}...")
            train_data = load_dataset(data_args.dataset_name, split="train")
            val_data = load_dataset(data_args.dataset_name, split="validation")

            rank0_print(f"✓ Using dataset's built-in splits:")
            rank0_print(f"  Train examples: {len(train_data)}")
            rank0_print(f"  Validation examples: {len(val_data)}")

            # Apply data limit if specified
            if data_args.data_limit:
                train_data = train_data.select(range(min(data_args.data_limit, len(train_data))))
                rank0_print(f"  Limited training data to {len(train_data)} examples")

            train_dataset = M2SVDataset(
                dataset_name_or_data=train_data,
                processor=processor
            )
            eval_dataset = M2SVDataset(
                dataset_name_or_data=val_data,
                processor=processor
            )

        except Exception as e:
            rank0_print(f"Warning: Could not load validation split: {e}")
            rank0_print("Falling back to manual train/val split...")
            data_args.use_dataset_validation_split = False

    # Fallback: Create validation split from training data
    if not data_args.use_dataset_validation_split:
        full_dataset = load_dataset(
            data_args.dataset_name,
            split=data_args.dataset_split
        )

        if data_args.data_limit:
            full_dataset = full_dataset.select(
                range(min(data_args.data_limit, len(full_dataset))))
            rank0_print(f"Limited dataset to {len(full_dataset)} examples")

        # Split into train/validation
        val_percentage = data_args.validation_split_percentage
        if val_percentage > 0:
            rank0_print(f"\nSplitting dataset: {int((1-val_percentage)*100)}% train, {int(val_percentage*100)}% validation")
            dataset_split = full_dataset.train_test_split(
                test_size=val_percentage,
                seed=data_args.validation_seed
            )
            train_data = dataset_split['train']
            val_data = dataset_split['test']

            rank0_print(f"Train examples: {len(train_data)}")
            rank0_print(f"Validation examples: {len(val_data)}")

            train_dataset = M2SVDataset(
                dataset_name_or_data=train_data,
                processor=processor
            )
            eval_dataset = M2SVDataset(
                dataset_name_or_data=val_data,
                processor=processor
            )
        else:
            rank0_print("No validation split (validation_split_percentage=0)")
            train_dataset = M2SVDataset(
                dataset_name_or_data=full_dataset,
                processor=processor
            )
            eval_dataset = None

    # Data collator
    data_collator = DataCollatorForM2SV(
        processor=processor,
        max_length=training_args.model_max_length
    )

    # Print training configuration visualization
    print_training_config(training_args, len(train_dataset))

    # Initialize trainer
    rank0_print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Added validation dataset
        data_collator=data_collator,
    )

    # Train
    rank0_print("\n" + "="*100)
    rank0_print("Starting training...")
    rank0_print("="*100 + "\n")

    trainer.train()

    # Save final model
    rank0_print("\nSaving final model...")
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

    rank0_print("\n" + "="*100)
    rank0_print("Training complete!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("="*100)


if __name__ == "__main__":
    train()
