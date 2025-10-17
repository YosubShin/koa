#!/usr/bin/env python3
"""
Validate YAML configuration files for training and evaluation.

Usage:
    python scripts/validate_config.py tune/configs/models/qwen3_8b_lora.yaml
    python scripts/validate_config.py eval/configs/qwen3_8b_full_eval.yaml
    python scripts/validate_config.py --all  # Validate all configs
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml


class ConfigValidator:
    """Validator for KOA-ML configuration files."""

    # Required fields for different config types
    TUNE_REQUIRED_FIELDS = {
        'model': ['model_name'],
        'data': ['train_dataset'],
        'training': ['per_device_train_batch_size', 'learning_rate', 'output_dir'],
    }

    EVAL_REQUIRED_FIELDS = {
        'model': ['model_name'],
        'tasks': None,  # List of task dicts
    }

    # Valid values for certain fields
    VALID_DTYPES = ['float16', 'bfloat16', 'float32', 'int8', 'int4']
    VALID_ATTENTION = ['eager', 'sdpa', 'flash_attention_2']
    VALID_LR_SCHEDULERS = ['linear', 'cosine', 'constant', 'constant_with_warmup']

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = None
        self.errors = []
        self.warnings = []

    def load_config(self) -> bool:
        """Load and parse the YAML config."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            return True
        except FileNotFoundError:
            self.errors.append(f"Config file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            self.errors.append(f"YAML parsing error: {e}")
            return False

    def is_tune_config(self) -> bool:
        """Check if this is a training config."""
        return self.config and 'training' in self.config

    def is_eval_config(self) -> bool:
        """Check if this is an evaluation config."""
        return self.config and 'tasks' in self.config

    def validate_required_fields(self):
        """Validate that all required fields are present."""
        if self.is_tune_config():
            required = self.TUNE_REQUIRED_FIELDS
        elif self.is_eval_config():
            required = self.EVAL_REQUIRED_FIELDS
        else:
            self.errors.append("Unable to determine config type (neither tune nor eval)")
            return

        for section, fields in required.items():
            if section not in self.config:
                self.errors.append(f"Missing required section: '{section}'")
                continue

            if fields is None:
                # Just check that the section exists and is not empty
                if not self.config[section]:
                    self.errors.append(f"Section '{section}' is empty")
            else:
                # Check for required fields in the section
                for field in fields:
                    if field not in self.config[section]:
                        self.errors.append(f"Missing required field: '{section}.{field}'")

    def validate_model_section(self):
        """Validate model configuration."""
        if 'model' not in self.config:
            return

        model = self.config['model']

        # Check dtype
        if 'dtype' in model:
            if model['dtype'] not in self.VALID_DTYPES:
                self.warnings.append(
                    f"Unusual dtype '{model['dtype']}'. "
                    f"Expected one of: {', '.join(self.VALID_DTYPES)}"
                )

        # Check attention implementation
        if 'attn_implementation' in model:
            if model['attn_implementation'] not in self.VALID_ATTENTION:
                self.warnings.append(
                    f"Unknown attention implementation '{model['attn_implementation']}'. "
                    f"Expected one of: {', '.join(self.VALID_ATTENTION)}"
                )

        # Check model_max_length is reasonable
        if 'model_max_length' in model:
            max_len = model['model_max_length']
            if max_len < 128:
                self.warnings.append(f"Very small model_max_length: {max_len}")
            elif max_len > 32768:
                self.warnings.append(f"Very large model_max_length: {max_len} (may cause OOM)")

    def validate_training_section(self):
        """Validate training configuration."""
        if 'training' not in self.config:
            return

        training = self.config['training']

        # Check batch size
        if 'per_device_train_batch_size' in training:
            batch_size = training['per_device_train_batch_size']
            if batch_size < 1:
                self.errors.append(f"Invalid batch size: {batch_size}")
            elif batch_size > 32:
                self.warnings.append(f"Large batch size: {batch_size} (may cause OOM)")

        # Check learning rate
        if 'learning_rate' in training:
            lr = training['learning_rate']
            if lr <= 0:
                self.errors.append(f"Invalid learning rate: {lr}")
            elif lr > 1e-2:
                self.warnings.append(f"Very high learning rate: {lr}")
            elif lr < 1e-6:
                self.warnings.append(f"Very low learning rate: {lr}")

        # Check gradient accumulation
        if 'gradient_accumulation_steps' in training:
            grad_accum = training['gradient_accumulation_steps']
            if grad_accum < 1:
                self.errors.append(f"Invalid gradient_accumulation_steps: {grad_accum}")

        # Check LR scheduler
        if 'lr_scheduler_type' in training:
            scheduler = training['lr_scheduler_type']
            if scheduler not in self.VALID_LR_SCHEDULERS:
                self.warnings.append(
                    f"Unknown lr_scheduler_type '{scheduler}'. "
                    f"Expected one of: {', '.join(self.VALID_LR_SCHEDULERS)}"
                )

        # Check epochs vs steps
        has_epochs = 'num_train_epochs' in training
        has_steps = 'max_steps' in training

        if not has_epochs and not has_steps:
            self.errors.append("Must specify either 'num_train_epochs' or 'max_steps'")
        elif has_epochs and has_steps:
            self.warnings.append("Both 'num_train_epochs' and 'max_steps' specified (max_steps takes precedence)")

    def validate_peft_section(self):
        """Validate PEFT (LoRA/QLoRA) configuration."""
        if 'peft' not in self.config:
            return

        peft = self.config['peft']

        # Check LoRA rank
        if 'lora_r' in peft:
            lora_r = peft['lora_r']
            if lora_r < 1:
                self.errors.append(f"Invalid lora_r: {lora_r}")
            elif lora_r > 64:
                self.warnings.append(f"Very large lora_r: {lora_r} (may be slow to train)")

        # Check LoRA alpha
        if 'lora_alpha' in peft:
            lora_alpha = peft['lora_alpha']
            if lora_alpha <= 0:
                self.errors.append(f"Invalid lora_alpha: {lora_alpha}")

        # Check r and alpha relationship
        if 'lora_r' in peft and 'lora_alpha' in peft:
            lora_r = peft['lora_r']
            lora_alpha = peft['lora_alpha']
            # Typical pattern is alpha = 2 * r
            if lora_alpha != 2 * lora_r:
                self.warnings.append(
                    f"Unusual lora_alpha/lora_r ratio: {lora_alpha}/{lora_r}. "
                    f"Common pattern is alpha = 2 * r"
                )

    def validate(self) -> bool:
        """Run all validation checks."""
        if not self.load_config():
            return False

        self.validate_required_fields()
        self.validate_model_section()
        self.validate_training_section()
        self.validate_peft_section()

        return len(self.errors) == 0

    def print_report(self):
        """Print validation report."""
        print(f"\nValidating: {self.config_path}")
        print("=" * 60)

        if self.errors:
            print(f"\n❌ {len(self.errors)} ERROR(S):")
            for error in self.errors:
                print(f"  • {error}")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNING(S):")
            for warning in self.warnings:
                print(f"  • {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid!")

        print("=" * 60)


def find_all_configs(base_dir: Path) -> List[Path]:
    """Find all YAML config files."""
    configs = []
    for pattern in ['**/*.yaml', '**/*.yml']:
        configs.extend(base_dir.glob(pattern))
    # Filter out example files and hidden files
    configs = [c for c in configs if not c.name.startswith('.') and 'example' not in c.name.lower()]
    return sorted(configs)


def main():
    parser = argparse.ArgumentParser(
        description="Validate KOA-ML configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        'config',
        nargs='?',
        type=Path,
        help="Path to config file to validate"
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help="Validate all config files in the repository"
    )

    args = parser.parse_args()

    if args.all:
        # Find and validate all configs
        base_dir = Path(__file__).parent.parent
        configs = []

        search_roots = [
            base_dir / 'configs' / 'recipes',
            base_dir / 'tune' / 'configs',  # Legacy configs (if still present)
            base_dir / 'eval' / 'configs',
        ]

        for config_dir in search_roots:
            if config_dir.exists():
                configs.extend(find_all_configs(config_dir))

        if not configs:
            print("No config files found")
            return 1

        print(f"Found {len(configs)} config file(s) to validate")

        total_errors = 0
        total_warnings = 0

        for config_path in configs:
            validator = ConfigValidator(config_path)
            validator.validate()
            validator.print_report()

            total_errors += len(validator.errors)
            total_warnings += len(validator.warnings)

        print(f"\n{'=' * 60}")
        print(f"Summary: {len(configs)} config(s) validated")
        print(f"  Errors: {total_errors}")
        print(f"  Warnings: {total_warnings}")
        print(f"{'=' * 60}\n")

        return 1 if total_errors > 0 else 0

    elif args.config:
        # Validate single config
        validator = ConfigValidator(args.config)
        is_valid = validator.validate()
        validator.print_report()

        return 0 if is_valid else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
