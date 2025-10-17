#!/usr/bin/env python3
"""
Evaluation script for language models using lm-evaluation-harness.
Supports standard HuggingFace benchmarks (MMLU, GSM8K, HellaSwag, etc.)

Usage:
    python eval/evaluate.py --config eval/configs/quickstart.yaml
    python eval/evaluate.py --config eval/configs/benchmarks/mmlu.yaml
    python eval/evaluate.py --model ./output/llama8b_lora --tasks mmlu,gsm8k
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Any

import torch
import yaml
from lm_eval import evaluator, tasks as lm_eval_tasks
from lm_eval.models.huggingface import HFLM


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict):
    """Create model wrapper for lm-evaluation-harness."""
    model_config = config['model']
    gen_config = config.get('generation', {})

    model_args = {
        'pretrained': model_config['model_name'],
        'dtype': model_config.get('dtype', 'bfloat16'),
        'batch_size': gen_config.get('per_device_batch_size', 'auto'),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'trust_remote_code': True,
    }

    # Add attention implementation if specified
    if model_config.get('attn_implementation'):
        model_args['attn_implementation'] = model_config['attn_implementation']

    # Create model wrapper
    model = HFLM(**model_args)

    return model


def save_results_csv(results: Dict[str, Any], output_file: Path, task_name: str):
    """Save results in CSV format."""
    if 'results' not in results:
        return

    task_results = results['results'].get(task_name, {})

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value', 'Stderr'])

        for metric, value in task_results.items():
            if not metric.endswith('_stderr'):
                stderr_key = f"{metric}_stderr"
                stderr = task_results.get(stderr_key, '')
                writer.writerow([metric, value, stderr])


def save_results_tsv(results: Dict[str, Any], output_file: Path, task_name: str):
    """Save results in TSV format (better for long responses)."""
    if 'results' not in results:
        return

    task_results = results['results'].get(task_name, {})

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Metric', 'Value', 'Stderr'])

        for metric, value in task_results.items():
            if not metric.endswith('_stderr'):
                stderr_key = f"{metric}_stderr"
                stderr = task_results.get(stderr_key, '')
                writer.writerow([metric, value, stderr])


def update_results_summary(results: Dict[str, Any], output_path: Path, task_name: str):
    """Aggregate task results into results.json for compatibility with downstream tooling."""
    summary_file = output_path / "results.json"

    # Structure that mirrors lm-eval outputs but allows incremental updates per task
    summary: Dict[str, Any] = {
        "results": {},
        "versions": {},
    }

    if summary_file.exists():
        try:
            with open(summary_file, "r") as existing:
                loaded = json.load(existing)
                if isinstance(loaded, dict):
                    summary.update({k: v for k, v in loaded.items() if isinstance(v, dict)})
        except json.JSONDecodeError:
            print(f"Warning: Could not parse existing results file at {summary_file}, recreating it.")

    summary.setdefault("results", {})
    summary.setdefault("versions", {})
    summary.setdefault("samples", {})

    task_results = results.get("results", {}).get(task_name, {})
    summary["results"][task_name] = task_results

    if "versions" in results and isinstance(results["versions"], dict):
        summary["versions"][task_name] = results["versions"].get(task_name)

    if "samples" in results:
        samples = results["samples"]
        if isinstance(samples, dict):
            summary["samples"][task_name] = samples.get(task_name, samples)
        else:
            summary["samples"][task_name] = samples

    with open(summary_file, "w") as handle:
        json.dump(summary, handle, indent=2)


def save_results_multi_format(results: Dict[str, Any], output_path: Path,
                              task_name: str, formats: List[str]):
    """Save results in multiple formats."""
    for fmt in formats:
        fmt = fmt.strip().lower()
        if fmt == 'json':
            output_file = output_path / f"{task_name}_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
        elif fmt == 'csv':
            output_file = output_path / f"{task_name}_results.csv"
            save_results_csv(results, output_file, task_name)
        elif fmt == 'tsv':
            output_file = output_path / f"{task_name}_results.tsv"
            save_results_tsv(results, output_file, task_name)
        else:
            print(f"Warning: Unknown format '{fmt}', skipping")

    update_results_summary(results, output_path, task_name)


def main():
    parser = argparse.ArgumentParser(description='Evaluate language models')
    parser.add_argument('-c', '--config', help='Path to config YAML file')
    parser.add_argument('--model', help='Override model name or path')
    parser.add_argument('--tasks', help='Comma-separated task names (overrides config)')
    parser.add_argument('--num_fewshot', type=int, help='Number of few-shot examples')
    parser.add_argument('--output_path', help='Override output path')
    parser.add_argument('--output_format', default='json',
                       help='Output format: json, csv, tsv, or comma-separated (e.g., json,csv) [default: json]')
    parser.add_argument('--limit', type=int, help='Limit number of examples (for testing)')
    args = parser.parse_args()

    # Parse output formats
    output_formats = [fmt.strip() for fmt in args.output_format.split(',')]

    # Load config if provided
    if args.config:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config)
    else:
        # Minimal config if using CLI args only
        config = {
            'model': {'model_name': args.model or 'meta-llama/Llama-3.1-8B-Instruct'},
            'generation': {},
            'tasks': []
        }

    # Apply CLI overrides
    if args.model:
        config['model']['model_name'] = args.model

    # Build task list
    if args.tasks:
        # Override with CLI tasks
        task_names = args.tasks.split(',')
        tasks_to_run = []
        for task_name in task_names:
            task_dict = {
                'backend': 'lm_harness',
                'task': task_name.strip(),
                'num_fewshot': args.num_fewshot or 0,
            }
            tasks_to_run.append(task_dict)
    else:
        # Use tasks from config
        tasks_to_run = config.get('tasks', [])

    if not tasks_to_run:
        print("Error: No tasks specified. Use --tasks or provide a config with tasks.")
        return 1

    # Create model
    print("\nLoading model...")
    print(f"Model: {config['model']['model_name']}")
    model = create_model(config)

    # Run evaluation for each task
    all_results = {}

    for task_config in tasks_to_run:
        task_name = task_config['task']
        num_fewshot = args.num_fewshot if args.num_fewshot is not None else task_config.get('num_fewshot', 0)
        output_path = args.output_path or task_config.get('output_path', './eval/results')

        print("\n" + "="*50)
        print(f"Evaluating: {task_name}")
        print(f"Few-shot: {num_fewshot}")
        print("="*50 + "\n")

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task_name],
            num_fewshot=num_fewshot,
            limit=args.limit,
            bootstrap_iters=100,
        )

        # Store results
        all_results[task_name] = results

        # Print results
        print("\n" + "-"*50)
        print(f"Results for {task_name}:")
        print("-"*50)

        if 'results' in results:
            task_results = results['results'].get(task_name, {})
            for metric, value in task_results.items():
                if not metric.endswith('_stderr'):
                    stderr_key = f"{metric}_stderr"
                    stderr = task_results.get(stderr_key, 0)
                    print(f"  {metric}: {value:.4f} ± {stderr:.4f}")

        # Save results to file in specified format(s)
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_results_multi_format(results, Path(output_path), task_name, output_formats)

        print(f"\n✓ Results saved to: {output_path} (formats: {', '.join(output_formats)})")

    # Save combined results
    if len(all_results) > 1:
        combined_output = Path(args.output_path or './eval/results') / 'combined_results.json'
        with open(combined_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Combined results saved to: {combined_output}")

    print("\n" + "="*50)
    print("Evaluation complete!")
    print("="*50)


if __name__ == '__main__':
    main()
