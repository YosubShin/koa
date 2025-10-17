#!/usr/bin/env python3
"""
Compare evaluation results between different model checkpoints.

Usage:
    python scripts/compare_results.py \
        --baseline eval/results/123456 \
        --checkpoint eval/results/789012 \
        --metrics mmlu,gsm8k,hellaswag

    python scripts/compare_results.py \
        --baseline eval/results/123456 \
        --checkpoint eval/results/789012 \
        --output comparison_report.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load evaluation results from a directory."""
    results_file = results_dir / "results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, 'r') as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any], metric_names: Optional[List[str]] = None) -> Dict[str, float]:
    """Extract specified metrics from results."""
    metrics = {}

    if "results" not in results:
        return metrics

    for task, task_results in results["results"].items():
        # If specific metrics are requested, only extract those
        if metric_names:
            for metric in metric_names:
                if task in metric_names or metric in task:
                    for key, value in task_results.items():
                        if isinstance(value, (int, float)):
                            metrics[f"{task}/{key}"] = value
        else:
            # Extract all numeric metrics
            for key, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task}/{key}"] = value

    return metrics


def calculate_delta(baseline_value: float, checkpoint_value: float) -> tuple[float, str]:
    """Calculate delta and format it with +/- and arrow."""
    delta = checkpoint_value - baseline_value
    delta_pct = (delta / baseline_value * 100) if baseline_value != 0 else 0

    if delta > 0:
        symbol = "↑"
        color = "green"
    elif delta < 0:
        symbol = "↓"
        color = "red"
    else:
        symbol = "→"
        color = "gray"

    return delta_pct, f"{symbol} {delta:+.4f} ({delta_pct:+.2f}%)"


def generate_markdown_report(baseline_dir: Path, checkpoint_dir: Path,
                            baseline_metrics: Dict[str, float],
                            checkpoint_metrics: Dict[str, float],
                            output_file: Optional[Path] = None) -> str:
    """Generate a markdown comparison report."""

    lines = []
    lines.append("# Evaluation Results Comparison\n")
    lines.append(f"**Baseline:** `{baseline_dir}`\n")
    lines.append(f"**Checkpoint:** `{checkpoint_dir}`\n")
    lines.append("")

    # Find common metrics
    common_metrics = set(baseline_metrics.keys()) & set(checkpoint_metrics.keys())

    if not common_metrics:
        lines.append("⚠️  No common metrics found between baseline and checkpoint.\n")
        report = "\n".join(lines)
        if output_file:
            output_file.write_text(report)
        return report

    lines.append("## Metrics Comparison\n")
    lines.append("| Metric | Baseline | Checkpoint | Delta |")
    lines.append("|--------|----------|------------|-------|")

    # Sort metrics by name
    for metric in sorted(common_metrics):
        baseline_val = baseline_metrics[metric]
        checkpoint_val = checkpoint_metrics[metric]
        delta_pct, delta_str = calculate_delta(baseline_val, checkpoint_val)

        lines.append(f"| {metric} | {baseline_val:.4f} | {checkpoint_val:.4f} | {delta_str} |")

    lines.append("")

    # Summary statistics
    lines.append("## Summary\n")

    improvements = sum(1 for m in common_metrics if checkpoint_metrics[m] > baseline_metrics[m])
    regressions = sum(1 for m in common_metrics if checkpoint_metrics[m] < baseline_metrics[m])
    unchanged = sum(1 for m in common_metrics if checkpoint_metrics[m] == baseline_metrics[m])

    lines.append(f"- **Improvements:** {improvements}/{len(common_metrics)}")
    lines.append(f"- **Regressions:** {regressions}/{len(common_metrics)}")
    lines.append(f"- **Unchanged:** {unchanged}/{len(common_metrics)}")
    lines.append("")

    # Calculate average delta
    deltas = [calculate_delta(baseline_metrics[m], checkpoint_metrics[m])[0]
              for m in common_metrics]
    avg_delta = sum(deltas) / len(deltas) if deltas else 0

    if avg_delta > 0:
        lines.append(f"✅ **Average improvement:** +{avg_delta:.2f}%")
    elif avg_delta < 0:
        lines.append(f"❌ **Average regression:** {avg_delta:.2f}%")
    else:
        lines.append("➖ **No average change**")

    report = "\n".join(lines)

    if output_file:
        output_file.write_text(report)
        print(f"Report saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--baseline",
        required=True,
        type=Path,
        help="Path to baseline results directory (e.g., eval/results/123456)"
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to checkpoint results directory (e.g., eval/results/789012)"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        help="Comma-separated list of metrics to compare (e.g., mmlu,gsm8k,hellaswag)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for markdown report (default: print to stdout)"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.baseline.exists():
        print(f"Error: Baseline directory not found: {args.baseline}")
        return 1

    if not args.checkpoint.exists():
        print(f"Error: Checkpoint directory not found: {args.checkpoint}")
        return 1

    # Parse metrics filter
    metric_names = args.metrics.split(",") if args.metrics else None

    try:
        # Load results
        print(f"Loading baseline results from {args.baseline}...")
        baseline_results = load_results(args.baseline)
        baseline_metrics = extract_metrics(baseline_results, metric_names)

        print(f"Loading checkpoint results from {args.checkpoint}...")
        checkpoint_results = load_results(args.checkpoint)
        checkpoint_metrics = extract_metrics(checkpoint_results, metric_names)

        # Generate report
        report = generate_markdown_report(
            args.baseline,
            args.checkpoint,
            baseline_metrics,
            checkpoint_metrics,
            args.output
        )

        if not args.output:
            print("\n" + report)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
