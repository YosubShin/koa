from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import Optional

from .config import Config, load_config
from .slurm import cancel_job, ensure_remote_workspace, list_jobs, run_health_checks, submit_job
from .ssh import (
    SSHError,
    sync_directory_to_remote,
    copy_to_remote,
    copy_from_remote,
    run_ssh,
)

DEFAULT_REFRESH_EXCLUDES: list[str] = [
    ".git/",
    ".gitignore",
    ".venv/",
    ".venv-vllm/",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.log",
    "*.tmp",
    ".DS_Store",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".coverage",
    ".idea/",
    ".vscode/",
    ".claude/",
    "node_modules/",
    "eval/results/",
]


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the KOA config file (defaults to ~/.config/koa-ml/config.yaml).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa-ml",
        description="Utilities for running KOA HPC (Slurm) jobs from your local machine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser(
        "check", help="Run KOA connectivity health checks."
    )
    _add_common_arguments(check_parser)

    jobs_parser = subparsers.add_parser(
        "jobs", help="List active KOA jobs for the configured user."
    )
    _add_common_arguments(jobs_parser)

    cancel_parser = subparsers.add_parser("cancel", help="Cancel a KOA job by id.")
    _add_common_arguments(cancel_parser)
    cancel_parser.add_argument("job_id", help="Slurm job id to cancel.")

    submit_parser = subparsers.add_parser(
        "submit", help="Submit a job script via sbatch."
    )
    _add_common_arguments(submit_parser)
    submit_parser.add_argument(
        "job_script", type=Path, help="Path to the local job script."
    )
    submit_parser.add_argument("--remote-name", help="Override the filename on KOA.")
    submit_parser.add_argument(
        "--partition",
        help="Slurm partition (queue) to submit to. Defaults to kill-shared.",
    )
    submit_parser.add_argument("--time", help="Walltime request (e.g. 02:00:00).")
    submit_parser.add_argument("--gpus", type=int, help="Number of GPUs to request.")
    submit_parser.add_argument("--cpus", type=int, help="Number of CPUs to request.")
    submit_parser.add_argument("--memory", help="Memory request (e.g. 32G).")
    submit_parser.add_argument("--account", help="Slurm account if required.")
    submit_parser.add_argument("--qos", help="Quality of service if required.")
    submit_parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments. Repeat for multiple flags.",
    )
    submit_parser.add_argument(
        "--no-auto-gpu",
        action="store_true",
        help="Disable automatic GPU selection (use GPU from script or --gpus flag).",
    )

    refresh_parser = subparsers.add_parser(
        "refresh", help="Sync the current directory to the remote KOA workdir."
    )
    _add_common_arguments(refresh_parser)
    refresh_parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Local directory to sync (defaults to current working directory).",
    )
    refresh_parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Exclude pattern for rsync (repeatable).",
    )

    auth_parser = subparsers.add_parser(
        "auth", help="Manage authentication tokens (HuggingFace, W&B) on KOA."
    )
    _add_common_arguments(auth_parser)
    auth_parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync local .env file to remote KOA workdir.",
    )
    auth_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if tokens are configured on KOA.",
    )
    auth_parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env file (defaults to .env in current directory).",
    )

    storage_parser = subparsers.add_parser(
        "storage", help="Inspect and initialize remote code/data directories."
    )
    _add_common_arguments(storage_parser)
    storage_subparsers = storage_parser.add_subparsers(
        dest="storage_command", required=True
    )

    storage_setup = storage_subparsers.add_parser(
        "setup", help="Create remote directories (code + scratch data)."
    )
    storage_setup.add_argument(
        "--link",
        action="store_true",
        help="Create/refresh symlinks under the code checkout pointing to scratch results.",
    )

    storage_subparsers.add_parser(
        "show", help="Display the configured remote code and data directories."
    )

    storage_subparsers.add_parser(
        "link", help="Create or refresh symlinks for train/eval results."
    )

    results_parser = subparsers.add_parser(
        "results", help="List or download scratch-hosted training/eval artefacts."
    )
    _add_common_arguments(results_parser)
    results_subparsers = results_parser.add_subparsers(
        dest="results_command", required=True
    )

    results_list = results_subparsers.add_parser(
        "list", help="List recent result directories on KOA scratch."
    )
    results_list.add_argument(
        "--kind",
        choices=["train", "eval", "all"],
        default="train",
        help="Which results to show (default: train).",
    )
    results_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of entries per section (default: 20).",
    )
    results_list.add_argument(
        "--full-path",
        action="store_true",
        help="Print full remote paths instead of just job IDs.",
    )

    results_pull = results_subparsers.add_parser(
        "pull", help="Download a result directory from KOA scratch."
    )
    results_pull.add_argument("job_id", help="Job ID or folder name to download.")
    results_pull.add_argument(
        "--kind",
        choices=["train", "eval"],
        default="train",
        help="Whether to pull training or evaluation results (default: train).",
    )
    results_pull.add_argument(
        "--dest",
        type=Path,
        default=None,
        help="Local destination directory (defaults to ./artifacts/<kind>/<job_id>).",
    )

    return parser


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _submit(args: argparse.Namespace, config: Config) -> int:
    sbatch_args: list[str] = []
    if args.partition:
        sbatch_args.extend(["--partition", args.partition])
    if args.time:
        sbatch_args.extend(["--time", args.time])
    if args.gpus:
        sbatch_args.append(f"--gres=gpu:{args.gpus}")
    if args.cpus:
        sbatch_args.extend(["--cpus-per-task", str(args.cpus)])
    if args.memory:
        sbatch_args.extend(["--mem", args.memory])
    if args.account:
        sbatch_args.extend(["--account", args.account])
    if args.qos:
        sbatch_args.extend(["--qos", args.qos])
    sbatch_args.extend(args.sbatch_arg or [])

    # Disable auto-GPU if user specifies --gpus or --no-auto-gpu
    auto_gpu = not (args.no_auto_gpu or args.gpus)

    job_id = submit_job(
        config,
        args.job_script,
        sbatch_args=sbatch_args,
        remote_name=args.remote_name,
        auto_gpu=auto_gpu,
    )
    print(f"Submitted KOA job {job_id}")
    return 0


def _cancel(args: argparse.Namespace, config: Config) -> int:
    cancel_job(config, args.job_id)
    print(f"Cancelled KOA job {args.job_id}")
    return 0


def _jobs(_: argparse.Namespace, config: Config) -> int:
    print(list_jobs(config), end="")
    return 0


def _check(_: argparse.Namespace, config: Config) -> int:
    print(run_health_checks(config), end="")
    return 0


def _refresh(args: argparse.Namespace, config: Config) -> int:
    local_path = Path(args.path).expanduser().resolve() if args.path else Path.cwd()
    excludes = list(DEFAULT_REFRESH_EXCLUDES)
    if args.exclude:
        excludes.extend(args.exclude)
    sync_directory_to_remote(
        config,
        local_path,
        config.remote_code_dir,
        excludes=excludes,
    )
    print(
        f"Synced {local_path} -> {config.login}:{config.remote_code_dir}"
    )
    return 0


def _auth(args: argparse.Namespace, config: Config) -> int:
    """Manage authentication tokens on KOA."""
    if args.check:
        # Check if tokens are configured
        print("Checking authentication configuration on KOA...")
        check_cmd = f"test -f {config.remote_code_dir}/.env && echo 'EXISTS' || echo 'NOT_FOUND'"
        result = run_ssh(config, check_cmd, capture_output=True, check=False)

        if "EXISTS" in result.stdout:
            print(f"✓ .env file found at {config.remote_code_dir}/.env")

            # Check for specific tokens
            tokens_to_check = ["HF_TOKEN", "WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]
            print("\nChecking for tokens:")
            for token in tokens_to_check:
                check_token_cmd = f"grep -q '^{token}=' {config.remote_code_dir}/.env && echo 'SET' || echo 'NOT_SET'"
                token_result = run_ssh(config, check_token_cmd, capture_output=True, check=False)
                status = "✓ Set" if "SET" in token_result.stdout else "✗ Not set"
                print(f"  {token}: {status}")
        else:
            print(f"✗ No .env file found at {config.remote_code_dir}/.env")
            print("\nTo sync your local .env file to KOA, run:")
            print("  koa-ml auth --sync")

        return 0

    if args.sync:
        # Sync .env file to remote
        env_file = Path(args.env_file).expanduser().resolve() if args.env_file else Path.cwd() / ".env"

        if not env_file.exists():
            print(f"Error: .env file not found at {env_file}", file=sys.stderr)
            print("\nTo create one, copy .env.example and fill in your tokens:", file=sys.stderr)
            print(f"  cp .env.example .env", file=sys.stderr)
            print(f"  # Edit .env and add your HF_TOKEN and WANDB_API_KEY", file=sys.stderr)
            return 1

        print(f"Syncing {env_file} -> {config.login}:{config.remote_code_dir}/.env")

        # Copy .env to remote
        remote_env_path = Path(config.remote_code_dir) / ".env"
        copy_to_remote(config, env_file, remote_env_path)

        # Set proper permissions (readable only by user)
        chmod_cmd = f"chmod 600 {config.remote_code_dir}/.env"
        run_ssh(config, chmod_cmd)

        print("✓ .env file synced successfully")
        print("\nYour tokens are now available on KOA.")
        print("SLURM jobs will automatically load them from the .env file.")

        return 0

    # Default: show help
    print("Usage: koa-ml auth [--sync | --check]")
    print("\nOptions:")
    print("  --sync   Sync local .env file to remote KOA")
    print("  --check  Check authentication status on KOA")
    return 0


def _create_remote_symlinks(config: Config) -> None:
    data_root = str(config.remote_data_dir)
    if data_root in {"", "."}:
        raise ValueError("remote_data_dir is not configured; update your koa-ml config")

    code_root = str(config.remote_code_dir)
    run_ssh(config, ["mkdir", "-p", f"{code_root}/train"])
    run_ssh(config, ["mkdir", "-p", f"{code_root}/eval"])
    run_ssh(config, ["rm", "-rf", f"{code_root}/train/results"])
    run_ssh(config, ["rm", "-rf", f"{code_root}/eval/results"])
    run_ssh(
        config,
        ["ln", "-sfn", f"{data_root}/train/results", f"{code_root}/train/results"],
    )
    run_ssh(
        config,
        ["ln", "-sfn", f"{data_root}/eval/results", f"{code_root}/eval/results"],
    )


def _storage(args: argparse.Namespace, config: Config) -> int:
    data_root = str(config.remote_data_dir)

    if args.storage_command == "setup":
        ensure_remote_workspace(config)
        print("Remote code directory:", config.remote_code_dir)
        print("Remote data directory:", config.remote_data_dir)
        if args.link:
            try:
                _create_remote_symlinks(config)
                print("Created symlinks for train/results and eval/results under the code checkout.")
            except ValueError as exc:
                print(f"Warning: {exc}", file=sys.stderr)
        return 0

    if args.storage_command == "show":
        print("Remote code directory:", config.remote_code_dir)
        print("Remote data directory:", config.remote_data_dir)
        return 0

    if args.storage_command == "link":
        ensure_remote_workspace(config)
        try:
            _create_remote_symlinks(config)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print("Symlinks updated:")
        print(f"  {config.remote_code_dir}/train/results -> {config.remote_data_dir}/train/results")
        print(f"  {config.remote_code_dir}/eval/results -> {config.remote_data_dir}/eval/results")
        return 0

    return 1


def _results(args: argparse.Namespace, config: Config) -> int:
    data_root = str(config.remote_data_dir)
    if data_root in {"", "."}:
        print(
            "remote_data_dir is not configured. Update your ~/.config/koa-ml/config.yaml",
            file=sys.stderr,
        )
        return 1

    if args.results_command == "list":
        kinds = []
        if args.kind in {"train", "all"}:
            kinds.append("train")
        if args.kind in {"eval", "all"}:
            kinds.append("eval")
        limit = max(1, args.limit)
        for kind in kinds:
            remote_dir = f"{data_root}/{kind}/results"
            print(f"[{kind}] {remote_dir}")

            quoted = shlex.quote(remote_dir)
            cmd = f"if [ -d {quoted} ]; then find {quoted} -mindepth 1 -maxdepth 1 -type d -print; fi"
            result = run_ssh(
                config,
                ["bash", "-lc", cmd],
                capture_output=True,
                check=False,
            )
            entries = [line.strip() for line in result.stdout.splitlines() if line.strip()]

            if not entries:
                print("  <empty>")
                continue
            entries = sorted(entries, reverse=True)[:limit]
            for entry in entries:
                name = entry.rsplit("/", 1)[-1]
                display = entry if args.full_path else name
                print(f"  {display}")
        return 0

    if args.results_command == "pull":
        remote_dir = config.remote_data_dir / args.kind / "results" / args.job_id
        quoted = shlex.quote(str(remote_dir))
        exists = run_ssh(
            config,
            ["bash", "-lc", f"test -d {quoted}"],
            capture_output=True,
            check=False,
        )
        if exists.returncode != 0:
            print(f"Remote results not found at {remote_dir}", file=sys.stderr)
            return 1

        if args.dest:
            dest_base = args.dest.expanduser()
        else:
            dest_base = (Path("artifacts") / args.kind / args.job_id).expanduser()
        dest_base.parent.mkdir(parents=True, exist_ok=True)
        try:
            copy_from_remote(config, remote_dir, dest_base, recursive=True)
        except SSHError as exc:
            print(f"Failed to download results: {exc}", file=sys.stderr)
            return 1

        print(f"Results copied to {dest_base.resolve()}")
        return 0

    return 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        config = _load(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    try:
        if args.command == "submit":
            return _submit(args, config)
        if args.command == "cancel":
            return _cancel(args, config)
        if args.command == "jobs":
            return _jobs(args, config)
        if args.command == "check":
            return _check(args, config)
        if args.command == "refresh":
            return _refresh(args, config)
        if args.command == "auth":
            return _auth(args, config)
        if args.command == "storage":
            return _storage(args, config)
        if args.command == "results":
            return _results(args, config)
    except (SSHError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
