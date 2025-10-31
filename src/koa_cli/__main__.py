from __future__ import annotations

import argparse
import importlib.resources as resources
import json
import os
import re
import shlex
import uuid
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from .config import Config, DEFAULT_CONFIG_PATH, load_config
from .slurm import (
    cancel_job,
    get_job_io_paths,
    list_jobs,
    run_health_checks,
    submit_job,
)
from .ssh import (
    SSHError,
    copy_to_remote,
    run_ssh,
)
from .manifest import update_manifest_metadata, write_run_manifest
from .runs import list_runs, record_submission, show_run, sync_statuses

DEFAULT_SNAPSHOT_EXCLUDES: list[str] = [
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
]

SNAPSHOT_IGNORE_PATTERNS = [pattern.rstrip("/") for pattern in DEFAULT_SNAPSHOT_EXCLUDES]
SNAPSHOT_IGNORE_PATTERNS.extend([
    "run_metadata",
    "runs",
    "results",
])

DEFAULT_ENV_WATCH = [
    "scripts/setup_env.sh",
    "requirements.txt",
    "requirements.lock",
    "requirements-dev.txt",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "environment.yml",
]


def _load_template(name: str) -> str:
    """Load a text template bundled with the CLI."""
    try:
        return resources.files("koa_cli.templates").joinpath(name).read_text(encoding="utf-8")
    except AttributeError:  # pragma: no cover - fallback for Python <3.9
        return resources.read_text("koa_cli.templates", name)


def _prompt(value: Optional[str], question: str, *, default: Optional[str] = None, required: bool = False) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()

    while True:
        suffix = f" [{default}]" if default else ""
        answer = input(f"{question}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None and default != "":
            return default
        if not required:
            return ""
        print("Value required.", file=sys.stderr)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the KOA config file (defaults to koa-config.yaml in the repository or ~/.config/koa/config.yaml).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="koa",
        description="Utilities for running KOA HPC (Slurm) jobs from your local machine.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup", help="Configure global KOA defaults (user, roots, modules)."
    )
    setup_parser.add_argument("--user", help="KOA username")
    setup_parser.add_argument("--host", help="KOA login host (default: koa.its.hawaii.edu)")
    setup_parser.add_argument(
        "--remote-root",
        help="Top-level remote workspace directory for KOA projects.",
    )
    setup_parser.add_argument(
        "--local-root",
        help="Top-level local workspace directory for KOA project mirrors.",
    )
    setup_parser.add_argument(
        "--python-module",
        help="Preferred KOA Python module to load in job scripts.",
    )
    setup_parser.add_argument(
        "--cuda-module",
        help="Preferred KOA CUDA module to load in job scripts.",
    )
    setup_parser.add_argument(
        "--default-partition",
        help="Default Slurm partition for submissions (e.g. kill-shared).",
    )

    init_parser = subparsers.add_parser(
        "init",
        help="Initialise KOA project configuration in the current repository.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing koa-config.yaml or scripts if present.",
    )

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
        "--desc",
        help="Optional description appended to the timestamped run directory name.",
    )
    submit_parser.add_argument(
        "--sbatch-arg",
        action="append",
        default=[],
        help="Additional raw sbatch arguments. Repeat for multiple flags.",
    )

    logs_parser = subparsers.add_parser(
        "logs", help="Stream or inspect a job's stdout/stderr log."
    )
    _add_common_arguments(logs_parser)
    logs_parser.add_argument("job_id", help="Job ID to inspect.")
    logs_parser.add_argument(
        "--stream",
        choices=["stdout", "stderr"],
        default="stdout",
        help="Select which stream to view (default: stdout).",
    )
    logs_parser.add_argument(
        "--lines",
        type=int,
        default=50,
        help="Number of lines to show when not following (default: 50).",
    )
    logs_parser.add_argument(
        "--follow",
        action="store_true",
        help="Follow log output in real time (tail -F).",
    )

    runs_parser = subparsers.add_parser(
        "runs", help="Manage and inspect recorded KOA job runs."
    )
    runs_subparsers = runs_parser.add_subparsers(dest="runs_command", required=True)
    runs_list = runs_subparsers.add_parser("list", help="List recorded runs.")
    runs_list.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of runs to display (default: 20).",
    )
    runs_subparsers.add_parser(
        "sync", help="Sync job statuses from KOA (updates local catalog)."
    )
    runs_show = runs_subparsers.add_parser("show", help="Display details for a run.")
    runs_show.add_argument("job_id", help="Job ID to inspect.")

    return parser


def _load(args: argparse.Namespace) -> Config:
    return load_config(args.config)


def _setup(args: argparse.Namespace) -> int:
    existing: dict = {}
    if DEFAULT_CONFIG_PATH.exists():
        try:
            existing = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Warning: failed to read existing config ({exc}); starting fresh.", file=sys.stderr)
            existing = {}

    default_user = args.user or existing.get("user") or os.getenv("KOA_USER") or os.getenv("USER") or ""
    default_host = args.host or existing.get("host") or os.getenv("KOA_HOST") or "koa.its.hawaii.edu"

    user = _prompt(args.user, "KOA username", default=default_user, required=True)
    host = _prompt(args.host, "KOA login host", default=default_host, required=True)

    suggested_remote_root = (
        args.remote_root
        or existing.get("remote_root")
        or (existing.get("remote", {}) or {}).get("root")
        or f"/mnt/lustre/koa/scratch/{user}/koa-cli"
    )
    remote_root = _prompt(args.remote_root, "Remote workspace root", default=suggested_remote_root, required=True)

    suggested_local_root = (
        args.local_root
        or existing.get("local_root")
        or (existing.get("local", {}) or {}).get("root")
        or str(Path("~/koa-projects").expanduser())
    )
    local_root = _prompt(args.local_root, "Local workspace root", default=suggested_local_root, required=True)

    default_partition = _prompt(
        args.default_partition,
        "Default Slurm partition",
        default=existing.get("default_partition") or "kill-shared",
        required=False,
    )

    default_python = _prompt(
        args.python_module,
        "Preferred Python module",
        default=(
            args.python_module
            or existing.get("python_module")
            or (existing.get("modules") or {}).get("python")
            or "lang/Python/3.11.5-GCCcore-13.2.0"
        ),
        required=False,
    )

    default_cuda = _prompt(
        args.cuda_module,
        "Preferred CUDA module",
        default=(
            args.cuda_module
            or existing.get("cuda_module")
            or (existing.get("modules") or {}).get("cuda")
            or "system/CUDA/12.2.0"
        ),
        required=False,
    )

    config_data = existing.copy()
    config_data.update(
        {
            "user": user,
            "host": host,
            "remote_root": remote_root,
            "local_root": local_root,
        }
    )

    if default_partition:
        config_data["default_partition"] = default_partition

    modules: dict = config_data.get("modules") or {}
    if default_python:
        modules["python"] = default_python
        config_data["python_module"] = default_python
    if default_cuda:
        modules["cuda"] = default_cuda
        config_data["cuda_module"] = default_cuda
    if modules:
        config_data["modules"] = modules

    # Ensure the config directory exists
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ensure the local root exists
    try:
        Path(local_root).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem issues
        print(f"Warning: unable to create local workspace root ({exc})", file=sys.stderr)

    DEFAULT_CONFIG_PATH.write_text(
        yaml.safe_dump(config_data, sort_keys=False),
        encoding="utf-8",
    )

    print("Updated KOA global configuration:")
    print(f"  File: {DEFAULT_CONFIG_PATH}")
    print(f"  User: {user}@{host}")
    print(f"  Remote workspace: {remote_root}")
    print(f"  Local workspace: {local_root}")
    if default_partition:
        print(f"  Default partition: {default_partition}")
    if default_python:
        print(f"  Python module: {default_python}")
    if default_cuda:
        print(f"  CUDA module: {default_cuda}")

    return 0


def _load_global_config_data() -> dict:
    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Global config not found at {DEFAULT_CONFIG_PATH}. Run `koa setup` first."
        )
    return yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}


def _render_setup_env_script(python_module: Optional[str], cuda_module: Optional[str]) -> str:
    module_lines = ["module purge >/dev/null 2>&1 || true"]
    if python_module:
        module_lines.append(f"module load {python_module} >/dev/null 2>&1 || true")
    else:
        module_lines.append("module load ${PYTHON_MODULE:-} >/dev/null 2>&1 || true")
    if cuda_module:
        module_lines.append(f"module load {cuda_module} >/dev/null 2>&1 || true")
    else:
        module_lines.append("module load ${CUDA_MODULE:-} >/dev/null 2>&1 || true")

    module_block = "\n".join(module_lines)
    if module_block:
        module_block += "\n"

    template = _load_template("setup_env.sh.tmpl")
    return template.replace("__MODULE_BLOCK__", module_block)


def _render_basic_job_template(
    project_name: str,
    default_partition: str,
    python_module: Optional[str],
    cuda_module: Optional[str],
) -> str:
    module_lines: list[str] = []
    if python_module:
        module_lines.append(f"module load {python_module} >/dev/null 2>&1 || true")
    else:
        module_lines.append("module load ${PYTHON_MODULE:-} >/dev/null 2>&1 || true")
    if cuda_module:
        module_lines.append(f"module load {cuda_module} >/dev/null 2>&1 || true")
    else:
        module_lines.append("module load ${CUDA_MODULE:-} >/dev/null 2>&1 || true")

    module_block = "\n".join(module_lines)
    if module_block:
        module_block += "\n"

    template = _load_template("basic_job.slurm.tmpl")
    return (
        template.replace("__JOB_NAME__", project_name)
        .replace("__DEFAULT_PARTITION__", default_partition)
        .replace("__MODULE_LINES__", module_block)
    )


def _write_file(path: Path, content: str, *, overwrite: bool = False, executable: bool = False) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | 0o111)


def _init_project(args: argparse.Namespace) -> int:
    data = _load_global_config_data()
    cwd = Path.cwd()
    project_name = cwd.name

    user = data.get("user")
    host = data.get("host")
    if not user or not host:
        raise ValueError("Global config missing user/host; run `koa setup` to fix.")

    remote_root = Path(data.get("remote_root") or f"/mnt/lustre/koa/scratch/{user}/koa-cli")
    local_root = Path(data.get("local_root") or Path("~/koa-projects").expanduser())

    python_module = data.get("python_module") or (data.get("modules") or {}).get("python")
    cuda_module = data.get("cuda_module") or (data.get("modules") or {}).get("cuda")
    default_partition = data.get("default_partition") or "kill-shared"

    config_path = cwd / "koa-config.yaml"

    modules_section: dict[str, str] = {}
    if python_module:
        modules_section["python"] = python_module
    if cuda_module:
        modules_section["cuda"] = cuda_module
    modules_block = ""
    if modules_section:
        module_lines = ["modules:"]
        for key, value in modules_section.items():
            module_lines.append(f"  {key}: {value}")
        modules_block = "\n".join(module_lines) + "\n\n"

    env_watch_lines = "\n".join(f"  - {item}" for item in DEFAULT_ENV_WATCH)

    config_template = _load_template("koa-config.yaml.tmpl")
    config_rendered = (
        config_template.replace("__PROJECT_NAME__", project_name)
        .replace("__DEFAULT_PARTITION__", default_partition)
        .replace("__MODULE_SECTION__", modules_block)
        .replace("__ENV_WATCH__", env_watch_lines)
    )
    _write_file(config_path, config_rendered, overwrite=args.force)

    scripts_dir = cwd / "scripts"
    _write_file(
        scripts_dir / "setup_env.sh",
        _render_setup_env_script(python_module, cuda_module),
        overwrite=args.force,
        executable=True,
    )

    _write_file(
        scripts_dir / "basic_job.slurm",
        _render_basic_job_template(project_name, default_partition, python_module, cuda_module),
        overwrite=args.force,
        executable=True,
    )

    remote_project_root = (remote_root / "projects" / project_name).resolve()
    local_project_root = (local_root / "projects" / project_name).expanduser().resolve()
    local_jobs_root = local_project_root / "jobs"
    local_jobs_root.mkdir(parents=True, exist_ok=True)

    print(f"Initialised KOA project '{project_name}'")
    print(f"  Config: {config_path}")
    print(f"  Remote project root: {remote_project_root}")
    print(f"  Local project root: {local_project_root}")
    return 0


def _create_repo_snapshot(source: Path, destination: Path, extra_excludes: Optional[list[str]] = None) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    patterns = list(SNAPSHOT_IGNORE_PATTERNS)
    if extra_excludes:
        patterns.extend(extra_excludes)
    ignore = shutil.ignore_patterns(*patterns)
    shutil.copytree(source, destination, ignore=ignore)

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

    with tempfile.TemporaryDirectory(prefix="koa-submit-") as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_path = tmp_path / "run_metadata"
        env_watch = config.env_watch_files or DEFAULT_ENV_WATCH
        write_run_manifest(manifest_path, env_watch=env_watch)
        update_manifest_metadata(
            manifest_path,
            sbatch_args=sbatch_args,
            job_script=str(args.job_script),
        )

        repo_snapshot_path = tmp_path / "repo"
        _create_repo_snapshot(Path.cwd(), repo_snapshot_path, config.snapshot_excludes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        desc = args.desc or ""
        if desc:
            desc = re.sub(r"[^A-Za-z0-9_-]+", "_", desc).strip("_-")
        job_folder = timestamp if not desc else f"{timestamp}_{desc}"

        remote_job_dir: Optional[Path] = None
        if config.remote_results_dir:
            remote_job_dir = config.remote_results_dir / job_folder
            try:
                run_ssh(config, ["mkdir", "-p", str(remote_job_dir)])
                run_ssh(config, ["mkdir", "-p", str(remote_job_dir / "results")])
                copy_to_remote(
                    config,
                    manifest_path,
                    remote_job_dir / "run_metadata",
                    recursive=True,
                )
                copy_to_remote(
                    config,
                    repo_snapshot_path,
                    remote_job_dir / "repo",
                    recursive=True,
                )
            except SSHError as exc:
                print(f"Warning: failed to stage files on KOA: {exc}", file=sys.stderr)
                remote_job_dir = None

        local_job_dir: Optional[Path] = None
        if config.local_results_dir:
            local_job_dir = (config.local_results_dir / job_folder).expanduser()
            if local_job_dir.exists():
                shutil.rmtree(local_job_dir)
            local_job_dir.mkdir(parents=True, exist_ok=True)
            (local_job_dir / "results").mkdir(exist_ok=True)
            shutil.copytree(manifest_path, local_job_dir / "run_metadata")
            shutil.copytree(repo_snapshot_path, local_job_dir / "repo")

        job_id = submit_job(
            config,
            args.job_script,
            sbatch_args=sbatch_args,
            remote_name=args.remote_name,
            run_dir=remote_job_dir,
        )

        update_manifest_metadata(
            manifest_path,
            job_id=job_id,
            remote_code_dir=str(config.remote_code_dir),
            remote_results_dir=str(config.remote_results_dir) if config.remote_results_dir else None,
        )

        manifest_data = json.loads((manifest_path / "manifest.json").read_text(encoding="utf-8"))

        record_submission(
            config,
            job_id=job_id,
            sbatch_args=sbatch_args,
            manifest=manifest_data,
            local_job_dir=local_job_dir,
            remote_job_dir=remote_job_dir,
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




def _logs(args: argparse.Namespace, config: Config) -> int:
    try:
        io_paths = get_job_io_paths(config, args.job_id)
    except SSHError as exc:
        print(f"Error querying job {args.job_id}: {exc}", file=sys.stderr)
        return 1

    target_path = io_paths.stdout if args.stream == "stdout" else io_paths.stderr

    if not target_path or target_path in {"UNKNOWN", "UNDEFINED"}:
        print(
            f"No {args.stream} log path reported for job {args.job_id}.",
            file=sys.stderr,
        )
        return 1

    quoted = shlex.quote(target_path)
    tail_flags = "-F" if args.follow else ""
    lines = max(0, args.lines)
    command = f"tail {tail_flags} -n {lines} {quoted}" if tail_flags else f"tail -n {lines} {quoted}"

    print(f"Streaming {args.stream} log: {target_path}")
    result = run_ssh(
        config,
        ["bash", "-lc", command],
        check=False,
    )
    return result.returncode


def _runs_list(args: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` before recording runs.")
        return 0
    runs = list_runs(config)
    if not runs:
        print("No recorded runs yet. Submit a job with `koa submit` to create one.")
        return 0

    limit = max(1, args.limit)
    print(f"Showing {min(limit, len(runs))} of {len(runs)} run(s):")
    for entry in runs[:limit]:
        job_id = entry.get("job_id")
        status = entry.get("status") or "UNKNOWN"
        submitted = entry.get("submitted_at") or "---"
        remote_dir = entry.get("remote_job_dir") or "<remote>"
        print(f"- {job_id}: {status} @ {submitted} -> {remote_dir}")
    return 0


def _runs_sync(_: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` first.")
        return 0
    updates = sync_statuses(config)
    if updates:
        print(f"Updated statuses for {updates} run(s).")
    else:
        print("No status changes detected.")
    return 0


def _runs_show(args: argparse.Namespace, config: Config) -> int:
    if not config.local_results_dir:
        print("No local results directory configured; run `koa init` first.")
        return 0
    entry = show_run(config, args.job_id)
    if not entry:
        print(f"No run recorded with job ID {args.job_id}.", file=sys.stderr)
        return 1
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "setup":
        return _setup(args)
    if args.command == "init":
        return _init_project(args)

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
        if args.command == "logs":
            return _logs(args, config)
        if args.command == "runs":
            if args.runs_command == "list":
                return _runs_list(args, config)
            if args.runs_command == "sync":
                return _runs_sync(args, config)
            if args.runs_command == "show":
                return _runs_show(args, config)
    except (SSHError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error(f"Unhandled command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
