from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .config import Config
from .ssh import SSHError, copy_to_remote, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
DEFAULT_PARTITION = "kill-shared"


@dataclass
class JobIOPaths:
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _has_partition_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg in {"--partition", "-p"}:
            return True
        if arg.startswith("--partition="):
            return True
        if arg.startswith("-p") and arg != "-p":
            return True
    return False


def _has_output_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg == "--output":
            return True
        if arg.startswith("--output="):
            return True
    return False


def _has_error_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg == "--error":
            return True
        if arg.startswith("--error="):
            return True
    return False


def ensure_remote_workspace(config: Config) -> None:
    run_ssh(config, ["mkdir", "-p", str(config.remote_code_dir)])
    if config.remote_results_dir:
        run_ssh(config, ["mkdir", "-p", str(config.remote_results_dir)])
    if config.shared_env_dir:
        run_ssh(config, ["mkdir", "-p", str(config.shared_env_dir)])


def submit_job(
    config: Config,
    local_job_script: Path,
    *,
    sbatch_args: Optional[Iterable[str]] = None,
    remote_name: Optional[str] = None,
) -> str:
    if not local_job_script.exists():
        raise FileNotFoundError(f"Job script not found: {local_job_script}")

    ensure_remote_workspace(config)

    remote_script = config.remote_code_dir / (remote_name or local_job_script.name)
    copy_to_remote(config, local_job_script, remote_script)

    env_vars: list[str] = [f"KOA_ML_CODE_ROOT={config.remote_code_dir}"]
    if config.remote_results_dir:
        env_vars.append(f"KOA_ML_RESULTS_ROOT={config.remote_results_dir}")
    if config.remote_project_root:
        env_vars.append(f"KOA_PROJECT_ROOT={config.remote_project_root}")
    if config.shared_env_dir:
        env_vars.append(f"KOA_SHARED_ENV={config.shared_env_dir}")

    args = ["env", *env_vars, "sbatch"]
    sbatch_args_list = list(sbatch_args or [])

    if not _has_partition_flag(sbatch_args_list):
        args.extend(["--partition", DEFAULT_PARTITION])

    if config.remote_results_dir:
        results_root = str(config.remote_results_dir)
        if not _has_output_flag(sbatch_args_list):
            args.extend(["--output", f"{results_root}/%j/job.log"])
        if not _has_error_flag(sbatch_args_list):
            args.extend(["--error", f"{results_root}/%j/job.err"])

    if sbatch_args_list:
        args.extend(sbatch_args_list)

    args.append(str(remote_script))

    result = run_ssh(config, args, capture_output=True)
    output = result.stdout.strip() if result.stdout else ""
    match = SBATCH_JOB_ID_PATTERN.search(output)
    if not match:
        raise SSHError(f"Unable to parse sbatch output for job id: {output}")
    return match.group(1)


def cancel_job(config: Config, job_id: str) -> None:
    run_ssh(config, ["scancel", job_id])


def list_jobs(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "squeue",
            "-u",
            config.user,
            "-o",
            r"%i|%j|%T|%M|%l|%D|%R",
        ],
        capture_output=True,
    )
    return result.stdout


def run_health_checks(config: Config) -> str:
    result = run_ssh(
        config,
        [
            "bash",
            "-lc",
            (
                "set -euo pipefail;"
                "echo '== hostname =='; hostname;"
                "echo '== sinfo =='; sinfo -o '%P %a %l %D %G %m'"
            ),
        ],
        capture_output=True,
    )
    return result.stdout


def get_job_io_paths(config: Config, job_id: str) -> JobIOPaths:
    """Return the stdout/stderr paths configured for the given job."""
    result = run_ssh(
        config,
        ["scontrol", "show", "job", str(job_id)],
        capture_output=True,
    )
    stdout_path = None
    stderr_path = None
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("StdOut="):
            stdout_path = line.split("=", 1)[1] or None
        elif line.startswith("StdErr="):
            stderr_path = line.split("=", 1)[1] or None
    return JobIOPaths(stdout=stdout_path, stderr=stderr_path)
