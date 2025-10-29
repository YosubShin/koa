from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import Config, GPURequest
from .ssh import SSHError, copy_to_remote, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
DEFAULT_PARTITION = "kill-shared"

# GPU priority ranking (higher score = better GPU) used as a fallback when no
# explicit user preference matches the live cluster inventory.
GPU_PRIORITY = {
    "nvidia_h200_nvl": 110,
    "nvidia_h100": 100,
    "nvidia_h100_pcie": 100,
    "nvidia_h100_nvl": 100,
    "nvidia_h200": 110,
    "nvidia_a100": 90,
    "nvidia_a30": 80,
    "nv_a30": 80,
    "nv_h100": 100,
    "nv_l40": 85,
    "nv_v100_sxm2": 70,
    "nv_rtx_a4000": 60,
    "nv_rtx5000": 55,
    "nv_rtx2080ti": 50,
    "nvidia_a30_1g_6gb": 45,
    "nvidia_a30_2g_12gb": 45,
}

# Default GPU to request if detection fails entirely
FALLBACK_GPU = "rtx2080ti"

# Map normalized GPU identifiers to SLURM GRES names
@dataclass
class GPUAvailability:
    """Summary of GPU availability within a partition."""

    nodes: int = 0
    max_per_node: int = 0
    slurm_name: str = "gpu"

    def supports(self, count: int) -> bool:
        return self.nodes > 0 and self.max_per_node >= count


@dataclass
class JobIOPaths:
    stdout: Optional[str] = None
    stderr: Optional[str] = None


def _canonical_gpu_key(name: Optional[str]) -> str:
    if not name:
        return "gpu"
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return key or "gpu"


def _canonical_user_preference(name: str) -> str:
    return _canonical_gpu_key(name)


def _extract_partition(args: Iterable[str]) -> Optional[str]:
    args_list = list(args)
    for idx, arg in enumerate(args_list):
        if arg in {"--partition", "-p"} and idx + 1 < len(args_list):
            return args_list[idx + 1]
        if arg.startswith("--partition="):
            return arg.split("=", 1)[1]
    return None


def _has_partition_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg in {"--partition", "-p"}:
            return True
        if arg.startswith("--partition="):
            return True
        if arg.startswith("-p") and arg != "-p":
            return True
    return False


def _has_gres_flag(args: Iterable[str]) -> bool:
    for arg in args:
        if arg in {"--gres", "--gpus", "--gpus-per-node"}:
            return True
        if arg.startswith("--gres=") or arg.startswith("--gpus="):
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


def get_available_gpus(config: Config, partition: str = DEFAULT_PARTITION) -> Dict[str, GPUAvailability]:
    """
    Query available GPUs on the specified partition.

    Returns:
        Dict keyed by normalized GPU name containing node counts and the maximum
        number of GPUs available per node.
    """
    try:
        result = run_ssh(
            config,
            [
                "sinfo",
                "-p",
                partition,
                "--Format=nodehost,gres:30,statecompact",
                "--noheader",
            ],
            capture_output=True,
        )
    except Exception as exc:  # pragma: no cover - network error
        print(f"Warning: Could not detect available GPUs: {exc}")
        return {}

    inventory: Dict[str, GPUAvailability] = {}
    lines = result.stdout.strip().splitlines() if result.stdout else []

    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue

        gres_field = parts[1]
        state = parts[2].lower()

        if state not in {"idle", "mix", "mixed"}:
            continue

        for entry in gres_field.split(","):
            entry = entry.strip()
            if "(" in entry:
                entry = entry.split("(", 1)[0]
            entry_lower = entry.lower()
            if not entry_lower.startswith("gpu"):
                continue

            pieces = entry_lower.split(":")
            if len(pieces) == 3:
                _, raw_type, raw_count = pieces
                gpu_type = raw_type
                try:
                    count = int(raw_count)
                except ValueError:
                    count = 1
            elif len(pieces) == 2:
                _, raw_count = pieces
                gpu_type = None
                try:
                    count = int(raw_count)
                except ValueError:
                    count = 1
            else:
                continue

            slurm_name = (gpu_type or "gpu").lower()
            canonical = _canonical_gpu_key(gpu_type)
            record = inventory.setdefault(canonical, GPUAvailability(slurm_name=slurm_name))
            record.nodes += 1
            if count > record.max_per_node:
                record.max_per_node = count

    return inventory


def _select_highest_priority_gpu(availability: Dict[str, GPUAvailability]) -> Optional[str]:
    best_key: Optional[str] = None
    best_score = -1
    for key, stats in availability.items():
        if stats.nodes <= 0:
            continue
        score = GPU_PRIORITY.get(key, 0)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key


def select_gpu_request(
    config: Config,
    partition: str = DEFAULT_PARTITION,
    availability: Optional[Dict[str, GPUAvailability]] = None,
) -> GPURequest:
    availability = availability or get_available_gpus(config, partition)
    preferences = config.gpu_preferences.for_partition(partition) if config.gpu_preferences else []

    for preference in preferences:
        if preference.type is None:
            return GPURequest(type=None, count=preference.count)

        canonical = _canonical_user_preference(preference.type)
        stats = availability.get(canonical)
        if stats and stats.supports(preference.count):
            return GPURequest(type=stats.slurm_name, count=preference.count)
        if canonical not in availability:
            available = ", ".join(sorted(stat.slurm_name for stat in availability.values()))
            raise ValueError(
                f"Requested GPU '{preference.type}' not recognised. Available GPUs: {available}"
            )

    best_key = _select_highest_priority_gpu(availability)
    if best_key:
        slurm_name = availability[best_key].slurm_name
        return GPURequest(type=slurm_name, count=1)

    fallback_normalized = _normalize_gpu_key(FALLBACK_GPU)
    return GPURequest(type=_to_slurm_gpu_name(fallback_normalized), count=1)


def select_best_gpu(config: Config, partition: str = DEFAULT_PARTITION) -> str:
    """Compatibility wrapper that returns only the GPU type."""
    request = select_gpu_request(config, partition)
    if request.type:
        print(f"Auto-selected GPU request: gpu:{request.type}:{request.count}")
        return request.type
    print(f"Auto-selected generic GPU request: gpu:{request.count}")
    return "gpu"


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
    auto_gpu: bool = True,
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

    if auto_gpu and not _has_gres_flag(sbatch_args_list):
        partition = _extract_partition(sbatch_args_list) or DEFAULT_PARTITION
        request = select_gpu_request(config, partition)
        if request.type:
            args.extend(["--gres", f"gpu:{request.type}:{request.count}"])
        else:
            args.extend(["--gres", f"gpu:{request.count}"])

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
    job_id = match.group(1)

    if config.remote_results_dir:
        results_root = str(config.remote_results_dir)
        try:
            run_ssh(config, ["mkdir", "-p", f"{results_root}/{job_id}"])
        except SSHError:
            # Non-fatal: job script can still create directories if needed.
            pass

    return job_id


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


def gpu_display_name(normalized: str) -> str:
    """Return a friendly label for a normalized GPU identifier."""
    return _to_slurm_gpu_name(normalized).replace("_", " ")


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
    """
    Return the stdout/stderr paths configured for the given job.
    """
    try:
        result = run_ssh(
            config,
            ["scontrol", "show", "job", str(job_id)],
            capture_output=True,
        )
    except SSHError:
        raise

    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None

    tokens = result.stdout.replace("\n", " ").split()
    for token in tokens:
        if token.startswith("StdOut="):
            stdout_path = token.split("=", 1)[1] or None
        elif token.startswith("StdErr="):
            stderr_path = token.split("=", 1)[1] or None

    return JobIOPaths(stdout=stdout_path, stderr=stderr_path)
