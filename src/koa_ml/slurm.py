from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

from .config import Config
from .ssh import SSHError, copy_to_remote, run_ssh

SBATCH_JOB_ID_PATTERN = re.compile(r"Submitted batch job (\d+)")
DEFAULT_PARTITION = "kill-shared"

# GPU priority ranking (higher score = better GPU)
GPU_PRIORITY = {
    "h200": 110,           # NVIDIA H200 (newest, best)
    "nvidiah200nvl": 110,  # H200 NVL variant
    "h100": 100,           # NVIDIA H100
    "nvidiah100": 100,     # H100 variant
    "a100": 90,            # NVIDIA A100
    "nvidiaa100": 90,      # A100 variant
    "a30": 80,             # NVIDIA A30
    "nvidiaa30": 80,       # A30 variant
    "v100": 70,            # NVIDIA V100
    "nvidiav100": 70,      # V100 variant
    "rtx2080ti": 50,       # RTX 2080 Ti
    "rtx_2080_ti": 50,
    "geforce_rtx_2080_ti": 50,
    "nvidiageforcertx2080ti": 50,
}

# Default GPU to request if detection fails
FALLBACK_GPU = "rtx2080ti"

# Map detected GPU names to SLURM GRES names (what SLURM expects in --gres)
GPU_NAME_MAP = {
    "nvidiah200nvl": "nvidia_h200_nvl",  # Normalize to SLURM format
    "nvidiah100": "nvidia_h100",
    "nvidiaa100": "nvidia_a100",
    "nvidiaa30": "nvidia_a30",
    "nvidiav100": "nvidia_v100",
    "nvidiageforcertx2080ti": "geforce_rtx_2080_ti",
}


def _has_partition_flag(args: Iterable[str]) -> bool:
    """Return True if any sbatch argument sets the partition."""
    for arg in args:
        if arg in {"--partition", "-p"}:
            return True
        if arg.startswith("--partition="):
            return True
        if arg.startswith("-p") and arg != "-p":
            return True
    return False


def _has_gres_flag(args: Iterable[str]) -> bool:
    """Return True if any sbatch argument sets the GPU/GRES."""
    for arg in args:
        if arg in {"--gres", "--gpus", "--gpus-per-node"}:
            return True
        if arg.startswith("--gres=") or arg.startswith("--gpus="):
            return True
    return False


def get_available_gpus(config: Config, partition: str = DEFAULT_PARTITION) -> Dict[str, int]:
    """
    Query available GPUs on KOA partition.

    Returns:
        Dict mapping GPU type to count of available nodes
        Example: {"h100": 2, "a100": 5, "rtx2080ti": 10}
    """
    try:
        # Query partition info with GPU details
        result = run_ssh(
            config,
            [
                "sinfo",
                "-p", partition,
                "--Format=nodehost,gres:30,statecompact",
                "--noheader"
            ],
            capture_output=True,
        )

        gpu_counts: Dict[str, int] = {}
        lines = result.stdout.strip().split("\n") if result.stdout else []

        for line in lines:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            # parts[1] contains GRES info like "gpu:h100:1" or "gpu:rtx2080ti:1"
            gres = parts[1].lower()
            state = parts[2].lower()

            # Only count idle or mixed nodes (available for jobs)
            if state not in ["idle", "mix", "mixed"]:
                continue

            # Extract GPU type from GRES (e.g., "gpu:h100:1" -> "h100")
            if "gpu:" in gres:
                gpu_parts = gres.split(":")
                if len(gpu_parts) >= 2:
                    gpu_type = gpu_parts[1].replace("_", "")  # Normalize names
                    gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0) + 1

        return gpu_counts

    except Exception as e:
        # On any error, return empty dict (will fall back to default)
        print(f"Warning: Could not detect available GPUs: {e}")
        return {}


def select_best_gpu(config: Config, partition: str = DEFAULT_PARTITION) -> str:
    """
    Automatically select the best available GPU based on priority ranking.

    Returns:
        GPU type string in SLURM GRES format (e.g., "nvidia_h100", "a100", "rtx2080ti")
    """
    available = get_available_gpus(config, partition)

    if not available:
        print(f"No GPU info available, using fallback: {FALLBACK_GPU}")
        return FALLBACK_GPU

    # Find highest priority GPU that's available
    best_gpu = None
    best_score = -1

    for gpu_type, count in available.items():
        if count > 0:  # Has available nodes
            score = GPU_PRIORITY.get(gpu_type, 0)
            if score > best_score:
                best_score = score
                best_gpu = gpu_type

    if best_gpu:
        # Map to SLURM GRES format if needed
        slurm_gpu_name = GPU_NAME_MAP.get(best_gpu, best_gpu)
        print(f"Auto-selected GPU: {best_gpu} -> {slurm_gpu_name} (priority: {best_score}, {available[best_gpu]} nodes available)")
        return slurm_gpu_name

    print(f"No recognized GPUs available, using fallback: {FALLBACK_GPU}")
    return FALLBACK_GPU


def ensure_remote_workspace(config: Config) -> None:
    run_ssh(config, ["mkdir", "-p", str(config.remote_code_dir)])

    data_root = str(config.remote_data_dir)
    if data_root not in {"", "."}:
        run_ssh(config, ["mkdir", "-p", data_root])
        run_ssh(config, ["mkdir", "-p", f"{data_root}/train/results"])
        run_ssh(config, ["mkdir", "-p", f"{data_root}/eval/results"])


def submit_job(
    config: Config,
    local_job_script: Path,
    *,
    sbatch_args: Optional[Iterable[str]] = None,
    remote_name: Optional[str] = None,
    auto_gpu: bool = True,
) -> str:
    """
    Copy a job script to KOA and submit it with sbatch. Returns the job id.

    Args:
        config: KOA configuration
        local_job_script: Path to SLURM script to submit
        sbatch_args: Additional sbatch arguments
        remote_name: Remote filename (default: same as local)
        auto_gpu: If True, automatically select best available GPU (default: True)
                 Set to False to use GPU specified in script or sbatch_args
    """
    if not local_job_script.exists():
        raise FileNotFoundError(f"Job script not found: {local_job_script}")

    ensure_remote_workspace(config)

    remote_script = config.remote_code_dir / (remote_name or local_job_script.name)
    copy_to_remote(config, local_job_script, remote_script)

    args = [
        "env",
        f"KOA_ML_CODE_ROOT={config.remote_code_dir}",
        f"KOA_ML_DATA_ROOT={config.remote_data_dir}",
        "sbatch",
    ]
    sbatch_args_list = list(sbatch_args or [])

    # Add default partition if not specified
    if not _has_partition_flag(sbatch_args_list):
        args.extend(["--partition", DEFAULT_PARTITION])

    # Auto-select best GPU if enabled and no GPU already specified
    if auto_gpu and not _has_gres_flag(sbatch_args_list):
        # Determine partition for GPU query
        partition = DEFAULT_PARTITION
        for i, arg in enumerate(sbatch_args_list):
            if arg in {"--partition", "-p"} and i + 1 < len(sbatch_args_list):
                partition = sbatch_args_list[i + 1]
            elif arg.startswith("--partition="):
                partition = arg.split("=", 1)[1]

        best_gpu = select_best_gpu(config, partition)
        args.extend(["--gres", f"gpu:{best_gpu}:1"])

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
