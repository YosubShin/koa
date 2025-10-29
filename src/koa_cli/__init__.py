"""
koa_cli package entrypoint.

Lightweight CLI and helpers for submitting and managing KOA SLURM jobs.
"""

from .config import (
    Config,
    GPUPreferences,
    GPURequest,
    discover_config_path,
    load_config,
)
from .manifest import update_manifest_metadata, write_run_manifest
from .runs import list_runs, record_submission, show_run, sync_statuses
from .slurm import (
    DEFAULT_PARTITION,
    JobIOPaths,
    cancel_job,
    get_available_gpus,
    get_job_io_paths,
    gpu_display_name,
    list_jobs,
    run_health_checks,
    select_best_gpu,
    select_gpu_request,
    submit_job,
)

__all__ = [
    "Config",
    "GPUPreferences",
    "GPURequest",
    "discover_config_path",
    "load_config",
    "write_run_manifest",
    "update_manifest_metadata",
    "record_submission",
    "list_runs",
    "show_run",
    "sync_statuses",
    "submit_job",
    "cancel_job",
    "list_jobs",
    "run_health_checks",
    "get_available_gpus",
    "get_job_io_paths",
    "gpu_display_name",
    "JobIOPaths",
    "select_gpu_request",
    "select_best_gpu",
    "DEFAULT_PARTITION",
]
