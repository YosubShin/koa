from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from .config import Config
from .ssh import copy_from_remote, run_ssh

RUN_INDEX_FILENAME = "runs.json"


def _index_path(local_results_dir: Path) -> Path:
    return local_results_dir / RUN_INDEX_FILENAME


def _load_index(local_results_dir: Path) -> Dict[str, dict]:
    path = _index_path(local_results_dir)
    if not path.exists():
        return {"version": 1, "runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"version": 1, "runs": {}}
    if "runs" not in data or not isinstance(data["runs"], dict):
        data["runs"] = {}
    return data


def _save_index(local_results_dir: Path, data: Dict[str, dict]) -> None:
    path = _index_path(local_results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def record_submission(
    config: Config,
    *,
    job_id: str,
    sbatch_args: list[str],
    manifest: dict,
    local_job_dir: Optional[Path],
    remote_job_dir: Optional[Path],
) -> None:
    if not config.local_results_dir:
        return
    local_results = config.local_results_dir.expanduser()
    index = _load_index(local_results)
    runs = index.setdefault("runs", {})

    env_hashes = manifest.get("env_hashes") or {}
    entry = {
        "job_id": job_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "user": config.user,
        "host": config.host,
        "status": "UNKNOWN",
        "sbatch_args": sbatch_args,
        "env_hashes": env_hashes,
        "remote_job_dir": str(remote_job_dir) if remote_job_dir else None,
        "local_job_dir": str(local_job_dir) if local_job_dir else None,
        "git": manifest.get("git"),
    }
    runs[job_id] = entry
    _save_index(local_results, index)


def list_runs(config: Config) -> list[dict]:
    if not config.local_results_dir:
        return []
    index = _load_index(config.local_results_dir.expanduser())
    runs = index.get("runs", {})
    entries = list(runs.values())
    entries.sort(key=lambda item: item.get("submitted_at") or "", reverse=True)
    return entries


def show_run(config: Config, job_id: str) -> Optional[dict]:
    runs = list_runs(config)
    for entry in runs:
        if entry.get("job_id") == job_id:
            return entry
    return None


def _batched(iterable: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def _squeue_states(config: Config, job_ids: Iterable[str]) -> Dict[str, str]:
    ids = list(job_ids)
    states: Dict[str, str] = {}
    for chunk in _batched(ids, 50):
        id_csv = ",".join(chunk)
        result = run_ssh(
            config,
            [
                "squeue",
                "-h",
                "-j",
                id_csv,
                "-o",
                r"%i|%T",
            ],
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2:
                states[parts[0]] = parts[1]
    return states


def _sacct_states(config: Config, job_ids: Iterable[str]) -> Dict[str, str]:
    ids = list(job_ids)
    states: Dict[str, str] = {}
    for chunk in _batched(ids, 50):
        id_csv = ",".join(chunk)
        result = run_ssh(
            config,
            [
                "sacct",
                "-P",
                "-n",
                "-j",
                id_csv,
                "-o",
                "JobIDRaw,State",
            ],
            capture_output=True,
            check=False,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|", 1)
            if len(parts) == 2 and parts[0].isdigit():
                states[parts[0]] = parts[1]
    return states


def sync_statuses(config: Config) -> int:
    if not config.local_results_dir:
        return 0
    local_results = config.local_results_dir.expanduser()
    index = _load_index(local_results)
    runs = index.get("runs", {})
    if not runs:
        return 0

    job_ids = list(runs.keys())
    active_states = _squeue_states(config, job_ids)
    historical_states = _sacct_states(config, job_ids)
    updates = 0

    for job_id, entry in runs.items():
        state = active_states.get(job_id) or historical_states.get(job_id)
        if state and entry.get("status") != state:
            entry["status"] = state
            updates += 1

        if state and state.upper().startswith("COMPLET"):
            entry.setdefault("synced_at", None)
            if entry.get("synced_at") is None:
                remote_dir = entry.get("remote_job_dir")
                local_dir = entry.get("local_job_dir")
                if remote_dir and local_dir:
                    remote_path = Path(remote_dir)
                    local_path = Path(local_dir).expanduser()
                    local_path.mkdir(parents=True, exist_ok=True)
                    try:
                        copy_from_remote(
                            config,
                            remote_path,
                            local_path,
                            recursive=True,
                        )
                        entry["synced_at"] = datetime.now(timezone.utc).isoformat()
                    except Exception as exc:
                        entry.setdefault("sync_errors", []).append(str(exc))

    if updates:
        _save_index(local_results, index)
    return updates
