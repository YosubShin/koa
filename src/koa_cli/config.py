from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/koa/config.yaml").expanduser()
LEGACY_CONFIG_PATHS = (
    Path("~/.config/koa-ml/config.yaml").expanduser(),
)
PROJECT_CONFIG_FILENAMES: tuple[str, ...] = ("koa-config.yaml", ".koa-config.yaml")


@dataclass
class Config:
    user: str
    host: str
    identity_file: Optional[Path] = None
    proxy_command: Optional[str] = None
    project_name: str = ""
    remote_root: Path = Path("~/koa-cli")
    local_root: Path = Path("./results")
    remote_project_root: Path = Path("~/koa-cli/projects/default")
    local_project_root: Path = Path("./results/projects/default")
    remote_code_dir: Path = Path("~/koa-cli/projects/default/jobs")
    remote_results_dir: Path = Path("~/koa-cli/projects/default/jobs")
    local_results_dir: Path = Path("./results/projects/default/jobs")
    shared_env_dir: Path = Path("~/koa-cli/projects/default/envs/uv")
    default_partition: Optional[str] = None
    python_module: Optional[str] = None
    cuda_module: Optional[str] = None
    env_watch_files: List[str] = field(default_factory=list)
    snapshot_excludes: List[str] = field(default_factory=list)

    @property
    def login(self) -> str:
        return f"{self.user}@{self.host}"

    @property
    def remote_workdir(self) -> Path:
        """Backward compatible alias for the legacy field name."""
        return self.remote_code_dir


PathLikeOrStr = Union[os.PathLike[str], str]


def discover_config_path(start: Optional[PathLikeOrStr] = None) -> Path:
    """
    Locate the project configuration file by walking parent directories from `start`.

    Falls back to ~/.config/koa/config.yaml and to the legacy ~/.config/koa-ml/config.yaml location when a project-level
    config is not available.
    """

    if start is None:
        current = Path.cwd().resolve()
    else:
        current = Path(start).expanduser().resolve()

    for directory in [current, *current.parents]:
        for candidate_name in PROJECT_CONFIG_FILENAMES:
            candidate = directory / candidate_name
            if candidate.exists():
                return candidate

    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    for legacy_path in LEGACY_CONFIG_PATHS:
        if legacy_path.exists():
            return legacy_path

    searched_locations = [
        str(Path.cwd().resolve() / name) for name in PROJECT_CONFIG_FILENAMES
    ]
    raise FileNotFoundError(
        "Unable to locate koa-config.yaml in this project. "
        f"Searched: {', '.join(searched_locations)}, {DEFAULT_CONFIG_PATH}, "
        + ", ".join(str(path) for path in LEGACY_CONFIG_PATHS)
        + ". Create one with `cp koa-config.example.yaml koa-config.yaml`."
    )


def load_config(config_path: Optional[PathLikeOrStr] = None) -> Config:
    """
    Load configuration from disk. When no path is provided we search for
    koa-config.yaml in the current project and fall back to ~/.config/koa/config.yaml
    or, for backwards compatibility, ~/.config/koa-ml/config.yaml.
    """

    if not DEFAULT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Global configuration file not found at {DEFAULT_CONFIG_PATH}. "
            "Run `koa setup` to create one."
        )

    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
        global_data = yaml.safe_load(fh) or {}

    project_config_path: Optional[Path] = None
    if config_path:
        project_config_path = Path(config_path).expanduser()
    else:
        current = Path.cwd().resolve()
        for directory in [current, *current.parents]:
            candidate = directory / "koa-config.yaml"
            if candidate.exists() and candidate != DEFAULT_CONFIG_PATH:
                project_config_path = candidate
                break

    project_data: dict = {}
    if project_config_path and project_config_path.exists() and project_config_path != DEFAULT_CONFIG_PATH:
        with project_config_path.open("r", encoding="utf-8") as fh:
            project_data = yaml.safe_load(fh) or {}

    def _merge_dicts(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = _merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    data = _merge_dicts(global_data, project_data)

    env_overrides = {
        "user": os.getenv("KOA_USER"),
        "host": os.getenv("KOA_HOST"),
        "identity_file": os.getenv("KOA_IDENTITY_FILE"),
        "remote_root": os.getenv("KOA_REMOTE_ROOT"),
        "local_root": os.getenv("KOA_LOCAL_ROOT"),
        "default_partition": os.getenv("KOA_DEFAULT_PARTITION"),
        "python_module": os.getenv("KOA_PYTHON_MODULE"),
        "cuda_module": os.getenv("KOA_CUDA_MODULE"),
        "env_watch": os.getenv("KOA_ENV_WATCH"),
        "snapshot_excludes": os.getenv("KOA_SNAPSHOT_EXCLUDES"),
        "proxy_command": os.getenv("KOA_PROXY_COMMAND"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            if key in {"env_watch", "snapshot_excludes"}:
                data[key] = [item.strip() for item in value.split(",") if item.strip()]
            else:
                data[key] = value

    missing = [key for key in ("user", "host") if not data.get(key)]
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    identity_file = data.get("identity_file") or None
    identity_path: Optional[Path] = None
    if identity_file:
        identity_path = Path(identity_file).expanduser()
        if not identity_path.exists():
            raise FileNotFoundError(
                f"Configured identity_file not found: {identity_path}. "
                "Update the path or remove the identity_file setting to rely on your SSH defaults."
            )

    config_dir = project_config_path.parent if project_config_path else Path.cwd()

    project_name = data.get("project")
    if not project_name:
        if project_config_path and project_config_path.parent != DEFAULT_CONFIG_PATH.parent:
            project_name = project_config_path.parent.name
        else:
            project_name = "default"

    remote_root_value = data.get("remote_root") or (data.get("remote") or {}).get("root")
    if not remote_root_value:
        raise ValueError("remote_root is not configured. Run `koa setup` or set remote_root in your global config.")
    remote_root = Path(remote_root_value).expanduser()

    local_root_value = data.get("local_root") or (data.get("local") or {}).get("root")
    if local_root_value:
        local_root = Path(local_root_value).expanduser()
        if not local_root.is_absolute():
            local_root = (config_dir / local_root).resolve()
    else:
        local_root = Path("./runs").resolve()

    remote_project_root = (remote_root / "projects" / project_name).resolve()
    remote_jobs_root = remote_project_root / "jobs"
    remote_env_dir = remote_project_root / "envs" / "uv"
    local_project_root = (local_root / "projects" / project_name).resolve()
    local_jobs_root = local_project_root / "jobs"

    env_watch_raw = data.get("env_watch") or data.get("env_watch_files") or []
    if isinstance(env_watch_raw, str):
        env_watch_files = [env_watch_raw]
    else:
        env_watch_files = list(env_watch_raw)

    snapshot_excludes_raw = data.get("snapshot_excludes") or []
    if isinstance(snapshot_excludes_raw, str):
        snapshot_excludes = [snapshot_excludes_raw]
    else:
        snapshot_excludes = list(snapshot_excludes_raw)

    return Config(
        user=data["user"],
        host=data["host"],
        identity_file=identity_path,
        proxy_command=data.get("proxy_command") or None,
        project_name=project_name,
        remote_root=remote_root,
        local_root=local_root,
        remote_project_root=remote_project_root,
        local_project_root=local_project_root,
        remote_code_dir=remote_jobs_root,
        remote_results_dir=remote_jobs_root,
        local_results_dir=local_jobs_root,
        shared_env_dir=remote_env_dir,
        default_partition=data.get("default_partition"),
        python_module=data.get("python_module"),
        cuda_module=data.get("cuda_module"),
        env_watch_files=env_watch_files,
        snapshot_excludes=snapshot_excludes,
    )
