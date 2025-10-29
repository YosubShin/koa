from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import yaml

DEFAULT_CONFIG_PATH = Path("~/.config/koa/config.yaml").expanduser()
LEGACY_CONFIG_PATHS = (
    Path("~/.config/koa-ml/config.yaml").expanduser(),
)
PROJECT_CONFIG_FILENAMES: tuple[str, ...] = ("koa-config.yaml", ".koa-config.yaml")


@dataclass
class GPURequest:
    """Desired GPU resource request."""

    type: Optional[str] = None  # SLURM GRES type (e.g. nvidia_h100). None = any GPU.
    count: int = 1


@dataclass
class GPUPreferences:
    """Ordered GPU preferences used by auto-selection."""

    default: List[GPURequest] = field(default_factory=list)
    partitions: Dict[str, List[GPURequest]] = field(default_factory=dict)

    def for_partition(self, partition: Optional[str]) -> List[GPURequest]:
        if partition and partition in self.partitions:
            return self.partitions[partition]
        return self.default


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
    gpu_preferences: GPUPreferences = field(default_factory=GPUPreferences)
    default_partition: Optional[str] = None
    python_module: Optional[str] = None
    cuda_module: Optional[str] = None
    env_watch_files: list[str] = field(default_factory=list)

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


def _parse_gpu_request(entry: Union[str, Dict[str, object]]) -> GPURequest:
    if isinstance(entry, str):
        cleaned = entry.strip()
        if not cleaned or cleaned.lower() in {"any", "auto"}:
            return GPURequest(type=None, count=1)
        return GPURequest(type=cleaned, count=1)

    if not isinstance(entry, dict):
        raise TypeError(f"Unsupported GPU preference entry: {entry!r}")

    gpu_type = entry.get("type") or entry.get("name") or entry.get("gres")
    if isinstance(gpu_type, str):
        gpu_type = gpu_type.strip()
        if gpu_type.lower() in {"", "any", "auto"}:
            gpu_type = None
    elif gpu_type is not None:
        raise TypeError(f"GPU preference type must be a string, got {gpu_type!r}")

    count_raw = entry.get("count", 1)
    try:
        count = int(count_raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"GPU preference count must be an integer (got {count_raw!r})") from exc
    if count < 1:
        count = 1

    return GPURequest(type=gpu_type, count=count)


def _parse_gpu_preferences(data: Optional[dict]) -> GPUPreferences:
    if not data:
        return GPUPreferences()

    def parse_list(values: Optional[Iterable[Union[str, Dict[str, object]]]]) -> List[GPURequest]:
        if values is None:
            return []
        if isinstance(values, (str, dict)):
            values = [values]
        result: List[GPURequest] = []
        for value in values:
            result.append(_parse_gpu_request(value))
        return result

    default_prefs = parse_list(data.get("default") or data.get("defaults"))
    partition_prefs: Dict[str, List[GPURequest]] = {}
    partitions_section = data.get("partitions") or {}
    if isinstance(partitions_section, dict):
        for partition, prefs in partitions_section.items():
            partition_prefs[str(partition)] = parse_list(prefs)

    return GPUPreferences(default=default_prefs, partitions=partition_prefs)


def load_config(config_path: Optional[PathLikeOrStr] = None) -> Config:
    """
    Load configuration from disk. When no path is provided we search for
    koa-config.yaml in the current project and fall back to ~/.config/koa/config.yaml
    or, for backwards compatibility, ~/.config/koa-ml/config.yaml.
    """

    global_data = {}
    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            global_data = yaml.safe_load(fh) or {}
    else:
        raise FileNotFoundError(
            f"Global configuration file not found at {DEFAULT_CONFIG_PATH}. "
            "Run `koa setup` to create one."
        )

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
        "remote_workdir": os.getenv("KOA_REMOTE_WORKDIR"),
        "remote_results_dir": os.getenv("KOA_REMOTE_RESULTS_DIR"),
        "local_results_dir": os.getenv("KOA_LOCAL_RESULTS_DIR"),
        "remote_root": os.getenv("KOA_REMOTE_ROOT"),
        "local_root": os.getenv("KOA_LOCAL_ROOT"),
        "default_partition": os.getenv("KOA_DEFAULT_PARTITION"),
        "python_module": os.getenv("KOA_PYTHON_MODULE"),
        "cuda_module": os.getenv("KOA_CUDA_MODULE"),
        "env_watch": os.getenv("KOA_ENV_WATCH"),
        "proxy_command": os.getenv("KOA_PROXY_COMMAND"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            if key == "env_watch":
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

    remote_root_value = (
        data.get("remote_root")
        or (data.get("remote") or {}).get("root")
    )
    if not remote_root_value:
        raise ValueError("remote_root is not configured. Run `koa setup` or set remote_root in your global config.")
    remote_root = Path(remote_root_value).expanduser()

    local_root_value = (
        data.get("local_root")
        or (data.get("local") or {}).get("root")
    )
    if local_root_value:
        local_root = Path(local_root_value).expanduser()
        if not local_root.is_absolute():
            local_root = (config_dir / local_root).resolve()
    else:
        local_root = Path("./runs").resolve()

    remote_project_root = (remote_root / "projects" / project_name).resolve()
    remote_jobs_root = remote_project_root / "jobs"
    local_project_root = (local_root / "projects" / project_name).resolve()
    local_jobs_root = local_project_root / "jobs"

    gpu_preferences = _parse_gpu_preferences(data.get("gpu_preferences"))

    default_partition = data.get("default_partition")
    python_module = data.get("python_module")
    cuda_module = data.get("cuda_module")
    modules_section = data.get("modules") or {}
    if not python_module:
        python_module = modules_section.get("python")
    if not cuda_module:
        cuda_module = modules_section.get("cuda")

    env_watch_raw = data.get("env_watch") or data.get("env_watch_files") or []
    env_watch_files: list[str]
    if isinstance(env_watch_raw, str):
        env_watch_files = [env_watch_raw]
    else:
        env_watch_files = list(env_watch_raw)

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
        gpu_preferences=gpu_preferences,
        default_partition=default_partition,
        python_module=python_module,
        cuda_module=cuda_module,
        env_watch_files=env_watch_files,
    )
