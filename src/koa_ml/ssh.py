from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from .config import Config


class SSHError(RuntimeError):
    """Raised when an SSH command returns a non-zero exit status."""


def _base_args(config: Config) -> List[str]:
    term_value = os.environ.get("TERM") or "xterm-256color"
    args = ["ssh", "-tt", "-o", f"SetEnv=TERM={term_value}", "-o", "LogLevel=ERROR"]
    if config.identity_file:
        args.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        args.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return args


def _scp_base_args(config: Config) -> List[str]:
    args = ["scp", "-o", "LogLevel=ERROR"]
    if config.identity_file:
        args.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        args.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return args


def _rsync_ssh_command(config: Config) -> str:
    ssh_parts = ["ssh", "-o", "LogLevel=ERROR"]
    if config.identity_file:
        ssh_parts.extend(["-i", str(config.identity_file)])
    if config.proxy_command:
        ssh_parts.extend(["-o", f"ProxyCommand={config.proxy_command}"])
    return " ".join(shlex.quote(part) for part in ssh_parts)


def run_ssh(
    config: Config,
    remote_command: Union[Iterable[str], str],
    *,
    check: bool = True,
    capture_output: bool = False,
    text: bool = True,
) -> subprocess.CompletedProcess:
    """
    Execute `remote_command` on the KOA host via ssh.
    """
    if isinstance(remote_command, str):
        command_str = remote_command
    else:
        command_str = " ".join(shlex.quote(part) for part in remote_command)

    ssh_command = [*_base_args(config), config.login, command_str]
    result = subprocess.run(
        ssh_command,
        check=False,
        capture_output=capture_output,
        text=text,
    )
    if check and result.returncode != 0:
        raise SSHError(
            f"SSH command failed ({result.returncode}): {' '.join(ssh_command)}\n"
            f"stderr: {result.stderr}"
        )
    return result


def copy_to_remote(
    config: Config,
    local_path: Path,
    remote_path: Path,
    *,
    recursive: bool = False,
) -> None:
    """
    Copy a local file or directory to the KOA host via scp.
    """
    args = _scp_base_args(config)
    if recursive:
        args.append("-r")
    scp_command = [
        *args,
        str(local_path),
        f"{config.login}:{remote_path}",
    ]
    result = subprocess.run(scp_command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise SSHError(
            f"SCP upload failed ({result.returncode}): {' '.join(scp_command)}\n"
            f"stderr: {result.stderr}"
        )


def copy_from_remote(
    config: Config,
    remote_path: Path,
    local_path: Path,
    *,
    recursive: bool = False,
) -> None:
    """
    Copy a file or directory from the KOA host to the local machine via scp.
    """
    args = _scp_base_args(config)
    if recursive:
        args.append("-r")
    scp_command = [
        *args,
        f"{config.login}:{remote_path}",
        str(local_path),
    ]
    result = subprocess.run(scp_command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise SSHError(
            f"SCP download failed ({result.returncode}): {' '.join(scp_command)}\n"
            f"stderr: {result.stderr}"
        )


def sync_directory_to_remote(
    config: Config,
    local_dir: Path,
    remote_dir: Path,
    *,
    excludes: Optional[Sequence[str]] = None,
) -> None:
    """
    Synchronize a local directory to the remote workdir via rsync.
    """
    local_dir = local_dir.expanduser().resolve()
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    excludes = excludes or []

    # Ensure the remote directory exists
    run_ssh(config, ["mkdir", "-p", str(remote_dir)])

    ssh_command = _rsync_ssh_command(config)

    rsync_command: list[str] = [
        "rsync",
        "-av",
        "--delete",
    ]
    for pattern in excludes:
        rsync_command.extend(["--exclude", pattern])

    rsync_command.extend(
        [
            "-e",
            ssh_command,
            f"{str(local_dir)}/",
            f"{config.login}:{remote_dir}",
        ]
    )

    result = subprocess.run(
        rsync_command,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SSHError(
            f"rsync failed ({result.returncode}): {' '.join(rsync_command)}\n"
            f"stderr: {result.stderr}"
        )
