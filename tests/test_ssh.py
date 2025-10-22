"""Tests for the ssh module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from koa_ml.config import Config
from koa_ml.ssh import SSHError, copy_from_remote, copy_to_remote, run_ssh


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        user="testuser",
        host="koa.its.hawaii.edu",
        identity_file=Path("/fake/id_rsa"),
        remote_code_dir=Path("/home/testuser/koa-ml"),
        remote_data_dir=Path("/mnt/lustre/koa/scratch/testuser/koa-ml"),
        proxy_command="ssh -W %h:%p jumphost",
    )


@pytest.fixture
def mock_config_minimal():
    """Create a minimal mock config without optional fields."""
    return Config(
        user="testuser",
        host="koa.its.hawaii.edu",
    )


class TestRunSSH:
    """Test the run_ssh function."""

    def test_run_ssh_with_string_command(self, mock_config_minimal, mocker):
        """Test running SSH with a string command."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="output", stderr="")

        result = run_ssh(mock_config_minimal, "hostname", capture_output=True)

        assert result.returncode == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "ssh" in call_args
        assert "testuser@koa.its.hawaii.edu" in call_args
        assert "hostname" in call_args

    def test_run_ssh_with_list_command(self, mock_config_minimal, mocker):
        """Test running SSH with a list command."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_ssh(mock_config_minimal, ["ls", "-la", "/tmp"], capture_output=True)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        # Should properly quote arguments
        assert "ls -la /tmp" in " ".join(call_args)

    def test_run_ssh_with_identity_file(self, mock_config, mocker):
        """Test SSH command includes identity file."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_ssh(mock_config, "hostname", capture_output=True)

        call_args = mock_run.call_args[0][0]
        assert "-i" in call_args
        assert "/fake/id_rsa" in call_args

    def test_run_ssh_with_proxy_command(self, mock_config, mocker):
        """Test SSH command includes proxy command."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        run_ssh(mock_config, "hostname", capture_output=True)

        call_args = mock_run.call_args[0][0]
        proxy_found = any("ProxyCommand=" in arg for arg in call_args)
        assert proxy_found

    def test_run_ssh_failure_raises_error(self, mock_config_minimal, mocker):
        """Test SSH failure raises SSHError."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=255, stdout="", stderr="Connection failed"
        )

        with pytest.raises(SSHError, match="SSH command failed"):
            run_ssh(mock_config_minimal, "hostname", capture_output=True)

    def test_run_ssh_no_check(self, mock_config_minimal, mocker):
        """Test SSH with check=False doesn't raise on failure."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Some error"
        )

        result = run_ssh(
            mock_config_minimal, "hostname", check=False, capture_output=True
        )

        assert result.returncode == 1
        # Should not raise


class TestCopyToRemote:
    """Test the copy_to_remote function."""

    def test_copy_file_to_remote(self, mock_config_minimal, mocker, tmp_path):
        """Test copying a file to remote."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")
        remote_path = Path("/remote/test.txt")

        copy_to_remote(mock_config_minimal, local_file, remote_path)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "scp" in call_args
        assert str(local_file) in call_args
        assert "testuser@koa.its.hawaii.edu:/remote/test.txt" in call_args

    def test_copy_directory_to_remote(self, mock_config_minimal, mocker, tmp_path):
        """Test copying a directory to remote with recursive flag."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        local_dir = tmp_path / "testdir"
        local_dir.mkdir()
        remote_path = Path("/remote/testdir")

        copy_to_remote(mock_config_minimal, local_dir, remote_path, recursive=True)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-r" in call_args

    def test_copy_to_remote_failure(self, mock_config_minimal, mocker, tmp_path):
        """Test SCP failure raises SSHError."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="Permission denied"
        )

        local_file = tmp_path / "test.txt"
        local_file.write_text("content")
        remote_path = Path("/remote/test.txt")

        with pytest.raises(SSHError, match="SCP upload failed"):
            copy_to_remote(mock_config_minimal, local_file, remote_path)


class TestCopyFromRemote:
    """Test the copy_from_remote function."""

    def test_copy_file_from_remote(self, mock_config_minimal, mocker, tmp_path):
        """Test copying a file from remote."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        remote_path = Path("/remote/test.txt")
        local_file = tmp_path / "test.txt"

        copy_from_remote(mock_config_minimal, remote_path, local_file)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "scp" in call_args
        assert "testuser@koa.its.hawaii.edu:/remote/test.txt" in call_args
        assert str(local_file) in call_args

    def test_copy_directory_from_remote(self, mock_config_minimal, mocker, tmp_path):
        """Test copying a directory from remote with recursive flag."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        remote_path = Path("/remote/testdir")
        local_dir = tmp_path / "testdir"

        copy_from_remote(mock_config_minimal, remote_path, local_dir, recursive=True)

        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert "-r" in call_args

    def test_copy_from_remote_failure(self, mock_config_minimal, mocker, tmp_path):
        """Test SCP download failure raises SSHError."""
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="No such file"
        )

        remote_path = Path("/remote/test.txt")
        local_file = tmp_path / "test.txt"

        with pytest.raises(SSHError, match="SCP download failed"):
            copy_from_remote(mock_config_minimal, remote_path, local_file)
