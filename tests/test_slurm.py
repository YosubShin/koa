"""Tests for the slurm module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from koa_ml.config import Config
from koa_ml.slurm import (
    cancel_job,
    ensure_remote_workspace,
    list_jobs,
    run_health_checks,
    submit_job,
)
from koa_ml.ssh import SSHError


@pytest.fixture
def mock_config():
    """Create a mock config for testing."""
    return Config(
        user="testuser",
        host="koa.its.hawaii.edu",
        remote_workdir=Path("/home/testuser/koa-ml"),
    )


class TestEnsureRemoteWorkspace:
    """Test the ensure_remote_workspace function."""

    def test_creates_remote_directory(self, mock_config, mocker):
        """Test that it creates the remote workspace directory."""
        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")

        ensure_remote_workspace(mock_config)

        mock_run_ssh.assert_called_once()
        call_args = mock_run_ssh.call_args[0]
        assert call_args[0] == mock_config
        assert "mkdir" in call_args[1]
        assert "-p" in call_args[1]


class TestSubmitJob:
    """Test the submit_job function."""

    def test_submit_job_success(self, mock_config, mocker, tmp_path):
        """Test successful job submission."""
        # Create a fake job script
        job_script = tmp_path / "test.slurm"
        job_script.write_text("#!/bin/bash\necho hello")

        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")
        mock_copy = mocker.patch("koa_ml.slurm.copy_to_remote")

        # Mock sbatch output
        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 12345\n"
        mock_run_ssh.return_value = mock_result

        job_id = submit_job(mock_config, job_script)

        assert job_id == "12345"
        mock_copy.assert_called_once()
        # Verify sbatch was called
        sbatch_call = [c for c in mock_run_ssh.call_args_list if "sbatch" in str(c)]
        assert len(sbatch_call) > 0
        call_args = sbatch_call[-1].args[1]
        assert "--partition" in call_args
        assert "kill-shared" in call_args

    def test_submit_job_with_sbatch_args(self, mock_config, mocker, tmp_path):
        """Test job submission with additional sbatch arguments."""
        job_script = tmp_path / "test.slurm"
        job_script.write_text("#!/bin/bash\necho hello")

        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")
        mock_copy = mocker.patch("koa_ml.slurm.copy_to_remote")

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 12345\n"
        mock_run_ssh.return_value = mock_result

        sbatch_args = ["--partition", "gpu", "--gres=gpu:1"]
        job_id = submit_job(mock_config, job_script, sbatch_args=sbatch_args)

        assert job_id == "12345"
        # Check that sbatch was called with the right args
        call_args = mock_run_ssh.call_args_list[-1].args[1]
        assert "--partition" in call_args
        assert "gpu" in call_args
        assert "kill-shared" not in call_args
        assert "--gres=gpu:1" in call_args

    def test_submit_job_with_remote_name(self, mock_config, mocker, tmp_path):
        """Test job submission with custom remote filename."""
        job_script = tmp_path / "test.slurm"
        job_script.write_text("#!/bin/bash\necho hello")

        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")
        mock_copy = mocker.patch("koa_ml.slurm.copy_to_remote")

        mock_result = MagicMock()
        mock_result.stdout = "Submitted batch job 12345\n"
        mock_run_ssh.return_value = mock_result

        submit_job(mock_config, job_script, remote_name="custom.slurm")

        # Verify the remote path uses the custom name
        call_args = mock_copy.call_args[0]
        remote_path = call_args[2]
        assert remote_path.name == "custom.slurm"

    def test_submit_job_nonexistent_script(self, mock_config):
        """Test error when job script doesn't exist."""
        nonexistent = Path("/nonexistent/script.slurm")

        with pytest.raises(FileNotFoundError, match="Job script not found"):
            submit_job(mock_config, nonexistent)

    def test_submit_job_parse_error(self, mock_config, mocker, tmp_path):
        """Test error when sbatch output can't be parsed."""
        job_script = tmp_path / "test.slurm"
        job_script.write_text("#!/bin/bash\necho hello")

        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")
        mocker.patch("koa_ml.slurm.copy_to_remote")

        mock_result = MagicMock()
        mock_result.stdout = "Unexpected output format\n"
        mock_run_ssh.return_value = mock_result

        with pytest.raises(SSHError, match="Unable to parse sbatch output"):
            submit_job(mock_config, job_script)


class TestCancelJob:
    """Test the cancel_job function."""

    def test_cancel_job(self, mock_config, mocker):
        """Test cancelling a job."""
        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")

        cancel_job(mock_config, "12345")

        mock_run_ssh.assert_called_once()
        call_args = mock_run_ssh.call_args[0][1]
        assert "scancel" in call_args
        assert "12345" in call_args


class TestListJobs:
    """Test the list_jobs function."""

    def test_list_jobs(self, mock_config, mocker):
        """Test listing jobs."""
        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")

        mock_result = MagicMock()
        mock_result.stdout = "JOBID|NAME|STATE|TIME|TIME_LIMIT|NODES|NODELIST\n12345|test|RUNNING|01:23|1:00:00|1|gpu001\n"
        mock_run_ssh.return_value = mock_result

        output = list_jobs(mock_config)

        assert "12345" in output
        assert "test" in output
        assert "RUNNING" in output
        mock_run_ssh.assert_called_once()
        call_args = mock_run_ssh.call_args[0][1]
        assert "squeue" in call_args
        assert "-u" in call_args
        assert "testuser" in call_args


class TestRunHealthChecks:
    """Test the run_health_checks function."""

    def test_run_health_checks(self, mock_config, mocker):
        """Test running health checks."""
        mock_run_ssh = mocker.patch("koa_ml.slurm.run_ssh")

        mock_result = MagicMock()
        mock_result.stdout = "== hostname ==\nkoa001\n== sinfo ==\ngpu available ...\n"
        mock_run_ssh.return_value = mock_result

        output = run_health_checks(mock_config)

        assert "hostname" in output
        assert "sinfo" in output
        assert "koa001" in output
        mock_run_ssh.assert_called_once()
        call_args = mock_run_ssh.call_args[0][1]
        assert "bash" in call_args
        assert "-lc" in call_args
