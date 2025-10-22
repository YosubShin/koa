"""Tests for the config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from koa_ml.config import Config, load_config


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "user": "testuser",
        "host": "koa.its.hawaii.edu",
        "remote_workdir": "/home/testuser/koa-ml",
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def temp_config_with_identity(tmp_path):
    """Create a temporary config file with identity_file."""
    # Create a fake SSH key file
    ssh_key = tmp_path / "id_rsa"
    ssh_key.write_text("fake key")

    config_path = tmp_path / "config.yaml"
    config_data = {
        "user": "testuser",
        "host": "koa.its.hawaii.edu",
        "identity_file": str(ssh_key),
        "remote_workdir": "/home/testuser/koa-ml",
        "proxy_command": "ssh -W %h:%p jumphost",
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestConfig:
    """Test the Config dataclass."""

    def test_config_creation(self):
        """Test creating a Config object."""
        config = Config(user="testuser", host="koa.its.hawaii.edu")
        assert config.user == "testuser"
        assert config.host == "koa.its.hawaii.edu"
        assert config.identity_file is None
        assert config.remote_workdir == Path("~/koa-ml")
        assert config.remote_data_dir == Path()
        assert config.proxy_command is None

    def test_login_property(self):
        """Test the login property."""
        config = Config(user="testuser", host="koa.its.hawaii.edu")
        assert config.login == "testuser@koa.its.hawaii.edu"

    def test_remote_workdir_not_expanded(self):
        """Test that remote_workdir is not expanded locally."""
        config = Config(user="testuser", host="koa.its.hawaii.edu")
        # Should remain as tilde, not expanded to local user's home
        assert str(config.remote_workdir) == "~/koa-ml"


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_basic_config(self, temp_config_file):
        """Test loading a basic config file."""
        config = load_config(temp_config_file)
        assert config.user == "testuser"
        assert config.host == "koa.its.hawaii.edu"
        assert config.identity_file is None
        assert config.remote_workdir == Path("/home/testuser/koa-ml")
        assert config.remote_data_dir == Path("/mnt/lustre/koa/scratch/testuser/koa-ml")

    def test_load_config_with_all_fields(self, temp_config_with_identity):
        """Test loading a config with all optional fields."""
        config = load_config(temp_config_with_identity)
        assert config.user == "testuser"
        assert config.host == "koa.its.hawaii.edu"
        assert config.identity_file is not None
        assert config.identity_file.exists()
        assert config.remote_workdir == Path("/home/testuser/koa-ml")
        assert config.remote_data_dir == Path("/mnt/lustre/koa/scratch/testuser/koa-ml")
        assert config.proxy_command == "ssh -W %h:%p jumphost"

    def test_missing_config_file(self, tmp_path):
        """Test error when config file doesn't exist."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            load_config(nonexistent)

    def test_missing_required_fields(self, tmp_path):
        """Test error when required fields are missing."""
        config_path = tmp_path / "config.yaml"
        config_data = {"user": "testuser"}  # Missing 'host'
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="Missing required config keys"):
            load_config(config_path)

    def test_missing_identity_file(self, tmp_path):
        """Test error when identity_file doesn't exist."""
        config_path = tmp_path / "config.yaml"
        config_data = {
            "user": "testuser",
            "host": "koa.its.hawaii.edu",
            "identity_file": "/nonexistent/key",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(FileNotFoundError, match="identity_file not found"):
            load_config(config_path)

    def test_env_overrides(self, temp_config_file, monkeypatch):
        """Test environment variable overrides."""
        monkeypatch.setenv("KOA_USER", "envuser")
        monkeypatch.setenv("KOA_HOST", "env.host.edu")
        monkeypatch.setenv("KOA_REMOTE_WORKDIR", "/env/workdir")
        monkeypatch.setenv("KOA_REMOTE_DATA_DIR", "/scratch/custom")

        config = load_config(temp_config_file)
        assert config.user == "envuser"
        assert config.host == "env.host.edu"
        assert config.remote_workdir == Path("/env/workdir")
        assert config.remote_data_dir == Path("/scratch/custom")

    def test_empty_config_file(self, tmp_path):
        """Test error with empty config file."""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("")

        with pytest.raises(ValueError, match="Missing required config keys"):
            load_config(config_path)

    def test_tilde_expansion_in_identity_file(self, tmp_path, monkeypatch):
        """Test that ~ is expanded in identity_file paths."""
        # Create a fake SSH key in the temp directory
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        ssh_dir = fake_home / ".ssh"
        ssh_dir.mkdir()
        ssh_key = ssh_dir / "id_rsa"
        ssh_key.write_text("fake key")

        # Mock expanduser to use our fake home
        original_expanduser = Path.expanduser

        def mock_expanduser(self):
            if str(self).startswith("~"):
                return fake_home / str(self)[2:]
            return original_expanduser(self)

        monkeypatch.setattr(Path, "expanduser", mock_expanduser)

        config_path = tmp_path / "config.yaml"
        config_data = {
            "user": "testuser",
            "host": "koa.its.hawaii.edu",
            "identity_file": "~/.ssh/id_rsa",
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.identity_file == fake_home / ".ssh/id_rsa"
