"""Tests for config module."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from koi.config import Config, create_default_config


def test_config_initialization():
    """Test Config class initialization with defaults."""
    config = Config()

    assert config.api_base == ""
    assert config.model == "openai/openai/gpt-5.2-codex"
    assert config.max_tokens == 4096
    assert config.context_window == 128000
    assert config.skills_paths == [".agent/skills"]
    assert config.temperature is None


def test_config_custom_values():
    """Test Config class with custom values."""
    config = Config(
        api_base="https://custom.api.com/v1",
        model="custom-model",
        max_tokens=2048,
        temperature=0.5,
    )

    assert config.api_base == "https://custom.api.com/v1"
    assert config.model == "custom-model"
    assert config.max_tokens == 2048
    assert config.temperature == 0.5


def test_config_save_and_load():
    """Test saving and loading config."""
    with TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"

        # Create and save config
        original_config = Config(
            api_base="https://test.api.com/v1",
            api_key="test-key",
            model="test-model",
            max_tokens=1024,
        )
        original_config.save(config_path)

        # Load config
        loaded_config = Config.load(config_path)

        assert loaded_config.api_base == original_config.api_base
        assert loaded_config.api_key == original_config.api_key
        assert loaded_config.model == original_config.model
        assert loaded_config.max_tokens == original_config.max_tokens


def test_config_load_nonexistent():
    """Test loading config from non-existent file."""
    with pytest.raises(FileNotFoundError):
        Config.load(Path("/nonexistent/config.json"))


def test_create_default_config():
    """Test creating default configuration dictionary."""
    default_config = create_default_config()

    assert default_config["api_base"] == ""
    assert default_config["model"] == "openai/openai/gpt-5.2-codex"
    assert default_config["max_tokens"] == 4096
    assert default_config["context_window"] == 128000
    assert default_config["skills_paths"] == [".agent/skills"]
    assert "temperature" not in default_config


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = Config(
        api_base="https://test.com/v1", api_key="test-key", model="test-model"
    )

    config_dict = config.to_dict()

    assert config_dict["api_base"] == "https://test.com/v1"
    assert config_dict["api_key"] == "test-key"
    assert config_dict["model"] == "test-model"
