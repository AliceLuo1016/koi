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
    assert config.skills_paths == [".koi/skills"]
    assert config.temperature is None
    assert config.api_format == "responses"


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
            api_format="chat_completions",
        )
        original_config.save(config_path)

        # Load config
        loaded_config = Config.load(config_path)

        assert loaded_config.api_base == original_config.api_base
        assert loaded_config.api_key == original_config.api_key
        assert loaded_config.model == original_config.model
        assert loaded_config.max_tokens == original_config.max_tokens
        assert loaded_config.api_format == "chat_completions"


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
    assert default_config["skills_paths"] == [".koi/skills"]
    assert default_config["api_format"] == "responses"
    assert "temperature" not in default_config


def test_create_default_config_with_params():
    """Test creating config with custom parameters."""
    cfg = create_default_config(
        model="us/aws/anthropic/bedrock-claude-opus-4-6",
        api_base="https://inference-api.nvidia.com/v1/chat/completions",
        api_key="sk-test",
        api_format="chat_completions",
        context_window=200000,
    )

    assert cfg["model"] == "us/aws/anthropic/bedrock-claude-opus-4-6"
    assert cfg["api_base"] == "https://inference-api.nvidia.com/v1/chat/completions"
    assert cfg["api_key"] == "sk-test"
    assert cfg["api_format"] == "chat_completions"
    assert cfg["context_window"] == 200000


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = Config(api_base="https://test.com/v1", api_key="test-key", model="test-model")

    config_dict = config.to_dict()

    assert config_dict["api_base"] == "https://test.com/v1"
    assert config_dict["api_key"] == "test-key"
    assert config_dict["model"] == "test-model"
    assert config_dict["api_format"] == "responses"


def test_api_format_auto_detect_anthropic():
    """Test that api_format auto-detects 'anthropic' for anthropic model names."""
    config = Config(model="us/aws/anthropic/bedrock-claude-opus-4-6")
    assert config.api_format == "anthropic"


def test_api_format_auto_detect_claude():
    """Test that api_format auto-detects 'anthropic' for claude model names."""
    config = Config(model="claude-3-opus")
    assert config.api_format == "anthropic"


def test_api_format_auto_detect_openai():
    """Test that api_format defaults to responses for non-anthropic models."""
    config = Config(model="openai/openai/gpt-5.2-codex")
    assert config.api_format == "responses"


def test_api_format_explicit_override():
    """Test that explicit api_format overrides auto-detection."""
    config = Config(
        model="us/aws/anthropic/bedrock-claude-opus-4-6",
        api_format="responses",
    )
    assert config.api_format == "responses"


def test_temperature_included_in_to_dict():
    """temperature appears in to_dict() only when set."""
    config_with_temp = Config(temperature=0.7)
    d = config_with_temp.to_dict()
    assert d["temperature"] == 0.7

    config_no_temp = Config()
    d2 = config_no_temp.to_dict()
    assert "temperature" not in d2


def test_temperature_included_in_save():
    """temperature is written to JSON only when set."""
    import json

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "config.json"

        Config(temperature=0.3).save(path)
        data = json.loads(path.read_text())
        assert data["temperature"] == 0.3

        path.unlink()
        Config().save(path)
        data2 = json.loads(path.read_text())
        assert "temperature" not in data2


def test_load_uses_default_path(monkeypatch, tmp_path):
    """Config.load() with no argument reads from cwd/.koi/config.json."""
    import json

    koi_dir = tmp_path / ".koi"
    koi_dir.mkdir()
    cfg_file = koi_dir / "config.json"
    cfg_file.write_text(json.dumps({"api_base": "https://x.com", "api_key": "", "model": "m"}))
    monkeypatch.chdir(tmp_path)
    config = Config.load()
    assert config.api_base == "https://x.com"
