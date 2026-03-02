"""Configuration management for koi agent."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

# Valid thinking levels (off disables, others set increasing reasoning effort)
THINK_LEVELS = ("off", "minimal", "low", "medium", "high")


def normalize_think_level(value: str) -> Optional[str]:
    """Normalize a user-provided thinking level string to a canonical level.

    Returns None for unrecognized inputs.
    """
    v = value.strip().lower()
    mapping = {
        "off": "off",
        "disabled": "off",
        "none": "off",
        "on": "low",
        "enable": "low",
        "enabled": "low",
        "min": "minimal",
        "minimal": "minimal",
        "think": "minimal",
        "low": "low",
        "med": "medium",
        "mid": "medium",
        "medium": "medium",
        "high": "high",
        "max": "high",
        "ultra": "high",
    }
    return mapping.get(v)


def load_claude_code_api_key() -> Optional[str]:
    """Load the Anthropic API key from Claude Code's config (~/.claude.json)."""
    claude_json = Path.home() / ".claude.json"
    if not claude_json.exists():
        return None
    try:
        with open(claude_json, "r") as f:
            data = json.load(f)
        key = data.get("primaryApiKey", "")
        if key and key.startswith("sk-ant-"):
            return key
    except (json.JSONDecodeError, OSError):
        pass
    return None


class Config:
    """Configuration for the koi agent."""

    def __init__(
        self,
        api_base: str = "",
        api_key: str = "",
        model: str = "openai/openai/gpt-5.2-codex",
        max_tokens: int = 4096,
        context_window: int = 128000,
        skills_paths: List[str] = None,
        temperature: float = None,
        api_format: str = None,
        thinking_level: str = "low",
        prompt_caching: bool = True,
        server: Dict[str, Any] = None,
        channels: Dict[str, Any] = None,
    ):
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.skills_paths = skills_paths or [".koi/skills"]
        self.temperature = temperature
        self.thinking_level = thinking_level if thinking_level in THINK_LEVELS else "low"
        self.prompt_caching = prompt_caching
        # Auto-detect api_format from model name if not explicitly set
        if api_format is not None:
            self.api_format = api_format
        elif "anthropic" in model or "claude" in model:
            self.api_format = "anthropic"
        else:
            self.api_format = "responses"

        # Server config (for `koi serve`)
        _server = server or {}
        self.server_enabled: bool = _server.get("enabled", False)
        self.server_host: str = _server.get("host", "0.0.0.0")
        self.server_port: int = _server.get("port", 8080)

        # Channel configs
        _channels = channels or {}
        self.channels: Dict[str, Any] = _channels

        # Resolve API key: explicit > env var > Claude Code config (for anthropic)
        if api_key:
            self.api_key = api_key
        elif os.getenv("KOI_API_KEY", ""):
            self.api_key = os.getenv("KOI_API_KEY", "")
        elif self.api_format == "anthropic":
            self.api_key = load_claude_code_api_key() or ""
        else:
            self.api_key = ""
    
    @classmethod
    def load(cls, config_path: Path = None) -> "Config":
        """Load configuration from .koi/config.json."""
        if config_path is None:
            config_path = Path.cwd() / ".koi" / "config.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, "r") as f:
            data = json.load(f)
        
        return cls(
            api_base=data.get("api_base", ""),
            api_key=data.get("api_key", ""),
            model=data.get("model", "openai/openai/gpt-5.2-codex"),
            max_tokens=data.get("max_tokens", 4096),
            context_window=data.get("context_window", 128000),
            skills_paths=data.get("skills_paths", [".koi/skills"]),
            temperature=data.get("temperature"),
            api_format=data.get("api_format"),
            thinking_level=data.get("thinking_level", "low"),
            prompt_caching=data.get("prompt_caching", True),
            server=data.get("server"),
            channels=data.get("channels"),
        )
    
    def save(self, config_path: Path = None):
        """Save configuration to .koi/config.json."""
        if config_path is None:
            config_path = Path.cwd() / ".koi" / "config.json"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "skills_paths": self.skills_paths,
            "api_format": self.api_format,
            "thinking_level": self.thinking_level,
            "prompt_caching": self.prompt_caching,
        }
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.server_enabled or self.server_host != "0.0.0.0" or self.server_port != 8080:
            data["server"] = {
                "enabled": self.server_enabled,
                "host": self.server_host,
                "port": self.server_port,
            }
        if self.channels:
            data["channels"] = self.channels

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def spawn_depth(self) -> int:
        """Current sub-agent spawn depth (from KOI_SPAWN_DEPTH env var)."""
        return int(os.environ.get("KOI_SPAWN_DEPTH", "0"))

    def effective_thinking_level(self) -> str:
        """Return thinking_level, or 'off' if the model doesn't support it."""
        if self.thinking_level == "off":
            return "off"
        # Lazy import to avoid circular dependency (llm imports config)
        from .llm import supports_thinking
        if supports_thinking(self.model, self.api_format):
            return self.thinking_level
        return "off"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        d = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "skills_paths": self.skills_paths,
            "api_format": self.api_format,
            "thinking_level": self.thinking_level,
            "prompt_caching": self.prompt_caching,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.server_enabled or self.server_host != "0.0.0.0" or self.server_port != 8080:
            d["server"] = {
                "enabled": self.server_enabled,
                "host": self.server_host,
                "port": self.server_port,
            }
        if self.channels:
            d["channels"] = self.channels
        return d


def create_default_config(
    model: str = "openai/openai/gpt-5.2-codex",
    api_base: str = "",
    api_key: str = "",
    api_format: str = "responses",
    context_window: int = 128000,
) -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        "api_base": api_base,
        "api_key": api_key,
        "model": model,
        "max_tokens": 4096,
        "context_window": context_window,
        "skills_paths": [".koi/skills"],
        "api_format": api_format,
    }