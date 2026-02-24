"""Configuration management for koi agent."""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


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
    ):
        self.api_base = api_base
        self.api_key = api_key or os.getenv("KOI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.skills_paths = skills_paths or [".koi/skills"]
        self.temperature = temperature
        # Auto-detect api_format from model name if not explicitly set
        if api_format is not None:
            self.api_format = api_format
        elif "anthropic" in model or "claude" in model:
            self.api_format = "chat_completions"
        else:
            self.api_format = "responses"
    
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
        }
        if self.temperature is not None:
            data["temperature"] = self.temperature

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)

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
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
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