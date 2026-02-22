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
    ):
        self.api_base = api_base
        self.api_key = api_key or os.getenv("KOI_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.skills_paths = skills_paths or [".agent/skills"]
        self.temperature = temperature
    
    @classmethod
    def load(cls, config_path: Path = None) -> "Config":
        """Load configuration from .agent/config.json."""
        if config_path is None:
            config_path = Path.cwd() / ".agent" / "config.json"
        
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
            skills_paths=data.get("skills_paths", [".agent/skills"]),
            temperature=data.get("temperature"),
        )
    
    def save(self, config_path: Path = None):
        """Save configuration to .agent/config.json."""
        if config_path is None:
            config_path = Path.cwd() / ".agent" / "config.json"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "api_base": self.api_base,
            "api_key": self.api_key,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "skills_paths": self.skills_paths,
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
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        return d


def create_default_config() -> Dict[str, Any]:
    """Create a default configuration dictionary."""
    return {
        "api_base": "",
        "api_key": "",
        "model": "openai/openai/gpt-5.2-codex",
        "max_tokens": 4096,
        "context_window": 128000,
        "skills_paths": [".agent/skills"],
    }