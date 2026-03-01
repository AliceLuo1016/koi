"""Token usage tracking and cost estimation."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


# Per 1M tokens pricing (approximate)
_PRICING = {
    "claude-opus-4": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.5,
        "cache_write": 18.75,
    },
    "claude-sonnet-4": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "claude-3-5-sonnet": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.3,
        "cache_write": 3.75,
    },
    "gpt-5.2": {"input": 2.5, "output": 10.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "o3": {"input": 10.0, "output": 40.0},
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
) -> float:
    """Estimate cost in USD based on model pricing.

    Returns 0.0 for unknown models.
    """
    model_lower = model.lower()
    rates = None
    for prefix, r in _PRICING.items():
        if prefix in model_lower:
            rates = r
            break

    if not rates:
        return 0.0

    return (
        input_tokens * rates.get("input", 0) / 1_000_000
        + output_tokens * rates.get("output", 0) / 1_000_000
        + cache_read * rates.get("cache_read", 0) / 1_000_000
        + cache_creation * rates.get("cache_write", 0) / 1_000_000
    )


@dataclass
class TokenUsage:
    """Accumulates token usage across API calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        input_t: int = 0,
        output_t: int = 0,
        cache_read: int = 0,
        cache_creation: int = 0,
    ):
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        self.total_requests += 1

    def summary(self, model: str = "") -> str:
        lines = [
            "Token Usage:",
            f"  Input:  {self.input_tokens:,} tokens",
            f"  Output: {self.output_tokens:,} tokens",
            f"  Total:  {self.total_tokens:,} tokens",
            f"  Requests: {self.total_requests}",
        ]
        if self.cache_read_tokens:
            lines.append(f"  Cache read:     {self.cache_read_tokens:,} tokens")
        if self.cache_creation_tokens:
            lines.append(
                f"  Cache creation: {self.cache_creation_tokens:,} tokens"
            )

        cost = estimate_cost(
            model,
            self.input_tokens,
            self.output_tokens,
            self.cache_read_tokens,
            self.cache_creation_tokens,
        )
        if cost > 0:
            lines.append(f"  Est. cost: ${cost:.4f}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_requests": self.total_requests,
        }


def log_usage(usage: TokenUsage, model: str, log_dir: Path) -> None:
    """Append session usage to .koi/usage-log.jsonl."""
    if usage.total_requests == 0:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "usage-log.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "session_tokens": usage.to_dict(),
        "estimated_cost": estimate_cost(
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_tokens,
            usage.cache_creation_tokens,
        ),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
