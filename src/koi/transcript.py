"""Debug transcript logger for post-mortem debugging.

Writes every LLM message (sent and received) to .koi/transcript.jsonl
in append mode. Each line is a JSON object with a timestamp, event type,
and payload. No-op when disabled.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class TranscriptLogger:
    """Lightweight JSONL transcript logger for debug sessions."""

    def __init__(self, koi_dir: Path, enabled: bool = False):
        self._enabled = enabled
        self._path = koi_dir / "transcript.jsonl" if enabled else None
        self._file = None

        if enabled:
            koi_dir.mkdir(parents=True, exist_ok=True)
            self._file = open(self._path, "a", encoding="utf-8")

    # -- public API --

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Write a single event line to the transcript."""
        if not self._enabled:
            return
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "data": data,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()

    def log_session_start(self, metadata: dict[str, Any]) -> None:
        """Log a session_start event (model, api_format, system_prompt hash, etc.)."""
        self.log_event("session_start", metadata)

    def log_message(self, event_type: str, message: dict[str, Any]) -> None:
        """Log a user_message, assistant_message, tool_call, or tool_result."""
        self.log_event(event_type, message)

    def log_compaction(self, before_count: int, after_count: int) -> None:
        """Log a compaction event."""
        self.log_event(
            "compaction",
            {
                "before_message_count": before_count,
                "after_message_count": after_count,
            },
        )

    def log_pruning(self, before_chars: int, after_chars: int) -> None:
        """Log a pruning event."""
        self.log_event(
            "pruning",
            {
                "before_chars": before_chars,
                "after_chars": after_chars,
            },
        )

    def close(self) -> None:
        """Flush and close the transcript file."""
        if self._file:
            self._file.close()
            self._file = None
