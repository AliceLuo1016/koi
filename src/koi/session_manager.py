"""Session persistence — save and load conversation state as JSONL files."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionManager:
    """Manages session persistence as JSONL files.

    Session files are stored at .koi/sessions/<timestamp>_<uuid>.jsonl
    Each line is a JSON object with a 'type' field.

    Entry types:
    - session: Header with metadata (first line, written once)
    - message: A conversation message (user, assistant, or tool)
    - compaction: Context compaction summary
    - model_change: Model switch event
    """

    VERSION = 1

    def __init__(self, koi_dir: Path, session_path: Optional[Path] = None):
        """Initialize SessionManager.

        Args:
            koi_dir: Path to .koi directory
            session_path: If provided, load/append to this session file.
                         If None, creates a new session.
        """
        self._koi_dir = koi_dir
        self._sessions_dir = koi_dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._path: Optional[Path] = None
        self._session_id: Optional[str] = None

        if session_path:
            self._path = session_path
            self._session_id = self._extract_session_id(session_path)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def session_path(self) -> Optional[Path]:
        return self._path

    def start_session(self, model: str, cwd: Optional[str] = None) -> str:
        """Start a new session. Writes the session header.

        Returns the session ID.
        """
        self._session_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{self._session_id}.jsonl"
        self._path = self._sessions_dir / filename

        self._file = open(self._path, "a", encoding="utf-8")

        header = {
            "type": "session",
            "version": self.VERSION,
            "id": self._session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cwd": cwd or str(Path.cwd()),
            "model": model,
        }
        self._write_entry(header)

        return self._session_id

    def resume_session(self) -> None:
        """Open an existing session file for appending."""
        if not self._path or not self._path.exists():
            raise FileNotFoundError(f"Session file not found: {self._path}")
        self._file = open(self._path, "a", encoding="utf-8")

    def save_message(self, message: Dict[str, Any]) -> None:
        """Persist a conversation message (user, assistant, or tool)."""
        entry = {
            "type": "message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
        }
        self._write_entry(entry)

    def save_compaction(self, summary: str, tokens_before: int) -> None:
        """Persist a compaction event."""
        entry = {
            "type": "compaction",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary,
            "tokens_before": tokens_before,
        }
        self._write_entry(entry)

    def save_model_change(self, old_model: str, new_model: str) -> None:
        """Persist a model change event."""
        entry = {
            "type": "model_change",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_model": old_model,
            "new_model": new_model,
        }
        self._write_entry(entry)

    def load_session(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Load a session from a JSONL file.

        Returns a dict with:
        - header: session header dict
        - messages: list of message dicts (ready for agent.messages)
        - compactions: list of compaction dicts
        - model_changes: list of model change dicts
        """
        target = path or self._path
        if not target or not target.exists():
            raise FileNotFoundError(f"Session file not found: {target}")

        header = None
        messages = []
        compactions = []
        model_changes = []

        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")
                if entry_type == "session":
                    header = entry
                elif entry_type == "message":
                    messages.append(entry["message"])
                elif entry_type == "compaction":
                    compactions.append(entry)
                elif entry_type == "model_change":
                    model_changes.append(entry)

        return {
            "header": header,
            "messages": messages,
            "compactions": compactions,
            "model_changes": model_changes,
        }

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions with metadata.

        Returns a list of session header dicts, most recent first.
        Each dict also includes a 'path' field with the file path and
        'message_count' with the number of messages.
        """
        sessions = []

        if not self._sessions_dir.exists():
            return sessions

        # Sort by filename (which starts with timestamp) descending
        files = sorted(
            self._sessions_dir.glob("*.jsonl"),
            key=lambda f: f.name,
            reverse=True,
        )

        for f in files[:limit]:
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    first_line = fh.readline().strip()
                    if not first_line:
                        continue
                    header = json.loads(first_line)
                    if header.get("type") != "session":
                        continue

                    # Count messages
                    message_count = 0
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            if entry.get("type") == "message":
                                message_count += 1
                        except json.JSONDecodeError:
                            continue

                    header["path"] = str(f)
                    header["message_count"] = message_count
                    sessions.append(header)
            except (json.JSONDecodeError, OSError):
                continue

        return sessions

    def get_latest_session(self) -> Optional[Path]:
        """Get the path to the most recent session file, or None."""
        if not self._sessions_dir.exists():
            return None

        files = sorted(
            self._sessions_dir.glob("*.jsonl"),
            key=lambda f: f.name,
            reverse=True,
        )

        return files[0] if files else None

    def fork_session(self, messages: List[Dict[str, Any]], model: str, cwd: Optional[str] = None) -> str:
        """Fork: close current session, create a new one with copied messages.

        Returns the new session ID.
        """
        self.close()
        new_id = self.start_session(model=model, cwd=cwd)
        for msg in messages:
            self.save_message(msg)
        return new_id

    def close(self) -> None:
        """Flush and close the session file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write a single JSONL entry."""
        if self._file is None:
            return
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    @staticmethod
    def _extract_session_id(path: Path) -> str:
        """Extract session ID from filename like 20260303_211500_abc123def456.jsonl"""
        stem = path.stem  # e.g. "20260303_211500_abc123def456"
        parts = stem.split("_")
        return parts[-1] if len(parts) >= 3 else stem
