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

    VERSION = 2

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
        self._entry_ids: set = set()
        self._leaf_id: Optional[str] = None

        if session_path:
            self._path = session_path
            self._session_id = self._extract_session_id(session_path)

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def session_path(self) -> Optional[Path]:
        return self._path

    @property
    def leaf_id(self) -> Optional[str]:
        return self._leaf_id

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
        """Open an existing session file for appending.

        Also populates _entry_ids and _leaf_id from existing entries
        so that new entries chain correctly via parentId.
        """
        if not self._path or not self._path.exists():
            raise FileNotFoundError(f"Session file not found: {self._path}")

        if not self._entry_ids:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    eid = entry.get("id")
                    if eid and entry.get("type") != "session":
                        self._entry_ids.add(eid)
                        self._leaf_id = eid

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

    def load_session(self, path: Optional[Path] = None, leaf_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a session from a JSONL file.

        If leaf_id is provided, walks from that entry to root to get the active branch.
        Otherwise uses the last entry as the leaf (default linear behavior).
        """
        target = path or self._path
        if not target or not target.exists():
            raise FileNotFoundError(f"Session file not found: {target}")

        header = None
        all_entries = []

        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") == "session":
                    header = entry
                else:
                    all_entries.append(entry)

        by_id = {e["id"]: e for e in all_entries if "id" in e}

        if leaf_id and leaf_id in by_id:
            leaf = by_id[leaf_id]
        elif all_entries:
            leaf = all_entries[-1]
        else:
            leaf = None

        if leaf and "id" in leaf:
            branch = []
            current = leaf
            while current:
                branch.append(current)
                pid = current.get("parentId")
                current = by_id.get(pid) if pid else None
            branch.reverse()
        else:
            branch = all_entries

        messages = []
        compactions = []
        model_changes = []

        for entry in branch:
            entry_type = entry.get("type")
            if entry_type == "message":
                messages.append(entry["message"])
            elif entry_type == "compaction":
                compactions.append(entry)
            elif entry_type == "model_change":
                model_changes.append(entry)

        self._entry_ids = {e["id"] for e in all_entries if "id" in e}
        if leaf and "id" in leaf:
            self._leaf_id = leaf["id"]

        return {
            "header": header,
            "messages": messages,
            "compactions": compactions,
            "model_changes": model_changes,
        }

    def fork(self, fork_from_id: Optional[str] = None) -> None:
        """Fork the session from a specific entry (or current leaf).

        After forking, new entries will branch from the fork point.
        The session file stays the same -- branching is in-place via parentId.
        """
        if fork_from_id:
            if fork_from_id not in self._entry_ids:
                raise ValueError(f"Entry ID not found: {fork_from_id}")
            self._leaf_id = fork_from_id

    def get_branches(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Get entries that have multiple children (branch points).

        Returns list of entry dicts that are parentId of more than one entry.
        """
        target = path or self._path
        if not target or not target.exists():
            return []

        all_entries = []
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("type") != "session" and "id" in entry:
                    all_entries.append(entry)

        children_count: Dict[str, int] = {}
        for e in all_entries:
            pid = e.get("parentId")
            if pid:
                children_count[pid] = children_count.get(pid, 0) + 1

        by_id = {e["id"]: e for e in all_entries}
        return [by_id[eid] for eid in children_count if children_count[eid] > 1 and eid in by_id]

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent sessions with metadata.

        Returns a list of session header dicts, most recent first.
        Each dict also includes a 'path' field with the file path and
        'message_count' with the number of messages.
        """
        sessions = []

        if not self._sessions_dir.exists():
            return sessions

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

    def close(self) -> None:
        """Flush and close the session file."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def _generate_id(self) -> str:
        """Generate a unique 8-char hex entry ID."""
        while True:
            eid = uuid.uuid4().hex[:8]
            if eid not in self._entry_ids:
                self._entry_ids.add(eid)
                return eid

    def _write_entry(self, entry: Dict[str, Any]) -> None:
        """Write a single JSONL entry."""
        if self._file is None:
            return
        if entry.get("type") != "session":
            entry["id"] = self._generate_id()
            entry["parentId"] = self._leaf_id
            self._leaf_id = entry["id"]
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    @staticmethod
    def _extract_session_id(path: Path) -> str:
        """Extract session ID from filename like 20260303_211500_abc123def456.jsonl"""
        stem = path.stem
        parts = stem.split("_")
        return parts[-1] if len(parts) >= 3 else stem
