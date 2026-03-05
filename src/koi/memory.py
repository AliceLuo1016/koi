"""Memory management for koi agent."""

from datetime import date, timedelta
from pathlib import Path


class Memory:
    """Handle loading and saving persistent memory."""

    def __init__(self, memory_path: Path | None = None):
        """Initialize memory with path to MEMORY.md file."""
        if memory_path is None:
            memory_path = Path.cwd() / ".koi" / "MEMORY.md"
        self.memory_path = memory_path
        self._memory_dir = self.memory_path.parent / "memory"

    def load(self) -> str:
        """Load memory content from MEMORY.md file."""
        if not self.memory_path.exists():
            return ""

        try:
            with open(self.memory_path, encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to load memory from {self.memory_path}: {e}")

    def save(self, content: str):
        """Save content to MEMORY.md file."""
        try:
            # Ensure parent directory exists
            self.memory_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.memory_path, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            raise RuntimeError(f"Failed to save memory to {self.memory_path}: {e}")

    def append(self, content: str):
        """Append content to MEMORY.md file."""
        existing_content = self.load()

        # Add newlines for proper formatting
        if existing_content and not existing_content.endswith("\n"):
            existing_content += "\n"

        if content and not content.startswith("\n"):
            content = "\n" + content

        new_content = existing_content + content
        self.save(new_content)

    def exists(self) -> bool:
        """Check if memory file exists."""
        return self.memory_path.exists()

    def get_path(self) -> Path:
        """Get the path to the memory file."""
        return self.memory_path

    # ── Daily logs ────────────────────────────────────────

    def _daily_log_path(self, day: date | None = None) -> Path:
        """Get path to a daily log file."""
        if day is None:
            day = date.today()
        return self._memory_dir / f"{day.isoformat()}.md"

    def load_daily(self, day: date | None = None) -> str:
        """Load a daily log file. Returns empty string if not found."""
        path = self._daily_log_path(day)
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def append_daily(self, content: str, day: date | None = None):
        """Append content to today's daily log."""
        path = self._daily_log_path(day)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if path.exists():
            try:
                existing = path.read_text(encoding="utf-8")
            except OSError:
                existing = ""

        if existing and not existing.endswith("\n"):
            existing += "\n"
        if content and not content.startswith("\n"):
            content = "\n" + content

        path.write_text(existing + content, encoding="utf-8")

    def load_recent_daily(self) -> str:
        """Load today and yesterday's daily logs concatenated."""
        today = date.today()
        yesterday = today - timedelta(days=1)
        parts = []
        for day in [yesterday, today]:
            text = self.load_daily(day)
            if text.strip():
                parts.append(f"## {day.isoformat()}\n{text}")
        return "\n\n".join(parts)
