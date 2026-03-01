"""Sandbox security — restricts file access, env vars, and shell commands."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


class Sandbox:
    """Enforces security boundaries for tool execution."""

    def __init__(self, project_root: Path = None):
        self.project_root = (project_root or Path.cwd()).resolve()
        self._load_config()

    def _load_config(self):
        config_path = self.project_root / ".koi" / "sandbox.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {}

        fs = cfg.get("filesystem", {})
        env = cfg.get("environment", {})
        cmd = cfg.get("commands", {})

        # Resolve paths
        self.allowed_paths = [self._resolve(p) for p in fs.get("allowed_paths", ["."])]
        self.readonly_paths = [self._resolve(p) for p in fs.get("readonly_paths", [])]
        self.blocked_paths = [self._resolve(p) for p in fs.get("blocked_paths", [])]
        

        self.env_allowlist = set(env.get("allowlist", ["PATH", "HOME", "USER", "SHELL", "LANG", "TERM", "SSH_AUTH_SOCK", "SSH_AGENT_PID"]))

        self.blocked_patterns = [re.compile(p, re.IGNORECASE) for p in cmd.get("blocked_patterns", [])]
        self.confirm_patterns = [re.compile(p, re.IGNORECASE) for p in cmd.get("confirm_patterns", [])]

    def _resolve(self, p: str) -> Path:
        """Resolve a path, expanding ~ and making relative paths project-relative."""
        expanded = Path(os.path.expanduser(p))
        if expanded.is_absolute():
            return expanded.resolve()
        return (self.project_root / expanded).resolve()

    # ── File access checks ──

    def check_read(self, path: str) -> Tuple[bool, Optional[str]]:
        """Check if reading a path is allowed. Returns (allowed, reason)."""
        resolved = self._resolve_file_path(path)
        return self._check_file_access(resolved, write=False)

    def check_write(self, path: str) -> Tuple[bool, Optional[str]]:
        """Check if writing to a path is allowed. Returns (allowed, reason)."""
        resolved = self._resolve_file_path(path)
        return self._check_file_access(resolved, write=True)

    def _resolve_file_path(self, path: str) -> Path:
        p = Path(os.path.expanduser(path))
        if p.is_absolute():
            return p.resolve()
        return (self.project_root / p).resolve()

    def _check_file_access(self, resolved: Path, write: bool) -> Tuple[bool, Optional[str]]:
        # Blocked paths always win
        for blocked in self.blocked_paths:
            try:
                resolved.relative_to(blocked)
                return False, f"Access denied: {resolved} is in blocked path {blocked}"
            except ValueError:
                continue

        # Check allowed paths (read+write)
        for allowed in self.allowed_paths:
            try:
                resolved.relative_to(allowed)
                return True, None
            except ValueError:
                continue

        # Check readonly paths (read only)
        if not write:
            for ro in self.readonly_paths:
                try:
                    resolved.relative_to(ro)
                    return True, None
                except ValueError:
                    continue

        action = "write" if write else "read"
        return False, f"Access denied: {resolved} is outside allowed paths ({action})"

    # ── Environment sandboxing ──

    def get_safe_env(self) -> Dict[str, str]:
        """Return a sanitized copy of os.environ with only allowlisted vars."""
        env = {k: v for k, v in os.environ.items() if k in self.env_allowlist}
        return env

    # ── Command checks ──

    def check_command(self, command: str) -> Tuple[bool, Optional[str], bool]:
        """
        Check a shell command.
        Returns (allowed, reason, needs_confirm).
        If allowed=False, the command is blocked.
        If needs_confirm=True, the command should ask for user confirmation.
        """
        for pattern in self.blocked_patterns:
            if pattern.search(command):
                return False, f"Blocked command pattern: {pattern.pattern}", False

        for pattern in self.confirm_patterns:
            if pattern.search(command):
                return True, f"Requires confirmation: {pattern.pattern}", True

        return True, None, False
