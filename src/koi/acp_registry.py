"""ACP agent registry — maps agent names to their spawn commands."""

import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentEntry:
    """An ACP-compatible agent."""

    name: str
    display_name: str
    command: List[str]
    check_binary: Optional[str] = None

    def is_available(self) -> bool:
        """Check if this agent's binary is installed."""
        if not self.check_binary:
            return True
        return shutil.which(self.check_binary) is not None


BUILTIN_AGENTS: Dict[str, AgentEntry] = {
    "claude-code": AgentEntry(
        name="claude-code",
        display_name="Claude Code",
        command=["claude", "--acp"],
        check_binary="claude",
    ),
    "codex": AgentEntry(
        name="codex",
        display_name="Codex CLI",
        command=["codex", "--acp"],
        check_binary="codex",
    ),
    "gemini": AgentEntry(
        name="gemini",
        display_name="Gemini CLI",
        command=["gemini", "--acp"],
        check_binary="gemini",
    ),
    "opencode": AgentEntry(
        name="opencode",
        display_name="OpenCode",
        command=["opencode", "--acp"],
        check_binary="opencode",
    ),
    "goose": AgentEntry(
        name="goose",
        display_name="Goose",
        command=["goose", "--acp"],
        check_binary="goose",
    ),
}


def get_agent(name: str, custom_agents: Optional[Dict] = None) -> Optional[AgentEntry]:
    """Look up an agent by name. Custom agents override builtins."""
    if custom_agents and name in custom_agents:
        entry = custom_agents[name]
        if isinstance(entry, dict):
            return AgentEntry(
                name=name,
                display_name=entry.get("display_name", name),
                command=entry.get("command", []),
                check_binary=entry.get("check_binary"),
            )
        return entry
    return BUILTIN_AGENTS.get(name)


def list_agents(custom_agents: Optional[Dict] = None) -> List[AgentEntry]:
    """List all known agents (builtins + custom)."""
    agents = dict(BUILTIN_AGENTS)
    if custom_agents:
        for name, entry in custom_agents.items():
            if isinstance(entry, dict):
                agents[name] = AgentEntry(
                    name=name,
                    display_name=entry.get("display_name", name),
                    command=entry.get("command", []),
                    check_binary=entry.get("check_binary"),
                )
            else:
                agents[name] = entry
    return list(agents.values())


def list_available_agents(custom_agents: Optional[Dict] = None) -> List[AgentEntry]:
    """List only agents whose binary is installed."""
    return [a for a in list_agents(custom_agents) if a.is_available()]
