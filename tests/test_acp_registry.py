"""Tests for ACP agent registry."""

from unittest.mock import patch
from koi.acp_registry import (
    AgentEntry, BUILTIN_AGENTS, get_agent, list_agents,
    list_available_agents,
)


class TestAgentEntry:
    def test_is_available_with_binary(self):
        entry = AgentEntry(name="test", display_name="Test", command=["test"], check_binary="python3")
        assert entry.is_available() is True

    def test_is_unavailable(self):
        entry = AgentEntry(name="test", display_name="Test", command=["test"], check_binary="nonexistent_binary_xyz")
        assert entry.is_available() is False

    def test_no_check_binary_always_available(self):
        entry = AgentEntry(name="test", display_name="Test", command=["test"])
        assert entry.is_available() is True


class TestGetAgent:
    def test_builtin_agent(self):
        agent = get_agent("claude-code")
        assert agent is not None
        assert agent.display_name == "Claude Code"
        assert "claude" in agent.command

    def test_unknown_agent(self):
        assert get_agent("nonexistent") is None

    def test_custom_agent_dict(self):
        custom = {"my-agent": {"display_name": "My Agent", "command": ["my-agent", "--acp"], "check_binary": "my-agent"}}
        agent = get_agent("my-agent", custom_agents=custom)
        assert agent is not None
        assert agent.display_name == "My Agent"

    def test_custom_overrides_builtin(self):
        custom = {"claude-code": {"display_name": "Custom Claude", "command": ["custom-claude"]}}
        agent = get_agent("claude-code", custom_agents=custom)
        assert agent.display_name == "Custom Claude"


class TestListAgents:
    def test_lists_all_builtins(self):
        agents = list_agents()
        names = {a.name for a in agents}
        assert "claude-code" in names
        assert "codex" in names
        assert "gemini" in names

    def test_includes_custom(self):
        custom = {"my-agent": {"display_name": "My Agent", "command": ["my-agent"]}}
        agents = list_agents(custom_agents=custom)
        names = {a.name for a in agents}
        assert "my-agent" in names


class TestListAvailableAgents:
    def test_only_available(self):
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/claude" if x == "claude" else None):
            agents = list_available_agents()
            names = {a.name for a in agents}
            assert "claude-code" in names
            # codex binary doesn't exist in mock
            assert "codex" not in names
