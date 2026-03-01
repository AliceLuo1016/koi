"""Tests for /status command and supporting helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from koi.agent import Agent, _fmt_num
from koi.compaction import ContextCompactor
from koi.config import Config
from koi.usage import TokenUsage


def _make_agent(**overrides):
    """Create a non-interactive Agent with sensible defaults for testing."""
    defaults = dict(
        api_base="https://api.example.com/v1/responses",
        api_key="sk-ant-test-1234567890abcdef",
        model="claude-sonnet-4-20250514",
        api_format="anthropic",
        thinking_level="low",
        prompt_caching=True,
    )
    defaults.update(overrides)
    config = Config(**defaults)
    return Agent(config, non_interactive=True)


# ── _fmt_num tests ──


class TestFmtNum:
    def test_zero(self):
        assert _fmt_num(0) == "0"

    def test_small_number(self):
        assert _fmt_num(999) == "999"

    def test_exactly_1k(self):
        assert _fmt_num(1000) == "1.0k"

    def test_thousands(self):
        assert _fmt_num(1500) == "1.5k"

    def test_large_thousands(self):
        assert _fmt_num(45000) == "45.0k"

    def test_millions(self):
        assert _fmt_num(1_500_000) == "1.5M"

    def test_exactly_1m(self):
        assert _fmt_num(1_000_000) == "1.0M"

    def test_large_millions(self):
        assert _fmt_num(200_000_000) == "200.0M"


# ── /status command tests ──


class TestStatusCommand:
    async def test_status_runs_without_error(self, capsys):
        """The /status command should print output without raising."""
        agent = _make_agent()
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Koi" in captured

    async def test_stats_alias(self, capsys):
        """/stats should produce the same output as /status."""
        agent = _make_agent()
        await agent._handle_command("/status")
        status_out = capsys.readouterr().out

        agent2 = _make_agent()
        await agent2._handle_command("/stats")
        stats_out = capsys.readouterr().out

        assert status_out == stats_out

    async def test_shows_model_name(self, capsys):
        agent = _make_agent(model="claude-opus-4-20250514")
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "claude-opus-4-20250514" in captured

    async def test_shows_masked_api_key(self, capsys):
        agent = _make_agent(api_key="sk-ant-test-1234567890abcdef")
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        # First 6 + ... + last 4
        assert "sk-ant" in captured
        assert "cdef" in captured
        # Full key should NOT appear
        assert "sk-ant-test-1234567890abcdef" not in captured

    async def test_short_api_key_masked(self, capsys):
        agent = _make_agent(api_key="short")
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "***" in captured

    async def test_shows_token_counts(self, capsys):
        agent = _make_agent()
        agent.llm_client.usage.add(input_t=1200, output_t=450)
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "1.2k" in captured
        assert "450" in captured

    async def test_shows_cache_info_when_active(self, capsys):
        agent = _make_agent()
        agent.llm_client.usage.add(
            input_t=1000, output_t=100, cache_read=10200, cache_creation=1100
        )
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Cache:" in captured
        assert "cached" in captured
        assert "new" in captured
        assert "%" in captured

    async def test_hides_cache_info_when_no_activity(self, capsys):
        agent = _make_agent()
        agent.llm_client.usage.add(input_t=500, output_t=200)
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Cache:" not in captured

    async def test_shows_context_usage_percent(self, capsys):
        agent = _make_agent(context_window=200000)
        # Add a message so context isn't empty
        agent.messages.append({"role": "user", "content": "hello"})
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Context:" in captured
        assert "%" in captured

    async def test_shows_compaction_count(self, capsys):
        agent = _make_agent()
        agent.compactor.compaction_count = 3
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Compactions: 3" in captured

    async def test_shows_thinking_level(self, capsys):
        agent = _make_agent(thinking_level="low")
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Think: low" in captured

    async def test_shows_prompt_cache_on(self, capsys):
        agent = _make_agent(prompt_caching=True)
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Prompt cache: on" in captured

    async def test_shows_prompt_cache_off(self, capsys):
        agent = _make_agent(prompt_caching=False)
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Prompt cache: off" in captured

    async def test_shows_api_format(self, capsys):
        agent = _make_agent(api_format="anthropic")
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "anthropic" in captured

    async def test_shows_subagents_when_active(self, capsys):
        agent = _make_agent()
        # Add a fake active run
        mock_run = MagicMock()
        mock_run.completed = False
        agent.subagent_manager.active_runs["test-1"] = mock_run
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Sub-agents: 1 active" in captured

    async def test_hides_subagents_when_none(self, capsys):
        agent = _make_agent()
        await agent._handle_command("/status")
        captured = capsys.readouterr().out
        assert "Sub-agents" not in captured


# ── Compaction count tests ──


class TestCompactionCount:
    def test_compaction_count_initializes_to_zero(self):
        config = Config(
            api_base="https://api.example.com",
            api_key="test",
            model="test-model",
            api_format="responses",
            thinking_level="off",
        )
        client = MagicMock()
        compactor = ContextCompactor(client, 128000)
        assert compactor.compaction_count == 0

    async def test_compaction_count_increments(self):
        config = Config(
            api_base="https://api.example.com",
            api_key="test",
            model="test-model",
            api_format="responses",
            thinking_level="off",
        )
        client = MagicMock()
        client.chat = AsyncMock(
            return_value={
                "choices": [
                    {"message": {"role": "assistant", "content": "Summary of conversation."}}
                ]
            }
        )
        compactor = ContextCompactor(client, 128000)

        messages = [
            {"role": "user", "content": f"Message {i}"} for i in range(10)
        ]
        result = await compactor.compact_messages(messages)

        assert compactor.compaction_count == 1
        assert any("summary" in m.get("content", "").lower() for m in result)

        # Compact again
        messages2 = result + [
            {"role": "user", "content": f"Follow-up {i}"} for i in range(5)
        ]
        await compactor.compact_messages(messages2)
        assert compactor.compaction_count == 2


# ── Help text test ──


class TestHelpIncludesStatus:
    def test_help_mentions_status(self):
        agent = _make_agent()
        # Capture the help text by calling _show_help through capsys
        # We'll just check the method exists and the help text mentions /status
        import io
        from rich.console import Console

        buf = io.StringIO()
        test_console = Console(file=buf, force_terminal=False)
        with patch("koi.agent.console", test_console):
            agent._show_help()
        output = buf.getvalue()
        assert "/status" in output
        assert "/stats" in output
