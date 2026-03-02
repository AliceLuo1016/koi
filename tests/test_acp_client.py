"""Tests for ACP client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from koi.acp_client import ACPResult, ACPSession, KoiACPClient


class TestKoiACPClient:
    def test_reset_clears_state(self):
        client = KoiACPClient()
        client._collected_text = "hello"
        client._collected_thoughts = "thinking"
        client._tool_calls = [{"id": "1"}]
        client.reset()
        assert client._collected_text == ""
        assert client._collected_thoughts == ""
        assert client._tool_calls == []

    async def test_request_permission_auto_approve(self):
        client = KoiACPClient(auto_approve=True)
        option = MagicMock()
        option.id = "opt1"
        resp = await client.request_permission(
            options=[option], session_id="s1", tool_call=MagicMock()
        )
        assert resp.outcome.outcome == "selected"

    async def test_request_permission_deny(self):
        client = KoiACPClient(auto_approve=False)
        resp = await client.request_permission(
            options=[MagicMock()], session_id="s1", tool_call=MagicMock()
        )
        assert resp.outcome.outcome == "cancelled"

    async def test_session_update_text(self):
        client = KoiACPClient()
        # Simulate AgentMessageChunk
        update = MagicMock()
        update.__class__ = type("AgentMessageChunk", (), {})
        # Use real isinstance check
        from koi.acp_client import AgentMessageChunk
        chunk = MagicMock(spec=AgentMessageChunk)
        chunk.content = MagicMock()
        chunk.content.text = "hello world"
        await client.session_update(session_id="s1", update=chunk)
        assert client._collected_text == "hello world"

    async def test_read_text_file(self, tmp_path):
        client = KoiACPClient()
        test_file = tmp_path / "test.txt"
        test_file.write_text("file content")
        resp = await client.read_text_file(str(test_file), session_id="s1")
        assert resp.content == "file content"

    async def test_read_text_file_not_found(self):
        client = KoiACPClient()
        resp = await client.read_text_file("/nonexistent/path", session_id="s1")
        assert "Error" in resp.content


class TestACPResult:
    def test_defaults(self):
        result = ACPResult(content="hello")
        assert result.content == "hello"
        assert result.stop_reason == "end_turn"
        assert result.tool_calls == []
        assert result.thoughts == ""
        assert result.usage is None


class TestACPSession:
    def test_init(self):
        session = ACPSession(command=["claude", "--acp"], cwd="/tmp")
        assert session.command == ["claude", "--acp"]
        assert session.cwd == "/tmp"
        assert session.session_id is None
        assert session.is_alive is False

    async def test_send_without_start_raises(self):
        session = ACPSession(command=["claude", "--acp"])
        with pytest.raises(RuntimeError, match="not started"):
            await session.send("hello")

    async def test_kill_no_process(self):
        session = ACPSession(command=["claude", "--acp"])
        # Should not raise
        await session.kill()
        assert session._process is None
