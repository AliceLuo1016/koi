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


# ── More coverage tests ──


class TestSessionUpdateThoughts:
    async def test_thought_chunk(self):
        from koi.acp_client import AgentThoughtChunk
        client = KoiACPClient()
        chunk = MagicMock(spec=AgentThoughtChunk)
        chunk.content = MagicMock()
        chunk.content.text = "I'm thinking..."
        await client.session_update(session_id="s1", update=chunk)
        assert client._collected_thoughts == "I'm thinking..."


class TestSessionUpdateToolCalls:
    async def test_tool_call_start(self):
        from koi.acp_client import ToolCallStart
        client = KoiACPClient()
        tc = MagicMock(spec=ToolCallStart)
        tc.tool_call_id = "tc1"
        tc.title = "Read file"
        tc.status = "pending"
        await client.session_update(session_id="s1", update=tc)
        assert len(client._tool_calls) == 1
        assert client._tool_calls[0]["title"] == "Read file"

    async def test_tool_call_progress(self):
        from koi.acp_client import ToolCallProgress
        client = KoiACPClient()
        tc = MagicMock(spec=ToolCallProgress)
        tc.tool_call_id = "tc1"
        tc.title = "Writing"
        tc.status = "in_progress"
        await client.session_update(session_id="s1", update=tc)
        assert len(client._tool_calls) == 1


class TestSessionUpdateUsage:
    async def test_usage_update(self):
        from koi.acp_client import UsageUpdate
        client = KoiACPClient()
        u = MagicMock(spec=UsageUpdate)
        u.usage = {"input_tokens": 100, "output_tokens": 50}
        await client.session_update(session_id="s1", update=u)
        assert client._usage == {"input_tokens": 100, "output_tokens": 50}

    async def test_usage_update_non_dict(self):
        from koi.acp_client import UsageUpdate
        client = KoiACPClient()
        u = MagicMock(spec=UsageUpdate)
        u.usage = MagicMock()  # not a dict
        u.usage.__class__ = type("NotDict", (), {})
        await client.session_update(session_id="s1", update=u)
        assert client._usage == {}


class TestWriteTextFile:
    async def test_write_success(self, tmp_path):
        client = KoiACPClient()
        path = str(tmp_path / "out.txt")
        await client.write_text_file("hello", path, session_id="s1")
        assert open(path).read() == "hello"

    async def test_write_failure(self):
        client = KoiACPClient()
        # Write to invalid path
        resp = await client.write_text_file("x", "/nonexistent/dir/file.txt", session_id="s1")
        # Should not raise, returns empty response


class TestTerminalMethods:
    async def test_create_terminal(self):
        client = KoiACPClient()
        resp = await client.create_terminal("bash", session_id="s1")
        assert resp.terminal_id == "unsupported"

    async def test_terminal_output(self):
        client = KoiACPClient()
        resp = await client.terminal_output(session_id="s1", terminal_id="t1")
        assert resp.output == ""

    async def test_release_terminal(self):
        client = KoiACPClient()
        resp = await client.release_terminal(session_id="s1", terminal_id="t1")
        assert resp is not None

    async def test_wait_for_terminal_exit(self):
        client = KoiACPClient()
        resp = await client.wait_for_terminal_exit(session_id="s1", terminal_id="t1")
        assert resp.exit_code == 1

    async def test_kill_terminal(self):
        client = KoiACPClient()
        resp = await client.kill_terminal_command(session_id="s1", terminal_id="t1")
        assert resp is not None


class TestACPSessionClose:
    async def test_close_without_start(self):
        session = ACPSession(command=["test"])
        await session.close()  # should not raise

    async def test_close_with_live_process(self):
        session = ACPSession(command=["test"])
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        session._process = mock_proc
        session._ctx = None
        await session.close()
        mock_proc.kill.assert_called_once()

    async def test_is_alive_false_after_kill(self):
        session = ACPSession(command=["test"])
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        session._process = mock_proc
        await session.kill()
        assert session.is_alive is False


class TestRequestPermissionEmpty:
    async def test_auto_approve_empty_options(self):
        client = KoiACPClient(auto_approve=True)
        from koi.acp_client import DeniedOutcome
        resp = await client.request_permission(options=[], session_id="s1", tool_call=MagicMock())
        assert resp.outcome.outcome == "cancelled"
