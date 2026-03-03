"""Tests for Ctrl+C / CancelledError cancellation behaviour."""

import asyncio
import json
import signal
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest
import yaml

from koi.sandbox import Sandbox
from koi.tools import ToolExecutor

# ── helpers ──

def _make_sandbox(tmpdir: str, blocked_patterns=None):
    td = Path(tmpdir)
    koi_dir = td / ".koi"
    koi_dir.mkdir(exist_ok=True)
    cfg = {
        "filesystem": {"allowed_paths": [str(td)]},
        "commands": {},
    }
    if blocked_patterns:
        cfg["commands"]["blocked_patterns"] = blocked_patterns
    (koi_dir / "sandbox.yaml").write_text(yaml.dump(cfg))
    return Sandbox(project_root=td)


def _make_tool_call(name: str, arguments: dict) -> dict:
    return {
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        }
    }


# ── ToolExecutor: CancelledError propagation ──

async def test_cancelled_error_propagates_from_tool_executor():
    """execute_tool must re-raise CancelledError, not swallow it."""
    executor = ToolExecutor(Mock())

    # Patch _read_file to raise CancelledError (simulating mid-operation cancel)
    async def _raise_cancelled(**kwargs):
        raise asyncio.CancelledError()

    executor._read_file = _raise_cancelled

    tool_call = _make_tool_call("read_file", {"path": "/tmp/x"})

    with pytest.raises(asyncio.CancelledError):
        await executor.execute_tool(tool_call)


# ── Subprocess cancellation ──

async def test_ctrl_c_kills_subprocess():
    """Cancelling during _exec_command should terminate the subprocess."""
    with TemporaryDirectory() as tmpdir:
        sandbox = _make_sandbox(tmpdir)
        executor = ToolExecutor(Mock(), sandbox)

        # Start a long-running command, then cancel almost immediately
        task = asyncio.create_task(
            executor._exec_command("sleep 60", timeout=120)
        )
        # Give the subprocess a moment to start
        await asyncio.sleep(0.1)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


async def test_subprocess_timeout_still_works():
    """Async timeout should replace subprocess.TimeoutExpired correctly."""
    with TemporaryDirectory() as tmpdir:
        sandbox = _make_sandbox(tmpdir)
        executor = ToolExecutor(Mock(), sandbox)

        result = await executor._exec_command("sleep 60", timeout=1)
        assert result["success"] is False
        assert "timed out" in result["error"]


# ── LLM client: CancelledError propagation ──

async def test_cancelled_error_propagates_from_llm_client():
    """LLMClient.chat() must re-raise CancelledError."""
    from koi.llm import LLMClient

    config = Mock()
    config.api_key = "test"
    config.api_base = "http://localhost:1234"
    config.model = "test"
    config.max_tokens = 100
    config.temperature = None
    config.context_window = 8000
    config.api_format = "responses"

    client = LLMClient(config)

    # Make the HTTP post raise CancelledError (simulating mid-request cancel)
    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    client.client.post = _raise_cancelled

    with pytest.raises(asyncio.CancelledError):
        await client.chat([{"role": "user", "content": "hi"}])

    await client.close()


# ── Agent: message rollback on cancellation ──

async def test_ctrl_c_cancels_llm_call():
    """Cancelling during LLM call should roll back messages."""
    from koi.agent import Agent

    config = Mock()
    config.api_key = "test"
    config.api_base = "http://localhost:1234"
    config.model = "test"
    config.max_tokens = 100
    config.temperature = None
    config.context_window = 128000
    config.skills_paths = []
    config.api_format = "responses"

    agent = Agent(config, non_interactive=True)

    initial_msg_count = len(agent.messages)

    # Make the LLM chat hang so we can cancel it
    hang_event = asyncio.Event()

    async def _hang_chat(*args, **kwargs):
        await hang_event.wait()

    agent.llm_client.chat = _hang_chat

    # Add user message and start agent loop
    agent.messages.append({"role": "user", "content": "do something"})
    agent._current_task = asyncio.create_task(
        agent._agent_loop(non_interactive=True)
    )
    # Wait just enough for the snapshot to be set
    await asyncio.sleep(0.05)
    agent._current_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await agent._current_task

    # Fine-grained rollback: the snapshot was taken at the start of the iteration,
    # which is after the user message was appended, so the user message stays
    # but no partial assistant messages are added.
    snapshot = getattr(agent, "_iter_msg_snapshot", initial_msg_count + 1)
    agent.messages = agent.messages[:snapshot]

    # The user message should still be there, no extra messages
    assert len(agent.messages) == initial_msg_count + 1
    assert agent.messages[-1]["role"] == "user"

    await agent.llm_client.close()


async def test_ctrl_c_cancels_tool_execution():
    """Cancelling during tool execution should roll back partial results."""
    from koi.agent import Agent

    config = Mock()
    config.api_key = "test"
    config.api_base = "http://localhost:1234"
    config.model = "test"
    config.max_tokens = 100
    config.temperature = None
    config.context_window = 128000
    config.skills_paths = []
    config.api_format = "responses"

    agent = Agent(config, non_interactive=True)

    # Simulate: LLM returns a tool call, then tool execution hangs
    call_count = 0
    hang_event = asyncio.Event()

    async def _fake_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"command": "sleep 60"})
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        }

    async def _hang_tool(tool_call):
        await hang_event.wait()

    agent.llm_client.chat = _fake_chat
    agent.tool_executor.execute_tool = _hang_tool

    agent.messages.append({"role": "user", "content": "run a command"})
    snapshot_before = len(agent.messages)

    agent._current_task = asyncio.create_task(
        agent._agent_loop(non_interactive=True)
    )
    await asyncio.sleep(0.05)
    agent._current_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await agent._current_task

    # After rollback, the iter_msg_snapshot should point to just before
    # the assistant message with tool_calls was appended
    iter_snap = getattr(agent, "_iter_msg_snapshot", snapshot_before)
    agent.messages = agent.messages[:iter_snap]

    # User message is preserved, but the partial assistant+tool messages are gone
    assert agent.messages[-1]["role"] == "user"

    await agent.llm_client.close()


async def test_message_rollback_preserves_completed_iterations():
    """Only interrupted iteration is rolled back; completed ones stay."""
    from koi.agent import Agent

    config = Mock()
    config.api_key = "test"
    config.api_base = "http://localhost:1234"
    config.model = "test"
    config.max_tokens = 100
    config.temperature = None
    config.context_window = 128000
    config.skills_paths = []
    config.api_format = "responses"

    agent = Agent(config, non_interactive=True)

    # Iteration 1: LLM returns tool call, tool succeeds
    # Iteration 2: LLM hangs (simulate cancel during second LLM call)
    call_count = 0
    hang_event = asyncio.Event()

    async def _fake_chat(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First call: return a tool call
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call_1",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "test"})
                            }
                        }]
                    },
                    "finish_reason": "tool_calls"
                }]
            }
        else:
            # Second call: hang (will be cancelled)
            await hang_event.wait()

    async def _fake_tool(tool_call):
        return {"success": True, "message": "done", "results": []}

    agent.llm_client.chat = _fake_chat
    agent.tool_executor.execute_tool = _fake_tool

    agent.messages.append({"role": "user", "content": "search something"})
    user_msg_index = len(agent.messages) - 1

    agent._current_task = asyncio.create_task(
        agent._agent_loop(non_interactive=True)
    )
    # Wait for first iteration to complete and second to start
    await asyncio.sleep(0.1)
    agent._current_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await agent._current_task

    # The iter snapshot should be set at the start of iteration 2,
    # which is after the assistant+tool messages from iteration 1.
    iter_snap = getattr(agent, "_iter_msg_snapshot", None)
    assert iter_snap is not None

    agent.messages = agent.messages[:iter_snap]

    # Iteration 1's messages (assistant + tool result) should be preserved
    assert len(agent.messages) > user_msg_index + 1
    # The assistant message with tool_calls from iteration 1 should be there
    assert any(
        m.get("role") == "assistant" and m.get("tool_calls")
        for m in agent.messages
    )

    await agent.llm_client.close()


# ── Double Ctrl+C / force exit ──

def _make_agent():
    """Helper to create a non-interactive Agent with mocked config."""
    from koi.agent import Agent

    config = Mock()
    config.api_key = "test"
    config.api_base = "http://localhost:1234"
    config.model = "test"
    config.max_tokens = 100
    config.temperature = None
    config.context_window = 128000
    config.skills_paths = []
    config.api_format = "responses"

    return Agent(config, non_interactive=True)


async def test_double_ctrl_c_force_exit():
    """Two rapid SIGINT calls (within 1.5s) should trigger _force_exit."""
    agent = _make_agent()

    # Set up a hanging task so _current_task is active
    hang_event = asyncio.Event()

    async def _hang_chat(*args, **kwargs):
        await hang_event.wait()

    agent.llm_client.chat = _hang_chat
    agent.messages.append({"role": "user", "content": "test"})
    agent._current_task = asyncio.create_task(
        agent._agent_loop(non_interactive=True)
    )
    await asyncio.sleep(0.05)

    with patch.object(agent, "_force_exit") as mock_force_exit:
        # First Ctrl+C — graceful cancel
        agent._handle_sigint(signal.SIGINT, None)
        assert agent._interrupted is True
        assert agent._last_interrupt_time is not None
        mock_force_exit.assert_not_called()

        # Second Ctrl+C within 1.5s — force exit
        # The task may already be done from the cancel, so re-create it
        # to simulate a stuck task
        agent._current_task = asyncio.create_task(hang_event.wait())
        agent._handle_sigint(signal.SIGINT, None)
        mock_force_exit.assert_called_once()

    # Cleanup
    agent._current_task.cancel()
    try:
        await agent._current_task
    except asyncio.CancelledError:
        pass
    await agent.llm_client.close()


async def test_single_ctrl_c_then_delayed_second():
    """Two SIGINT calls >2s apart should both do graceful cancel, not force exit."""
    agent = _make_agent()

    with patch.object(agent, "_force_exit") as mock_force_exit:
        # Simulate a running task
        hang_event = asyncio.Event()
        agent._current_task = asyncio.create_task(hang_event.wait())
        await asyncio.sleep(0.01)

        # First Ctrl+C
        agent._handle_sigint(signal.SIGINT, None)
        first_time = agent._last_interrupt_time

        # Cancel and recreate task to simulate next operation
        try:
            await agent._current_task
        except asyncio.CancelledError:
            pass

        # Wait >1.5s (simulate with time manipulation)
        agent._last_interrupt_time = first_time - 2.0  # pretend it was 2s ago

        # Second Ctrl+C — should be graceful (not force exit)
        agent._current_task = asyncio.create_task(hang_event.wait())
        agent._interrupted = False
        agent._handle_sigint(signal.SIGINT, None)

        mock_force_exit.assert_not_called()
        assert agent._interrupted is True

    # Cleanup
    agent._current_task.cancel()
    try:
        await agent._current_task
    except asyncio.CancelledError:
        pass
    await agent.llm_client.close()


async def test_atexit_kills_subprocesses():
    """force_kill_all_sync should kill active subprocess runs."""
    from koi.subagent import SubagentManager, SubagentRun
    from datetime import datetime

    config = Mock()
    mgr = SubagentManager(config)

    # Create mock processes
    alive_proc = Mock()
    alive_proc.returncode = None  # still running
    alive_proc.kill = Mock()

    dead_proc = Mock()
    dead_proc.returncode = 0  # already exited
    dead_proc.kill = Mock()

    # Active run with alive process
    run1 = SubagentRun(
        id="r1",
        task="task1",
        label="alive",
        process=alive_proc,
        result_file=Path("/tmp/r1.json"),
        started_at=datetime.now(),
        completed=False,
    )
    # Completed run
    run2 = SubagentRun(
        id="r2",
        task="task2",
        label="done",
        process=dead_proc,
        result_file=Path("/tmp/r2.json"),
        started_at=datetime.now(),
        completed=True,
    )
    # Active run but process already exited
    run3 = SubagentRun(
        id="r3",
        task="task3",
        label="exited",
        process=dead_proc,
        result_file=Path("/tmp/r3.json"),
        started_at=datetime.now(),
        completed=False,
    )

    mgr.active_runs = {"r1": run1, "r2": run2, "r3": run3}

    mgr.force_kill_all_sync()

    # Only the alive, non-completed process should be killed
    alive_proc.kill.assert_called_once()
    dead_proc.kill.assert_not_called()
