"""Tests for sub-agent spawning and management."""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from koi.acp_client import ACPResult
from koi.config import Config
from koi.subagent import SubagentManager, SubagentRun
from koi.tools import ToolExecutor, get_tool_definitions

# ── Helpers ──────────────────────────────────────────────────


def _make_config(**overrides):
    """Create a minimal Config for testing."""
    defaults = dict(
        api_base="http://localhost:8080",
        api_key="test-key",
        model="test-model",
    )
    defaults.update(overrides)
    return Config(**defaults)


def _mock_process(returncode=0, stdout=b"done", stderr=b""):
    """Create a mock asyncio.subprocess.Process."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    proc.kill = MagicMock()
    proc.pid = 12345
    return proc


# ── SubagentManager.spawn ────────────────────────────────────


async def test_spawn_creates_process():
    """spawn() creates a subprocess and returns run_id."""
    config = _make_config()
    mgr = SubagentManager(config)

    mock_proc = _mock_process()

    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await mgr.spawn(task="say hello", cwd=tmpdir)

        assert result["status"] == "accepted"
        assert "run_id" in result
        assert result["run_id"] in mgr.active_runs

        # Give the background _wait_for_completion task a tick
        await asyncio.sleep(0.05)


async def test_spawn_returns_run_id_and_label():
    """spawn() stores label on the run."""
    config = _make_config()
    mgr = SubagentManager(config)
    mock_proc = _mock_process()

    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await mgr.spawn(task="do work", label="my-job", cwd=tmpdir)

        run_id = result["run_id"]
        run = mgr.active_runs[run_id]
        assert run.label == "my-job"
        assert run.task == "do work"

        await asyncio.sleep(0.05)


# ── Depth limit ──────────────────────────────────────────────


async def test_spawn_respects_max_depth():
    """spawn() rejects when current depth >= max_depth."""
    config = _make_config()
    mgr = SubagentManager(config, max_depth=2)
    mgr._depth = 2  # Already at max

    result = await mgr.spawn(task="too deep")

    assert result["status"] == "error"
    assert "depth" in result["error"].lower()


async def test_spawn_depth_env_propagated():
    """spawn() passes KOI_SPAWN_DEPTH incremented by 1 to child env."""
    config = _make_config()
    mgr = SubagentManager(config)
    mgr._depth = 1

    mock_proc = _mock_process()
    captured_kwargs = {}

    async def capture_exec(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_proc

    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", side_effect=capture_exec):
            await mgr.spawn(task="child task", cwd=tmpdir)

        assert captured_kwargs["env"]["KOI_SPAWN_DEPTH"] == "2"

        await asyncio.sleep(0.05)


# ── Children limit ───────────────────────────────────────────


async def test_spawn_respects_max_children():
    """spawn() rejects when active (non-completed) children >= max_children."""
    config = _make_config()
    mgr = SubagentManager(config, max_children=2)

    # Create 2 fake running runs
    for i in range(2):
        run = SubagentRun(
            id=f"run-{i}",
            task="busy",
            label=None,
            process=AsyncMock(),
            result_file=Path(f"/tmp/fake-{i}.json"),
            started_at=__import__("datetime").datetime.now(),
            completed=False,
        )
        mgr.active_runs[run.id] = run

    result = await mgr.spawn(task="one too many")

    assert result["status"] == "error"
    assert "Max children" in result["error"]


async def test_spawn_allows_after_completed():
    """spawn() counts only non-completed runs toward the limit."""
    config = _make_config()
    mgr = SubagentManager(config, max_children=1)

    # One completed run shouldn't count
    run = SubagentRun(
        id="done-1",
        task="finished",
        label=None,
        process=AsyncMock(),
        result_file=Path("/tmp/fake.json"),
        started_at=__import__("datetime").datetime.now(),
        completed=True,
    )
    mgr.active_runs[run.id] = run

    mock_proc = _mock_process()
    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await mgr.spawn(task="should work", cwd=tmpdir)

        assert result["status"] == "accepted"

        await asyncio.sleep(0.05)


# ── list_runs ────────────────────────────────────────────────


async def test_list_runs_returns_correct_status():
    """list_runs() shows running and completed runs."""
    config = _make_config()
    mgr = SubagentManager(config)

    from datetime import datetime

    mgr.active_runs["a"] = SubagentRun(
        id="a",
        task="running task",
        label="runner",
        process=AsyncMock(),
        result_file=Path("/tmp/a.json"),
        started_at=datetime.now(),
        completed=False,
    )
    mgr.active_runs["b"] = SubagentRun(
        id="b",
        task="done task",
        label="doner",
        process=AsyncMock(),
        result_file=Path("/tmp/b.json"),
        started_at=datetime.now(),
        completed=True,
        result={"summary": "all good"},
    )

    runs = mgr.list_runs()
    assert len(runs) == 2

    by_id = {r["id"]: r for r in runs}
    assert by_id["a"]["status"] == "running"
    assert by_id["a"]["label"] == "runner"
    assert by_id["b"]["status"] == "completed"
    assert "all good" in by_id["b"]["result_summary"]


# ── kill ─────────────────────────────────────────────────────


async def test_kill_terminates_process():
    """kill() calls process.kill() and marks run completed."""
    config = _make_config()
    mgr = SubagentManager(config)

    mock_proc = MagicMock()
    mock_proc.returncode = None
    mock_proc.kill = MagicMock()

    from datetime import datetime

    run = SubagentRun(
        id="k1",
        task="long task",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/k1.json"),
        started_at=datetime.now(),
        completed=False,
    )
    mgr.active_runs["k1"] = run

    result = await mgr.kill("k1")
    assert result["status"] == "killed"
    assert run.completed is True
    assert run.error == "Killed by parent"
    mock_proc.kill.assert_called_once()


async def test_kill_nonexistent_returns_error():
    """kill() returns error for unknown run_id."""
    config = _make_config()
    mgr = SubagentManager(config)

    result = await mgr.kill("no-such-id")
    assert "error" in result


async def test_kill_already_completed():
    """kill() on completed run doesn't re-kill."""
    config = _make_config()
    mgr = SubagentManager(config)

    from datetime import datetime

    mock_proc = MagicMock()
    run = SubagentRun(
        id="done1",
        task="finished",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/done1.json"),
        started_at=datetime.now(),
        completed=True,
    )
    mgr.active_runs["done1"] = run

    result = await mgr.kill("done1")
    assert result["status"] == "killed"
    mock_proc.kill.assert_not_called()


# ── Result file reading ──────────────────────────────────────


async def test_wait_for_completion_reads_result_file():
    """_wait_for_completion reads the JSON result file written by child."""
    config = _make_config()
    mgr = SubagentManager(config)

    with TemporaryDirectory() as tmpdir:
        result_file = Path(tmpdir) / "result.json"
        result_file.write_text(json.dumps({"summary": "task done", "response": "full output"}))

        mock_proc = _mock_process(returncode=0, stdout=b"stdout text", stderr=b"")

        from datetime import datetime

        run = SubagentRun(
            id="r1",
            task="test",
            label=None,
            process=mock_proc,
            result_file=result_file,
            started_at=datetime.now(),
        )

        await mgr._wait_for_completion(run)

        assert run.completed is True
        assert run.exit_code == 0
        assert run.result["summary"] == "task done"
        assert run.stdout == "stdout text"


async def test_wait_for_completion_missing_result_file():
    """_wait_for_completion handles missing result file gracefully."""
    config = _make_config()
    mgr = SubagentManager(config)

    mock_proc = _mock_process()

    from datetime import datetime

    run = SubagentRun(
        id="r2",
        task="test",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/nonexistent_koi_test_file.json"),
        started_at=datetime.now(),
    )

    await mgr._wait_for_completion(run)

    assert run.completed is True
    assert run.result is None


# ── Timeout ──────────────────────────────────────────────────


async def test_wait_for_completion_timeout_kills_process():
    """_wait_for_completion kills process on timeout."""
    config = _make_config()
    mgr = SubagentManager(config)

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()
    mock_proc.returncode = None

    from datetime import datetime

    run = SubagentRun(
        id="t1",
        task="slow task",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/t1.json"),
        started_at=datetime.now(),
        timeout_seconds=1,
    )

    await mgr._wait_for_completion(run)

    assert run.completed is True
    assert run.error == "Timed out"
    mock_proc.kill.assert_called_once()


# ── on_complete callback ─────────────────────────────────────


async def test_on_complete_callback_called():
    """_on_complete callback is invoked when a run finishes."""
    config = _make_config()
    mgr = SubagentManager(config)

    callback_runs = []

    async def on_complete(run):
        callback_runs.append(run)

    mgr._on_complete = on_complete

    mock_proc = _mock_process()

    from datetime import datetime

    run = SubagentRun(
        id="cb1",
        task="callback test",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/nonexistent_cb.json"),
        started_at=datetime.now(),
    )

    await mgr._wait_for_completion(run)

    assert len(callback_runs) == 1
    assert callback_runs[0].id == "cb1"


async def test_on_complete_callback_on_timeout():
    """_on_complete callback is called even when the sub-agent times out."""
    config = _make_config()
    mgr = SubagentManager(config)

    callback_runs = []

    async def on_complete(run):
        callback_runs.append(run)

    mgr._on_complete = on_complete

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_proc.kill = MagicMock()

    from datetime import datetime

    run = SubagentRun(
        id="to1",
        task="timeout callback",
        label=None,
        process=mock_proc,
        result_file=Path("/tmp/to1.json"),
        started_at=datetime.now(),
        timeout_seconds=1,
    )

    await mgr._wait_for_completion(run)

    assert len(callback_runs) == 1
    assert callback_runs[0].error == "Timed out"


# ── Tool definitions include subagent tools ──────────────────


def test_tool_definitions_include_subagent_tools():
    """get_tool_definitions() includes spawn_subagent, list_subagents, kill_subagent."""
    tools = get_tool_definitions()
    tool_names = {t["function"]["name"] for t in tools}

    assert "spawn_subagent" in tool_names
    assert "list_subagents" in tool_names
    assert "kill_subagent" in tool_names


# ── Tool executor: spawn_subagent ────────────────────────────


async def test_tool_executor_spawn_subagent_calls_manager():
    """spawn_subagent tool delegates to SubagentManager.spawn."""
    mock_mgr = AsyncMock()
    mock_mgr.spawn = AsyncMock(return_value={"status": "accepted", "run_id": "abc123", "note": "started"})

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "spawn_subagent",
            "arguments": json.dumps({"task": "analyze logs"}),
        }
    }

    result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert result["run_id"] == "abc123"
    mock_mgr.spawn.assert_called_once_with(
        task="analyze logs",
        label=None,
        model=None,
        thinking=None,
        timeout_seconds=0,
        cwd=None,
    )


async def test_tool_executor_spawn_subagent_with_options():
    """spawn_subagent passes optional parameters through."""
    mock_mgr = AsyncMock()
    mock_mgr.spawn = AsyncMock(return_value={"status": "accepted", "run_id": "xyz", "note": "ok"})

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "spawn_subagent",
            "arguments": json.dumps(
                {
                    "task": "test",
                    "label": "my-label",
                    "model": "gpt-4",
                    "thinking": "high",
                    "timeout_seconds": 120,
                    "cwd": "/tmp",
                }
            ),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is True

    mock_mgr.spawn.assert_called_once_with(
        task="test",
        label="my-label",
        model="gpt-4",
        thinking="high",
        timeout_seconds=120,
        cwd="/tmp",
    )


async def test_tool_executor_spawn_subagent_error_propagated():
    """spawn_subagent returns error when manager returns error status."""
    mock_mgr = AsyncMock()
    mock_mgr.spawn = AsyncMock(return_value={"status": "error", "error": "Max children reached (5)"})

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "spawn_subagent",
            "arguments": json.dumps({"task": "overflow"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is False
    assert "Max children" in result["error"]


async def test_tool_executor_spawn_subagent_no_manager():
    """spawn_subagent returns error when subagent_manager is None."""
    executor = ToolExecutor(Mock(), subagent_manager=None)

    tool_call = {
        "function": {
            "name": "spawn_subagent",
            "arguments": json.dumps({"task": "no manager"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is False
    assert "not available" in result["error"]


# ── Tool executor: list_subagents ────────────────────────────


async def test_tool_executor_list_subagents():
    """list_subagents tool returns runs from manager."""
    mock_mgr = MagicMock()
    mock_mgr.list_runs.return_value = [
        {
            "id": "a",
            "task": "one",
            "label": None,
            "status": "running",
            "started": "2025-01-01T00:00:00",
            "result_summary": None,
        }
    ]

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "list_subagents",
            "arguments": json.dumps({}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is True
    assert result["count"] == 1
    assert result["runs"][0]["id"] == "a"


async def test_tool_executor_list_subagents_no_manager():
    """list_subagents returns error when subagent_manager is None."""
    executor = ToolExecutor(Mock(), subagent_manager=None)

    tool_call = {
        "function": {
            "name": "list_subagents",
            "arguments": json.dumps({}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is False


# ── Tool executor: kill_subagent ─────────────────────────────


async def test_tool_executor_kill_subagent():
    """kill_subagent tool delegates to manager.kill."""
    mock_mgr = AsyncMock()
    mock_mgr.kill = AsyncMock(return_value={"status": "killed", "id": "x1"})

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "kill_subagent",
            "arguments": json.dumps({"run_id": "x1"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is True
    assert "killed" in result["message"].lower()
    mock_mgr.kill.assert_called_once_with("x1")


async def test_tool_executor_kill_subagent_not_found():
    """kill_subagent returns error when run not found."""
    mock_mgr = AsyncMock()
    mock_mgr.kill = AsyncMock(return_value={"error": "No run with id nope"})

    executor = ToolExecutor(Mock(), subagent_manager=mock_mgr)

    tool_call = {
        "function": {
            "name": "kill_subagent",
            "arguments": json.dumps({"run_id": "nope"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is False


async def test_tool_executor_kill_subagent_no_manager():
    """kill_subagent returns error when subagent_manager is None."""
    executor = ToolExecutor(Mock(), subagent_manager=None)

    tool_call = {
        "function": {
            "name": "kill_subagent",
            "arguments": json.dumps({"run_id": "x"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is False


# ── Config: spawn_depth ──────────────────────────────────────


def test_config_spawn_depth_default():
    """Config.spawn_depth returns 0 when KOI_SPAWN_DEPTH is unset."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("KOI_SPAWN_DEPTH", None)
        config = _make_config()
        assert config.spawn_depth == 0


def test_config_spawn_depth_from_env():
    """Config.spawn_depth reads KOI_SPAWN_DEPTH."""
    with patch.dict(os.environ, {"KOI_SPAWN_DEPTH": "3"}):
        config = _make_config()
        assert config.spawn_depth == 3


# ── CLI: --result-file writes JSON ───────────────────────────


async def test_run_task_writes_result_file():
    """_run_task writes a JSON result file when result_file is set."""
    from koi.cli import _run_task

    with TemporaryDirectory() as tmpdir:
        result_path = Path(tmpdir) / "out.json"

        # Create a mock agent whose messages simulate a completed conversation
        mock_agent = AsyncMock()
        mock_agent.messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        mock_agent.run_task = AsyncMock()

        await _run_task(mock_agent, "hello", non_interactive=True, result_file=str(result_path))

        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["response"] == "Hi there!"
        assert data["message_count"] == 2


async def test_run_task_no_result_file():
    """_run_task does not create a file when result_file is None."""
    from koi.cli import _run_task

    mock_agent = AsyncMock()
    mock_agent.messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "ok"},
    ]
    mock_agent.run_task = AsyncMock()

    # Should not raise
    await _run_task(mock_agent, "hello", non_interactive=True, result_file=None)


# ── Integration: spawn builds correct command ────────────────


async def test_spawn_builds_correct_command():
    """spawn() passes --result-file, --model, --thinking to subprocess."""
    config = _make_config()
    mgr = SubagentManager(config)

    captured_args = []

    async def capture_exec(*args, **kwargs):
        captured_args.extend(args)
        return _mock_process()

    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", side_effect=capture_exec):
            await mgr.spawn(
                task="do stuff",
                model="gpt-4",
                thinking="high",
                cwd=tmpdir,
            )

        cmd = captured_args
        assert "--task" in cmd
        assert "do stuff" in cmd
        assert "--non-interactive" in cmd
        assert "--result-file" in cmd
        assert "--model" in cmd
        assert "gpt-4" in cmd
        assert "--thinking" in cmd
        assert "high" in cmd

        await asyncio.sleep(0.05)


async def test_spawn_minimal_command():
    """spawn() with only task produces a minimal command."""
    config = _make_config()
    mgr = SubagentManager(config)

    captured_args = []

    async def capture_exec(*args, **kwargs):
        captured_args.extend(args)
        return _mock_process()

    with TemporaryDirectory() as tmpdir:
        with patch("koi.subagent.asyncio.create_subprocess_exec", side_effect=capture_exec):
            await mgr.spawn(task="simple task", cwd=tmpdir)

        cmd = captured_args
        assert "--task" in cmd
        assert "--model" not in cmd
        assert "--thinking" not in cmd

        await asyncio.sleep(0.05)


# ── get_result ───────────────────────────────────────────────


async def test_get_result_completed():
    """get_result returns result dict for a completed run."""
    config = _make_config()
    mgr = SubagentManager(config)

    from datetime import datetime

    run = SubagentRun(
        id="gr1",
        task="done",
        label="lbl",
        process=AsyncMock(),
        result_file=Path("/tmp/gr1.json"),
        started_at=datetime.now(),
        completed=True,
        result={"summary": "ok"},
        exit_code=0,
        stdout="output",
        stderr="",
    )
    mgr.active_runs["gr1"] = run

    res = mgr.get_result("gr1")
    assert res is not None
    assert res["result"]["summary"] == "ok"
    assert res["exit_code"] == 0


async def test_get_result_running_returns_none():
    """get_result returns None for a still-running sub-agent."""
    config = _make_config()
    mgr = SubagentManager(config)

    from datetime import datetime

    run = SubagentRun(
        id="gr2",
        task="running",
        label=None,
        process=AsyncMock(),
        result_file=Path("/tmp/gr2.json"),
        started_at=datetime.now(),
        completed=False,
    )
    mgr.active_runs["gr2"] = run

    assert mgr.get_result("gr2") is None


async def test_get_result_unknown_returns_none():
    """get_result returns None for unknown run_id."""
    config = _make_config()
    mgr = SubagentManager(config)
    assert mgr.get_result("unknown") is None


# ── Persistent subagent tests ──


class TestSpawnSession:
    async def test_spawn_session_creates_run(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        with patch("koi.subagent.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_exec.return_value = mock_proc

            # Patch idle watcher to not actually run
            with patch.object(mgr, "_idle_watcher", new_callable=AsyncMock):
                result = await mgr.spawn_session(label="test-session")

        assert result["status"] == "accepted"
        assert result["label"] == "test-session"
        assert "run_id" in result
        run = mgr.active_runs[result["run_id"]]
        assert run.mode == "session"
        assert run.label == "test-session"
        assert run.last_activity is not None

    async def test_spawn_session_label_uniqueness(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        with patch("koi.subagent.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_exec.return_value = mock_proc

            with patch.object(mgr, "_idle_watcher", new_callable=AsyncMock):
                r1 = await mgr.spawn_session(label="dup")
                r2 = await mgr.spawn_session(label="dup")

        assert r1["status"] == "accepted"
        assert r2["status"] == "error"
        assert "already exists" in r2["error"]

    async def test_spawn_session_depth_guard(self):
        config = MagicMock()
        mgr = SubagentManager(config, max_depth=2)
        mgr._depth = 2
        result = await mgr.spawn_session(label="deep")
        assert result["status"] == "error"
        assert "depth" in result["error"].lower()

    async def test_spawn_session_children_guard(self):
        config = MagicMock()
        mgr = SubagentManager(config, max_children=1)

        with patch("koi.subagent.asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.stdin = MagicMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stderr = MagicMock()
            mock_exec.return_value = mock_proc

            with patch.object(mgr, "_idle_watcher", new_callable=AsyncMock):
                await mgr.spawn_session(label="first")
                result = await mgr.spawn_session(label="second")

        assert result["status"] == "error"
        assert "Max children" in result["error"]


class TestSendToSubagent:
    async def test_send_success(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        # Create a fake session run
        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        response_json = json.dumps({"type": "response", "content": "Done!", "usage": {}})
        mock_proc.stdout = MagicMock()
        mock_proc.stdout.readline = AsyncMock(return_value=(response_json + "\n").encode())

        run = SubagentRun(
            id="abc",
            task="[session:test]",
            label="test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        result = await mgr.send("test", "do something")
        assert result["type"] == "response"
        assert result["content"] == "Done!"
        mock_proc.stdin.write.assert_called_once()

    async def test_send_not_found(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        result = await mgr.send("nonexistent", "hello")
        assert "error" in result

    async def test_send_to_completed_session_by_label(self):
        """Label lookup skips completed sessions → 'not found'."""
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="[session:done]",
            label="done",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            completed=True,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        result = await mgr.send("done", "hello")
        assert "error" in result

    async def test_send_to_completed_session_by_id(self):
        """ID lookup finds completed session → 'has ended'."""
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="[session:done]",
            label="done",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            completed=True,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        result = await mgr.send("abc", "hello")
        assert "error" in result
        assert "ended" in result["error"]

    async def test_send_to_oneshot_run(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="one-shot",
            label="runner",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="run",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        # Use ID to find it (label lookup skips non-session runs)
        result = await mgr.send("abc", "hello")
        assert "error" in result
        assert "not a persistent" in result["error"].lower()

    async def test_send_broken_pipe(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock(side_effect=BrokenPipeError)

        run = SubagentRun(
            id="abc",
            task="[session:broken]",
            label="broken",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        result = await mgr.send("broken", "hello")
        assert "error" in result
        assert run.completed is True


class TestFindSession:
    def test_find_by_label(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="[session:my-session]",
            label="my-session",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        assert mgr._find_session("my-session") is run
        assert mgr._find_session("abc") is run
        assert mgr._find_session("nonexistent") is None

    def test_find_skips_completed(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="[session:old]",
            label="old",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            completed=True,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        # ID lookup still works (returns the run even if completed)
        assert mgr._find_session("abc") is run
        # Label lookup skips completed
        assert mgr._find_session("old") is None


class TestListRunsWithSessions:
    def test_includes_session_info(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="[session:test]",
            label="test",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        runs = mgr.list_runs()
        assert len(runs) == 1
        assert runs[0]["mode"] == "session"
        assert "idle_seconds" in runs[0]


# ── Lifecycle fixes tests ──


class TestKillRun:
    async def test_kill_run_marks_completed(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()

        run = SubagentRun(
            id="abc",
            task="test",
            label="test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="run",
        )
        mgr.active_runs["abc"] = run

        await mgr._kill_run(run, "test reason")
        assert run.completed is True
        assert run.error == "test reason"
        mock_proc.kill.assert_called_once()

    async def test_kill_run_acp_closes_gracefully(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        mock_acp = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # already exited after close

        run = SubagentRun(
            id="abc",
            task="[acp]",
            label="acp-test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            harness="acp",
            acp_session=mock_acp,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        await mgr._kill_run(run, "cleanup")
        mock_acp.close.assert_called_once()
        assert run.completed is True

    async def test_kill_run_pipe_sends_shutdown(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdin = MagicMock()
        mock_proc.stdin.write = MagicMock()
        mock_proc.stdin.drain = AsyncMock()
        mock_proc.wait = AsyncMock(side_effect=asyncio.TimeoutError)
        mock_proc.kill = MagicMock()

        run = SubagentRun(
            id="abc",
            task="[session:test]",
            label="test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        await mgr._kill_run(run, "idle")
        mock_proc.stdin.write.assert_called_once()  # shutdown message
        mock_proc.kill.assert_called_once()  # force kill after timeout

    async def test_kill_already_completed_noop(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        run = SubagentRun(
            id="abc",
            task="done",
            label="done",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="run",
            completed=True,
        )
        mgr.active_runs["abc"] = run

        await mgr._kill_run(run, "should not change")
        assert run.error is None  # unchanged


class TestKillAll:
    async def test_kills_all_active(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        for i in range(3):
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.kill = MagicMock()
            run = SubagentRun(
                id=f"r{i}",
                task=f"task{i}",
                label=f"label{i}",
                process=mock_proc,
                result_file=Path(f"/tmp/fake{i}.json"),
                started_at=datetime.now(),
                mode="run",
            )
            mgr.active_runs[f"r{i}"] = run

        # Complete one
        mgr.active_runs["r0"].completed = True

        result = await mgr.kill_all()
        assert result["count"] == 2
        assert "r1" in result["ids"]
        assert "r2" in result["ids"]


class TestProcessHealthCheck:
    async def test_send_detects_dead_process(self):
        config = MagicMock()
        mgr = SubagentManager(config)

        mock_proc = MagicMock()
        mock_proc.returncode = 1  # already dead

        run = SubagentRun(
            id="abc",
            task="[session:dead]",
            label="dead",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run

        result = await mgr.send("abc", "hello")
        assert "error" in result
        assert "exit code 1" in result["error"]
        assert run.completed is True


class TestCleanupCompleted:
    def test_removes_old_completed(self):
        from datetime import timedelta

        config = MagicMock()
        mgr = SubagentManager(config)

        old_run = SubagentRun(
            id="old",
            task="old task",
            label="old",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now() - timedelta(hours=2),
            mode="run",
            completed=True,
        )
        new_run = SubagentRun(
            id="new",
            task="new task",
            label="new",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="run",
            completed=True,
        )
        active_run = SubagentRun(
            id="active",
            task="active task",
            label="active",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now() - timedelta(hours=2),
            mode="run",
            completed=False,
        )

        mgr.active_runs = {"old": old_run, "new": new_run, "active": active_run}
        removed = mgr.cleanup_completed(max_age_seconds=3600)
        assert removed == 1
        assert "old" not in mgr.active_runs
        assert "new" in mgr.active_runs  # too recent
        assert "active" in mgr.active_runs  # not completed


# ── ACP session lifecycle tests ──


class TestSpawnACPSession:
    async def test_unknown_agent(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        result = await mgr.spawn_acp_session(agent_name="nonexistent", label="test")
        assert result["status"] == "error"
        assert "Unknown agent" in result["error"]

    async def test_unavailable_agent(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        with patch("koi.subagent.get_agent") as mock_get:
            agent = MagicMock()
            agent.is_available.return_value = False
            agent.check_binary = "fake-binary"
            mock_get.return_value = agent
            result = await mgr.spawn_acp_session(agent_name="fake", label="test")
        assert result["status"] == "error"
        assert "not installed" in result["error"]

    async def test_depth_guard(self):
        config = MagicMock()
        mgr = SubagentManager(config, max_depth=1)
        mgr._depth = 1
        result = await mgr.spawn_acp_session(agent_name="claude-code", label="test")
        assert result["status"] == "error"
        assert "depth" in result["error"].lower()

    async def test_children_guard(self):
        config = MagicMock()
        mgr = SubagentManager(config, max_children=0)
        result = await mgr.spawn_acp_session(agent_name="claude-code", label="test")
        assert result["status"] == "error"
        assert "Max children" in result["error"]

    async def test_label_uniqueness(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        # Add existing session with same label
        run = SubagentRun(
            id="existing",
            task="[session:dup]",
            label="dup",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            last_activity=datetime.now(),
        )
        mgr.active_runs["existing"] = run
        result = await mgr.spawn_acp_session(agent_name="claude-code", label="dup")
        assert result["status"] == "error"
        assert "already exists" in result["error"]


class TestSendACPSession:
    async def test_send_acp_not_found(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        result = await mgr.send_acp("nope", "hello")
        assert "error" in result

    async def test_send_acp_not_acp(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        run = SubagentRun(
            id="abc",
            task="[session:test]",
            label="test",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            harness="koi",
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run
        result = await mgr.send_acp("abc", "hello")
        assert "error" in result
        assert "not an ACP" in result["error"]

    async def test_send_acp_success(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        mock_acp = AsyncMock()
        mock_acp.send = AsyncMock(
            return_value=ACPResult(content="Done!", stop_reason="end_turn", tool_calls=[], thoughts="")
        )
        run = SubagentRun(
            id="abc",
            task="[acp]",
            label="acp-test",
            process=MagicMock(),
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            harness="acp",
            acp_session=mock_acp,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run
        result = await mgr.send_acp("abc", "do stuff")
        assert result["content"] == "Done!"
        assert result["type"] == "response"

    async def test_send_acp_timeout_kills(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        mock_acp = AsyncMock()
        mock_acp.send = AsyncMock(return_value=ACPResult(content="", stop_reason="timeout"))
        mock_acp.close = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        run = SubagentRun(
            id="abc",
            task="[acp]",
            label="timeout-test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            harness="acp",
            acp_session=mock_acp,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run
        result = await mgr.send_acp("abc", "slow task")
        assert "error" in result
        assert run.completed is True

    async def test_send_acp_exception_kills(self):
        config = MagicMock()
        mgr = SubagentManager(config)
        mock_acp = AsyncMock()
        mock_acp.send = AsyncMock(side_effect=RuntimeError("connection lost"))
        mock_acp.close = AsyncMock()
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        run = SubagentRun(
            id="abc",
            task="[acp]",
            label="err-test",
            process=mock_proc,
            result_file=Path("/tmp/fake.json"),
            started_at=datetime.now(),
            mode="session",
            harness="acp",
            acp_session=mock_acp,
            last_activity=datetime.now(),
        )
        mgr.active_runs["abc"] = run
        result = await mgr.send_acp("abc", "hello")
        assert "error" in result
        assert run.completed is True
