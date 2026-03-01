"""Tests for tools module."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import yaml

from koi.sandbox import Sandbox
from koi.tools import ToolExecutor, get_tool_definitions


def _make_sandbox(tmpdir: str, blocked_patterns=None):
    """Create a Sandbox with a sandbox.yaml allowing tmpdir."""
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


def test_get_tool_definitions():
    """Test that tool definitions are properly formatted."""
    tools = get_tool_definitions()

    assert isinstance(tools, list)
    assert len(tools) > 0

    # Check that all tools have proper structure
    for tool in tools:
        assert tool["type"] == "function"
        assert "function" in tool

        func = tool["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params


def test_tool_names():
    """Test that expected tools are defined."""
    tools = get_tool_definitions()
    tool_names = [tool["function"]["name"] for tool in tools]

    expected_tools = [
        "read_file",
        "write_file",
        "edit_file",
        "exec_command",
        "web_search",
        "web_fetch",
        "read_skill",
    ]

    for expected in expected_tools:
        assert expected in tool_names


def test_all_tool_names_present():
    """Verify all 20 tools are defined."""
    tools = get_tool_definitions()
    tool_names = {tool["function"]["name"] for tool in tools}
    expected = {
        "read_file",
        "write_file",
        "edit_file",
        "exec_command",
        "glob_files",
        "grep_files",
        "web_search",
        "web_fetch",
        "update_memory",
        "read_skill",
        "add_cron_job",
        "list_cron_jobs",
        "remove_cron_job",
        "remove_file",
        "create_alert",
        "list_alerts",
        "resolve_alert",
        "spawn_subagent",
        "list_subagents",
        "kill_subagent",
    }
    assert tool_names == expected


async def test_tool_executor_read_file():
    """Test read_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)

        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "Line 1\nLine 2\nLine 3\n"
        test_file.write_text(test_content)

        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(test_file)}),
            }
        }

        result = await executor.execute_tool(tool_call)

        assert result["success"] is True
        assert result["content"] == test_content
        assert result["lines_read"] == 3


async def test_tool_executor_read_file_not_found():
    """Test read_file with non-existent file."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(Path(temp_dir) / "nope.txt")}),
            }
        }

        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "not found" in result["error"]


async def test_tool_executor_read_file_with_offset_limit():
    """Test read_file with offset and limit."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        test_file = Path(temp_dir) / "lines.txt"
        test_file.write_text("a\nb\nc\nd\ne\n")

        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps(
                    {"path": str(test_file), "offset": 2, "limit": 2}
                ),
            }
        }

        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["content"].startswith("b\nc\n")
        assert "output truncated" in result["content"]
        assert result["lines_read"] == 2


async def test_tool_executor_write_file():
    """Test write_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        test_file = Path(temp_dir) / "new_file.txt"
        test_content = "This is new content."

        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "write_file",
                "arguments": json.dumps(
                    {"path": str(test_file), "content": test_content}
                ),
            }
        }

        result = await executor.execute_tool(tool_call)

        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == test_content


async def test_tool_executor_edit_file():
    """Test edit_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        test_file = Path(temp_dir) / "edit_test.txt"
        original_content = "Hello world!\nThis is a test."
        test_file.write_text(original_content)

        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "path": str(test_file),
                        "old_text": "Hello world!",
                        "new_text": "Hello universe!",
                    }
                ),
            }
        }

        result = await executor.execute_tool(tool_call)

        assert result["success"] is True

        new_content = test_file.read_text()
        assert "Hello universe!" in new_content
        assert "Hello world!" not in new_content
        assert "This is a test." in new_content


async def test_tool_executor_edit_file_text_not_found():
    """Test edit_file when old_text is not in the file."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        test_file = Path(temp_dir) / "edit_miss.txt"
        test_file.write_text("some content")

        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps(
                    {
                        "path": str(test_file),
                        "old_text": "nonexistent text",
                        "new_text": "replacement",
                    }
                ),
            }
        }

        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "not found" in result["error"].lower()


async def test_tool_executor_exec_command_safe():
    """Test exec_command with safe command."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "exec_command",
                "arguments": json.dumps({"command": "echo 'Hello, World!'"}),
            }
        }

        result = await executor.execute_tool(tool_call)

        assert result["success"] is True
        assert "Hello, World!" in result["stdout"]
        assert result["exit_code"] == 0


async def test_sandbox_blocks_dangerous_commands():
    """Test that sandbox blocks commands matching blocked_patterns."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(
            temp_dir, blocked_patterns=[r"rm\s+-rf\s+/", r"DROP\s+TABLE"]
        )

        allowed, reason, _ = sandbox.check_command("rm -rf /")
        assert not allowed
        assert "Blocked command pattern" in reason

        allowed2, reason2, _ = sandbox.check_command("DROP TABLE users")
        assert not allowed2
        assert "Blocked command pattern" in reason2

        allowed3, _, _ = sandbox.check_command("echo hello")
        assert allowed3


async def test_tool_executor_exec_command_dangerous():
    """Test exec_command blocks dangerous commands via sandbox."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir, blocked_patterns=[r"rm\s+-rf\s+/"])
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "exec_command",
                "arguments": json.dumps({"command": "rm -rf /"}),
            }
        }

        result = await executor.execute_tool(tool_call)

        assert result["success"] is False
        assert "Blocked command pattern" in result["error"]


async def test_tool_executor_invalid_json():
    """Test tool executor with invalid JSON arguments."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)

    tool_call = {
        "function": {
            "name": "read_file",
            "arguments": "invalid json",
        }
    }

    result = await executor.execute_tool(tool_call)

    assert result["success"] is False
    assert "Failed to parse arguments" in result["error"]


async def test_tool_executor_unknown_function():
    """Test tool executor with unknown function."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)

    tool_call = {
        "function": {
            "name": "unknown_function",
            "arguments": json.dumps({}),
        }
    }

    result = await executor.execute_tool(tool_call)

    assert result["success"] is False
    assert "Unknown function" in result["error"]


async def test_tool_executor_web_search_placeholder():
    """Test web_search returns TODO placeholder."""
    executor = ToolExecutor(Mock())

    tool_call = {
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"query": "test query"}),
        }
    }

    result = await executor.execute_tool(tool_call)
    assert result["success"] is True
    assert "TODO" in result["message"]


async def test_tool_executor_update_memory():
    """Test update_memory appends to memory."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        # Create .koi dir for memory
        koi_dir = Path(temp_dir) / ".koi"
        koi_dir.mkdir(exist_ok=True)
        mem_file = koi_dir / "MEMORY.md"
        mem_file.write_text("# Memory\n")

        executor = ToolExecutor(Mock(), sandbox)

        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            tool_call = {
                "function": {
                    "name": "update_memory",
                    "arguments": json.dumps({"content": "remember this"}),
                }
            }

            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert "remember this" in mem_file.read_text()
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_read_skill():
    """Test read_skill loads skill content."""
    with TemporaryDirectory() as temp_dir:
        # Create a skill
        skill_dir = Path(temp_dir) / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test Skill\nDoes testing.")

        from koi.skills import SkillsManager

        sm = SkillsManager([str(Path(temp_dir) / "skills")])
        executor = ToolExecutor(sm)

        tool_call = {
            "function": {
                "name": "read_skill",
                "arguments": json.dumps({"skill_name": "Test Skill"}),
            }
        }

        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert "Test Skill" in result["content"]


async def test_tool_executor_create_alert():
    """Test create_alert creates an alert file."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            (Path(temp_dir) / ".koi").mkdir()
            executor = ToolExecutor(Mock())

            tool_call = {
                "function": {
                    "name": "create_alert",
                    "arguments": json.dumps(
                        {
                            "title": "Test Alert",
                            "summary": "Something happened",
                            "severity": "high",
                            "proposed_fix": "Fix it",
                        }
                    ),
                }
            }

            result = await executor.execute_tool(tool_call)
            assert result["success"] is True

            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            alert_files = list(alerts_dir.glob("*.md"))
            assert len(alert_files) == 1
            content = alert_files[0].read_text()
            assert "Test Alert" in content
            assert "pending" in content
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_list_alerts():
    """Test list_alerts lists by status."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            alerts_dir.mkdir(parents=True)
            (alerts_dir / "alert1.md").write_text(
                "# Alert One\n- **Status:** pending\n- **Severity:** high\n"
                "- **Detected:** 2025-01-01\n- **Summary:** test\n"
            )
            (alerts_dir / "alert2.md").write_text(
                "# Alert Two\n- **Status:** dismissed\n- **Severity:** low\n"
                "- **Detected:** 2025-01-01\n- **Summary:** old\n"
            )

            executor = ToolExecutor(Mock())

            tool_call = {
                "function": {
                    "name": "list_alerts",
                    "arguments": json.dumps({"status": "pending"}),
                }
            }

            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert result["count"] == 1
            assert result["alerts"][0]["title"] == "Alert One"
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_resolve_alert():
    """Test resolve_alert changes status in file."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            alerts_dir.mkdir(parents=True)
            alert_file = alerts_dir / "test_alert.md"
            alert_file.write_text(
                "# Test\n- **Status:** pending\n- **Severity:** high\n"
            )

            executor = ToolExecutor(Mock())

            tool_call = {
                "function": {
                    "name": "resolve_alert",
                    "arguments": json.dumps(
                        {
                            "alert_file": "test_alert.md",
                            "resolution": "approved",
                        }
                    ),
                }
            }

            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert "approved" in alert_file.read_text()
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_resolve_alert_not_found():
    """resolve_alert returns error when alert file doesn't exist."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            (Path(temp_dir) / ".koi" / "alerts").mkdir(parents=True)
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "resolve_alert",
                    "arguments": json.dumps(
                        {"alert_file": "missing.md", "resolution": "dismissed"}
                    ),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is False
            assert "not found" in result["error"].lower()
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_resolve_alert_returns_fix_command():
    """resolve_alert approved with fix_command includes it in result."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            alerts_dir.mkdir(parents=True)
            (alerts_dir / "fix_alert.md").write_text(
                "# DB Full\n- **Status:** pending\n"
                "- **Fix Command:** `vacuum db`\n"
            )
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "resolve_alert",
                    "arguments": json.dumps(
                        {"alert_file": "fix_alert.md", "resolution": "approved"}
                    ),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert "fix_command" in result
            assert result["fix_command"] == "vacuum db"
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_create_alert_with_fix_command():
    """create_alert stores fix_command in the alert file."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            (Path(temp_dir) / ".koi").mkdir()
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "create_alert",
                    "arguments": json.dumps(
                        {
                            "title": "Disk Alert",
                            "summary": "Disk is full",
                            "severity": "critical",
                            "proposed_fix": "Delete old logs",
                            "fix_command": "rm -rf /var/log/old/",
                        }
                    ),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            content = list(alerts_dir.glob("*.md"))[0].read_text()
            assert "rm -rf /var/log/old/" in content
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_list_alerts_approved():
    """list_alerts filters by approved status."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            alerts_dir = Path(temp_dir) / ".koi" / "alerts"
            alerts_dir.mkdir(parents=True)
            (alerts_dir / "a1.md").write_text(
                "# Alert A\n- **Status:** approved\n- **Severity:** low\n"
                "- **Detected:** 2025-01-01\n- **Summary:** done\n"
            )
            (alerts_dir / "a2.md").write_text(
                "# Alert B\n- **Status:** pending\n- **Severity:** high\n"
                "- **Detected:** 2025-01-01\n- **Summary:** todo\n"
            )
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "list_alerts",
                    "arguments": json.dumps({"status": "approved"}),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert result["count"] == 1
            assert result["alerts"][0]["title"] == "Alert A"
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_exec_command_needs_confirm():
    """exec_command returns needs_confirmation when command matches confirm pattern."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        # Add a confirm pattern for git push
        td = Path(temp_dir)
        koi_dir = td / ".koi"
        cfg = {
            "filesystem": {"allowed_paths": [str(td)]},
            "commands": {"confirm_patterns": [r"git\s+push"]},
        }
        (koi_dir / "sandbox.yaml").write_text(yaml.dump(cfg))
        sandbox = Sandbox(project_root=td)
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "exec_command",
                "arguments": json.dumps({"command": "git push origin main"}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert result.get("needs_confirmation") is True


async def test_tool_executor_exec_command_timeout():
    """exec_command returns error when command exceeds timeout."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "exec_command",
                "arguments": json.dumps({"command": "sleep 60", "timeout": 1}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "timed out" in result["error"].lower()


async def test_tool_executor_remove_file_outside_koi():
    """remove_file denies paths outside .koi/."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            (Path(temp_dir) / ".koi").mkdir()
            target = Path(temp_dir) / "sensitive.txt"
            target.write_text("secret")
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "remove_file",
                    "arguments": json.dumps({"path": str(target)}),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is False
            assert "denied" in result["error"].lower()
            assert target.exists()  # File should NOT be deleted
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_remove_file_directory():
    """remove_file removes a directory under .koi/."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            koi_dir = Path(temp_dir) / ".koi"
            koi_dir.mkdir()
            target_dir = koi_dir / "old-skill"
            target_dir.mkdir()
            (target_dir / "SKILL.md").write_text("content")
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "remove_file",
                    "arguments": json.dumps({"path": str(target_dir)}),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert not target_dir.exists()
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_remove_file_not_found():
    """remove_file returns error when path doesn't exist."""
    with TemporaryDirectory() as temp_dir:
        import os

        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            koi_dir = Path(temp_dir) / ".koi"
            koi_dir.mkdir()
            executor = ToolExecutor(Mock())
            tool_call = {
                "function": {
                    "name": "remove_file",
                    "arguments": json.dumps({"path": str(koi_dir / "nonexistent.txt")}),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is False
            assert "not found" in result["error"].lower()
        finally:
            os.chdir(old_cwd)


async def test_tool_executor_web_fetch_html_parsing():
    """web_fetch parses HTML content, strips scripts, and prepends title."""
    import unittest.mock as mock

    executor = ToolExecutor(Mock())

    html = """<html>
<head><title>Test Page</title></head>
<body>
<script>var x = 1;</script>
<p>Hello world content.</p>
</body></html>"""

    mock_response = mock.MagicMock()
    mock_response.raise_for_status = lambda: None
    mock_response.text = html

    mock_client = mock.AsyncMock()
    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=False)
    mock_client.get = mock.AsyncMock(return_value=mock_response)

    with mock.patch("koi.tools.httpx.AsyncClient", return_value=mock_client):
        tool_call = {
            "function": {
                "name": "web_fetch",
                "arguments": json.dumps({"url": "https://example.com"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert "Test Page" in result["content"]
    assert "Hello world content." in result["content"]
    assert "var x = 1" not in result["content"]  # Script stripped


async def test_tool_executor_web_fetch_http_error():
    """web_fetch returns error on HTTP failure."""
    import unittest.mock as mock
    import httpx

    executor = ToolExecutor(Mock())

    mock_client = mock.AsyncMock()
    mock_client.__aenter__ = mock.AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = mock.AsyncMock(return_value=False)
    mock_client.get = mock.AsyncMock(side_effect=Exception("Connection refused"))

    with mock.patch("koi.tools.httpx.AsyncClient", return_value=mock_client):
        tool_call = {
            "function": {
                "name": "web_fetch",
                "arguments": json.dumps({"url": "https://bad.example.com"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is False
    assert "Connection refused" in result["error"]


# ── Additional coverage: edge cases and exception paths ──


async def test_tool_executor_read_file_is_directory():
    """read_file returns error when path points to a directory."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)

        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "not a file" in result["error"].lower()


async def test_tool_executor_edit_file_long_text_uses_summary_diff():
    """edit_file produces character-count summary when text exceeds 200 chars."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        test_file = Path(temp_dir) / "big.txt"
        old_text = "A" * 250
        new_text = "B" * 250
        test_file.write_text(old_text)

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps({
                    "path": str(test_file),
                    "old_text": old_text,
                    "new_text": new_text,
                }),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert "characters" in result["message"]


async def test_tool_executor_execute_tool_outer_exception():
    """execute_tool outer except catches TypeError from bad kwargs."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)

        # Pass an unexpected kwarg so the dispatch call raises TypeError
        # before entering _read_file's own try/except
        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": temp_dir, "bogus_kwarg": True}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "Tool execution failed" in result["error"]


async def test_tool_executor_add_cron_job_success():
    """add_cron_job tool returns job_id from CronManager."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.add_job.return_value = "abc-123"
        tool_call = {
            "function": {
                "name": "add_cron_job",
                "arguments": json.dumps({"schedule": "0 * * * *", "task": "run checks"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert result["job_id"] == "abc-123"
    assert "abc-123" in result["message"]


async def test_tool_executor_add_cron_job_error():
    """add_cron_job returns error when CronManager.add_job raises."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.add_job.side_effect = RuntimeError("cron failed")
        tool_call = {
            "function": {
                "name": "add_cron_job",
                "arguments": json.dumps({"schedule": "bad", "task": "thing"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is False
    assert "cron failed" in result["error"]


async def test_tool_executor_list_cron_jobs_success():
    """list_cron_jobs returns job list and count."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.list_jobs.return_value = [
            {"id": "j1", "schedule": "0 * * * *", "task": "do stuff"}
        ]
        tool_call = {
            "function": {
                "name": "list_cron_jobs",
                "arguments": json.dumps({}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert result["count"] == 1
    assert result["jobs"][0]["id"] == "j1"


async def test_tool_executor_list_cron_jobs_empty():
    """list_cron_jobs returns empty list when no jobs exist."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.list_jobs.return_value = []
        tool_call = {
            "function": {
                "name": "list_cron_jobs",
                "arguments": json.dumps({}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert result["count"] == 0


async def test_tool_executor_list_cron_jobs_error():
    """list_cron_jobs returns error when CronManager raises."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.list_jobs.side_effect = RuntimeError("no cron")
        tool_call = {
            "function": {
                "name": "list_cron_jobs",
                "arguments": json.dumps({}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is False
    assert "no cron" in result["error"]


async def test_tool_executor_remove_cron_job_success():
    """remove_cron_job removes a job and returns success message."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.remove_job.return_value = None
        tool_call = {
            "function": {
                "name": "remove_cron_job",
                "arguments": json.dumps({"job_id": "j1"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is True
    assert "j1" in result["message"]


async def test_tool_executor_remove_cron_job_error():
    """remove_cron_job returns error when CronManager raises."""
    executor = ToolExecutor(Mock())

    with patch("koi.cron.CronManager") as MockCron:
        MockCron.return_value.remove_job.side_effect = KeyError("job not found")
        tool_call = {
            "function": {
                "name": "remove_cron_job",
                "arguments": json.dumps({"job_id": "missing"}),
            }
        }
        result = await executor.execute_tool(tool_call)

    assert result["success"] is False


async def test_tool_executor_update_memory_error():
    """update_memory returns error when Memory.append raises."""
    import os

    with TemporaryDirectory() as temp_dir:
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            executor = ToolExecutor(Mock())
            with patch("koi.memory.Memory") as MockMem:
                MockMem.return_value.append.side_effect = OSError("disk full")
                tool_call = {
                    "function": {
                        "name": "update_memory",
                        "arguments": json.dumps({"content": "something"}),
                    }
                }
                result = await executor.execute_tool(tool_call)
            assert result["success"] is False
            assert "disk full" in result["error"]
        finally:
            os.chdir(old_cwd)


# ── glob_files ──


async def test_tool_executor_glob_files_basic():
    """glob_files returns matching files."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "foo.py").write_text("x")
        (td / "bar.py").write_text("x")
        (td / "readme.md").write_text("x")

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "glob_files",
                "arguments": json.dumps({"pattern": "**/*.py", "base_dir": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 2
        assert all(m.endswith(".py") for m in result["matches"])


async def test_tool_executor_glob_files_no_matches():
    """glob_files returns empty list when pattern matches nothing."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "glob_files",
                "arguments": json.dumps({"pattern": "**/*.rs", "base_dir": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 0
        assert result["matches"] == []


async def test_tool_executor_glob_files_skips_hidden_dirs():
    """glob_files excludes results inside .git and node_modules."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "src").mkdir()
        (td / "src" / "main.py").write_text("x")
        (td / ".git").mkdir()
        (td / ".git" / "config.py").write_text("x")  # should be excluded
        (td / "node_modules").mkdir()
        (td / "node_modules" / "dep.py").write_text("x")  # should be excluded

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "glob_files",
                "arguments": json.dumps({"pattern": "**/*.py", "base_dir": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"] == ["src/main.py"]


async def test_tool_executor_glob_files_sandbox_blocked():
    """glob_files returns error when base_dir is outside sandbox."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "glob_files",
                "arguments": json.dumps({"pattern": "**/*.py", "base_dir": "/etc"}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False


async def test_tool_executor_glob_files_default_base_dir():
    """glob_files defaults to CWD when base_dir is omitted."""
    import os
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "hello.py").write_text("x")
        old_cwd = os.getcwd()
        os.chdir(temp_dir)
        try:
            executor = ToolExecutor(Mock(), sandbox)
            tool_call = {
                "function": {
                    "name": "glob_files",
                    "arguments": json.dumps({"pattern": "*.py"}),
                }
            }
            result = await executor.execute_tool(tool_call)
            assert result["success"] is True
            assert "hello.py" in result["matches"]
        finally:
            os.chdir(old_cwd)


# ── grep_files ──


async def test_tool_executor_grep_files_basic():
    """grep_files returns matching lines with file and line number."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "a.py").write_text("def foo():\n    pass\n")
        (td / "b.py").write_text("def bar():\n    return 1\n")

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "def foo", "path": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["file"] == "a.py"
        assert result["matches"][0]["line"] == 1
        assert "def foo" in result["matches"][0]["text"]


async def test_tool_executor_grep_files_no_matches():
    """grep_files returns empty list when pattern not found."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        (Path(temp_dir) / "x.py").write_text("hello world\n")
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "zzznomatch", "path": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 0


async def test_tool_executor_grep_files_case_insensitive():
    """grep_files case_insensitive=True matches regardless of case."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        (Path(temp_dir) / "x.py").write_text("Hello World\n")
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({
                    "pattern": "hello world",
                    "path": temp_dir,
                    "case_insensitive": True,
                }),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1


async def test_tool_executor_grep_files_file_glob_filter():
    """grep_files file_glob restricts search to matching filenames."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "code.py").write_text("import os\n")
        (td / "notes.txt").write_text("import something\n")  # should be excluded

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({
                    "pattern": "import",
                    "path": temp_dir,
                    "file_glob": "*.py",
                }),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["file"] == "code.py"


async def test_tool_executor_grep_files_single_file():
    """grep_files works when path is a single file."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        target = Path(temp_dir) / "target.py"
        target.write_text("line one\nline two\nline three\n")

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "two", "path": str(target)}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["line"] == 2


async def test_tool_executor_grep_files_invalid_regex():
    """grep_files returns error for invalid regex pattern."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "[unclosed", "path": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
        assert "Invalid regex" in result["error"]


async def test_tool_executor_grep_files_skips_binary():
    """grep_files silently skips files that can't be decoded as UTF-8."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        td = Path(temp_dir)
        (td / "binary.bin").write_bytes(b"\xff\xfe binary \x00\x01")
        (td / "text.py").write_text("find me\n")

        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "find me", "path": temp_dir}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is True
        assert result["count"] == 1
        assert result["matches"][0]["file"] == "text.py"


async def test_tool_executor_grep_files_sandbox_blocked():
    """grep_files returns error when path is outside sandbox."""
    with TemporaryDirectory() as temp_dir:
        sandbox = _make_sandbox(temp_dir)
        executor = ToolExecutor(Mock(), sandbox)
        tool_call = {
            "function": {
                "name": "grep_files",
                "arguments": json.dumps({"pattern": "root", "path": "/etc"}),
            }
        }
        result = await executor.execute_tool(tool_call)
        assert result["success"] is False
