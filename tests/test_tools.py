"""Tests for tools module."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import yaml

from koi.sandbox import Sandbox
from koi.tools import ToolExecutor, get_tool_definitions


def _make_sandbox(tmpdir: str, blocked_patterns=None):
    """Create a Sandbox with a sandbox.yaml allowing tmpdir."""
    td = Path(tmpdir)
    agent_dir = td / ".agent"
    agent_dir.mkdir(exist_ok=True)
    cfg = {
        "filesystem": {"allowed_paths": [str(td)]},
        "commands": {},
    }
    if blocked_patterns:
        cfg["commands"]["blocked_patterns"] = blocked_patterns
    (agent_dir / "sandbox.yaml").write_text(yaml.dump(cfg))
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
    """Verify all 14 tools are defined."""
    tools = get_tool_definitions()
    tool_names = {tool["function"]["name"] for tool in tools}
    expected = {
        "read_file",
        "write_file",
        "edit_file",
        "exec_command",
        "web_search",
        "web_fetch",
        "update_memory",
        "read_skill",
        "add_cron_job",
        "list_cron_jobs",
        "remove_cron_job",
        "create_alert",
        "list_alerts",
        "resolve_alert",
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
        assert result["content"] == "b\nc\n"
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
        # Create .agent dir for memory
        agent_dir = Path(temp_dir) / ".agent"
        agent_dir.mkdir(exist_ok=True)
        mem_file = agent_dir / "MEMORY.md"
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
            (Path(temp_dir) / ".agent").mkdir()
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

            alerts_dir = Path(temp_dir) / ".agent" / "alerts"
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
            alerts_dir = Path(temp_dir) / ".agent" / "alerts"
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
            alerts_dir = Path(temp_dir) / ".agent" / "alerts"
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
