"""Tests for prompts module."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from koi.config import Config
from koi.prompts import (
    _build_alerts_section,
    _build_memory_section,
    _build_project_section,
    _format_tool_result,
    build_system_prompt,
    build_tool_call_message,
    build_tool_result_message,
)


def test_build_tool_result_message():
    """Verifies structure of tool result message."""
    tool_call = {"id": "call_123", "function": {"name": "read_file"}}
    result = {"success": True, "content": "file contents"}

    msg = build_tool_result_message(tool_call, result)
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_123"
    assert msg["content"] == "file contents"


def test_build_tool_call_message():
    """Verifies structure of tool call message."""
    tool_call = {
        "id": "call_456",
        "type": "function",
        "function": {"name": "exec_command", "arguments": '{"command": "ls"}'},
    }

    msg = build_tool_call_message(tool_call)
    assert msg["role"] == "assistant"
    assert msg["content"] is None
    assert msg["tool_calls"] == [tool_call]


def test_format_tool_result_error():
    """Error result formatting."""
    result = {"success": False, "error": "File not found"}
    formatted = _format_tool_result(result)
    assert formatted == "Error: File not found"


def test_format_tool_result_content():
    """Content result."""
    result = {"success": True, "content": "hello world"}
    formatted = _format_tool_result(result)
    assert formatted == "hello world"


def test_format_tool_result_message():
    """Message result."""
    result = {"success": True, "message": "Done successfully"}
    formatted = _format_tool_result(result)
    assert formatted == "Done successfully"


def test_format_tool_result_stdout():
    """stdout + stderr result."""
    result = {"success": True, "stdout": "output here", "stderr": "warning"}
    formatted = _format_tool_result(result)
    assert "output here" in formatted
    assert "[stderr]: warning" in formatted


def test_format_tool_result_stdout_no_stderr():
    """stdout only, no stderr."""
    result = {"success": True, "stdout": "output here"}
    formatted = _format_tool_result(result)
    assert formatted == "output here"
    assert "stderr" not in formatted


def test_format_tool_result_json_fallback():
    """Complex result becomes JSON."""
    result = {"success": True, "data": {"key": "value"}, "count": 42}
    formatted = _format_tool_result(result)
    assert '"key"' in formatted
    assert '"value"' in formatted


def test_build_system_prompt_contains_sections():
    """Verify prompt has tools and context sections."""
    config = Config()
    prompt = build_system_prompt(config)

    assert "Available Tools:" in prompt
    assert "Current Context:" in prompt
    assert "Working Directory:" in prompt
    assert "read_file" in prompt
    assert "exec_command" in prompt


def test_build_system_prompt_non_interactive():
    """Non-interactive mode appends cron note to prompt."""
    config = Config()
    prompt = build_system_prompt(config, non_interactive=True)
    assert "non-interactive" in prompt.lower() or "cron" in prompt.lower()


def test_build_project_section_with_agents_md():
    """_build_project_section returns content when AGENTS.md exists."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            koi_dir = Path(td) / ".koi"
            koi_dir.mkdir()
            (koi_dir / "AGENTS.md").write_text("# Project Rules\nAlways test.")
            section = _build_project_section()
            assert "Project Instructions:" in section
            assert "Always test." in section
        finally:
            os.chdir(old_cwd)


def test_build_project_section_no_agents_md():
    """_build_project_section returns empty string when AGENTS.md is absent."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            (Path(td) / ".koi").mkdir()
            assert _build_project_section() == ""
        finally:
            os.chdir(old_cwd)


def test_build_memory_section_with_content():
    """_build_memory_section returns content when MEMORY.md has text."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            koi_dir = Path(td) / ".koi"
            koi_dir.mkdir()
            (koi_dir / "MEMORY.md").write_text("Remember: use Python 3.9.")
            section = _build_memory_section()
            assert "Memory:" in section
            assert "Remember: use Python 3.9." in section
        finally:
            os.chdir(old_cwd)


def test_build_memory_section_empty():
    """_build_memory_section returns empty string when memory is blank."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            (Path(td) / ".koi").mkdir()
            assert _build_memory_section() == ""
        finally:
            os.chdir(old_cwd)


def test_build_alerts_section_with_pending():
    """_build_alerts_section reports pending alert count and titles."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            alerts_dir = Path(td) / ".koi" / "alerts"
            alerts_dir.mkdir(parents=True)
            (alerts_dir / "alert1.md").write_text(
                "# Disk Full\n- **Status:** pending\n"
            )
            (alerts_dir / "alert2.md").write_text(
                "# CPU High\n- **Status:** dismissed\n"
            )
            section = _build_alerts_section()
            assert "pending" in section.lower() or "alert" in section.lower()
            assert "Disk Full" in section
            assert "CPU High" not in section  # dismissed, not shown
        finally:
            os.chdir(old_cwd)


def test_build_alerts_section_no_dir():
    """_build_alerts_section returns empty string when alerts dir missing."""
    with TemporaryDirectory() as td:
        old_cwd = os.getcwd()
        os.chdir(td)
        try:
            (Path(td) / ".koi").mkdir()
            assert _build_alerts_section() == ""
        finally:
            os.chdir(old_cwd)
