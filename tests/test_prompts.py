"""Tests for prompts module."""

from koi.config import Config
from koi.prompts import (
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
