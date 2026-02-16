"""Tests for tools module."""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from koi.tools import ToolExecutor, get_tool_definitions, DANGEROUS_PATTERNS


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
        "read_skill"
    ]
    
    for expected in expected_tools:
        assert expected in tool_names


@pytest.mark.asyncio
async def test_tool_executor_read_file():
    """Test read_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        # Create test file
        test_file = Path(temp_dir) / "test.txt"
        test_content = "Line 1\nLine 2\nLine 3\n"
        test_file.write_text(test_content)
        
        # Create tool executor
        skills_manager = Mock()
        executor = ToolExecutor(skills_manager)
        
        # Test tool call
        tool_call = {
            "function": {
                "name": "read_file",
                "arguments": json.dumps({"path": str(test_file)})
            }
        }
        
        result = await executor.execute_tool(tool_call)
        
        assert result["success"] is True
        assert result["content"] == test_content
        assert result["lines_read"] == 3


@pytest.mark.asyncio
async def test_tool_executor_write_file():
    """Test write_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "new_file.txt"
        test_content = "This is new content."
        
        # Create tool executor
        skills_manager = Mock()
        executor = ToolExecutor(skills_manager)
        
        # Test tool call
        tool_call = {
            "function": {
                "name": "write_file",
                "arguments": json.dumps({
                    "path": str(test_file),
                    "content": test_content
                })
            }
        }
        
        result = await executor.execute_tool(tool_call)
        
        assert result["success"] is True
        assert test_file.exists()
        assert test_file.read_text() == test_content


@pytest.mark.asyncio
async def test_tool_executor_edit_file():
    """Test edit_file tool execution."""
    with TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "edit_test.txt"
        original_content = "Hello world!\nThis is a test."
        test_file.write_text(original_content)
        
        # Create tool executor
        skills_manager = Mock()
        executor = ToolExecutor(skills_manager)
        
        # Test tool call
        tool_call = {
            "function": {
                "name": "edit_file",
                "arguments": json.dumps({
                    "path": str(test_file),
                    "old_text": "Hello world!",
                    "new_text": "Hello universe!"
                })
            }
        }
        
        result = await executor.execute_tool(tool_call)
        
        assert result["success"] is True
        
        # Check file was edited
        new_content = test_file.read_text()
        assert "Hello universe!" in new_content
        assert "Hello world!" not in new_content
        assert "This is a test." in new_content


@pytest.mark.asyncio
async def test_tool_executor_exec_command_safe():
    """Test exec_command with safe command."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)
    
    # Test safe command
    tool_call = {
        "function": {
            "name": "exec_command",
            "arguments": json.dumps({"command": "echo 'Hello, World!'"})
        }
    }
    
    result = await executor.execute_tool(tool_call)
    
    assert result["success"] is True
    assert "Hello, World!" in result["stdout"]
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_tool_executor_exec_command_dangerous():
    """Test exec_command blocks dangerous commands."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)
    
    # Test dangerous command
    tool_call = {
        "function": {
            "name": "exec_command",
            "arguments": json.dumps({"command": "rm -rf /"})
        }
    }
    
    result = await executor.execute_tool(tool_call)
    
    assert result["success"] is False
    assert "Dangerous command detected" in result["error"]


@pytest.mark.asyncio
async def test_tool_executor_invalid_json():
    """Test tool executor with invalid JSON arguments."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)
    
    # Test invalid JSON
    tool_call = {
        "function": {
            "name": "read_file",
            "arguments": "invalid json"
        }
    }
    
    result = await executor.execute_tool(tool_call)
    
    assert result["success"] is False
    assert "Failed to parse arguments" in result["error"]


@pytest.mark.asyncio
async def test_tool_executor_unknown_function():
    """Test tool executor with unknown function."""
    skills_manager = Mock()
    executor = ToolExecutor(skills_manager)
    
    # Test unknown function
    tool_call = {
        "function": {
            "name": "unknown_function",
            "arguments": json.dumps({})
        }
    }
    
    result = await executor.execute_tool(tool_call)
    
    assert result["success"] is False
    assert "Unknown function" in result["error"]


def test_dangerous_patterns():
    """Test that dangerous command patterns are properly defined."""
    dangerous_commands = [
        "rm -rf /",
        "sudo rm -rf /home",
        "DROP TABLE users",
        "aws s3 rm --recursive s3://bucket",
        "format C:\\",
        "del /q /s C:\\*",
        "mkfs.ext4 /dev/sda1",
        "dd if=/dev/zero of=/dev/sda"
    ]
    
    import re
    
    for cmd in dangerous_commands:
        is_dangerous = any(
            re.search(pattern, cmd, re.IGNORECASE) 
            for pattern in DANGEROUS_PATTERNS
        )
        assert is_dangerous, f"Command '{cmd}' should be detected as dangerous"