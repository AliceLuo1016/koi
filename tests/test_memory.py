"""Tests for memory module."""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from koi.memory import Memory


def test_memory_initialization():
    """Test Memory class initialization."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "MEMORY.md"
        memory = Memory(memory_path)
        
        assert memory.get_path() == memory_path
        assert not memory.exists()


def test_memory_save_and_load():
    """Test saving and loading memory content."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "MEMORY.md"
        memory = Memory(memory_path)
        
        test_content = "This is a test memory entry.\n\nIt has multiple lines."
        
        # Save content
        memory.save(test_content)
        
        # Check file exists
        assert memory.exists()
        assert memory_path.exists()
        
        # Load content
        loaded_content = memory.load()
        assert loaded_content == test_content


def test_memory_load_nonexistent():
    """Test loading from non-existent memory file."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "nonexistent.md"
        memory = Memory(memory_path)
        
        # Should return empty string for non-existent file
        content = memory.load()
        assert content == ""


def test_memory_append():
    """Test appending content to memory."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "MEMORY.md"
        memory = Memory(memory_path)
        
        # Initial content
        initial_content = "Initial memory content."
        memory.save(initial_content)
        
        # Append content
        additional_content = "Additional memory content."
        memory.append(additional_content)
        
        # Check result
        final_content = memory.load()
        assert "Initial memory content." in final_content
        assert "Additional memory content." in final_content


def test_memory_append_to_empty():
    """Test appending content to empty memory."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "MEMORY.md"
        memory = Memory(memory_path)
        
        # Append to non-existent file
        content = "First memory entry."
        memory.append(content)
        
        # Check result
        loaded_content = memory.load()
        assert "First memory entry." in loaded_content


def test_memory_append_formatting():
    """Test that append properly handles newlines."""
    with TemporaryDirectory() as temp_dir:
        memory_path = Path(temp_dir) / "MEMORY.md"
        memory = Memory(memory_path)
        
        # Save initial content without trailing newline
        memory.save("Line 1")
        
        # Append content
        memory.append("Line 2")
        
        # Check formatting
        content = memory.load()
        lines = content.split('\n')
        assert len(lines) >= 2
        assert "Line 1" in content
        assert "Line 2" in content


def test_memory_default_path():
    """Test memory with default path."""
    memory = Memory()
    
    # Should use .agent/MEMORY.md as default
    expected_path = Path.cwd() / ".agent" / "MEMORY.md"
    assert memory.get_path() == expected_path