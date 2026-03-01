"""Test agent.py module."""
import signal
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from koi.agent import Agent, _fmt_num, strip_thinking_tags
from koi.config import Config


def test_fmt_num():
    """Test _fmt_num helper function."""
    assert _fmt_num(500) == "500"
    assert _fmt_num(1500) == "1.5k"
    assert _fmt_num(1_000_000) == "1.0M"
    assert _fmt_num(2_500_000) == "2.5M"


def test_strip_thinking_tags():
    """Test strip_thinking_tags function."""
    # No thinking tags
    visible, thinking = strip_thinking_tags("Just regular text")
    assert visible == "Just regular text"
    assert thinking == ""
    
    # With thinking tags
    text = "<think>This is internal</think>This is visible"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "This is visible"
    assert thinking == "This is internal"
    
    # Multiple thinking blocks (note: actual implementation joins with \n)
    text = "<think>Think 1</think>Visible 1<think>Think 2</think>Visible 2"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Visible 1Visible 2"
    assert thinking == "Think 1\nThink 2"
    
    # With final tags
    text = "<think>Internal</think><final>Final answer</final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Final answer"
    assert thinking == "Internal"


def test_agent_init():
    """Test Agent initialization."""
    config = Config({
        "model": "test-model",
        "api_key": "test-key",
        "max_tokens": 1000,
        "context_window": 8000,
        "temperature": 0.7
    })
    
    with patch('koi.agent.LLMClient'), \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'):
        
        agent = Agent(config)
        
        assert agent.config == config
        assert isinstance(agent.messages, list)
        assert agent.messages == []


def test_agent_init_non_interactive():
    """Test Agent initialization in non-interactive mode."""
    config = Config({
        "model": "test-model",
        "api_key": "test-key"
    })
    
    with patch('koi.agent.LLMClient'), \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'):
        
        # Non-interactive mode is passed to run_task, not stored as attribute
        agent = Agent(config, non_interactive=True)
        assert isinstance(agent, Agent)


@pytest.mark.asyncio
async def test_handle_command_help():
    """Test _handle_command with /help command."""
    config = Config({"model": "test", "api_key": "test"})
    
    with patch('koi.agent.LLMClient'), \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'), \
         patch('koi.agent.console') as mock_console:
        
        agent = Agent(config)
        await agent._handle_command("/help")
        
        # Should have printed help text
        assert mock_console.print.called


@pytest.mark.asyncio
async def test_handle_command_new():
    """Test _handle_command with /new command."""
    config = Config({"model": "test", "api_key": "test"})
    
    with patch('koi.agent.LLMClient'), \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'), \
         patch('koi.agent.console') as mock_console:
        
        agent = Agent(config)
        agent.messages = [{"role": "user", "content": "test"}]
        
        await agent._handle_command("/new")
        
        # Should clear messages
        assert agent.messages == []
        assert mock_console.print.called


@pytest.mark.asyncio
async def test_handle_command_usage():
    """Test _handle_command with /usage command."""
    config = Config({"model": "test", "api_key": "test"})
    
    with patch('koi.agent.LLMClient') as mock_llm, \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'), \
         patch('koi.agent.console') as mock_console, \
         patch('koi.usage.get_usage_history') as mock_history:
        
        mock_llm.return_value.usage.summary.return_value = "Test usage"
        mock_history.return_value = "Test history"
        
        agent = Agent(config)
        await agent._handle_command("/usage")
        
        # Should print usage and history
        assert mock_console.print.call_count >= 2
        mock_history.assert_called_once()


@pytest.mark.asyncio
async def test_on_subagent_complete():
    """Test _on_subagent_complete callback."""
    config = Config({"model": "test", "api_key": "test"})
    
    with patch('koi.agent.LLMClient'), \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor'), \
         patch('koi.agent.console') as mock_console:
        
        agent = Agent(config)
        
        # Mock subagent run
        mock_run = Mock()
        mock_run.id = "test-id"
        mock_run.exit_code = 0
        mock_run.label = "test-label"
        mock_run.result = {"summary": "Test completed successfully"}
        mock_run.error = None
        mock_run.stdout = None
        
        await agent._on_subagent_complete(mock_run)
        
        # Should print completion notification and prompt
        assert mock_console.print.called
        assert mock_console.file.write.called
        assert mock_console.file.flush.called
        
        # Should add result to pending messages
        assert len(agent._pending_subagent_results) == 1

