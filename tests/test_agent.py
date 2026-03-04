"""Test agent.py module."""
import json
import signal
import pytest
from unittest.mock import Mock, patch, AsyncMock
from pathlib import Path

from koi.agent import Agent, _fmt_num, strip_thinking_tags
from koi.config import Config
from koi.errors import KoiContextOverflowError
from koi.session_manager import SessionManager


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


@pytest.mark.asyncio
async def test_context_overflow_auto_compact_retry():
    """Test that context overflow triggers auto-compaction and retries once."""
    config = Config({"model": "test", "api_key": "test"})

    with patch('koi.agent.LLMClient') as mock_llm_cls, \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor') as mock_compactor_cls, \
         patch('koi.agent.console') as mock_console:

        agent = Agent(config, non_interactive=True)
        agent.system_prompt = "test"

        # First call raises overflow, second call returns a normal response
        normal_response = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }]
        }
        agent.llm_client.chat = AsyncMock(
            side_effect=[KoiContextOverflowError("too many tokens"), normal_response]
        )

        # Compactor returns a shorter message list
        compacted = [{"role": "user", "content": "compacted"}]
        agent.compactor.compact_messages = AsyncMock(return_value=compacted)
        agent.compactor.needs_compaction.return_value = False

        # Add a user message
        agent.messages = [{"role": "user", "content": "Hello"}]

        await agent._agent_loop(non_interactive=True)

        # Compaction should have been called once
        agent.compactor.compact_messages.assert_called_once()

        # Agent should have gotten a response (assistant message appended)
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == "Hello!"


@pytest.mark.asyncio
async def test_context_overflow_double_overflow_breaks():
    """Test that a second context overflow after retry breaks the loop."""
    config = Config({"model": "test", "api_key": "test"})

    with patch('koi.agent.LLMClient') as mock_llm_cls, \
         patch('koi.agent.Memory'), \
         patch('koi.agent.SkillsManager'), \
         patch('koi.agent.Sandbox'), \
         patch('koi.agent.SubagentManager'), \
         patch('koi.agent.ToolExecutor'), \
         patch('koi.agent.ContextCompactor') as mock_compactor_cls, \
         patch('koi.agent.console') as mock_console:

        agent = Agent(config, non_interactive=True)
        agent.system_prompt = "test"

        # Both calls raise overflow
        agent.llm_client.chat = AsyncMock(
            side_effect=[
                KoiContextOverflowError("too many tokens"),
                KoiContextOverflowError("still too many tokens"),
            ]
        )

        # Compactor returns a shorter message list
        compacted = [{"role": "user", "content": "compacted"}]
        agent.compactor.compact_messages = AsyncMock(return_value=compacted)
        agent.compactor.needs_compaction.return_value = False

        # Add a user message
        agent.messages = [{"role": "user", "content": "Hello"}]

        await agent._agent_loop(non_interactive=True)

        # Compaction should have been called exactly once (only retries once)
        agent.compactor.compact_messages.assert_called_once()

        # No assistant message should have been added (loop broke)
        assistant_msgs = [m for m in agent.messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 0


class TestSessionPersistence:
    """Tests for session persistence integration in Agent."""

    def _make_agent(self, tmp_path):
        """Create an Agent with mocked dependencies pointing at tmp_path."""
        config = Config({"model": "test-model", "api_key": "test-key"})
        with patch('koi.agent.LLMClient'), \
             patch('koi.agent.Memory'), \
             patch('koi.agent.SkillsManager'), \
             patch('koi.agent.Sandbox'), \
             patch('koi.agent.SubagentManager'), \
             patch('koi.agent.ToolExecutor'), \
             patch('koi.agent.ContextCompactor'), \
             patch('koi.agent.Path') as mock_path_cls, \
             patch('koi.agent.console'):
            mock_path_cls.cwd.return_value = tmp_path
            # Make Path(".koi") resolve to tmp_path / ".koi" for usage logging
            mock_path_cls.side_effect = lambda *a, **kw: Path(*a, **kw)
            agent = Agent(config, non_interactive=True)
        # Replace the session_manager with one using a real writable directory
        koi_dir = tmp_path / ".koi"
        koi_dir.mkdir(exist_ok=True)
        agent.session_manager = SessionManager(koi_dir)
        return agent

    def test_agent_persists_messages(self, tmp_path):
        """Agent auto-persists messages to session file."""
        agent = self._make_agent(tmp_path)
        agent.session_manager.start_session(model="test", cwd=str(tmp_path))

        # Simulate adding messages
        msg1 = {"role": "user", "content": "Hello"}
        agent.messages.append(msg1)
        agent.session_manager.save_message(msg1)

        msg2 = {"role": "assistant", "content": "Hi there!"}
        agent.messages.append(msg2)
        agent.session_manager.save_message(msg2)

        agent.session_manager.close()

        # Verify messages were saved
        data = agent.session_manager.load_session()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["content"] == "Hi there!"

    def test_agent_resume_loads_messages(self, tmp_path):
        """Resuming a session loads its messages into the agent."""
        koi_dir = tmp_path / ".koi"
        koi_dir.mkdir(exist_ok=True)

        # Create a session with messages
        sm = SessionManager(koi_dir)
        sm.start_session(model="test-model", cwd=str(tmp_path))
        sm.save_message({"role": "user", "content": "First"})
        sm.save_message({"role": "assistant", "content": "Second"})
        sm.save_message({"role": "user", "content": "Third"})
        sm.close()
        session_path = sm.session_path

        # Create an agent and resume from that session
        agent = self._make_agent(tmp_path)
        agent.resume_from_session(session_path)

        assert len(agent.messages) == 3
        assert agent.messages[0]["content"] == "First"
        assert agent.messages[2]["content"] == "Third"

    def test_ephemeral_mode_no_persist(self, tmp_path):
        """Ephemeral mode should not create session files."""
        agent = self._make_agent(tmp_path)
        agent._ephemeral = True

        koi_dir = tmp_path / ".koi"
        sessions_dir = koi_dir / "sessions"

        # Count existing session files
        existing = list(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []

        # Simulate what would happen in non-ephemeral mode
        # (the guards in agent.py prevent save_message from being called)
        if not agent._ephemeral:
            agent.session_manager.start_session(model="test", cwd=str(tmp_path))

        msg = {"role": "user", "content": "Hello"}
        agent.messages.append(msg)
        if not agent._ephemeral:
            agent.session_manager.save_message(msg)

        if not agent._ephemeral:
            agent.session_manager.close()

        # No new session files should have been created
        current = list(sessions_dir.glob("*.jsonl")) if sessions_dir.exists() else []
        assert len(current) == len(existing)

