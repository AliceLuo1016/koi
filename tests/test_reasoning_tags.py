"""Tests for non-native reasoning tag support (<think>/<final> tags)."""

import pytest

from koi.config import Config
from koi.llm import LLMClient, supports_thinking, uses_reasoning_tags
from koi.prompts import build_system_prompt
from koi.agent import strip_thinking_tags


# ── uses_reasoning_tags() ──


def test_uses_reasoning_tags_true_for_unknown_model_with_thinking():
    """Unknown model that doesn't support native thinking → uses tags."""
    assert uses_reasoning_tags("my-custom-llm", "responses", "low") is True


def test_uses_reasoning_tags_true_for_gpt4_with_thinking():
    """GPT-4 doesn't support native thinking → uses tags as fallback."""
    assert uses_reasoning_tags("gpt-4o", "responses", "medium") is True


def test_uses_reasoning_tags_false_when_native_thinking_supported():
    """Models with native thinking (e.g., o3) → don't use tags."""
    assert uses_reasoning_tags("o3", "responses", "high") is False


def test_uses_reasoning_tags_false_for_claude_with_thinking():
    """Claude 4 has native thinking → don't use tags."""
    assert uses_reasoning_tags("claude-4-sonnet", "anthropic", "low") is False


def test_uses_reasoning_tags_false_when_thinking_off():
    """Thinking is off → no tags regardless of model."""
    assert uses_reasoning_tags("my-custom-llm", "responses", "off") is False


def test_uses_reasoning_tags_false_when_thinking_off_unsupported_model():
    """Even an unsupported model gets False when thinking is off."""
    assert uses_reasoning_tags("gpt-4o", "chat_completions", "off") is False


def test_uses_reasoning_tags_various_thinking_levels():
    """All non-off levels trigger tags for unsupported models."""
    for level in ("minimal", "low", "medium", "high"):
        assert uses_reasoning_tags("some-local-model", "responses", level) is True


# ── LLMClient.use_reasoning_tags flag ──


def test_llm_client_sets_use_reasoning_tags_true():
    """LLMClient sets use_reasoning_tags=True for unknown models with thinking."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="my-local-model",
        api_format="responses",
        thinking_level="low",
    )
    client = LLMClient(config)
    assert client.use_reasoning_tags is True


def test_llm_client_sets_use_reasoning_tags_false_native():
    """LLMClient sets use_reasoning_tags=False for models with native thinking."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="o3",
        api_format="responses",
        thinking_level="high",
    )
    client = LLMClient(config)
    assert client.use_reasoning_tags is False


def test_llm_client_sets_use_reasoning_tags_false_off():
    """LLMClient sets use_reasoning_tags=False when thinking is off."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="my-local-model",
        api_format="responses",
        thinking_level="off",
    )
    client = LLMClient(config)
    assert client.use_reasoning_tags is False


# ── System prompt: reasoning format section ──


def test_system_prompt_includes_reasoning_format_when_tags_needed():
    """System prompt contains reasoning format section when use_reasoning_tags=True."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="local-model",
        api_format="responses",
    )
    prompt = build_system_prompt(config, use_reasoning_tags=True)
    assert "## Reasoning Format" in prompt
    assert "<think>" in prompt
    assert "<final>" in prompt


def test_system_prompt_excludes_reasoning_format_when_not_needed():
    """System prompt omits reasoning format section when use_reasoning_tags=False."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="local-model",
        api_format="responses",
    )
    prompt = build_system_prompt(config, use_reasoning_tags=False)
    assert "## Reasoning Format" not in prompt


def test_system_prompt_excludes_reasoning_format_by_default():
    """System prompt omits reasoning format section by default."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="local-model",
        api_format="responses",
    )
    prompt = build_system_prompt(config)
    assert "## Reasoning Format" not in prompt


def test_system_prompt_reasoning_format_before_tools():
    """Reasoning format section appears before the tools section."""
    config = Config(
        api_base="http://localhost:8080",
        api_key="test",
        model="local-model",
        api_format="responses",
    )
    prompt = build_system_prompt(config, use_reasoning_tags=True)
    reasoning_idx = prompt.index("## Reasoning Format")
    tools_idx = prompt.index("Available Tools:")
    assert reasoning_idx < tools_idx


# ── strip_thinking_tags() ──


def test_strip_thinking_tags_basic():
    """Basic case: <think> stripped, <final> content shown."""
    text = "<think>I need to greet the user.</think><final>Hello!</final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Hello!"
    assert thinking == "I need to greet the user."


def test_strip_thinking_tags_no_tags():
    """No tags at all — text returned as-is."""
    text = "Just a normal response."
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Just a normal response."
    assert thinking == ""


def test_strip_thinking_tags_empty_string():
    """Empty string returns empty tuple."""
    visible, thinking = strip_thinking_tags("")
    assert visible == ""
    assert thinking == ""


def test_strip_thinking_tags_think_only():
    """Only <think> block, no <final> — visible text is empty."""
    text = "<think>All reasoning, no output.</think>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == ""
    assert thinking == "All reasoning, no output."


def test_strip_thinking_tags_no_final_with_trailing_text():
    """<think> block followed by plain text (no <final>) — show the trailing text."""
    text = "<think>Let me think about this.</think>Here is the answer."
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Here is the answer."
    assert thinking == "Let me think about this."


def test_strip_thinking_tags_multiple_think_blocks():
    """Multiple <think> blocks are all collected."""
    text = (
        "<think>First thought.</think>"
        "<final>Answer one.</final>"
        "<think>Second thought.</think>"
        "<final>Answer two.</final>"
    )
    visible, thinking = strip_thinking_tags(text)
    assert "Answer one." in visible
    assert "Answer two." in visible
    assert "First thought." in thinking
    assert "Second thought." in thinking


def test_strip_thinking_tags_multiline_think():
    """<think> blocks can span multiple lines."""
    text = "<think>\nLine 1\nLine 2\n</think><final>Result</final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Result"
    assert "Line 1" in thinking
    assert "Line 2" in thinking


def test_strip_thinking_tags_whitespace_handling():
    """Whitespace around content inside tags is stripped."""
    text = "<think>  padded thinking  </think><final>  padded answer  </final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "padded answer"
    assert thinking == "padded thinking"


def test_strip_thinking_tags_nested_angle_brackets():
    """Content with angle brackets inside <think> is handled correctly."""
    text = "<think>if x > 0 and y < 10</think><final>Code reviewed.</final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Code reviewed."
    assert "x > 0" in thinking


def test_strip_thinking_tags_partial_tags_preserved():
    """Incomplete/malformed tags are left as-is in visible text."""
    text = "<think>reasoning</think>Some text with <final unclosed"
    visible, thinking = strip_thinking_tags(text)
    assert "Some text with <final unclosed" in visible
    assert thinking == "reasoning"


def test_strip_thinking_tags_final_only():
    """<final> without <think> — shows final content."""
    text = "<final>Direct answer.</final>"
    visible, thinking = strip_thinking_tags(text)
    assert visible == "Direct answer."
    assert thinking == ""
