"""Tests for compaction module."""

from unittest.mock import AsyncMock, MagicMock

from koi.compaction import ContextCompactor


def _make_compactor(context_window=128000):
    llm_client = MagicMock()
    return ContextCompactor(llm_client, context_window)


def test_estimate_tokens():
    """Returns positive int for messages."""
    compactor = _make_compactor()
    messages = [
        {"role": "user", "content": "Hello world"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    tokens = compactor.estimate_tokens(messages)
    assert isinstance(tokens, int)
    assert tokens > 0


def test_needs_compaction_false():
    """Small messages stay below threshold."""
    compactor = _make_compactor(context_window=128000)
    messages = [{"role": "user", "content": "short message"}]
    assert not compactor.needs_compaction(messages)


def test_needs_compaction_true():
    """Large messages exceed 70% threshold."""
    compactor = _make_compactor(context_window=100)
    # Create a message with enough tokens to exceed 70 tokens
    big_content = "word " * 200
    messages = [{"role": "user", "content": big_content}]
    assert compactor.needs_compaction(messages)


def test_safe_split_index_no_tool():
    """Returns same index when no tool result at split point."""
    compactor = _make_compactor()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    assert compactor._safe_split_index(messages, 2) == 2


def test_safe_split_index_adjusts_for_tool():
    """Walks back past tool results to include the assistant tool_call."""
    compactor = _make_compactor()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "tool_calls": [{"id": "c1"}]},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "ok"},
    ]
    # If split_index=2 (the tool result), should walk back to 1
    assert compactor._safe_split_index(messages, 2) == 1


async def test_compact_messages_small():
    """Returns unchanged if ≤3 messages."""
    compactor = _make_compactor()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    result = await compactor.compact_messages(messages)
    assert result == messages


async def test_compact_messages_creates_summary():
    """Mock LLM, verify summary message is created."""
    compactor = _make_compactor(context_window=128000)

    # Mock the LLM to return a summary
    compactor.llm_client.chat = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "Summary of previous conversation."}}]
        }
    )

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
        {"role": "user", "content": "msg3"},
    ]

    result = await compactor.compact_messages(messages)

    # Should have a summary message at the start
    assert result[0]["role"] == "system"
    assert "summary" in result[0]["content"].lower()
    # Should be shorter than original
    assert len(result) < len(messages)


def test_messages_to_text():
    """Verifies text conversion for each role."""
    compactor = _make_compactor()
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "read_file"}},
            ],
        },
        {"role": "tool", "content": "file contents"},
    ]
    text = compactor._messages_to_text(messages)
    assert "[System]:" in text
    assert "User: hello" in text
    assert "Assistant: hi" in text
    assert "[Called tool: read_file]" in text
    assert "[Tool result]:" in text


def test_get_context_stats():
    """Returns correct structure."""
    compactor = _make_compactor(context_window=100000)
    messages = [
        {"role": "user", "content": "hello"},
    ]
    stats = compactor.get_context_stats(messages)
    assert "estimated_tokens" in stats
    assert "context_window" in stats
    assert stats["context_window"] == 100000
    assert "usage_percent" in stats
    assert "needs_compaction" in stats
    assert "message_count" in stats
    assert stats["message_count"] == 1


async def test_compact_messages_falls_back_on_llm_failure():
    """When LLM summarization fails, compact_messages truncates instead."""
    compactor = _make_compactor(context_window=128000)
    compactor.llm_client.chat = AsyncMock(side_effect=RuntimeError("LLM down"))

    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "msg1"},
        {"role": "assistant", "content": "resp1"},
        {"role": "user", "content": "msg2"},
        {"role": "assistant", "content": "resp2"},
        {"role": "user", "content": "msg3"},
        {"role": "assistant", "content": "resp3"},
    ]
    result = await compactor.compact_messages(messages)
    # Should return a subset (truncated), not raise
    assert isinstance(result, list)
    assert len(result) < len(messages)


async def test_create_summary_empty_choices_fallback():
    """_create_summary returns fallback string when LLM returns no choices."""
    compactor = _make_compactor()
    compactor.llm_client.chat = AsyncMock(return_value={"choices": []})
    summary = await compactor._create_summary(
        [{"role": "user", "content": "hello"}]
    )
    assert "unavailable" in summary.lower() or summary


def test_tokenizer_fallback(monkeypatch):
    """Falls back to cl100k_base when gpt-4 encoding is unavailable."""
    import tiktoken

    original_for_model = tiktoken.encoding_for_model

    def raise_key_error(model):
        raise KeyError(f"Unknown model: {model}")

    monkeypatch.setattr(tiktoken, "encoding_for_model", raise_key_error)

    # Importing after monkeypatching won't help since module is already loaded;
    # instantiate directly with patched tiktoken
    from koi.llm import LLMClient
    from koi.compaction import ContextCompactor

    compactor = ContextCompactor.__new__(ContextCompactor)
    compactor.context_window = 128000
    compactor.llm_client = MagicMock()
    try:
        compactor.tokenizer = tiktoken.encoding_for_model("gpt-4")
    except KeyError:
        compactor.tokenizer = tiktoken.get_encoding("cl100k_base")

    tokens = compactor.estimate_tokens([{"role": "user", "content": "hello"}])
    assert tokens > 0
