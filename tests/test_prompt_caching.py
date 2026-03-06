"""Tests for Anthropic prompt caching in LLMClient."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import pytest

from koi.config import Config
from koi.llm import LLMClient

# ── Fixtures ──


@pytest.fixture
def anthropic_client():
    """Anthropic client with prompt caching enabled (default)."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-opus-4-6",
        api_format="anthropic",
        prompt_caching=True,
    )
    return LLMClient(config)


@pytest.fixture
def anthropic_client_no_cache():
    """Anthropic client with prompt caching disabled."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-opus-4-6",
        api_format="anthropic",
        prompt_caching=False,
    )
    return LLMClient(config)


@pytest.fixture
def responses_client():
    """Non-Anthropic client (Responses API)."""
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model="test-model",
        api_format="responses",
        prompt_caching=True,
    )
    return LLMClient(config)


def _make_fake_post(captured):
    """Return an async fake_post that captures the payload."""

    async def fake_post(url, headers=None, json=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "msg_test",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        return mock_resp

    return fake_post


# ── System prompt caching ──


async def test_system_prompt_array_with_cache_control(anthropic_client):
    """When caching enabled, system prompt is array with cache_control."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    await anthropic_client.chat(
        [{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful.",
    )

    system = captured["system"]
    assert isinstance(system, list)
    assert len(system) == 1
    assert system[0]["type"] == "text"
    assert system[0]["text"] == "You are helpful."
    assert system[0]["cache_control"] == {"type": "ephemeral"}


async def test_system_prompt_plain_string_when_cache_disabled(
    anthropic_client_no_cache,
):
    """When caching disabled, system prompt remains a plain string."""
    captured = {}
    anthropic_client_no_cache.client.post = _make_fake_post(captured)

    await anthropic_client_no_cache.chat(
        [{"role": "user", "content": "Hi"}],
        system_prompt="You are helpful.",
    )

    system = captured["system"]
    assert isinstance(system, str)
    assert system == "You are helpful."


async def test_cache_control_format(anthropic_client):
    """cache_control has the exact format {type: ephemeral}."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    await anthropic_client.chat(
        [{"role": "user", "content": "hi"}],
        system_prompt="test",
    )

    cc = captured["system"][0]["cache_control"]
    assert cc == {"type": "ephemeral"}
    assert list(cc.keys()) == ["type"]


# ── Tool result caching ──


async def test_last_tool_result_gets_cache_control(anthropic_client):
    """The last user message with tool_result content gets
    cache_control on its last block."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    messages = [
        {"role": "user", "content": "do something"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        {"role": "assistant", "content": "Here's the file."},
        {"role": "user", "content": "thanks"},
    ]
    await anthropic_client.chat(messages, system_prompt="sys")

    # Find the user message with tool_result content
    tool_result_msgs = [
        m
        for m in captured["messages"]
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_msgs) == 1
    last_block = tool_result_msgs[0]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


async def test_multiple_tool_results_only_last_gets_cache_control(anthropic_client):
    """With multiple tool result groups, only the LAST one gets cache_control."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    messages = [
        {"role": "user", "content": "do something"},
        # First tool use cycle
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "contents of a"},
        # Second tool use cycle
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_2", "content": "contents of b"},
    ]
    await anthropic_client.chat(messages, system_prompt="sys")

    # Find all user messages with tool_result content
    tool_result_msgs = [
        m
        for m in captured["messages"]
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_result_msgs) == 2

    # First tool result group should NOT have cache_control
    first_block = tool_result_msgs[0]["content"][-1]
    assert "cache_control" not in first_block

    # Last tool result group SHOULD have cache_control
    last_block = tool_result_msgs[1]["content"][-1]
    assert last_block["cache_control"] == {"type": "ephemeral"}


async def test_no_tool_results_only_system_gets_cache_control(anthropic_client):
    """With no tool results, only the system prompt gets cache_control."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    messages = [
        {"role": "user", "content": "Hello"},
    ]
    await anthropic_client.chat(messages, system_prompt="You are helpful.")

    # System prompt should have cache_control
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}

    # No message should have cache_control
    for msg in captured["messages"]:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                assert "cache_control" not in block
        # string content won't have cache_control


# ── Config load/save ──


def test_config_default_prompt_caching_true():
    """prompt_caching defaults to True."""
    config = Config()
    assert config.prompt_caching is True


def test_config_prompt_caching_false():
    """prompt_caching can be set to False."""
    config = Config(prompt_caching=False)
    assert config.prompt_caching is False


def test_config_load_saves_prompt_caching():
    """prompt_caching is persisted through save/load cycle."""
    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"

        # Save with prompt_caching=False
        config = Config(
            api_base="https://api.example.com",
            api_key="test",
            prompt_caching=False,
        )
        config.save(config_path)

        # Reload and verify
        loaded = Config.load(config_path)
        assert loaded.prompt_caching is False


def test_config_load_defaults_prompt_caching_when_missing():
    """When prompt_caching is missing from config.json, defaults to True."""
    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        # Write a config without prompt_caching
        config_path.write_text(
            json.dumps(
                {
                    "api_base": "https://api.example.com",
                    "api_key": "test",
                    "model": "test-model",
                }
            )
        )

        loaded = Config.load(config_path)
        assert loaded.prompt_caching is True


def test_config_to_dict_includes_prompt_caching():
    """to_dict includes prompt_caching."""
    config = Config(prompt_caching=False)
    d = config.to_dict()
    assert "prompt_caching" in d
    assert d["prompt_caching"] is False


# ── Non-anthropic formats unaffected ──


async def test_responses_format_unaffected(responses_client):
    """Responses API format uses instructions for system_prompt."""
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"id": "r", "output": []}
        return mock_resp

    responses_client.client.post = fake_post

    await responses_client.chat(
        [{"role": "user", "content": "hi"}],
        system_prompt="sys prompt",
    )

    # Responses API uses "instructions", not "system"
    assert "system" not in captured
    assert captured.get("instructions") == "sys prompt"


async def test_chat_completions_format_unaffected():
    """Chat Completions format prepends system_prompt as messages[0]."""
    config = Config(
        api_base="https://api.example.com/v1/chat/completions",
        api_key="test",
        model="gpt-4",
        api_format="chat_completions",
        prompt_caching=True,
    )
    client = LLMClient(config)
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "r",
            "choices": [
                {
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                }
            ],
        }
        return mock_resp

    client.client.post = fake_post

    await client.chat(
        [{"role": "user", "content": "hi"}],
        system_prompt="sys prompt",
    )

    # CC format passes system_prompt as messages[0] with no cache_control
    assert "system" not in captured  # CC uses messages array, not system key
    assert captured["messages"][0] == {"role": "system", "content": "sys prompt"}
    for msg in captured.get("messages", []):
        assert "cache_control" not in msg
    await client.close()


# ── Streaming path also applies caching ──


async def test_streaming_anthropic_applies_cache_control():
    """The streaming Anthropic path also applies prompt caching."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-opus-4-6",
        api_format="anthropic",
        prompt_caching=True,
        thinking_level="off",
    )
    client = LLMClient(config)
    captured = {}

    class _StreamCtx:
        def __init__(self):
            self._resp = _MockStreamResp()

        async def __aenter__(self):
            return self._resp

        async def __aexit__(self, *_):
            pass

    class _MockStreamResp:
        def __init__(self):
            self.status_code = 200
            self.headers = {}

        @property
        def is_error(self):
            return self.status_code >= 400

        async def aread(self):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield ('data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "hi"}}')
            yield 'data: {"type": "message_stop"}'

    def capture_stream(*a, **kw):
        captured.update(kw.get("json", {}))
        return _StreamCtx()

    client.client.stream = capture_stream

    events = []
    async for event in client.stream_chat(
        [{"role": "user", "content": "hi"}],
        system_prompt="Be helpful.",
    ):
        events.append(event)

    # System should be array with cache_control
    system = captured["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    await client.close()


# ── Edge cases ──


async def test_no_system_prompt_with_caching(anthropic_client):
    """When there's no system prompt, caching still works (no system key)."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    await anthropic_client.chat(
        [
            {"role": "user", "content": "Hi"},
        ]
    )

    # No system prompt means no system key in payload
    assert "system" not in captured


async def test_grouped_tool_results_cache_on_last_block(anthropic_client):
    """Multiple consecutive tool results grouped into one user msg --
    cache_control on last block."""
    captured = {}
    anthropic_client.client.post = _make_fake_post(captured)

    messages = [
        {"role": "user", "content": "run both"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "b.py"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "content a"},
        {"role": "tool", "tool_call_id": "call_2", "content": "content b"},
    ]
    await anthropic_client.chat(messages)

    # Both tool results are grouped into one user message
    tool_user_msgs = [
        m
        for m in captured["messages"]
        if m.get("role") == "user"
        and isinstance(m.get("content"), list)
        and any(b.get("type") == "tool_result" for b in m["content"])
    ]
    assert len(tool_user_msgs) == 1
    blocks = tool_user_msgs[0]["content"]
    assert len(blocks) == 2
    # Only the LAST block in the group has cache_control
    assert "cache_control" not in blocks[0]
    assert blocks[1]["cache_control"] == {"type": "ephemeral"}
