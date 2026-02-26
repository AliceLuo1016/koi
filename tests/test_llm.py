"""Tests for llm module — Responses API conversion logic."""

import asyncio
import json as _json
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest

from koi.config import Config
from koi.llm import LLMClient


# ── Streaming helpers ──


class _MockStreamResponse:
    """Simulate an httpx streaming response that yields SSE lines."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code
        self.headers = {}

    def raise_for_status(self):
        if self.status_code >= 400:
            mock_req = MagicMock()
            mock_resp = MagicMock()
            mock_resp.status_code = self.status_code
            mock_resp.text = f"HTTP {self.status_code}"
            raise httpx.HTTPStatusError(
                f"{self.status_code}", request=mock_req, response=mock_resp
            )

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StreamCtx:
    """Async context manager wrapping a _MockStreamResponse."""

    def __init__(self, lines, status_code=200):
        self._resp = _MockStreamResponse(lines, status_code)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *_):
        pass


@pytest.fixture
def client():
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model="test-model",
        api_format="responses",
    )
    return LLMClient(config)


@pytest.fixture
def cc_client():
    """Client configured for Chat Completions format."""
    config = Config(
        api_base="https://api.example.com/v1/chat/completions",
        api_key="test-key",
        model="us/aws/anthropic/bedrock-claude-opus-4-6",
        api_format="chat_completions",
    )
    return LLMClient(config)


# ── _convert_messages_to_input ──


def test_system_message_becomes_instructions(client):
    messages = [{"role": "system", "content": "You are helpful."}]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions == "You are helpful."
    # Also injected as a developer message
    assert len(input_items) == 1
    assert input_items[0]["role"] == "developer"
    assert input_items[0]["content"] == "You are helpful."


def test_user_message(client):
    messages = [{"role": "user", "content": "Hello"}]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions is None
    assert input_items == [{"role": "user", "content": "Hello"}]


def test_assistant_text_message(client):
    messages = [{"role": "assistant", "content": "Hi there"}]
    _, input_items = client._convert_messages_to_input(messages)
    assert input_items == [{"role": "assistant", "content": "Hi there"}]


def test_assistant_tool_calls_become_function_call_items(client):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.txt"}',
                    },
                }
            ],
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert len(input_items) == 1
    assert input_items[0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path": "test.txt"}',
    }


def test_assistant_with_content_and_tool_calls(client):
    messages = [
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert len(input_items) == 2
    assert input_items[0] == {"role": "assistant", "content": "Let me check that."}
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["name"] == "exec_command"


def test_tool_result_becomes_function_call_output(client):
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file contents here",
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert input_items == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "file contents here",
        }
    ]


def test_full_conversation_roundtrip(client):
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read test.txt"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello world"},
        {"role": "assistant", "content": "The file says hello world."},
        {"role": "user", "content": "Thanks"},
    ]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions == "Be helpful."
    # developer msg + user + function_call + function_call_output + assistant + user
    assert len(input_items) == 6
    assert input_items[0]["role"] == "developer"
    assert input_items[1]["role"] == "user"
    assert input_items[2]["type"] == "function_call"
    assert input_items[3]["type"] == "function_call_output"
    assert input_items[4]["role"] == "assistant"
    assert input_items[5]["role"] == "user"


# ── _convert_tools ──


def test_convert_tools_flattens_function(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    converted = client._convert_tools(tools)
    assert len(converted) == 1
    assert converted[0] == {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    # Must NOT have nested "function" key
    assert "function" not in converted[0]


def test_convert_tools_multiple(client):
    from koi.tools import get_tool_definitions

    tools = get_tool_definitions()
    converted = client._convert_tools(tools)
    assert len(converted) == len(tools)
    for c in converted:
        assert "name" in c
        assert "function" not in c


# ── _convert_response ──


def test_convert_text_response(client):
    api_response = {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello!"
    assert "tool_calls" not in msg


def test_convert_tool_call_response(client):
    api_response = {
        "id": "resp_2",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": '{"path": "x.py"}',
            }
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"] == '{"path": "x.py"}'


def test_convert_multiple_tool_calls(client):
    api_response = {
        "id": "resp_3",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "a.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": '{"path": "b.py"}',
            },
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 2


def test_convert_mixed_text_and_tool_calls(client):
    api_response = {
        "id": "resp_4",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "call_id": "call_x",
                "name": "exec_command",
                "arguments": '{"command": "ls"}',
            },
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1


def test_convert_empty_output(client):
    api_response = {"id": "resp_5", "output": []}
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert "tool_calls" not in msg


# ── chat() request construction ──


async def test_chat_builds_correct_payload(client):
    """Verify the payload sent to the API matches Responses API format."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    captured_payload = {}

    async def fake_post(url, headers=None, json=None):
        captured_payload.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hi!"}],
                }
            ],
        }
        return mock_resp

    client.client.post = fake_post

    result = await client.chat(messages, tools=tools)

    # Verify payload structure
    assert captured_payload["model"] == "test-model"
    assert captured_payload["instructions"] == "You are helpful."
    # input includes developer message + user message
    assert captured_payload["input"] == [
        {"role": "developer", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    # Tools should be flattened
    assert captured_payload["tools"][0]["name"] == "read_file"
    assert "function" not in captured_payload["tools"][0]
    # No stream for non-streaming call
    assert "stream" not in captured_payload

    # Verify response was converted back
    assert result["choices"][0]["message"]["content"] == "Hi!"


async def test_chat_url_uses_api_base_directly(client):
    """Verify we POST to api_base, not api_base + /chat/completions."""
    captured_url = None

    async def fake_post(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"id": "r", "output": []}
        return mock_resp

    client.client.post = fake_post
    await client.chat([{"role": "user", "content": "test"}])

    assert captured_url == "https://api.example.com/v1/responses"
    assert "chat/completions" not in captured_url


# ── Chat Completions format tests ──


def test_build_cc_payload_basic(cc_client):
    """Test Chat Completions payload is built correctly."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    payload = cc_client._build_cc_payload(messages)

    assert payload["model"] == "us/aws/anthropic/bedrock-claude-opus-4-6"
    assert payload["messages"] == messages
    assert payload["max_tokens"] == 4096
    assert "stream" not in payload


def test_build_cc_payload_with_tools(cc_client):
    """Test Chat Completions payload includes tools in CC format."""
    messages = [{"role": "user", "content": "Hi"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
            },
        }
    ]
    payload = cc_client._build_cc_payload(messages, tools=tools)

    # Tools are passed through as-is (already CC format)
    assert payload["tools"] == tools
    assert payload["tools"][0]["function"]["name"] == "read_file"


def test_build_cc_payload_with_stream(cc_client):
    """Test Chat Completions payload with streaming enabled."""
    messages = [{"role": "user", "content": "Hi"}]
    payload = cc_client._build_cc_payload(messages, stream=True)

    assert payload["stream"] is True


def test_convert_cc_response_passthrough(cc_client):
    """Test Chat Completions response passthrough."""
    cc_response = {
        "id": "chatcmpl-1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }
        ],
    }
    result = cc_client._convert_cc_response(cc_response)
    assert result == cc_response


async def test_chat_completions_builds_correct_payload(cc_client):
    """Verify CC path sends Chat Completions format payload."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    captured_payload = {}

    async def fake_post(url, headers=None, json=None):
        captured_payload.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "chatcmpl-1",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hi!"},
                    "finish_reason": "stop",
                }
            ],
        }
        return mock_resp

    cc_client.client.post = fake_post

    result = await cc_client.chat(messages, tools=tools)

    # Verify CC payload structure
    assert captured_payload["model"] == "us/aws/anthropic/bedrock-claude-opus-4-6"
    assert captured_payload["messages"] == messages
    assert captured_payload["max_tokens"] == 4096
    assert captured_payload["tools"] == tools
    assert "input" not in captured_payload
    assert "instructions" not in captured_payload

    # Verify response is passed through
    assert result["choices"][0]["message"]["content"] == "Hi!"


# ── Temperature in payloads ──


async def test_temperature_included_in_responses_payload(client):
    """When temperature is set, it appears in Responses API payload."""
    client.config.temperature = 0.5
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"id": "r", "output": []}
        return mock_resp

    client.client.post = fake_post
    await client.chat([{"role": "user", "content": "hi"}])
    assert captured.get("temperature") == 0.5


async def test_temperature_omitted_when_none(client):
    """When temperature is None, key must not appear in payload."""
    client.config.temperature = None
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"id": "r", "output": []}
        return mock_resp

    client.client.post = fake_post
    await client.chat([{"role": "user", "content": "hi"}])
    assert "temperature" not in captured


async def test_temperature_included_in_cc_payload(cc_client):
    """When temperature is set, it appears in Chat Completions payload."""
    cc_client.config.temperature = 0.7
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "r",
            "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        }
        return mock_resp

    cc_client.client.post = fake_post
    await cc_client.chat([{"role": "user", "content": "hi"}])
    assert captured.get("temperature") == 0.7


# ── Retry logic ──


async def test_retry_non_retryable_status_raises_immediately(client):
    """Non-retryable HTTP status (e.g. 404) raises RuntimeError without retrying."""
    call_count = 0

    async def always_404(url, headers=None, json=None):
        nonlocal call_count
        call_count += 1
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_resp.text = "Not found"
        raise httpx.HTTPStatusError("404", request=MagicMock(), response=mock_resp)

    client.client.post = always_404

    with pytest.raises(RuntimeError, match="HTTP 404"):
        await client.chat([{"role": "user", "content": "hi"}])

    assert call_count == 1  # No retries


async def test_retry_all_exhausted_raises(client):
    """After MAX_RETRIES retries of a 429, RuntimeError is raised."""
    with patch("asyncio.sleep", new_callable=AsyncMock):
        async def always_429(url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.headers = {}
            mock_resp.text = "rate limited"
            raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_resp)

        client.client.post = always_429
        with pytest.raises(RuntimeError, match="retries"):
            await client.chat([{"role": "user", "content": "hi"}])


async def test_retry_after_header_parsed(client):
    """retry-after header value is used as sleep duration."""
    sleep_delays = []

    async def fake_sleep(delay):
        sleep_delays.append(delay)

    call_count = 0

    async def retryable_then_ok(url, headers=None, json=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.headers = {"retry-after": "5"}
            mock_resp.text = "slow down"
            raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_resp)
        ok = MagicMock()
        ok.raise_for_status = lambda: None
        ok.json.return_value = {"id": "r", "output": []}
        return ok

    client.client.post = retryable_then_ok
    with patch("asyncio.sleep", side_effect=fake_sleep):
        await client.chat([{"role": "user", "content": "hi"}])

    assert sleep_delays[0] == 5.0


async def test_retry_on_connect_error(client):
    """ConnectError is retried and succeeds on second attempt."""
    call_count = 0

    async def connect_error_then_ok(url, headers=None, json=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise httpx.ConnectError("Connection refused")
        ok = MagicMock()
        ok.raise_for_status = lambda: None
        ok.json.return_value = {"id": "r", "output": []}
        return ok

    client.client.post = connect_error_then_ok
    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = await client.chat([{"role": "user", "content": "hi"}])

    assert call_count == 2
    assert "choices" in result


# ── Streaming — Responses API ──


async def test_stream_chat_assembles_text_deltas(client):
    """_stream_chat accumulates output_text.delta events into content."""
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "Hello"}',
        'data: {"type": "response.output_text.delta", "delta": " world"}',
        "data: [DONE]",
    ]
    client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await client._stream_chat("https://api.example.com", {})
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_stream_chat_assembles_tool_call(client):
    """_stream_chat collects function_call name and argument deltas."""
    lines = [
        'data: {"type": "response.output_item.added", "item": {"type": "function_call", "call_id": "c1", "name": "read_file"}}',
        'data: {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": "{\\"path\\": "}',
        'data: {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": "\\"x.txt\\"}"}',
        "data: [DONE]",
    ]
    client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await client._stream_chat("https://api.example.com", {})
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"
    assert "x.txt" in tc["function"]["arguments"]


async def test_stream_chat_response_completed_event(client):
    """response.completed event triggers immediate return via _convert_response."""
    completed_resp = {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done!"}],
            }
        ],
    }
    lines = [
        f'data: {{"type": "response.completed", "response": {_json.dumps(completed_resp)}}}',
        "data: [DONE]",
    ]
    client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await client._stream_chat("https://api.example.com", {})
    assert result["choices"][0]["message"]["content"] == "Done!"


async def test_stream_chat_ignores_malformed_json(client):
    """_stream_chat silently skips lines with invalid JSON."""
    lines = [
        "data: NOT_JSON",
        'data: {"type": "response.output_text.delta", "delta": "ok"}',
        "data: [DONE]",
    ]
    client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await client._stream_chat("https://api.example.com", {})
    assert result["choices"][0]["message"]["content"] == "ok"


# ── Streaming — Chat Completions ──


async def test_stream_chat_completions_assembles_text(cc_client):
    """_stream_chat_completions accumulates content deltas."""
    lines = [
        'data: {"choices": [{"delta": {"content": "Hi"}}]}',
        'data: {"choices": [{"delta": {"content": " there"}}]}',
        "data: [DONE]",
    ]
    cc_client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await cc_client._stream_chat_completions("https://api.example.com", {})
    assert result["choices"][0]["message"]["content"] == "Hi there"


async def test_stream_chat_completions_assembles_tool_calls(cc_client):
    """_stream_chat_completions collects tool call deltas by index."""
    lines = [
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": ""}}]}}]}',
        'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"path\\": \\"a.txt\\"}"}}]}}]}',
        "data: [DONE]",
    ]
    cc_client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await cc_client._stream_chat_completions("https://api.example.com", {})
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    assert msg["tool_calls"][0]["function"]["name"] == "read_file"


# ── stream_chat token generator ──


async def test_stream_chat_yields_tokens_responses_format(client):
    """stream_chat() yields text tokens from output_text.delta events."""
    lines = [
        'data: {"type": "response.output_text.delta", "delta": "tok1"}',
        'data: {"type": "response.output_text.delta", "delta": "tok2"}',
        "data: [DONE]",
    ]
    client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    tokens = []
    async for token in client.stream_chat([{"role": "user", "content": "hi"}]):
        tokens.append(token)
    assert tokens == ["tok1", "tok2"]


async def test_stream_chat_yields_tokens_cc_format(cc_client):
    """stream_chat() yields tokens from Chat Completions delta content."""
    lines = [
        'data: {"choices": [{"delta": {"content": "A"}}]}',
        'data: {"choices": [{"delta": {"content": "B"}}]}',
        "data: [DONE]",
    ]
    cc_client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    tokens = []
    async for token in cc_client.stream_chat([{"role": "user", "content": "hi"}]):
        tokens.append(token)
    assert tokens == ["A", "B"]


# ── Anthropic Messages API ──


@pytest.fixture
def anthropic_client():
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-opus-4-6",
        api_format="anthropic",
    )
    return LLMClient(config)


def test_convert_messages_to_anthropic_system(anthropic_client):
    """System message is extracted as system_prompt, not appended to messages."""
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Hi"},
    ]
    system, msgs = anthropic_client._convert_messages_to_anthropic(messages)
    assert system == "Be helpful."
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "Hi"


def test_convert_messages_to_anthropic_tool_calls(anthropic_client):
    """Tool calls become tool_use content blocks in assistant message."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "x.py"}',
                    },
                }
            ],
        }
    ]
    _, msgs = anthropic_client._convert_messages_to_anthropic(messages)
    assert len(msgs) == 1
    content = msgs[0]["content"]
    assert content[0]["type"] == "tool_use"
    assert content[0]["name"] == "read_file"
    assert content[0]["input"] == {"path": "x.py"}


def test_convert_messages_to_anthropic_tool_results(anthropic_client):
    """Tool results become user message with tool_result content blocks."""
    messages = [
        {"role": "tool", "tool_call_id": "call_1", "content": "file contents"},
        {"role": "tool", "tool_call_id": "call_2", "content": "more contents"},
    ]
    _, msgs = anthropic_client._convert_messages_to_anthropic(messages)
    # Multiple consecutive tool results are grouped into ONE user message
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    results = msgs[0]["content"]
    assert len(results) == 2
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "call_1"
    assert results[1]["tool_use_id"] == "call_2"


def test_convert_anthropic_tools(anthropic_client):
    """OpenAI tool format is converted to Anthropic input_schema format."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    converted = anthropic_client._convert_anthropic_tools(tools)
    assert len(converted) == 1
    c = converted[0]
    assert c["name"] == "read_file"
    assert c["description"] == "Read a file"
    assert "input_schema" in c
    assert "function" not in c
    assert c["input_schema"]["type"] == "object"


def test_convert_anthropic_response_text(anthropic_client):
    """Anthropic text response converts to Chat Completions format."""
    api_response = {
        "id": "msg_1",
        "content": [{"type": "text", "text": "Hello there!"}],
        "stop_reason": "end_turn",
    }
    result = anthropic_client._convert_anthropic_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello there!"
    assert "tool_calls" not in msg


def test_convert_anthropic_response_tool_use(anthropic_client):
    """Anthropic tool_use response converts to Chat Completions tool_calls."""
    api_response = {
        "id": "msg_2",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
                "input": {"path": "x.py"},
            }
        ],
        "stop_reason": "tool_use",
    }
    result = anthropic_client._convert_anthropic_response(api_response)
    msg = result["choices"][0]["message"]
    assert "tool_calls" in msg
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert tc["function"]["name"] == "read_file"
    import json as _j
    assert _j.loads(tc["function"]["arguments"]) == {"path": "x.py"}


async def test_chat_routes_to_anthropic(anthropic_client):
    """chat() uses _chat_anthropic when api_format == 'anthropic'."""
    captured = {}

    async def fake_post(url, headers=None, json=None):
        captured.update(json or {})
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "msg_x",
            "content": [{"type": "text", "text": "Hi!"}],
            "stop_reason": "end_turn",
        }
        return mock_resp

    anthropic_client.client.post = fake_post
    result = await anthropic_client.chat(
        [{"role": "user", "content": "Hello"}]
    )
    # Anthropic payload uses "messages" (not "input") and "max_tokens"
    assert "messages" in captured
    assert "max_tokens" in captured
    assert "input" not in captured
    assert result["choices"][0]["message"]["content"] == "Hi!"


async def test_stream_anthropic_assembles_text(anthropic_client):
    """_stream_anthropic accumulates text_delta events."""
    lines = [
        '{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}',
        '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}',
        '{"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}',
        '{"type": "message_stop"}',
    ]
    # Anthropic doesn't use "data: " prefix for regular events but our code still filters for it
    # Actually looking at the code, it does filter for "data: " - let me check the streaming code
    sse_lines = [f"data: {l}" for l in lines]
    anthropic_client.client.stream = lambda *a, **kw: _StreamCtx(sse_lines)
    result = await anthropic_client._stream_anthropic(
        "https://api.anthropic.com/v1/messages", {}
    )
    assert result["choices"][0]["message"]["content"] == "Hello world"


async def test_stream_chat_yields_tokens_anthropic_format(anthropic_client):
    """stream_chat() routes to _stream_anthropic_tokens for anthropic format."""
    lines = [
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "tok1"}}',
        'data: {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "tok2"}}',
        "data: [DONE]",
    ]
    anthropic_client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    tokens = []
    async for token in anthropic_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        tokens.append(token)
    assert tokens == ["tok1", "tok2"]


async def test_chat_completions_url(cc_client):
    """Verify CC path POSTs to the correct URL."""
    captured_url = None

    async def fake_post(url, **kwargs):
        nonlocal captured_url
        captured_url = url
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

    cc_client.client.post = fake_post
    await cc_client.chat([{"role": "user", "content": "test"}])

    assert captured_url == "https://api.example.com/v1/chat/completions"


# ── Additional coverage: conversion edge cases ──


def test_convert_tools_passthrough_non_function(client):
    """Non-function tools are passed through unchanged in _convert_tools."""
    raw_tool = {"type": "computer_use", "display_width_px": 1024}
    result = client._convert_tools([raw_tool])
    assert result == [raw_tool]


def test_convert_anthropic_tools_passthrough_non_function(client):
    """Non-function tools pass through unchanged in _convert_anthropic_tools."""
    raw_tool = {"type": "bash", "name": "bash"}
    result = client._convert_anthropic_tools([raw_tool])
    assert result == [raw_tool]


def test_convert_messages_to_anthropic_assistant_with_text_and_tool_call(client):
    """Assistant message with text content AND tool calls produces both block types."""
    messages = [
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path": "/tmp/x"}'},
                }
            ],
        }
    ]
    _, anthropic_msgs = client._convert_messages_to_anthropic(messages)
    blocks = anthropic_msgs[0]["content"]
    block_types = [b["type"] for b in blocks]
    assert "text" in block_types
    assert "tool_use" in block_types


def test_convert_messages_to_anthropic_malformed_tool_args_defaults_empty(client):
    """Malformed JSON in tool_calls arguments is silently replaced with {}."""
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "not-valid-json"},
                }
            ],
        }
    ]
    _, anthropic_msgs = client._convert_messages_to_anthropic(messages)
    tool_block = anthropic_msgs[0]["content"][0]
    assert tool_block["type"] == "tool_use"
    assert tool_block["input"] == {}


# ── Retry: invalid retry-after header falls back to backoff ──


async def test_retry_invalid_retry_after_falls_back_to_backoff(client):
    """Non-numeric retry-after header falls back to exponential backoff delay."""
    call_count = 0
    sleep_delays = []

    async def fake_sleep(d):
        sleep_delays.append(d)

    async def retryable_then_ok(url, headers=None, json=None):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            mock_resp = MagicMock()
            mock_resp.status_code = 429
            mock_resp.headers = {"retry-after": "not-a-number"}
            mock_resp.text = "rate limited"
            raise httpx.HTTPStatusError("429", request=MagicMock(), response=mock_resp)
        ok = MagicMock()
        ok.raise_for_status = lambda: None
        ok.json.return_value = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]
        }
        return ok

    client.client.post = retryable_then_ok
    with patch("asyncio.sleep", side_effect=fake_sleep):
        result = await client.chat([{"role": "user", "content": "hi"}])

    # Should have fallen back to exponential backoff (not crash)
    assert call_count == 3
    assert len(sleep_delays) == 2
    # Backoff values should be positive numbers, not from the bad header
    assert all(d > 0 for d in sleep_delays)
    assert result["choices"][0]["message"]["content"] == "ok"


# ── Retry: non-HTTP exception raises immediately ──


async def test_retry_non_http_exception_raises_immediately(client):
    """Unexpected non-HTTP exceptions raise RuntimeError without retrying."""
    call_count = 0

    async def bad_post(url, headers=None, json=None):
        nonlocal call_count
        call_count += 1
        raise ValueError("unexpected internal error")

    client.client.post = bad_post
    with pytest.raises(RuntimeError, match="Request failed"):
        await client.chat([{"role": "user", "content": "hi"}])

    assert call_count == 1  # No retries


# ── Anthropic stream: blank/non-prefixed/malformed lines skipped ──


async def test_stream_anthropic_skips_blank_and_malformed_lines():
    """_stream_anthropic silently skips blank, non-data, and malformed-JSON lines."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-3",
        api_format="anthropic",
    )
    anthro_client = LLMClient(config)

    lines = [
        "",                                # blank → skip
        "event: ping",                     # no "data: " prefix → skip
        "data: {malformed json",           # bad JSON → skip
        'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}',
        'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}',
        'data: {"type": "message_stop"}',
    ]
    anthro_client.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await anthro_client._stream_anthropic(
        "https://api.anthropic.com/v1/messages", {}
    )
    assert result["choices"][0]["message"]["content"] == "hello"
    await anthro_client.close()


# ── CC stream: blank/malformed/empty-choices lines skipped ──


async def test_stream_cc_skips_blank_malformed_empty_choices():
    """_stream_chat_completions skips blank, malformed JSON, and empty-choices lines."""
    cc2 = LLMClient(
        Config(
            api_base="https://api.example.com/v1/chat/completions",
            api_key="k",
            model="gpt-4",
            api_format="chat_completions",
        )
    )
    lines = [
        "",                                        # blank → skip
        "data: {bad json",                         # malformed → skip
        'data: {"choices": []}',                   # empty choices → skip
        'data: {"choices": [{"delta": {"content": "world"}}]}',
        "data: [DONE]",
    ]
    cc2.client.stream = lambda *a, **kw: _StreamCtx(lines)
    result = await cc2._stream_chat_completions(
        "https://api.example.com/v1/chat/completions", {}
    )
    assert result["choices"][0]["message"]["content"] == "world"
    await cc2.close()


# ── Temperature in Anthropic payload ──


async def test_anthropic_payload_includes_temperature():
    """chat() includes temperature in the Anthropic payload when set."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-3",
        api_format="anthropic",
        temperature=0.7,
    )
    anthro_client = LLMClient(config)
    captured: dict = {}

    async def mock_post(url, headers=None, json=None):
        captured.update(json)
        resp = MagicMock()
        resp.raise_for_status = lambda: None
        resp.json.return_value = {
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
        }
        return resp

    anthro_client.client.post = mock_post
    await anthro_client.chat([{"role": "user", "content": "hi"}])
    assert captured.get("temperature") == 0.7
    await anthro_client.close()


# ── Token stream: HTTP errors raise RuntimeError ──


async def test_stream_chat_http_error_raises_runtime_error(client):
    """stream_chat (Responses format) raises RuntimeError on HTTP error."""
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    error = httpx.HTTPStatusError("500", request=mock_req, response=mock_resp)

    class _ErrorCtx:
        async def __aenter__(self):
            raise error
        async def __aexit__(self, *_):
            pass

    client.client.stream = lambda *a, **kw: _ErrorCtx()
    with pytest.raises(RuntimeError, match="HTTP 500"):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await client.close()


async def test_stream_cc_tokens_http_error_raises_runtime_error(cc_client):
    """_stream_chat_completions_tokens raises RuntimeError on HTTP error."""
    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "Service Unavailable"
    error = httpx.HTTPStatusError("503", request=mock_req, response=mock_resp)

    class _ErrorCtx:
        async def __aenter__(self):
            raise error
        async def __aexit__(self, *_):
            pass

    cc_client.client.stream = lambda *a, **kw: _ErrorCtx()
    with pytest.raises(RuntimeError, match="HTTP 503"):
        async for _ in cc_client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await cc_client.close()


async def test_stream_anthropic_tokens_http_error_raises_runtime_error():
    """_stream_anthropic_tokens raises RuntimeError on HTTP error."""
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-3",
        api_format="anthropic",
    )
    anthro_client = LLMClient(config)

    mock_req = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized"
    error = httpx.HTTPStatusError("401", request=mock_req, response=mock_resp)

    class _ErrorCtx:
        async def __aenter__(self):
            raise error
        async def __aexit__(self, *_):
            pass

    anthro_client.client.stream = lambda *a, **kw: _ErrorCtx()
    with pytest.raises(RuntimeError, match="HTTP 401"):
        async for _ in anthro_client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await anthro_client.close()


async def test_stream_chat_general_exception_raises_runtime_error(client):
    """stream_chat (Responses format) wraps unexpected exceptions in RuntimeError."""
    class _ErrorCtx:
        async def __aenter__(self):
            raise ConnectionError("network down")
        async def __aexit__(self, *_):
            pass

    client.client.stream = lambda *a, **kw: _ErrorCtx()
    with pytest.raises(RuntimeError, match="Stream request failed"):
        async for _ in client.stream_chat([{"role": "user", "content": "hi"}]):
            pass
    await client.close()
