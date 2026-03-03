"""Tests for streaming display — stream_chat sets _last_stream_response."""

import json
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

import httpx
import pytest

from koi.config import Config
from koi.llm import LLMClient, TOOL_CALL_START


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
def responses_client():
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model="test-model",
        api_format="responses",
    )
    return LLMClient(config)


@pytest.fixture
def cc_client():
    config = Config(
        api_base="https://api.example.com/v1/chat/completions",
        api_key="test-key",
        model="test-model",
        api_format="chat_completions",
    )
    return LLMClient(config)


@pytest.fixture
def anthropic_client():
    config = Config(
        api_base="https://api.anthropic.com/v1/messages",
        api_key="test-key",
        model="claude-sonnet-4-20250514",
        api_format="anthropic",
    )
    return LLMClient(config)


# ── Responses API: text-only stream ──


async def test_responses_stream_chat_sets_last_response_text(responses_client):
    """stream_chat yields text tokens and stores text in _last_stream_response."""
    lines = [
        'data: {"type":"response.output_text.delta","delta":"Hello"}',
        'data: {"type":"response.output_text.delta","delta":" world"}',
        "data: [DONE]",
    ]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in responses_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        tokens.append(token)

    assert tokens == ["Hello", " world"]
    resp = responses_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hello world"
    assert "tool_calls" not in msg


# ── Responses API: tool call stream ──


async def test_responses_stream_chat_sets_last_response_tool_calls(responses_client):
    """stream_chat assembles tool_calls in _last_stream_response."""
    lines = [
        'data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"c1","name":"read_file"}}',
        'data: {"type":"response.function_call_arguments.delta","call_id":"c1","delta":"{\\"path\\": \\"/tmp/x\\"}"}',
        "data: [DONE]",
    ]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in responses_client.stream_chat(
        [{"role": "user", "content": "read /tmp/x"}]
    ):
        tokens.append(token)

    # Only sentinel token for tool calls, no text
    assert tokens == [TOOL_CALL_START]
    resp = responses_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert "content" not in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"
    assert json.loads(tc["function"]["arguments"]) == {"path": "/tmp/x"}


# ── Responses API: response.completed event ──


async def test_responses_stream_chat_uses_completed_event(responses_client):
    """When response.completed fires, the converted response is used."""
    completed_response = {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Done!"}],
            }
        ],
    }
    lines = [
        'data: {"type":"response.output_text.delta","delta":"Done!"}',
        f'data: {{"type":"response.completed","response":{json.dumps(completed_response)}}}',
    ]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in responses_client.stream_chat(
        [{"role": "user", "content": "do it"}]
    ):
        tokens.append(token)

    assert tokens == ["Done!"]
    resp = responses_client._last_stream_response
    assert resp is not None
    assert resp["id"] == "resp_123"
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Done!"


# ── Chat Completions: text-only stream ──


async def test_cc_stream_chat_sets_last_response_text(cc_client):
    """Chat Completions stream_chat yields text and stores response."""
    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
    ]

    cc_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in cc_client.stream_chat(
        [{"role": "user", "content": "hello"}]
    ):
        tokens.append(token)

    assert tokens == ["Hi", " there"]
    resp = cc_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hi there"
    assert "tool_calls" not in msg


# ── Chat Completions: tool call stream ──


async def test_cc_stream_chat_sets_last_response_tool_calls(cc_client):
    """Chat Completions stream_chat assembles tool_calls."""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"run_command","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"cmd\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"ls\\"}"}}]}}]}',
        "data: [DONE]",
    ]

    cc_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in cc_client.stream_chat(
        [{"role": "user", "content": "list files"}]
    ):
        tokens.append(token)

    assert tokens == [TOOL_CALL_START]
    resp = cc_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "tc1"
    assert tc["function"]["name"] == "run_command"
    assert json.loads(tc["function"]["arguments"]) == {"cmd": "ls"}


# ── Anthropic: text-only stream ──


async def test_anthropic_stream_chat_sets_last_response_text(anthropic_client):
    """Anthropic stream_chat yields text and stores response."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hey"}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in anthropic_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        tokens.append(token)

    assert tokens == ["Hey", "!"]
    resp = anthropic_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hey!"
    assert "tool_calls" not in msg


# ── Anthropic: tool call stream ──


async def test_anthropic_stream_chat_sets_last_response_tool_calls(anthropic_client):
    """Anthropic stream_chat assembles tool_calls."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"read_file"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"/x\\"}"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in anthropic_client.stream_chat(
        [{"role": "user", "content": "read /x"}]
    ):
        tokens.append(token)

    assert tokens == [TOOL_CALL_START]
    resp = anthropic_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"] == '{"path": "/x"}'


# ── Anthropic: thinking deltas are skipped ──


async def test_anthropic_stream_chat_skips_thinking(anthropic_client):
    """Thinking deltas are not yielded as text tokens."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"internal reasoning"}}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Answer"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in anthropic_client.stream_chat(
        [{"role": "user", "content": "think about this"}]
    ):
        tokens.append(token)

    assert tokens == ["Answer"]
    resp = anthropic_client._last_stream_response
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Answer"


# ── Empty stream ──


async def test_empty_stream_sets_empty_response(responses_client):
    """An empty stream produces an empty response in _last_stream_response."""
    lines = ["data: [DONE]"]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    tokens = []
    async for token in responses_client.stream_chat(
        [{"role": "user", "content": ""}]
    ):
        tokens.append(token)

    assert tokens == []
    resp = responses_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert "tool_calls" not in msg


# ── _last_stream_response is None before streaming ──


async def test_last_stream_response_none_before_streaming(responses_client):
    """_last_stream_response is None before any streaming call."""
    assert responses_client._last_stream_response is None


# ── agent._stream_response reasoning tag stripping ──


async def test_stream_response_strips_reasoning_tags():
    """_stream_response buffers text and strips <think>/<final> tags."""
    from koi.agent import Agent

    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model="test-model",
        api_format="responses",
    )

    agent = Agent(config, non_interactive=True)
    # Simulate reasoning tag mode
    agent.llm_client.use_reasoning_tags = True

    # Mock stream_chat to yield tokens that include think/final tags
    async def mock_stream_chat(messages, tools=None, system_prompt=None):
        text = "<think>internal</think><final>visible answer</final>"
        for ch in [text[:7], text[7:25], text[25:]]:
            yield ch

    agent.llm_client.stream_chat = mock_stream_chat
    agent.llm_client._last_stream_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "<think>internal</think><final>visible answer</final>",
                },
                "finish_reason": "stop",
            }
        ]
    }

    response = await agent._stream_response(
        [{"role": "user", "content": "test"}], []
    )

    assert response is not None
    msg = response["choices"][0]["message"]
    assert msg["content"] == "<think>internal</think><final>visible answer</final>"
