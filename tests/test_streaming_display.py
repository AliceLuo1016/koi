"""Tests for streaming display — stream_chat yields StreamEvent objects."""

import json
from unittest.mock import MagicMock, AsyncMock, patch
from io import StringIO

import httpx
import pytest

from koi.config import Config
from koi.llm import LLMClient
from koi.stream_events import StreamEvent


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
    """stream_chat yields StreamEvent objects and stores text in _last_stream_response."""
    lines = [
        'data: {"type":"response.output_text.delta","delta":"Hello"}',
        'data: {"type":"response.output_text.delta","delta":" world"}',
        "data: [DONE]",
    ]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in responses_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Hello", " world"]
    resp = responses_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hello world"
    assert "tool_calls" not in msg


# ── Responses API: tool call stream ──


async def test_responses_stream_chat_sets_last_response_tool_calls(responses_client):
    """stream_chat yields toolcall events and assembles tool_calls in _last_stream_response."""
    lines = [
        'data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"c1","name":"read_file"}}',
        'data: {"type":"response.function_call_arguments.delta","call_id":"c1","delta":"{\\"path\\": \\"/tmp/x\\"}"}',
        "data: [DONE]",
    ]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in responses_client.stream_chat(
        [{"role": "user", "content": "read /tmp/x"}]
    ):
        events.append(event)

    # Should have toolcall_start and toolcall_delta events
    types = [e.type for e in events]
    assert "toolcall_start" in types
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

    events = []
    async for event in responses_client.stream_chat(
        [{"role": "user", "content": "do it"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Done!"]
    resp = responses_client._last_stream_response
    assert resp is not None
    assert resp["id"] == "resp_123"
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Done!"


# ── Chat Completions: text-only stream ──


async def test_cc_stream_chat_sets_last_response_text(cc_client):
    """Chat Completions stream_chat yields StreamEvent and stores response."""
    lines = [
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',
        'data: {"choices":[{"delta":{"content":" there"}}]}',
        "data: [DONE]",
    ]

    cc_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in cc_client.stream_chat(
        [{"role": "user", "content": "hello"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Hi", " there"]
    resp = cc_client._last_stream_response
    assert resp is not None
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "Hi there"
    assert "tool_calls" not in msg


# ── Chat Completions: tool call stream ──


async def test_cc_stream_chat_sets_last_response_tool_calls(cc_client):
    """Chat Completions stream_chat yields toolcall events and assembles tool_calls."""
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"tc1","function":{"name":"run_command","arguments":""}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"cmd\\":"}}]}}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"ls\\"}"}}]}}]}',
        "data: [DONE]",
    ]

    cc_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in cc_client.stream_chat(
        [{"role": "user", "content": "list files"}]
    ):
        events.append(event)

    types = [e.type for e in events]
    assert "toolcall_start" in types
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
    """Anthropic stream_chat yields StreamEvent objects."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hey"}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"!"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Hey", "!"]


# ── Anthropic: tool call stream ──


async def test_anthropic_stream_chat_sets_last_response_tool_calls(anthropic_client):
    """Anthropic stream_chat yields toolcall events."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"read_file"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"/x\\"}"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client.stream_chat(
        [{"role": "user", "content": "read /x"}]
    ):
        events.append(event)

    types = [e.type for e in events]
    assert "toolcall_start" in types
    start = [e for e in events if e.type == "toolcall_start"][0]
    assert start.tool_name == "read_file"
    assert start.tool_call_id == "tu1"


# ── Anthropic: thinking deltas are skipped ──


async def test_anthropic_stream_chat_skips_thinking(anthropic_client):
    """Thinking events are yielded as thinking_delta, text as text_delta."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"internal reasoning"}}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Answer"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client.stream_chat(
        [{"role": "user", "content": "think about this"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Answer"]


# ── Empty stream ──


async def test_empty_stream_sets_empty_response(responses_client):
    """An empty stream produces an empty response in _last_stream_response."""
    lines = ["data: [DONE]"]

    responses_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in responses_client.stream_chat(
        [{"role": "user", "content": ""}]
    ):
        events.append(event)

    assert events == []
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

    # Mock stream_chat to yield StreamEvent objects
    async def mock_stream_chat(messages, tools=None, system_prompt=None):
        text = "<think>internal</think><final>visible answer</final>"
        for ch in [text[:7], text[7:25], text[25:]]:
            yield StreamEvent(type="text_delta", delta=ch)

    agent.llm_client.stream_chat = mock_stream_chat

    response = await agent._stream_response(
        [{"role": "user", "content": "test"}], []
    )

    assert response is not None
    msg = response["choices"][0]["message"]
    assert msg["content"] == "<think>internal</think><final>visible answer</final>"


# ── StreamEvent: Anthropic text response ──


async def test_anthropic_stream_events_text(anthropic_client):
    """_stream_anthropic_events yields text_start, text_delta, text_end, done."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client._stream_anthropic_events(
        [{"role": "user", "content": "hi"}]
    ):
        events.append(event)

    types = [e.type for e in events]
    assert types == ["text_start", "text_delta", "text_delta", "text_end", "done"]

    # Check text_delta contents
    deltas = [e.delta for e in events if e.type == "text_delta"]
    assert deltas == ["Hello", " world"]

    # Check text_end has full accumulated content
    text_end = [e for e in events if e.type == "text_end"][0]
    assert text_end.content == "Hello world"

    # Check done event
    done = [e for e in events if e.type == "done"][0]
    assert done.finish_reason == "stop"


# ── StreamEvent: Anthropic tool call ──


async def test_anthropic_stream_events_tool_call(anthropic_client):
    """_stream_anthropic_events yields toolcall_start, toolcall_delta, toolcall_end, done."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tu1","name":"read_file"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"/x\\"}"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client._stream_anthropic_events(
        [{"role": "user", "content": "read /x"}]
    ):
        events.append(event)

    types = [e.type for e in events]
    assert types == ["toolcall_start", "toolcall_delta", "toolcall_delta", "toolcall_end", "done"]

    # Check toolcall_start
    start = events[0]
    assert start.tool_name == "read_file"
    assert start.tool_call_id == "tu1"

    # Check toolcall_end has full arguments
    end = [e for e in events if e.type == "toolcall_end"][0]
    assert end.tool_name == "read_file"
    assert end.tool_call_id == "tu1"
    assert end.arguments == '{"path": "/x"}'


# ── StreamEvent: thinking skipped in stream_chat ──


async def test_anthropic_stream_events_thinking_skipped_in_stream_chat(anthropic_client):
    """stream_chat yields all events including thinking; consumer filters."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"internal reasoning"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}',
        'data: {"type":"content_block_stop","index":1}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client.stream_chat(
        [{"role": "user", "content": "think about this"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Answer"]
    # Thinking events are also present
    thinking_deltas = [e.delta for e in events if e.type == "thinking_delta"]
    assert thinking_deltas == ["internal reasoning"]


# ── StreamEvent: usage event ──


async def test_anthropic_stream_events_usage(anthropic_client):
    """usage events carry correct token counts."""
    lines = [
        'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0,"cache_read_input_tokens":5,"cache_creation_input_tokens":2}}}',
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
        'data: {"type":"content_block_stop","index":0}',
        'data: {"type":"message_delta","usage":{"output_tokens":3}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client._stream_anthropic_events(
        [{"role": "user", "content": "hi"}]
    ):
        events.append(event)

    usage_events = [e for e in events if e.type == "usage"]
    assert len(usage_events) == 2

    # First usage from message_start
    u0 = usage_events[0].usage
    assert u0["input_tokens"] == 10
    assert u0["cache_read_input_tokens"] == 5
    assert u0["cache_creation_input_tokens"] == 2

    # Second usage from message_delta
    u1 = usage_events[1].usage
    assert u1["output_tokens"] == 3


# ── StreamEvent: backward compat with stream_chat ──


async def test_anthropic_stream_chat_yields_events(anthropic_client):
    """stream_chat yields StreamEvent objects with correct types and deltas."""
    lines = [
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hey"}}',
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}',
        'data: {"type":"message_stop"}',
    ]

    anthropic_client.client.stream = MagicMock(return_value=_StreamCtx(lines))

    events = []
    async for event in anthropic_client.stream_chat(
        [{"role": "user", "content": "hi"}]
    ):
        events.append(event)

    text_deltas = [e.delta for e in events if e.type == "text_delta"]
    assert text_deltas == ["Hey", "!"]
    # All events are StreamEvent instances
    assert all(isinstance(e, StreamEvent) for e in events)
