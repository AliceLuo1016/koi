"""Tests for token usage tracking."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from koi.config import Config
from koi.llm import LLMClient
from koi.usage import TokenUsage, estimate_cost, log_usage


# ── TokenUsage unit tests ──


class TestTokenUsage:
    def test_add_accumulates(self):
        u = TokenUsage()
        u.add(input_t=100, output_t=50)
        u.add(input_t=200, output_t=30)
        assert u.input_tokens == 300
        assert u.output_tokens == 80
        assert u.total_requests == 2

    def test_total_tokens(self):
        u = TokenUsage()
        u.add(input_t=400, output_t=100)
        assert u.total_tokens == 500

    def test_cache_tokens_tracked(self):
        u = TokenUsage()
        u.add(input_t=100, output_t=50, cache_read=30, cache_creation=20)
        assert u.cache_read_tokens == 30
        assert u.cache_creation_tokens == 20

    def test_summary_includes_all_fields(self):
        u = TokenUsage()
        u.add(input_t=1000, output_t=500, cache_read=200, cache_creation=100)
        s = u.summary("claude-sonnet-4")
        assert "1,000" in s
        assert "500" in s
        assert "1,500" in s  # total
        assert "Requests: 1" in s
        assert "Cache read" in s
        assert "Cache creation" in s
        assert "Est. cost: $" in s

    def test_summary_no_cache_lines_when_zero(self):
        u = TokenUsage()
        u.add(input_t=100, output_t=50)
        s = u.summary("gpt-4o")
        assert "Cache read" not in s
        assert "Cache creation" not in s

    def test_summary_no_cost_for_unknown_model(self):
        u = TokenUsage()
        u.add(input_t=100, output_t=50)
        s = u.summary("some-unknown-model")
        assert "Est. cost" not in s

    def test_to_dict_round_trips(self):
        u = TokenUsage()
        u.add(input_t=100, output_t=50, cache_read=10, cache_creation=5)
        u.add(input_t=200, output_t=100)
        d = u.to_dict()
        assert d == {
            "input_tokens": 300,
            "output_tokens": 150,
            "cache_read_tokens": 10,
            "cache_creation_tokens": 5,
            "total_requests": 2,
        }
        # Reconstruct from dict
        u2 = TokenUsage(**d)
        assert u2.input_tokens == 300
        assert u2.total_tokens == 450

    def test_default_values(self):
        u = TokenUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.cache_read_tokens == 0
        assert u.cache_creation_tokens == 0
        assert u.total_requests == 0
        assert u.total_tokens == 0


# ── estimate_cost tests ──


class TestEstimateCost:
    def test_known_model_claude_sonnet(self):
        cost = estimate_cost("claude-sonnet-4-20250514", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.0 + 15.0)

    def test_known_model_with_cache(self):
        cost = estimate_cost(
            "claude-sonnet-4",
            input_tokens=1_000_000,
            output_tokens=0,
            cache_read=1_000_000,
            cache_creation=1_000_000,
        )
        # input: 3.0, cache_read: 0.3, cache_write: 3.75
        assert cost == pytest.approx(3.0 + 0.3 + 3.75)

    def test_known_model_gpt4o(self):
        cost = estimate_cost("gpt-4o-2025-01-01", 1_000_000, 1_000_000)
        assert cost == pytest.approx(2.5 + 10.0)

    def test_known_model_o3(self):
        cost = estimate_cost("o3-mini", 1_000_000, 1_000_000)
        assert cost == pytest.approx(10.0 + 40.0)

    def test_unknown_model_returns_zero(self):
        assert estimate_cost("llama-3-70b", 10000, 5000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("claude-sonnet-4", 0, 0) == 0.0

    def test_case_insensitive(self):
        cost = estimate_cost("Claude-Sonnet-4", 1_000_000, 0)
        assert cost == pytest.approx(3.0)


# ── log_usage tests ──


class TestLogUsage:
    def test_log_creates_file(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            u = TokenUsage()
            u.add(input_t=100, output_t=50)
            log_usage(u, "test-model", log_dir)

            log_path = log_dir / "usage-log.jsonl"
            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 1
            entry = json.loads(lines[0])
            assert entry["model"] == "test-model"
            assert entry["session_tokens"]["input_tokens"] == 100
            assert entry["session_tokens"]["output_tokens"] == 50
            assert "timestamp" in entry
            assert "estimated_cost" in entry

    def test_log_appends(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            u1 = TokenUsage()
            u1.add(input_t=100, output_t=50)
            log_usage(u1, "model-a", log_dir)

            u2 = TokenUsage()
            u2.add(input_t=200, output_t=100)
            log_usage(u2, "model-b", log_dir)

            log_path = log_dir / "usage-log.jsonl"
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_log_skips_zero_requests(self):
        with TemporaryDirectory() as tmp:
            log_dir = Path(tmp)
            u = TokenUsage()
            log_usage(u, "test-model", log_dir)
            log_path = log_dir / "usage-log.jsonl"
            assert not log_path.exists()


# ── LLMClient usage extraction tests ──

# Helpers


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
    def __init__(self, lines, status_code=200):
        self._resp = _MockStreamResponse(lines, status_code)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *_):
        pass


def _make_client(api_format="responses", model="test-model"):
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model=model,
        api_format=api_format,
        thinking_level="off",
    )
    return LLMClient(config)


class TestLLMClientUsageExtraction:
    def test_responses_api_usage(self):
        client = _make_client("responses")
        data = {
            "id": "resp-1",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ],
            "usage": {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        }
        client._convert_response(data)
        assert client.usage.input_tokens == 100
        assert client.usage.output_tokens == 50
        assert client.usage.total_requests == 1

    def test_chat_completions_usage(self):
        client = _make_client("chat_completions")
        data = {
            "id": "chatcmpl-1",
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 80,
                "total_tokens": 280,
            },
        }
        client._convert_cc_response(data)
        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 80
        assert client.usage.total_requests == 1

    def test_anthropic_usage(self):
        client = _make_client("anthropic", model="claude-sonnet-4")
        data = {
            "id": "msg-1",
            "content": [{"type": "text", "text": "hello"}],
            "usage": {
                "input_tokens": 500,
                "output_tokens": 120,
                "cache_read_input_tokens": 300,
                "cache_creation_input_tokens": 100,
            },
        }
        client._convert_anthropic_response(data)
        assert client.usage.input_tokens == 500
        assert client.usage.output_tokens == 120
        assert client.usage.cache_read_tokens == 300
        assert client.usage.cache_creation_tokens == 100
        assert client.usage.total_requests == 1

    def test_anthropic_no_cache_fields(self):
        client = _make_client("anthropic", model="claude-sonnet-4")
        data = {
            "id": "msg-2",
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        client._convert_anthropic_response(data)
        assert client.usage.cache_read_tokens == 0
        assert client.usage.cache_creation_tokens == 0

    def test_multiple_responses_accumulate(self):
        client = _make_client("responses")
        for _ in range(3):
            client._convert_response(
                {
                    "output": [],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            )
        assert client.usage.input_tokens == 300
        assert client.usage.output_tokens == 150
        assert client.usage.total_requests == 3

    def test_no_usage_field(self):
        """No crash when usage field is absent."""
        client = _make_client("responses")
        client._convert_response({"output": []})
        assert client.usage.total_requests == 0


class TestLLMClientStreamingUsage:
    async def test_streaming_responses_usage(self):
        """Usage extracted from response.completed in Responses API streaming."""
        client = _make_client("responses")
        resp_json = json.dumps(
            {
                "id": "resp-1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 150, "output_tokens": 60},
            }
        )
        lines = [
            'data: {"type":"response.output_text.delta","delta":"hello"}',
            f'data: {{"type":"response.completed","response":{resp_json}}}',
        ]

        with patch.object(client.client, "stream", return_value=_StreamCtx(lines)):
            result = await client._stream_chat(
                "https://api.example.com/v1/responses", {}
            )

        assert client.usage.input_tokens == 150
        assert client.usage.output_tokens == 60

    async def test_streaming_cc_usage(self):
        """Usage extracted from final chunk in Chat Completions streaming."""
        client = _make_client("chat_completions")
        lines = [
            'data: {"choices":[{"delta":{"content":"hi"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":200,"completion_tokens":90}}',
            "data: [DONE]",
        ]

        with patch.object(client.client, "stream", return_value=_StreamCtx(lines)):
            result = await client._stream_chat_completions(
                "https://api.example.com/v1/chat/completions", {}
            )

        assert client.usage.input_tokens == 200
        assert client.usage.output_tokens == 90

    async def test_streaming_anthropic_usage(self):
        """Usage extracted from message_start and message_delta in Anthropic streaming."""
        client = _make_client("anthropic", model="claude-sonnet-4")
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":400,"output_tokens":0,"cache_read_input_tokens":100,"cache_creation_input_tokens":50}}}',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":30}}',
            'data: {"type":"message_stop"}',
        ]

        with patch.object(client.client, "stream", return_value=_StreamCtx(lines)):
            result = await client._stream_anthropic(
                "https://api.anthropic.com/v1/messages", {}
            )

        assert client.usage.input_tokens == 400
        assert client.usage.output_tokens == 30
        assert client.usage.cache_read_tokens == 100
        assert client.usage.cache_creation_tokens == 50
        # message_start + message_delta = 2 extraction calls
        assert client.usage.total_requests == 2

    async def test_streaming_anthropic_tokens_usage(self):
        """Usage extracted from _stream_anthropic_tokens generator."""
        client = _make_client("anthropic", model="claude-sonnet-4")
        lines = [
            'data: {"type":"message_start","message":{"usage":{"input_tokens":300,"output_tokens":0}}}',
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world"}}',
            'data: {"type":"message_delta","usage":{"output_tokens":20}}',
            'data: {"type":"message_stop"}',
        ]

        with patch.object(client.client, "stream", return_value=_StreamCtx(lines)):
            tokens = []
            async for t in client._stream_anthropic_tokens(
                [{"role": "user", "content": "hi"}]
            ):
                tokens.append(t)

        assert tokens == ["world"]
        assert client.usage.input_tokens == 300
        assert client.usage.output_tokens == 20


# ── /usage command test ──


class TestUsageCommand:
    async def test_usage_command(self, capsys):
        """The /usage command prints usage summary."""
        config = Config(
            api_base="https://api.example.com/v1/responses",
            api_key="test-key",
            model="claude-sonnet-4",
            api_format="responses",
            thinking_level="off",
        )
        from koi.agent import Agent

        agent = Agent(config, non_interactive=True)
        agent.llm_client.usage.add(input_t=1000, output_t=500)

        await agent._handle_command("/usage")
        captured = capsys.readouterr().out
        assert "1,000" in captured
        assert "500" in captured
