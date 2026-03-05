"""Tests for model-aware thinking parameter safety."""

from unittest.mock import MagicMock

import httpx
import pytest

from koi.config import Config
from koi.llm import LLMClient, supports_thinking

# ── supports_thinking: Anthropic models ──


class TestSupportsThinkingAnthropic:
    def test_claude_4_supported(self):
        assert supports_thinking("claude-4", "anthropic") is True

    def test_claude_sonnet_4_supported(self):
        assert supports_thinking("claude-sonnet-4-20250514", "anthropic") is True

    def test_claude_opus_4_supported(self):
        assert supports_thinking("claude-opus-4-6", "anthropic") is True

    def test_claude_3_5_sonnet_supported(self):
        assert supports_thinking("claude-3-5-sonnet-20241022", "anthropic") is True

    def test_claude_3_dot_5_sonnet_supported(self):
        assert supports_thinking("claude-3.5-sonnet", "anthropic") is True

    def test_claude_3_5_haiku_not_supported(self):
        """claude-3-5-haiku does NOT support extended thinking."""
        assert supports_thinking("claude-3-5-haiku-20241022", "anthropic") is False

    def test_claude_3_haiku_not_supported(self):
        assert supports_thinking("claude-3-haiku-20240307", "anthropic") is False

    def test_claude_3_opus_not_supported(self):
        assert supports_thinking("claude-3-opus-20240229", "anthropic") is False

    def test_claude_3_sonnet_not_supported(self):
        assert supports_thinking("claude-3-sonnet-20240229", "anthropic") is False


# ── supports_thinking: OpenAI / Responses models ──


class TestSupportsThinkingOpenAI:
    def test_o1_preview_supported(self):
        assert supports_thinking("o1-preview", "responses") is True

    def test_o1_mini_supported(self):
        assert supports_thinking("o1-mini", "responses") is True

    def test_o3_mini_supported(self):
        assert supports_thinking("o3-mini", "responses") is True

    def test_o4_mini_supported(self):
        assert supports_thinking("o4-mini", "chat_completions") is True

    def test_gpt_5_2_supported(self):
        assert supports_thinking("gpt-5.2", "responses") is True

    def test_gpt_5_2_codex_supported(self):
        assert supports_thinking("openai/gpt-5.2-codex", "responses") is True

    def test_gpt_4o_not_supported(self):
        assert supports_thinking("gpt-4o", "responses") is False

    def test_gpt_4_turbo_not_supported(self):
        assert supports_thinking("gpt-4-turbo", "chat_completions") is False

    def test_gpt_4_not_supported(self):
        assert supports_thinking("gpt-4", "chat_completions") is False


# ── supports_thinking: Other models ──


class TestSupportsThinkingOther:
    def test_qwen3_supported(self):
        assert supports_thinking("qwen3-72b", "chat_completions") is True

    def test_qwen_3_dash_supported(self):
        assert supports_thinking("qwen-3-235b", "responses") is True

    def test_deepseek_r1_not_supported(self):
        """DeepSeek R1 has always-on reasoning — don't send effort param."""
        assert supports_thinking("deepseek-r1", "chat_completions") is False

    def test_deepseek_reasoner_not_supported(self):
        assert supports_thinking("deepseek-reasoner", "responses") is False

    def test_unknown_model_not_supported(self):
        """Unknown models default to False (safe)."""
        assert supports_thinking("llama-3.1-70b", "chat_completions") is False

    def test_unknown_model_responses_not_supported(self):
        assert supports_thinking("mistral-large", "responses") is False


# ── Anthropic payload: thinking params gated ──


class TestAnthropicPayloadGating:
    async def test_no_thinking_for_unsupported_anthropic_model(self):
        """claude-3-haiku should NOT get thinking params
        even with thinking_level=high."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-3-haiku-20240307",
            api_format="anthropic",
            thinking_level="high",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "id": "msg_1",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "thinking" not in captured
        # Should have temperature instead (if set)
        assert "anthropic-beta" not in client.headers

    async def test_thinking_included_for_supported_anthropic_model(self):
        """claude-sonnet-4 should get thinking params when thinking_level != off."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            api_format="anthropic",
            thinking_level="medium",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "id": "msg_1",
                "content": [{"type": "text", "text": "hi"}],
                "stop_reason": "end_turn",
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "thinking" in captured
        assert captured["thinking"]["type"] == "enabled"


# ── Chat Completions payload: reasoning_effort gated ──


class TestCCPayloadGating:
    async def test_no_reasoning_effort_for_gpt4o(self):
        """gpt-4o should NOT get reasoning_effort."""
        config = Config(
            api_base="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model="gpt-4o",
            api_format="chat_completions",
            thinking_level="high",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "reasoning_effort" not in captured

    async def test_reasoning_effort_for_o3(self):
        """o3-mini should get reasoning_effort."""
        config = Config(
            api_base="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model="o3-mini",
            api_format="chat_completions",
            thinking_level="high",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "reasoning_effort" in captured

    async def test_no_reasoning_effort_for_deepseek_r1(self):
        """deepseek-r1 has always-on reasoning — no effort param."""
        config = Config(
            api_base="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model="deepseek-r1",
            api_format="chat_completions",
            thinking_level="high",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "reasoning_effort" not in captured


# ── Responses API payload gating ──


class TestResponsesPayloadGating:
    async def test_no_reasoning_for_unknown_model(self):
        """Unknown model should NOT get reasoning params."""
        config = Config(
            api_base="https://api.example.com/v1/responses",
            api_key="test-key",
            model="llama-3.1-70b",
            api_format="responses",
            thinking_level="high",
        )
        client = LLMClient(config)
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hi"}],
                    }
                ],
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "reasoning" not in captured


# ── Fallback: thinking disabled after API error ──


class TestThinkingFallback:
    async def test_fallback_disables_thinking_on_error(self):
        """After a thinking-related 400 error, thinking is disabled
        and request retried."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            api_format="anthropic",
            thinking_level="medium",
        )
        client = LLMClient(config)
        call_count = 0
        payloads = []

        async def fake_post(url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            payloads.append(dict(json))
            if call_count == 1:
                # First call: error about thinking not supported
                mock_resp = MagicMock()
                mock_resp.status_code = 400
                mock_resp.text = "thinking is not supported for this model"
                mock_resp.headers = {}
                raise httpx.HTTPStatusError("400", request=MagicMock(), response=mock_resp)
            # Second call: success
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "id": "msg_1",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            }
            return resp

        client.client.post = fake_post
        result = await client.chat([{"role": "user", "content": "test"}])

        assert call_count == 2
        # First payload had thinking, second should NOT
        assert "thinking" in payloads[0]
        assert "thinking" not in payloads[1]
        # Flag is now set for the rest of the session
        assert client._thinking_disabled_fallback is True
        assert result["choices"][0]["message"]["content"] == "ok"

    async def test_fallback_flag_persists_across_calls(self):
        """Once fallback is set, subsequent calls also skip thinking."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            api_format="anthropic",
            thinking_level="high",
        )
        client = LLMClient(config)
        client._thinking_disabled_fallback = True  # pretend already failed
        captured = {}

        async def fake_post(url, headers=None, json=None):
            captured.update(json)
            resp = MagicMock()
            resp.raise_for_status = lambda: None
            resp.json.return_value = {
                "id": "msg_1",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            }
            return resp

        client.client.post = fake_post
        await client.chat([{"role": "user", "content": "test"}])

        assert "thinking" not in captured

    async def test_non_thinking_error_not_caught_by_fallback(self):
        """A 400 error NOT about thinking should still raise normally."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-sonnet-4-20250514",
            api_format="anthropic",
            thinking_level="medium",
        )
        client = LLMClient(config)

        async def fake_post(url, headers=None, json=None):
            mock_resp = MagicMock()
            mock_resp.status_code = 400
            mock_resp.text = "invalid_request_error: messages is required"
            mock_resp.headers = {}
            raise httpx.HTTPStatusError("400", request=MagicMock(), response=mock_resp)

        from koi.errors import KoiAPIError

        client.client.post = fake_post
        with pytest.raises(KoiAPIError):
            await client.chat([{"role": "user", "content": "test"}])
        assert client._thinking_disabled_fallback is False


# ── Config.effective_thinking_level ──


class TestEffectiveThinkingLevel:
    def test_returns_off_when_thinking_off(self):
        config = Config(model="claude-sonnet-4", api_format="anthropic", thinking_level="off")
        assert config.effective_thinking_level() == "off"

    def test_returns_level_for_supported_model(self):
        config = Config(model="claude-sonnet-4", api_format="anthropic", thinking_level="high")
        assert config.effective_thinking_level() == "high"

    def test_returns_off_for_unsupported_model(self):
        config = Config(model="claude-3-haiku", api_format="anthropic", thinking_level="high")
        assert config.effective_thinking_level() == "off"

    def test_returns_off_for_unknown_model(self):
        config = Config(
            model="llama-3.1-70b",
            api_format="chat_completions",
            thinking_level="medium",
        )
        assert config.effective_thinking_level() == "off"

    def test_returns_level_for_o3(self):
        config = Config(model="o3-mini", api_format="responses", thinking_level="medium")
        assert config.effective_thinking_level() == "medium"
