"""Tests for thinking/reasoning support across config, LLM, and CLI."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, AsyncMock, patch

import httpx
import pytest
from click.testing import CliRunner

from koi.config import Config, normalize_think_level, THINK_LEVELS
from koi.llm import (
    LLMClient,
    _ANTHROPIC_THINKING_BUDGETS,
    _ANTHROPIC_MODEL_MAX_TOKENS,
    _CC_REASONING_EFFORT,
    _adjust_max_tokens_for_thinking,
    _get_anthropic_model_max,
)


# ── normalize_think_level ──


class TestNormalizeThinkLevel:
    def test_canonical_levels(self):
        for level in THINK_LEVELS:
            assert normalize_think_level(level) == level

    def test_aliases(self):
        assert normalize_think_level("on") == "low"
        assert normalize_think_level("enable") == "low"
        assert normalize_think_level("enabled") == "low"
        assert normalize_think_level("disabled") == "off"
        assert normalize_think_level("none") == "off"
        assert normalize_think_level("min") == "minimal"
        assert normalize_think_level("think") == "minimal"
        assert normalize_think_level("med") == "medium"
        assert normalize_think_level("mid") == "medium"
        assert normalize_think_level("max") == "high"
        assert normalize_think_level("ultra") == "high"

    def test_case_insensitive(self):
        assert normalize_think_level("HIGH") == "high"
        assert normalize_think_level("Off") == "off"
        assert normalize_think_level("  Medium  ") == "medium"

    def test_unrecognized_returns_none(self):
        assert normalize_think_level("banana") is None
        assert normalize_think_level("") is None
        assert normalize_think_level("super") is None


# ── Config ──


class TestConfigThinking:
    def test_default_thinking_level(self):
        config = Config()
        assert config.thinking_level == "low"

    def test_explicit_thinking_level(self):
        config = Config(thinking_level="high")
        assert config.thinking_level == "high"

    def test_invalid_thinking_level_defaults_to_low(self):
        config = Config(thinking_level="banana")
        assert config.thinking_level == "low"

    def test_load_thinking_level_from_json(self):
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "api_base": "https://example.com",
                "api_key": "test",
                "model": "test-model",
                "thinking_level": "high",
            }))
            config = Config.load(config_path)
            assert config.thinking_level == "high"

    def test_load_missing_thinking_level_defaults_to_low(self):
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({
                "api_base": "https://example.com",
                "api_key": "test",
                "model": "test-model",
            }))
            config = Config.load(config_path)
            assert config.thinking_level == "low"

    def test_save_includes_thinking_level(self):
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config = Config(thinking_level="medium")
            config.save(config_path)
            data = json.loads(config_path.read_text())
            assert data["thinking_level"] == "medium"

    def test_to_dict_includes_thinking_level(self):
        config = Config(thinking_level="minimal")
        d = config.to_dict()
        assert d["thinking_level"] == "minimal"


# ── LLM Client: Anthropic ──


class TestAnthropicThinking:
    def _make_client(self, thinking_level="low"):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level=thinking_level,
        )
        return LLMClient(config)

    def test_beta_header_set_when_thinking_enabled(self):
        client = self._make_client("low")
        assert "anthropic-beta" in client.headers
        assert client.headers["anthropic-beta"] == "interleaved-thinking-2025-05-14"

    def test_beta_header_not_set_when_thinking_off(self):
        client = self._make_client("off")
        assert "anthropic-beta" not in client.headers

    @pytest.mark.asyncio
    async def test_payload_includes_thinking_when_enabled(self):
        client = self._make_client("medium")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert "thinking" in payload
            assert payload["thinking"]["type"] == "enabled"
            assert payload["thinking"]["budget_tokens"] == _ANTHROPIC_THINKING_BUDGETS["medium"]

    @pytest.mark.asyncio
    async def test_payload_excludes_thinking_when_off(self):
        client = self._make_client("off")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert "thinking" not in payload

    @pytest.mark.asyncio
    async def test_temperature_removed_when_thinking_enabled(self):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level="high",
            temperature=0.7,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert "temperature" not in payload
            assert "thinking" in payload

    @pytest.mark.asyncio
    async def test_temperature_present_when_thinking_off(self):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level="off",
            temperature=0.5,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert payload["temperature"] == 0.5
            assert "thinking" not in payload

    @pytest.mark.asyncio
    async def test_thinking_budget_per_level(self):
        for level, expected_budget in _ANTHROPIC_THINKING_BUDGETS.items():
            client = self._make_client(level)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "msg_1",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
            }
            mock_response.raise_for_status = MagicMock()

            with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                await client.chat([{"role": "user", "content": "test"}])
                payload = mock_post.call_args[1]["json"]
                assert payload["thinking"]["budget_tokens"] == expected_budget


# ── LLM Client: Thinking blocks stripped ──


class TestThinkingBlocksStripped:
    def test_thinking_blocks_stripped_from_response(self):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level="medium",
        )
        client = LLMClient(config)

        anthropic_response = {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
                {"type": "thinking", "thinking": "I should double check..."},
                {"type": "text", "text": " And that's final."},
            ],
            "stop_reason": "end_turn",
        }

        result = client._convert_anthropic_response(anthropic_response)
        message = result["choices"][0]["message"]
        assert message["content"] == "The answer is 42. And that's final."
        # Ensure no thinking content leaked
        assert "reason" not in message["content"].lower()

    def test_response_with_only_thinking_blocks(self):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
        )
        client = LLMClient(config)

        anthropic_response = {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "Hmm..."},
            ],
            "stop_reason": "end_turn",
        }

        result = client._convert_anthropic_response(anthropic_response)
        message = result["choices"][0]["message"]
        assert "content" not in message

    def test_thinking_blocks_with_tool_use(self):
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
        )
        client = LLMClient(config)

        anthropic_response = {
            "id": "msg_1",
            "content": [
                {"type": "thinking", "thinking": "I need to call a tool."},
                {"type": "text", "text": "Let me check that."},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "read_file",
                    "input": {"path": "test.py"},
                },
            ],
            "stop_reason": "tool_use",
        }

        result = client._convert_anthropic_response(anthropic_response)
        message = result["choices"][0]["message"]
        assert message["content"] == "Let me check that."
        assert len(message["tool_calls"]) == 1
        assert message["tool_calls"][0]["function"]["name"] == "read_file"


# ── LLM Client: Chat Completions ──


class TestChatCompletionsThinking:
    def _make_client(self, thinking_level="low"):
        config = Config(
            api_base="https://api.example.com/v1/chat/completions",
            api_key="test-key",
            model="o3-mini",
            api_format="chat_completions",
            thinking_level=thinking_level,
        )
        return LLMClient(config)

    @pytest.mark.asyncio
    async def test_reasoning_effort_included(self):
        client = self._make_client("high")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert payload["reasoning_effort"] == "high"

    @pytest.mark.asyncio
    async def test_reasoning_effort_omitted_when_off(self):
        client = self._make_client("off")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert "reasoning_effort" not in payload

    @pytest.mark.asyncio
    async def test_reasoning_effort_mapping(self):
        for level, expected_effort in _CC_REASONING_EFFORT.items():
            client = self._make_client(level)
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{"message": {"role": "assistant", "content": "hi"}}],
            }
            mock_response.raise_for_status = MagicMock()

            with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
                mock_post.return_value = mock_response
                await client.chat([{"role": "user", "content": "test"}])
                payload = mock_post.call_args[1]["json"]
                assert payload["reasoning_effort"] == expected_effort


# ── LLM Client: Responses API ──


class TestResponsesAPIThinking:
    def _make_client(self, thinking_level="low"):
        config = Config(
            api_base="https://api.example.com/v1/responses",
            api_key="test-key",
            model="o3-mini",
            api_format="responses",
            thinking_level=thinking_level,
        )
        return LLMClient(config)

    @pytest.mark.asyncio
    async def test_reasoning_included(self):
        client = self._make_client("medium")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_1",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert payload["reasoning"] == {"effort": "medium"}

    @pytest.mark.asyncio
    async def test_reasoning_omitted_when_off(self):
        client = self._make_client("off")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "resp_1",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert "reasoning" not in payload


# ── CLI ──


class TestCLIThinking:
    @pytest.fixture(autouse=True)
    def _mock_agent_import(self):
        """Mock prompt_toolkit to avoid ImportError when loading koi.cli.

        Only remove koi.cli and koi.agent from the module cache (they need
        prompt_toolkit). Leave other koi.* modules intact so downstream
        tests that mock koi.cron etc. are not affected by cache pollution.
        """
        import sys
        import types

        # Create a proper mock package with submodules
        pt_pkg = types.ModuleType("prompt_toolkit")
        pt_pkg.PromptSession = MagicMock()
        pt_kb = types.ModuleType("prompt_toolkit.key_binding")
        pt_kb.KeyBindings = MagicMock()

        modules = {
            "prompt_toolkit": pt_pkg,
            "prompt_toolkit.key_binding": pt_kb,
        }
        # Only evict the modules that depend on prompt_toolkit
        cli_modules = ["koi.cli", "koi.agent"]
        saved = {k: sys.modules.pop(k) for k in cli_modules if k in sys.modules}
        with patch.dict(sys.modules, modules):
            yield
        # Restore evicted modules so other tests see them unchanged
        for k, v in saved.items():
            sys.modules[k] = v

    def test_run_accepts_thinking_flag(self):
        from koi.cli import main
        runner = CliRunner()
        with patch("koi.cli.Config.load") as mock_load, \
             patch("koi.cli.Agent") as mock_agent, \
             patch("koi.cli.asyncio") as mock_asyncio:
            mock_config = Config(
                api_base="https://example.com",
                api_key="test",
                model="test-model",
                thinking_level="low",
            )
            mock_load.return_value = mock_config
            mock_agent_instance = MagicMock()
            mock_agent.return_value = mock_agent_instance

            result = runner.invoke(main, ["run", "--thinking", "high", "--task", "hello"])

            assert mock_config.thinking_level == "high"

    def test_run_without_thinking_flag_uses_config_default(self):
        from koi.cli import main
        runner = CliRunner()
        with patch("koi.cli.Config.load") as mock_load, \
             patch("koi.cli.Agent") as mock_agent, \
             patch("koi.cli.asyncio") as mock_asyncio:
            mock_config = Config(
                api_base="https://example.com",
                api_key="test",
                model="test-model",
                thinking_level="medium",
            )
            mock_load.return_value = mock_config
            mock_agent_instance = MagicMock()
            mock_agent.return_value = mock_agent_instance

            result = runner.invoke(main, ["run", "--task", "hello"])

            assert mock_config.thinking_level == "medium"

    def test_run_thinking_off_flag(self):
        from koi.cli import main
        runner = CliRunner()
        with patch("koi.cli.Config.load") as mock_load, \
             patch("koi.cli.Agent") as mock_agent, \
             patch("koi.cli.asyncio") as mock_asyncio:
            mock_config = Config(
                api_base="https://example.com",
                api_key="test",
                model="test-model",
                thinking_level="high",
            )
            mock_load.return_value = mock_config
            mock_agent_instance = MagicMock()
            mock_agent.return_value = mock_agent_instance

            result = runner.invoke(main, ["run", "--thinking", "off", "--task", "hello"])

            assert mock_config.thinking_level == "off"


# ── _adjust_max_tokens_for_thinking helper ──


class TestAdjustMaxTokensForThinking:
    def test_basic_adjustment(self):
        """max_tokens should be base + budget when under model max."""
        max_tokens, budget = _adjust_max_tokens_for_thinking(4096, 2048, 64000)
        assert max_tokens == 6144
        assert budget == 2048

    def test_capped_at_model_max(self):
        """max_tokens should not exceed model_max."""
        max_tokens, budget = _adjust_max_tokens_for_thinking(60000, 16384, 64000)
        assert max_tokens == 64000
        assert budget == 16384

    def test_budget_reduced_when_max_tokens_lte_budget(self):
        """If max_tokens <= budget, budget is reduced to reserve 1024 for output."""
        # model_max=4096, base=1024, budget=8192 → max_tokens=min(9216,4096)=4096
        # 4096 <= 8192 → budget = max(0, 4096-1024) = 3072
        max_tokens, budget = _adjust_max_tokens_for_thinking(1024, 8192, 4096)
        assert max_tokens == 4096
        assert budget == 3072

    def test_budget_reduced_to_zero_when_model_max_tiny(self):
        """With a very small model_max, budget can go to 0."""
        max_tokens, budget = _adjust_max_tokens_for_thinking(512, 8192, 512)
        assert max_tokens == 512
        assert budget == 0

    def test_exact_equal_case(self):
        """When base + budget exactly equals model_max."""
        max_tokens, budget = _adjust_max_tokens_for_thinking(4096, 2048, 6144)
        assert max_tokens == 6144
        assert budget == 2048

    def test_default_model_max(self):
        """Default model_max is 64000."""
        max_tokens, budget = _adjust_max_tokens_for_thinking(4096, 2048)
        assert max_tokens == 6144
        assert budget == 2048


# ── Model max token lookup ──


class TestGetAnthropicModelMax:
    def test_claude_opus_4(self):
        assert _get_anthropic_model_max("claude-opus-4-20250514") == 64000

    def test_claude_sonnet_4(self):
        assert _get_anthropic_model_max("claude-sonnet-4-20250514") == 64000

    def test_claude_3_5_sonnet(self):
        assert _get_anthropic_model_max("claude-3-5-sonnet-20241022") == 8192

    def test_unknown_model_gets_default(self):
        assert _get_anthropic_model_max("some-unknown-model") == 64000


# ── Low budget value ──


class TestLowBudgetValue:
    def test_low_budget_is_2048(self):
        """Low thinking budget should be 2048 (matching pi-ai)."""
        assert _ANTHROPIC_THINKING_BUDGETS["low"] == 2048


# ── max_tokens adjusted in Anthropic payloads ──


class TestAnthropicMaxTokensAdjusted:
    @pytest.mark.asyncio
    async def test_max_tokens_increased_for_thinking(self):
        """max_tokens in the payload should be base + budget, not just base."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level="low",
            max_tokens=4096,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            budget = _ANTHROPIC_THINKING_BUDGETS["low"]  # 2048
            assert payload["max_tokens"] == 4096 + budget  # 6144
            assert payload["thinking"]["budget_tokens"] == budget

    @pytest.mark.asyncio
    async def test_max_tokens_capped_at_model_max(self):
        """max_tokens should not exceed the model's maximum."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-3-5-sonnet-20241022",
            api_format="anthropic",
            thinking_level="high",
            max_tokens=4096,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            # 4096 + 16384 = 20480, but model max is 8192
            assert payload["max_tokens"] == 8192

    @pytest.mark.asyncio
    async def test_budget_reduced_when_model_max_small(self):
        """When model max is very constraining, budget is reduced to reserve output tokens."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-3-5-sonnet-20241022",
            api_format="anthropic",
            thinking_level="high",
            max_tokens=4096,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            # model max 8192, budget 16384 → max_tokens=8192, 8192 <= 16384
            # → budget = max(0, 8192-1024) = 7168
            assert payload["max_tokens"] == 8192
            assert payload["thinking"]["budget_tokens"] == 7168

    @pytest.mark.asyncio
    async def test_max_tokens_unchanged_when_thinking_off(self):
        """When thinking is off, max_tokens should be the raw config value."""
        config = Config(
            api_base="https://api.anthropic.com/v1/messages",
            api_key="sk-ant-test",
            model="claude-opus-4-20250514",
            api_format="anthropic",
            thinking_level="off",
            max_tokens=4096,
        )
        client = LLMClient(config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "msg_1",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
        }
        mock_response.raise_for_status = MagicMock()

        with patch.object(client.client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.chat([{"role": "user", "content": "test"}])

            payload = mock_post.call_args[1]["json"]
            assert payload["max_tokens"] == 4096
