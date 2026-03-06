"""LLM client for OpenAI-compatible Responses API."""

import asyncio
import json
import re
from collections.abc import AsyncGenerator
from typing import Any

import httpx

from .config import Config
from .errors import (
    KoiAPIError,
    KoiConnectionError,
    KoiRateLimitError,
    classify_http_error,
    extract_retry_delay,
)
from .stream_events import StreamEvent
from .usage import TokenUsage

# Anthropic thinking budget tokens by level
_ANTHROPIC_THINKING_BUDGETS = {
    "minimal": 1024,
    "low": 2048,
    "medium": 8192,
    "high": 16384,
}

# Chat Completions / Responses API reasoning_effort mapping
_CC_REASONING_EFFORT = {
    "minimal": "low",
    "low": "medium",
    "medium": "medium",
    "high": "high",
}

# Anthropic model max output tokens for adjusting thinking budget headroom
_ANTHROPIC_MODEL_MAX_TOKENS = {
    "claude-4": 64000,
    "claude-opus-4": 64000,
    "claude-sonnet-4": 64000,
    "claude-3-5-sonnet": 8192,
    "claude-3.5-sonnet": 8192,
}
_ANTHROPIC_MODEL_MAX_TOKENS_DEFAULT = 64000


def _get_anthropic_model_max(model: str) -> int:
    """Return the max output tokens for an Anthropic model."""
    m = model.lower()
    for prefix, limit in _ANTHROPIC_MODEL_MAX_TOKENS.items():
        if prefix in m:
            return limit
    return _ANTHROPIC_MODEL_MAX_TOKENS_DEFAULT


def _adjust_max_tokens_for_thinking(base_max: int, budget: int, model_max: int = 64000) -> tuple:
    """Adjust max_tokens so it covers both thinking budget and output.

    Returns (max_tokens, budget) where max_tokens = min(base_max + budget, model_max).
    If the resulting max_tokens would leave no room for output
    (i.e. max_tokens <= budget),
    the budget is reduced to reserve at least 1024 tokens for output.
    """
    max_tokens = min(base_max + budget, model_max)
    if max_tokens <= budget:
        budget = max(0, max_tokens - 1024)
    return max_tokens, budget


# Error message patterns that indicate unsupported thinking/reasoning params
_THINKING_ERROR_PATTERNS = re.compile(
    r"thinking|reasoning|budget_tokens|not.?supported.*reason|"
    r"reasoning_effort|enable_thinking",
    re.IGNORECASE,
)


def supports_thinking(model: str, api_format: str) -> bool:
    """Return True if the model is known to support thinking/reasoning params.

    When unsure, defaults to False (safer to omit thinking params).
    """
    m = model.lower()

    if api_format == "anthropic":
        # Claude 4.x family — all support extended thinking
        if re.search(r"claude[- ]?4", m):
            return True
        # Claude 3.5-sonnet supports it; 3.5-haiku does NOT
        if re.search(r"claude[- ]?3[.-]5[- ]?(sonnet|sonnet-v2)", m):
            return True
        if re.search(r"claude[- ]?3\.5[- ]?(sonnet|sonnet-v2)", m):
            return True
        # Claude sonnet-4, opus-4 naming variants
        if re.search(r"claude[- ]?(sonnet|opus)[- ]?4", m):
            return True
        # Older Claude models (3-haiku, 3-opus, 3-sonnet, 3.5-haiku) — no
        return False

    # Non-Anthropic formats: responses, chat_completions
    # OpenAI reasoning models
    if re.search(r"\bo[134]\b", m):
        return True
    # GPT-5.x reasoning
    if "gpt-5" in m:
        return True
    # Qwen 3.x has enable_thinking
    if re.search(r"qwen[- ]?3", m):
        return True
    # DeepSeek reasoning models — reasoning is always-on, don't send effort param
    if "deepseek-r1" in m or "deepseek-reasoner" in m:
        return False
    # GPT-4o, GPT-4-turbo, GPT-4 — no reasoning support
    if re.search(r"gpt-4", m):
        return False

    # Unknown model — default to NOT sending thinking params
    return False


def uses_reasoning_tags(model: str, api_format: str, thinking_level: str) -> bool:
    """Return True if the model should use prompt-based <think>/<final> tags.

    This is the fallback for models that don't support native thinking APIs
    but can still reason when prompted with structured tags.  Returns True
    when thinking is requested (level != 'off') but the model lacks native
    support.
    """
    if thinking_level == "off":
        return False
    if supports_thinking(model, api_format):
        return False
    # Thinking requested but model has no native support → use tags
    return True


class LLMClient:
    """Client for OpenAI Responses API.

    Translates internally-used Chat Completions message format to/from the
    Responses API format so the rest of the codebase stays unchanged.
    """

    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    MAX_RETRIES = 6
    MAX_BACKOFF = 60
    MAX_RETRY_DELAY = 60  # seconds — fail fast if server wants longer

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._thinking_disabled_fallback = False
        # For abort on Ctrl+C
        self._active_stream_response: httpx.Response | None = None
        self.usage = TokenUsage()
        # try stream_options.include_usage; disable on error
        self._stream_include_usage = True
        self.use_reasoning_tags = uses_reasoning_tags(config.model, config.api_format, config.thinking_level)
        if config.api_format == "anthropic":
            self.headers = {
                "Content-Type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            }
            if config.thinking_level != "off" and supports_thinking(config.model, config.api_format):
                self.headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"
        else:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            }

    def _should_send_thinking(self) -> bool:
        """Return True if thinking params should be included in requests."""
        if self._thinking_disabled_fallback:
            return False
        if self.config.thinking_level == "off":
            return False
        return supports_thinking(self.config.model, self.config.api_format)

    # ── Format conversion helpers ──

    def _convert_messages_to_input(self, messages: list[dict[str, Any]], system_prompt: str | None = None) -> tuple:
        """Convert Chat Completions messages → (instructions, input).

        Returns (instructions, input_items) where *instructions* is the
        system prompt and *input_items* is the conversation history in
        Responses API format.

        The system prompt is passed in separately — it is never extracted
        from the messages array.
        """
        input_items: list[dict[str, Any]] = []

        # Inject system prompt as a developer message so proxies that
        # ignore the 'instructions' field still see it.
        if system_prompt:
            input_items.append(
                {
                    "role": "developer",
                    "content": system_prompt,
                }
            )

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                input_items.append(
                    {
                        "role": "user",
                        "content": msg.get("content", ""),
                    }
                )

            elif role == "assistant":
                content = msg.get("content")
                if content:
                    input_items.append(
                        {
                            "role": "assistant",
                            "content": content,
                        }
                    )

                # Tool calls → function_call items
                for tc in msg.get("tool_calls", []):
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        }
                    )

            elif role == "tool":
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.get("tool_call_id", ""),
                        "output": msg.get("content", ""),
                    }
                )

        return system_prompt, input_items

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Flatten Chat Completions tool defs → Responses API format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted.append(
                    {
                        "type": "function",
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "parameters": func.get("parameters", {}),
                    }
                )
            else:
                converted.append(tool)
        return converted

    def _extract_usage(self, data: dict[str, Any], fmt: str = "responses") -> None:
        """Extract token usage from an API response dict."""
        usage = data.get("usage", {})
        if not usage:
            return
        if fmt == "anthropic":
            self.usage.add(
                input_t=usage.get("input_tokens", 0),
                output_t=usage.get("output_tokens", 0),
                cache_read=usage.get("cache_read_input_tokens", 0),
                cache_creation=usage.get("cache_creation_input_tokens", 0),
            )
        elif fmt == "chat_completions":
            self.usage.add(
                input_t=usage.get("prompt_tokens", 0),
                output_t=usage.get("completion_tokens", 0),
            )
        else:
            # Responses API
            self.usage.add(
                input_t=usage.get("input_tokens", 0),
                output_t=usage.get("output_tokens", 0),
            )

    def _convert_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert Responses API response → Chat Completions format."""
        self._extract_usage(data, "responses")
        output = data.get("output", [])

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for item in output:
            item_type = item.get("type")

            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_parts.append(part.get("text", ""))

            elif item_type == "function_call":
                tool_calls.append(
                    {
                        "id": item.get("call_id", item.get("id", "")),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    }
                )

        message: dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": data.get("id", ""),
            "choices": [{"message": message, "finish_reason": "stop"}],
        }

    # ── Chat Completions helpers ──

    def _build_cc_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Build a Chat Completions API payload.

        The system prompt is prepended as messages[0] only in the payload,
        not in the stored array.
        """
        payload_messages = list(messages)
        if system_prompt:
            payload_messages.insert(0, {"role": "system", "content": system_prompt})
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": payload_messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if tools:
            payload["tools"] = tools  # already in CC format
        if stream:
            payload["stream"] = True
            if self._stream_include_usage:
                payload["stream_options"] = {"include_usage": True}
        # Add reasoning_effort for CC providers (if supported)
        if self._should_send_thinking():
            effort = _CC_REASONING_EFFORT.get(self.config.thinking_level)
            if effort:
                payload["reasoning_effort"] = effort
        return payload

    def _convert_cc_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize a Chat Completions response (mostly passthrough)."""
        self._extract_usage(data, "chat_completions")
        return data

    # ── Anthropic Messages API helpers ──

    def _convert_messages_to_anthropic(self, messages: list[dict[str, Any]], system_prompt: str | None = None) -> tuple:
        """Convert Chat Completions messages → (system, anthropic_messages).

        Returns (system_prompt, messages) in Anthropic Messages API format.
        The system prompt is passed in separately — it is never extracted
        from the messages array.
        """
        anthropic_msgs: list[dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if role == "user":
                anthropic_msgs.append(
                    {
                        "role": "user",
                        "content": msg.get("content", ""),
                    }
                )

            elif role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": text})

                for tc in msg.get("tool_calls", []):
                    arguments = tc["function"]["arguments"]
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (json.JSONDecodeError, TypeError):
                            arguments = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": arguments,
                        }
                    )

                if content_blocks:
                    anthropic_msgs.append(
                        {
                            "role": "assistant",
                            "content": content_blocks,
                        }
                    )

            elif role == "tool":
                tool_results: list[dict[str, Any]] = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_msg.get("tool_call_id", ""),
                            "content": tool_msg.get("content", ""),
                        }
                    )
                    i += 1
                anthropic_msgs.append(
                    {
                        "role": "user",
                        "content": tool_results,
                    }
                )
                continue  # skip the i += 1 at end

            i += 1

        return system_prompt, anthropic_msgs

    def _convert_anthropic_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI tool defs → Anthropic tool format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted.append(
                    {
                        "name": func["name"],
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {"type": "object"}),
                    }
                )
            else:
                converted.append(tool)
        return converted

    def _convert_anthropic_response(self, data: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic Messages response → Chat Completions format.

        Thinking blocks (type=='thinking') are stripped from the output.
        """
        self._extract_usage(data, "anthropic")
        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "thinking":
                # Strip thinking blocks from the final response
                continue
            if block_type == "text":
                content_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    }
                )

        message: dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": data.get("id", ""),
            "choices": [
                {
                    "message": message,
                    "finish_reason": data.get("stop_reason", "stop"),
                }
            ],
        }

    # ── API calls ──

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat request using the configured API format."""
        if self.config.api_format == "anthropic":
            return await self._chat_anthropic(
                messages,
                tools,
                stream,
                system_prompt=system_prompt,
            )
        if self.config.api_format == "chat_completions":
            return await self._chat_completions(
                messages,
                tools,
                stream,
                system_prompt=system_prompt,
            )
        return await self._chat_responses(
            messages,
            tools,
            stream,
            system_prompt=system_prompt,
        )

    async def _chat_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Send a request to the Responses API."""
        if stream:
            full_content = ""
            tool_calls: dict[str, dict[str, Any]] = {}
            async for event in self._stream_responses_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                if event.type == "text_delta":
                    full_content += event.delta
                elif event.type == "toolcall_start":
                    cid = event.tool_call_id
                    tool_calls[cid] = {
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": event.tool_name,
                            "arguments": "",
                        },
                    }
                elif event.type == "toolcall_delta":
                    # Find the most recent tool call to append
                    for tc in reversed(list(tool_calls.values())):
                        tc["function"]["arguments"] += event.delta
                        break
                elif event.type == "usage":
                    self._extract_usage_from_event(event)

            message: dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = list(tool_calls.values())
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

        instructions, input_items = self._convert_messages_to_input(messages, system_prompt=system_prompt)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
            "max_output_tokens": self.config.max_tokens,
        }

        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = self._convert_tools(tools)

        # Add reasoning for Responses API providers (only if model supports it)
        if self._should_send_thinking():
            effort = _CC_REASONING_EFFORT.get(self.config.thinking_level)
            if effort:
                payload["reasoning"] = {"effort": effort}

        url = self.config.api_base.rstrip("/")
        return await self._post_with_retries(url, payload, self._convert_response)

    def _apply_prompt_caching(
        self,
        system_prompt: str | None,
        anthropic_msgs: list[dict[str, Any]],
    ) -> tuple:
        """Apply Anthropic prompt caching to system prompt and last tool result.

        Returns (system_value, anthropic_msgs) with cache_control annotations.
        Only modifies values when prompt_caching is enabled.
        """
        if not self.config.prompt_caching:
            return system_prompt, anthropic_msgs

        # Convert system prompt to array format with cache_control
        system_value: Any = system_prompt
        if system_prompt:
            system_value = [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # Add cache_control to the last user message that contains tool_result blocks
        for msg in reversed(anthropic_msgs):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            if any(b.get("type") == "tool_result" for b in content):
                content[-1]["cache_control"] = {"type": "ephemeral"}
                break

        return system_value, anthropic_msgs

    async def _chat_anthropic(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Send a request to the Anthropic Messages API."""
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages, system_prompt=system_prompt)
        system_value, anthropic_msgs = self._apply_prompt_caching(system_prompt, anthropic_msgs)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
        }

        if system_value:
            payload["system"] = system_value

        if self._should_send_thinking():
            thinking_level = self.config.thinking_level
            budget = _ANTHROPIC_THINKING_BUDGETS.get(thinking_level, 2048)
            model_max = _get_anthropic_model_max(self.config.model)
            adjusted_max, budget = _adjust_max_tokens_for_thinking(self.config.max_tokens, budget, model_max)
            payload["max_tokens"] = adjusted_max
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
            # Anthropic requires temperature to be omitted when thinking is enabled
        else:
            if self.config.temperature is not None:
                payload["temperature"] = self.config.temperature

        if tools:
            payload["tools"] = self._convert_anthropic_tools(tools)
        if stream:
            payload["stream"] = True

        url = self.config.api_base.rstrip("/")

        if stream:
            # Consume events directly — no separate _stream_anthropic method needed
            full_content = ""
            tool_calls: dict[int, dict[str, Any]] = {}

            async for event in self._stream_anthropic_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                if event.type == "text_delta":
                    full_content += event.delta
                elif event.type == "toolcall_start":
                    idx = event.content_index
                    tool_calls[idx] = {
                        "id": event.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": event.tool_name,
                            "arguments": "",
                        },
                    }
                elif event.type == "toolcall_delta":
                    idx = event.content_index
                    if idx in tool_calls:
                        tool_calls[idx]["function"]["arguments"] += event.delta
                elif event.type == "usage":
                    self._extract_usage_from_event(event)

            message: dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

            return {"choices": [{"message": message, "finish_reason": "stop"}]}

        return await self._post_with_retries(url, payload, self._convert_anthropic_response)

    async def _chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Send a request to a Chat Completions API endpoint."""
        if stream:
            full_content = ""
            tool_calls: dict[int, dict[str, Any]] = {}
            async for event in self._stream_cc_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                if event.type == "text_delta":
                    full_content += event.delta
                elif event.type == "toolcall_start":
                    tool_calls[event.content_index] = {
                        "id": event.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": event.tool_name,
                            "arguments": "",
                        },
                    }
                elif event.type == "toolcall_delta":
                    idx = event.content_index
                    if idx in tool_calls:
                        tool_calls[idx]["function"]["arguments"] += event.delta
                elif event.type == "usage":
                    self._extract_usage_from_event(event)

            message: dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
            return {"choices": [{"message": message, "finish_reason": "stop"}]}

        payload = self._build_cc_payload(
            messages,
            tools,
            stream,
            system_prompt=system_prompt,
        )
        url = self.config.api_base.rstrip("/")
        return await self._post_with_retries(url, payload, self._convert_cc_response)

    def _strip_thinking_params(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a copy of payload with all thinking/reasoning params removed."""
        cleaned = dict(payload)
        cleaned.pop("thinking", None)
        cleaned.pop("reasoning", None)
        cleaned.pop("reasoning_effort", None)
        return cleaned

    async def _post_with_retries(
        self,
        url: str,
        payload: dict[str, Any],
        convert_fn,
    ) -> dict[str, Any]:
        """POST with exponential-backoff retries, then convert the response."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                return convert_fn(response.json())

            except httpx.HTTPStatusError as e:
                error_text = getattr(e.response, "text", "")

                # Detect thinking-related API errors and retry without thinking
                if (
                    not self._thinking_disabled_fallback
                    and _THINKING_ERROR_PATTERNS.search(error_text)
                    and e.response.status_code not in self.RETRYABLE_STATUS_CODES
                ):
                    self._thinking_disabled_fallback = True
                    payload = self._strip_thinking_params(payload)
                    # Remove anthropic-beta header if present
                    self.headers.pop("anthropic-beta", None)
                    continue  # retry immediately with cleaned payload

                # Extract retry delay from headers + body
                headers_dict = dict(e.response.headers)
                retry_delay = extract_retry_delay(error_text, headers_dict)

                # Classify the error
                classified = classify_http_error(e.response.status_code, error_text, retry_after=retry_delay)
                last_error = classified

                if not classified.retryable:
                    raise classified

                # Check max retry delay cap
                if retry_delay and retry_delay > self.MAX_RETRY_DELAY:
                    raise KoiRateLimitError(
                        f"Rate limited — server wants {retry_delay:.0f}s wait (max: {self.MAX_RETRY_DELAY}s)",
                        status_code=e.response.status_code,
                        error_text=error_text,
                        retry_after=retry_delay,
                    )

                delay = retry_delay if retry_delay else min(2 ** (attempt + 1), self.MAX_BACKOFF)
                await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = KoiConnectionError(f"Connection error: {e}")
                delay = min(2 ** (attempt + 1), self.MAX_BACKOFF)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                raise

            except KoiAPIError:
                raise  # Already classified, don't wrap

            except Exception as e:
                raise KoiAPIError(f"Request failed: {e}", retryable=False)

        # Exhausted retries
        if last_error:
            raise last_error
        raise KoiAPIError("Request failed after retries", retryable=False)

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvent objects from a streaming LLM response.

        The consumer (agent.py) is responsible for accumulating text,
        tool calls, and assembling the final response dict.
        """
        if self.config.api_format == "anthropic":
            async for event in self._stream_anthropic_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                yield event
        elif self.config.api_format == "chat_completions":
            async for event in self._stream_cc_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                yield event
        else:
            async for event in self._stream_responses_events(
                messages,
                tools,
                system_prompt=system_prompt,
            ):
                yield event

    async def _stream_responses_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvent objects from a Responses API streaming response.

        This is the unified event adapter for the Responses API path.
        It handles SSE parsing and yields structured events; the consumer
        handles accumulation and response assembly.
        """
        instructions, input_items = self._convert_messages_to_input(messages, system_prompt=system_prompt)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": input_items,
            "max_output_tokens": self.config.max_tokens,
            "stream": True,
        }

        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = self._convert_tools(tools)

        # Add reasoning for Responses API streaming (only if model supports it)
        if self._should_send_thinking():
            effort = _CC_REASONING_EFFORT.get(self.config.thinking_level)
            if effort:
                payload["reasoning"] = {"effort": effort}

        url = self.config.api_base.rstrip("/")

        content_len = 0
        tool_args_len = 0
        seen_call_ids: set = set()
        completed = False

        try:
            async with self.client.stream("POST", url, headers=self.headers, json=payload) as response:
                self._active_stream_response = response
                try:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip() or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type", "")

                        if event_type == "response.output_text.delta":
                            delta = data.get("delta", "")
                            content_len += len(delta)
                            yield StreamEvent(type="text_delta", delta=delta)

                        elif event_type == "response.function_call_arguments.delta":
                            cid = data.get("call_id", data.get("item_id", ""))
                            if cid not in seen_call_ids:
                                seen_call_ids.add(cid)
                                yield StreamEvent(
                                    type="toolcall_start",
                                    tool_call_id=cid,
                                )
                            arg_delta = data.get("delta", "")
                            tool_args_len += len(arg_delta)
                            yield StreamEvent(type="toolcall_delta", delta=arg_delta)

                        elif event_type in (
                            "response.function_call.name",
                            "response.output_item.added",
                        ):
                            item = data.get("item", data)
                            if item.get("type") == "function_call":
                                cid = item.get("call_id", item.get("id", ""))
                                name = item.get("name", "")
                                if cid not in seen_call_ids:
                                    seen_call_ids.add(cid)
                                    yield StreamEvent(
                                        type="toolcall_start",
                                        tool_call_id=cid,
                                        tool_name=name,
                                    )

                        elif event_type == "response.completed":
                            resp = data.get("response", {})
                            if resp:
                                usage = resp.get("usage", {})
                                if usage:
                                    yield StreamEvent(
                                        type="usage",
                                        usage={
                                            "input_tokens": usage.get("input_tokens", 0),
                                            "output_tokens": usage.get("output_tokens", 0),
                                        },
                                    )
                                completed = True
                finally:
                    self._active_stream_response = None

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            error_text = getattr(e.response, "text", "")
            raise classify_http_error(e.response.status_code, error_text)
        except KoiAPIError:
            raise
        except Exception as e:
            raise KoiAPIError(f"Stream request failed: {e}", retryable=False)

        # Fallback usage estimation if no response.completed
        if not completed and (content_len or tool_args_len):
            estimated = content_len // 4 + tool_args_len // 4
            yield StreamEvent(
                type="usage",
                usage={
                    "input_tokens": 0,
                    "output_tokens": max(estimated, 1),
                },
            )

        yield StreamEvent(type="done")

    async def _stream_cc_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvent objects from a Chat Completions streaming response.

        This is the unified event adapter for the Chat Completions path.
        It handles SSE parsing, stream_options negotiation, and yields
        structured events; the consumer handles accumulation and response assembly.
        """
        payload = self._build_cc_payload(
            messages,
            tools,
            stream=True,
            system_prompt=system_prompt,
        )
        url = self.config.api_base.rstrip("/")

        # stream_options negotiation: try with include_usage; if 400, disable and retry
        if self._stream_include_usage and payload.get("stream_options"):
            try:
                async with self.client.stream("POST", url, headers=self.headers, json=payload) as response:
                    self._active_stream_response = response
                    try:
                        if response.status_code == 400:
                            await response.aread()
                            self._stream_include_usage = False
                            payload.pop("stream_options", None)
                        else:
                            if response.is_error:
                                await response.aread()
                            response.raise_for_status()
                            async for event in self._parse_cc_sse(response):
                                yield event
                            return
                    finally:
                        self._active_stream_response = None
            except httpx.HTTPStatusError:
                self._stream_include_usage = False
                payload.pop("stream_options", None)

        # Main attempt (or retry without stream_options)
        content_len = 0
        tool_args_len = 0
        usage_found = False
        seen_indices: set = set()

        try:
            async with self.client.stream("POST", url, headers=self.headers, json=payload) as response:
                self._active_stream_response = response
                try:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip() or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Usage from final chunk
                        if "usage" in data:
                            usage = data["usage"]
                            yield StreamEvent(
                                type="usage",
                                usage={
                                    "prompt_tokens": usage.get("prompt_tokens", 0),
                                    "completion_tokens": usage.get("completion_tokens", 0),
                                },
                            )
                            usage_found = True

                        choices = data.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})

                        if "content" in delta and delta["content"]:
                            content_len += len(delta["content"])
                            yield StreamEvent(type="text_delta", delta=delta["content"])

                        for tc_delta in delta.get("tool_calls", []):
                            idx = tc_delta.get("index", 0)
                            is_new = idx not in seen_indices
                            if is_new:
                                seen_indices.add(idx)
                                yield StreamEvent(
                                    type="toolcall_start",
                                    content_index=idx,
                                    tool_call_id=tc_delta.get("id", ""),
                                    tool_name=tc_delta.get("function", {}).get("name", ""),
                                )
                            func = tc_delta.get("function", {})
                            if func.get("arguments"):
                                tool_args_len += len(func["arguments"])
                                yield StreamEvent(
                                    type="toolcall_delta",
                                    content_index=idx,
                                    delta=func["arguments"],
                                )
                finally:
                    self._active_stream_response = None

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            error_text = getattr(e.response, "text", "")
            raise classify_http_error(e.response.status_code, error_text)
        except KoiAPIError:
            raise
        except Exception as e:
            raise KoiAPIError(f"Stream request failed: {e}", retryable=False)

        # Fallback usage estimation
        if not usage_found and (content_len or tool_args_len):
            estimated = content_len // 4 + tool_args_len // 4
            yield StreamEvent(
                type="usage",
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": max(estimated, 1),
                },
            )

        yield StreamEvent(type="done")

    async def _parse_cc_sse(self, response: httpx.Response) -> AsyncGenerator[StreamEvent, None]:
        """Parse CC SSE lines from an already-opened response."""
        content_len = 0
        tool_args_len = 0
        usage_found = False
        seen_indices: set = set()

        async for line in response.aiter_lines():
            if not line.strip() or not line.startswith("data: "):
                continue

            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break

            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            if "usage" in data:
                usage = data["usage"]
                yield StreamEvent(
                    type="usage",
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                    },
                )
                usage_found = True

            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})

            if "content" in delta and delta["content"]:
                content_len += len(delta["content"])
                yield StreamEvent(type="text_delta", delta=delta["content"])

            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                is_new = idx not in seen_indices
                if is_new:
                    seen_indices.add(idx)
                    yield StreamEvent(
                        type="toolcall_start",
                        content_index=idx,
                        tool_call_id=tc_delta.get("id", ""),
                        tool_name=tc_delta.get("function", {}).get("name", ""),
                    )
                func = tc_delta.get("function", {})
                if func.get("arguments"):
                    tool_args_len += len(func["arguments"])
                    yield StreamEvent(
                        type="toolcall_delta",
                        content_index=idx,
                        delta=func["arguments"],
                    )

        if not usage_found and (content_len or tool_args_len):
            estimated = content_len // 4 + tool_args_len // 4
            yield StreamEvent(
                type="usage",
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": max(estimated, 1),
                },
            )

        yield StreamEvent(type="done")

    async def _stream_anthropic_events(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Yield StreamEvent objects from an Anthropic Messages streaming response.

        This is the unified event adapter for the Anthropic path. It handles
        SSE parsing and yields structured events; the consumer handles
        accumulation and response assembly.
        """
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages, system_prompt=system_prompt)
        system_value, anthropic_msgs = self._apply_prompt_caching(system_prompt, anthropic_msgs)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        if system_value:
            payload["system"] = system_value

        if self._should_send_thinking():
            thinking_level = self.config.thinking_level
            budget = _ANTHROPIC_THINKING_BUDGETS.get(thinking_level, 2048)
            model_max = _get_anthropic_model_max(self.config.model)
            adjusted_max, budget = _adjust_max_tokens_for_thinking(self.config.max_tokens, budget, model_max)
            payload["max_tokens"] = adjusted_max
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        else:
            if self.config.temperature is not None:
                payload["temperature"] = self.config.temperature

        if tools:
            payload["tools"] = self._convert_anthropic_tools(tools)

        url = self.config.api_base.rstrip("/")

        # Track accumulated content per block for *_end events
        block_content: dict[int, str] = {}  # index → accumulated text/thinking
        block_args: dict[int, str] = {}  # index → accumulated tool arguments
        # index -> "text" | "thinking" | "tool_use"
        block_types: dict[int, str] = {}
        block_meta: dict[int, dict] = {}  # index → {name, id}

        try:
            async with self.client.stream("POST", url, headers=self.headers, json=payload) as response:
                self._active_stream_response = response
                try:
                    if response.is_error:
                        await response.aread()
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.strip() or not line.startswith("data: "):
                            continue

                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        event_type = data.get("type", "")

                        if event_type == "message_start":
                            msg = data.get("message", {})
                            usage = msg.get("usage")
                            if usage:
                                yield StreamEvent(type="usage", usage=usage)

                        elif event_type == "message_delta":
                            usage = data.get("usage")
                            if usage:
                                yield StreamEvent(type="usage", usage=usage)

                        elif event_type == "content_block_start":
                            idx = data.get("index", 0)
                            block = data.get("content_block", {})
                            btype = block.get("type")
                            block_types[idx] = btype
                            block_content[idx] = ""
                            if btype == "text":
                                yield StreamEvent(type="text_start", content_index=idx)
                            elif btype == "thinking":
                                yield StreamEvent(
                                    type="thinking_start",
                                    content_index=idx,
                                )
                            elif btype == "tool_use":
                                block_args[idx] = ""
                                block_meta[idx] = {
                                    "name": block.get("name", ""),
                                    "id": block.get("id", ""),
                                }
                                yield StreamEvent(
                                    type="toolcall_start",
                                    content_index=idx,
                                    tool_name=block.get("name", ""),
                                    tool_call_id=block.get("id", ""),
                                )

                        elif event_type == "content_block_delta":
                            idx = data.get("index", 0)
                            delta = data.get("delta", {})
                            delta_type = delta.get("type")
                            if delta_type == "text_delta":
                                text = delta.get("text", "")
                                block_content[idx] = block_content.get(idx, "") + text
                                yield StreamEvent(
                                    type="text_delta",
                                    content_index=idx,
                                    delta=text,
                                )
                            elif delta_type == "thinking_delta":
                                thinking = delta.get("thinking", "")
                                block_content[idx] = block_content.get(idx, "") + thinking
                                yield StreamEvent(
                                    type="thinking_delta",
                                    content_index=idx,
                                    delta=thinking,
                                )
                            elif delta_type == "input_json_delta":
                                partial = delta.get("partial_json", "")
                                block_args[idx] = block_args.get(idx, "") + partial
                                yield StreamEvent(
                                    type="toolcall_delta",
                                    content_index=idx,
                                    delta=partial,
                                )

                        elif event_type == "content_block_stop":
                            idx = data.get("index", 0)
                            btype = block_types.get(idx)
                            if btype == "text":
                                yield StreamEvent(
                                    type="text_end",
                                    content_index=idx,
                                    content=block_content.get(idx, ""),
                                )
                            elif btype == "thinking":
                                yield StreamEvent(
                                    type="thinking_end",
                                    content_index=idx,
                                    content=block_content.get(idx, ""),
                                )
                            elif btype == "tool_use":
                                meta = block_meta.get(idx, {})
                                yield StreamEvent(
                                    type="toolcall_end",
                                    content_index=idx,
                                    tool_name=meta.get("name", ""),
                                    tool_call_id=meta.get("id", ""),
                                    arguments=block_args.get(idx, ""),
                                )

                        elif event_type == "message_stop":
                            yield StreamEvent(type="done", finish_reason="stop")
                            break
                finally:
                    self._active_stream_response = None

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            error_text = getattr(e.response, "text", "")
            raise classify_http_error(e.response.status_code, error_text)
        except KoiAPIError:
            raise
        except Exception as e:
            raise KoiAPIError(f"Stream request failed: {e}", retryable=False)

    def _extract_usage_from_event(self, event: StreamEvent) -> None:
        """Extract usage from a StreamEvent's usage dict (all formats)."""
        if not event.usage:
            return
        u = event.usage
        self.usage.add(
            input_t=u.get("input_tokens", 0) or u.get("prompt_tokens", 0),
            output_t=u.get("output_tokens", 0) or u.get("completion_tokens", 0),
            cache_read=u.get("cache_read_input_tokens", 0),
            cache_creation=u.get("cache_creation_input_tokens", 0),
        )

    def abort_stream(self):
        """Abort any in-flight streaming response.

        Called from the SIGINT handler (sync context) to unblock httpx
        immediately rather than waiting for CancelledError to propagate.
        """
        resp = self._active_stream_response
        if resp is not None:
            try:
                resp.close()  # sync close — unblocks the async iterator
            except Exception:
                pass
            self._active_stream_response = None

    async def close(self):
        await self.client.aclose()
