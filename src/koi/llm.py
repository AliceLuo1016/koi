"""LLM client for OpenAI-compatible Responses API."""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

from .config import Config
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


def _adjust_max_tokens_for_thinking(
    base_max: int, budget: int, model_max: int = 64000
) -> tuple:
    """Adjust max_tokens so it covers both thinking budget and output.

    Returns (max_tokens, budget) where max_tokens = min(base_max + budget, model_max).
    If the resulting max_tokens would leave no room for output (i.e. max_tokens <= budget),
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

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
        self._thinking_disabled_fallback = False
        self._last_stream_response: Optional[Dict[str, Any]] = None
        self.usage = TokenUsage()
        self.use_reasoning_tags = uses_reasoning_tags(
            config.model, config.api_format, config.thinking_level
        )
        if config.api_format == "anthropic":
            self.headers = {
                "Content-Type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            }
            if config.thinking_level != "off" and supports_thinking(
                config.model, config.api_format
            ):
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

    def _convert_messages_to_input(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> tuple:
        """Convert Chat Completions messages → (instructions, input).

        Returns (instructions, input_items) where *instructions* is the
        system prompt and *input_items* is the conversation history in
        Responses API format.

        The system prompt is passed in separately — it is never extracted
        from the messages array.
        """
        input_items: List[Dict[str, Any]] = []

        # Inject system prompt as a developer message so proxies that
        # ignore the 'instructions' field still see it.
        if system_prompt:
            input_items.append({
                "role": "developer",
                "content": system_prompt,
            })

        for msg in messages:
            role = msg.get("role")

            if role == "user":
                input_items.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            elif role == "assistant":
                content = msg.get("content")
                if content:
                    input_items.append({
                        "role": "assistant",
                        "content": content,
                    })

                # Tool calls → function_call items
                for tc in msg.get("tool_calls", []):
                    input_items.append({
                        "type": "function_call",
                        "call_id": tc["id"],
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    })

            elif role == "tool":
                input_items.append({
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": msg.get("content", ""),
                })

        return system_prompt, input_items

    def _convert_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Flatten Chat Completions tool defs → Responses API format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted.append({
                    "type": "function",
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "parameters": func.get("parameters", {}),
                })
            else:
                converted.append(tool)
        return converted

    def _extract_usage(self, data: Dict[str, Any], fmt: str = "responses") -> None:
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

    def _convert_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Responses API response → Chat Completions format."""
        self._extract_usage(data, "responses")
        output = data.get("output", [])

        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for item in output:
            item_type = item.get("type")

            if item_type == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        content_parts.append(part.get("text", ""))

            elif item_type == "function_call":
                tool_calls.append({
                    "id": item.get("call_id", item.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", "{}"),
                    },
                })

        message: Dict[str, Any] = {"role": "assistant"}
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
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Chat Completions API payload.

        The system prompt is prepended as messages[0] only in the payload,
        not in the stored array.
        """
        payload_messages = list(messages)
        if system_prompt:
            payload_messages.insert(0, {"role": "system", "content": system_prompt})
        payload: Dict[str, Any] = {
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
        # Add reasoning_effort for Chat Completions providers (only if model supports it)
        if self._should_send_thinking():
            effort = _CC_REASONING_EFFORT.get(self.config.thinking_level)
            if effort:
                payload["reasoning_effort"] = effort
        return payload

    def _convert_cc_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Chat Completions response (mostly passthrough)."""
        self._extract_usage(data, "chat_completions")
        return data

    # ── Anthropic Messages API helpers ──

    def _convert_messages_to_anthropic(
        self, messages: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> tuple:
        """Convert Chat Completions messages → (system, anthropic_messages).

        Returns (system_prompt, messages) in Anthropic Messages API format.
        The system prompt is passed in separately — it is never extracted
        from the messages array.
        """
        anthropic_msgs: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if role == "user":
                anthropic_msgs.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            elif role == "assistant":
                content_blocks: List[Dict[str, Any]] = []
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
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": arguments,
                    })

                if content_blocks:
                    anthropic_msgs.append({
                        "role": "assistant",
                        "content": content_blocks,
                    })

            elif role == "tool":
                tool_results: List[Dict[str, Any]] = []
                while i < len(messages) and messages[i].get("role") == "tool":
                    tool_msg = messages[i]
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_msg.get("tool_call_id", ""),
                        "content": tool_msg.get("content", ""),
                    })
                    i += 1
                anthropic_msgs.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue  # skip the i += 1 at end

            i += 1

        return system_prompt, anthropic_msgs

    def _convert_anthropic_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert OpenAI tool defs → Anthropic tool format."""
        converted = []
        for tool in tools:
            if tool.get("type") == "function" and "function" in tool:
                func = tool["function"]
                converted.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object"}),
                })
            else:
                converted.append(tool)
        return converted

    def _convert_anthropic_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Anthropic Messages response → Chat Completions format.

        Thinking blocks (type=='thinking') are stripped from the output.
        """
        self._extract_usage(data, "anthropic")
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in data.get("content", []):
            block_type = block.get("type")
            if block_type == "thinking":
                # Strip thinking blocks from the final response
                continue
            if block_type == "text":
                content_parts.append(block.get("text", ""))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                })

        message: Dict[str, Any] = {"role": "assistant"}
        if content_parts:
            message["content"] = "".join(content_parts)
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "id": data.get("id", ""),
            "choices": [{"message": message, "finish_reason": data.get("stop_reason", "stop")}],
        }

    # ── API calls ──

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a chat request using the configured API format."""
        if self.config.api_format == "anthropic":
            return await self._chat_anthropic(messages, tools, stream, system_prompt=system_prompt)
        if self.config.api_format == "chat_completions":
            return await self._chat_completions(messages, tools, stream, system_prompt=system_prompt)
        return await self._chat_responses(messages, tools, stream, system_prompt=system_prompt)

    async def _chat_responses(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a request to the Responses API."""
        instructions, input_items = self._convert_messages_to_input(messages, system_prompt=system_prompt)

        payload: Dict[str, Any] = {
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
        if stream:
            payload["stream"] = True

        # Add reasoning for Responses API providers (only if model supports it)
        if self._should_send_thinking():
            effort = _CC_REASONING_EFFORT.get(self.config.thinking_level)
            if effort:
                payload["reasoning"] = {"effort": effort}

        url = self.config.api_base.rstrip("/")

        if stream:
            return await self._stream_chat(url, payload)

        return await self._post_with_retries(url, payload, self._convert_response)

    def _apply_prompt_caching(
        self,
        system_prompt: Optional[str],
        anthropic_msgs: List[Dict[str, Any]],
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
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a request to the Anthropic Messages API."""
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages, system_prompt=system_prompt)
        system_value, anthropic_msgs = self._apply_prompt_caching(
            system_prompt, anthropic_msgs
        )

        payload: Dict[str, Any] = {
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
            adjusted_max, budget = _adjust_max_tokens_for_thinking(
                self.config.max_tokens, budget, model_max
            )
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
            return await self._stream_anthropic(url, payload)

        return await self._post_with_retries(url, payload, self._convert_anthropic_response)

    async def _chat_completions(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a request to a Chat Completions API endpoint."""
        payload = self._build_cc_payload(messages, tools, stream, system_prompt=system_prompt)
        url = self.config.api_base.rstrip("/")

        if stream:
            return await self._stream_chat_completions(url, payload)

        return await self._post_with_retries(url, payload, self._convert_cc_response)

    def _strip_thinking_params(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a copy of payload with all thinking/reasoning params removed."""
        cleaned = dict(payload)
        cleaned.pop("thinking", None)
        cleaned.pop("reasoning", None)
        cleaned.pop("reasoning_effort", None)
        return cleaned

    async def _post_with_retries(
        self,
        url: str,
        payload: Dict[str, Any],
        convert_fn,
    ) -> Dict[str, Any]:
        """POST with exponential-backoff retries, then convert the response."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.post(
                    url, headers=self.headers, json=payload
                )
                response.raise_for_status()
                return convert_fn(response.json())

            except httpx.HTTPStatusError as e:
                last_error = e
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

                if e.response.status_code not in self.RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"HTTP {e.response.status_code}: {e.response.text}"
                    )
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = min(float(retry_after), self.MAX_BACKOFF)
                    except ValueError:
                        delay = min(2 ** (attempt + 1), self.MAX_BACKOFF)
                else:
                    delay = min(2 ** (attempt + 1), self.MAX_BACKOFF)
                await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                delay = min(2 ** (attempt + 1), self.MAX_BACKOFF)
                await asyncio.sleep(delay)

            except asyncio.CancelledError:
                raise

            except Exception as e:
                raise RuntimeError(f"Request failed: {e}")

        if isinstance(last_error, httpx.HTTPStatusError):
            raise RuntimeError(
                f"HTTP {last_error.response.status_code} after {self.MAX_RETRIES} retries: {last_error.response.text}"
            )
        raise RuntimeError(f"Request failed after {self.MAX_RETRIES} retries: {last_error}")

    async def _stream_chat(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle streaming response and return assembled result."""
        async with self.client.stream(
            "POST", url, headers=self.headers, json=payload
        ) as response:
            response.raise_for_status()

            full_content = ""
            tool_calls: Dict[str, Dict[str, Any]] = {}

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

                # Text deltas
                if event_type == "response.output_text.delta":
                    full_content += data.get("delta", "")

                # Tool-call argument deltas
                elif event_type == "response.function_call_arguments.delta":
                    cid = data.get("call_id", data.get("item_id", ""))
                    if cid not in tool_calls:
                        tool_calls[cid] = {
                            "id": cid,
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    tool_calls[cid]["function"]["arguments"] += data.get(
                        "delta", ""
                    )

                # Tool-call name
                elif event_type in (
                    "response.function_call.name",
                    "response.output_item.added",
                ):
                    item = data.get("item", data)
                    if item.get("type") == "function_call":
                        cid = item.get("call_id", item.get("id", ""))
                        if cid not in tool_calls:
                            tool_calls[cid] = {
                                "id": cid,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tool_calls[cid]["function"]["name"] = item.get(
                            "name", ""
                        )

                # Final complete response — parse directly
                elif event_type == "response.completed":
                    resp = data.get("response", {})
                    if resp:
                        return self._convert_response(resp)

            # Assemble from accumulated deltas
            message: Dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = list(tool_calls.values())

            return {
                "choices": [{"message": message, "finish_reason": "stop"}]
            }

    async def _stream_chat_completions(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Chat Completions streaming and return assembled result."""
        async with self.client.stream(
            "POST", url, headers=self.headers, json=payload
        ) as response:
            response.raise_for_status()

            full_content = ""
            tool_calls: Dict[int, Dict[str, Any]] = {}

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

                # Extract usage from final chunk
                if "usage" in data:
                    self._extract_usage(data, "chat_completions")

                choices = data.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                # Text content
                if "content" in delta and delta["content"]:
                    full_content += delta["content"]

                # Tool calls
                for tc_delta in delta.get("tool_calls", []):
                    idx = tc_delta.get("index", 0)
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tc_delta.get("id", ""),
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc_delta.get("id"):
                        tool_calls[idx]["id"] = tc_delta["id"]
                    func = tc_delta.get("function", {})
                    if func.get("name"):
                        tool_calls[idx]["function"]["name"] = func["name"]
                    if func.get("arguments"):
                        tool_calls[idx]["function"]["arguments"] += func["arguments"]

            message: Dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = [
                    tool_calls[i] for i in sorted(tool_calls)
                ]

            return {
                "choices": [{"message": message, "finish_reason": "stop"}]
            }

    async def _stream_anthropic(
        self, url: str, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle Anthropic streaming and return assembled result."""
        async with self.client.stream(
            "POST", url, headers=self.headers, json=payload
        ) as response:
            response.raise_for_status()

            full_content = ""
            tool_calls: Dict[int, Dict[str, Any]] = {}
            current_block_idx = -1
            current_block_type = None

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
                    if "usage" in msg:
                        self._extract_usage(msg, "anthropic")

                elif event_type == "message_delta":
                    if "usage" in data:
                        self._extract_usage(data, "anthropic")

                elif event_type == "content_block_start":
                    current_block_idx = data.get("index", 0)
                    block = data.get("content_block", {})
                    current_block_type = block.get("type")
                    if current_block_type == "tool_use":
                        tool_calls[current_block_idx] = {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": "",
                            },
                        }

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    delta_type = delta.get("type")
                    # Skip thinking deltas
                    if delta_type == "thinking_delta":
                        continue
                    if delta_type == "text_delta":
                        full_content += delta.get("text", "")
                    elif delta_type == "input_json_delta":
                        idx = data.get("index", current_block_idx)
                        if idx in tool_calls:
                            tool_calls[idx]["function"]["arguments"] += delta.get(
                                "partial_json", ""
                            )

                elif event_type == "message_stop":
                    break

            message: Dict[str, Any] = {"role": "assistant"}
            if full_content:
                message["content"] = full_content
            if tool_calls:
                message["tool_calls"] = [
                    tool_calls[i] for i in sorted(tool_calls)
                ]

            return {
                "choices": [{"message": message, "finish_reason": "stop"}]
            }

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text content token-by-token for live display.

        After iteration completes, the full assembled response (including
        any tool_calls) is available via ``self._last_stream_response``.
        """
        self._last_stream_response = None

        if self.config.api_format == "anthropic":
            async for token in self._stream_anthropic_tokens(messages, tools, system_prompt=system_prompt):
                yield token
            return

        if self.config.api_format == "chat_completions":
            async for token in self._stream_chat_completions_tokens(
                messages, tools, system_prompt=system_prompt
            ):
                yield token
            return

        instructions, input_items = self._convert_messages_to_input(messages, system_prompt=system_prompt)

        payload: Dict[str, Any] = {
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

        full_content = ""
        tool_calls: Dict[str, Dict[str, Any]] = {}

        try:
            async with self.client.stream(
                "POST", url, headers=self.headers, json=payload
            ) as response:
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
                        full_content += delta
                        yield delta

                    elif event_type == "response.function_call_arguments.delta":
                        cid = data.get("call_id", data.get("item_id", ""))
                        if cid not in tool_calls:
                            tool_calls[cid] = {
                                "id": cid,
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        tool_calls[cid]["function"]["arguments"] += data.get(
                            "delta", ""
                        )

                    elif event_type in (
                        "response.function_call.name",
                        "response.output_item.added",
                    ):
                        item = data.get("item", data)
                        if item.get("type") == "function_call":
                            cid = item.get("call_id", item.get("id", ""))
                            if cid not in tool_calls:
                                tool_calls[cid] = {
                                    "id": cid,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            tool_calls[cid]["function"]["name"] = item.get(
                                "name", ""
                            )

                    elif event_type == "response.completed":
                        resp = data.get("response", {})
                        if resp:
                            self._last_stream_response = self._convert_response(resp)
                            return

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

        # Assemble response from accumulated deltas
        message: Dict[str, Any] = {"role": "assistant"}
        if full_content:
            message["content"] = full_content
        if tool_calls:
            message["tool_calls"] = list(tool_calls.values())

        self._last_stream_response = {
            "choices": [{"message": message, "finish_reason": "stop"}]
        }

    async def _stream_chat_completions_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text tokens from a Chat Completions streaming response.

        Also accumulates tool calls and stores the full assembled response
        in ``self._last_stream_response`` after the stream ends.
        """
        payload = self._build_cc_payload(messages, tools, stream=True, system_prompt=system_prompt)
        url = self.config.api_base.rstrip("/")

        full_content = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}

        try:
            async with self.client.stream(
                "POST", url, headers=self.headers, json=payload
            ) as response:
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

                    # Extract usage from final chunk
                    if "usage" in data:
                        self._extract_usage(data, "chat_completions")

                    choices = data.get("choices", [])
                    if not choices:
                        continue
                    delta = choices[0].get("delta", {})

                    if "content" in delta and delta["content"]:
                        full_content += delta["content"]
                        yield delta["content"]

                    for tc_delta in delta.get("tool_calls", []):
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_calls:
                            tool_calls[idx] = {
                                "id": tc_delta.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        if tc_delta.get("id"):
                            tool_calls[idx]["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            tool_calls[idx]["function"]["name"] = func["name"]
                        if func.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += func[
                                "arguments"
                            ]

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

        message: Dict[str, Any] = {"role": "assistant"}
        if full_content:
            message["content"] = full_content
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        self._last_stream_response = {
            "choices": [{"message": message, "finish_reason": "stop"}]
        }

    async def _stream_anthropic_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text tokens from an Anthropic Messages streaming response.

        Also accumulates tool calls and stores the full assembled response
        in ``self._last_stream_response`` after the stream ends.
        """
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages, system_prompt=system_prompt)
        system_value, anthropic_msgs = self._apply_prompt_caching(
            system_prompt, anthropic_msgs
        )

        payload: Dict[str, Any] = {
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
            adjusted_max, budget = _adjust_max_tokens_for_thinking(
                self.config.max_tokens, budget, model_max
            )
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

        full_content = ""
        tool_calls: Dict[int, Dict[str, Any]] = {}
        current_block_idx = -1

        try:
            async with self.client.stream(
                "POST", url, headers=self.headers, json=payload
            ) as response:
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
                        if "usage" in msg:
                            self._extract_usage(msg, "anthropic")

                    elif event_type == "message_delta":
                        if "usage" in data:
                            self._extract_usage(data, "anthropic")

                    elif event_type == "content_block_start":
                        current_block_idx = data.get("index", 0)
                        block = data.get("content_block", {})
                        if block.get("type") == "tool_use":
                            tool_calls[current_block_idx] = {
                                "id": block.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": block.get("name", ""),
                                    "arguments": "",
                                },
                            }

                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        delta_type = delta.get("type")
                        if delta_type == "thinking_delta":
                            continue
                        if delta_type == "text_delta":
                            text = delta.get("text", "")
                            full_content += text
                            yield text
                        elif delta_type == "input_json_delta":
                            idx = data.get("index", current_block_idx)
                            if idx in tool_calls:
                                tool_calls[idx]["function"][
                                    "arguments"
                                ] += delta.get("partial_json", "")

                    elif event_type == "message_stop":
                        break

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

        message: Dict[str, Any] = {"role": "assistant"}
        if full_content:
            message["content"] = full_content
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]

        self._last_stream_response = {
            "choices": [{"message": message, "finish_reason": "stop"}]
        }

    async def close(self):
        await self.client.aclose()
