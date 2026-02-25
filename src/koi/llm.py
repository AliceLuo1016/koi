"""LLM client for OpenAI-compatible Responses API."""

import asyncio
import json
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

from .config import Config


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
        if config.api_format == "anthropic":
            self.headers = {
                "Content-Type": "application/json",
                "x-api-key": config.api_key,
                "anthropic-version": "2023-06-01",
            }
        else:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_key}",
            }

    # ── Format conversion helpers ──

    def _convert_messages_to_input(
        self, messages: List[Dict[str, Any]]
    ) -> tuple:
        """Convert Chat Completions messages → (instructions, input).

        Returns (instructions, input_items) where *instructions* is the
        system prompt and *input_items* is the conversation history in
        Responses API format.
        """
        instructions = None
        input_items: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                instructions = msg.get("content", "")
                # Also inject as a user message so proxies that
                # ignore the 'instructions' field still see it.
                input_items.insert(0, {
                    "role": "developer",
                    "content": msg.get("content", ""),
                })

            elif role == "user":
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

        return instructions, input_items

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

    def _convert_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Responses API response → Chat Completions format."""
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
    ) -> Dict[str, Any]:
        """Build a Chat Completions API payload (messages passed through)."""
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if tools:
            payload["tools"] = tools  # already in CC format
        if stream:
            payload["stream"] = True
        return payload

    def _convert_cc_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Chat Completions response (mostly passthrough)."""
        # Already in CC format — just ensure the structure we expect
        return data

    # ── Anthropic Messages API helpers ──

    def _convert_messages_to_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> tuple:
        """Convert Chat Completions messages → (system, anthropic_messages).

        Returns (system_prompt, messages) in Anthropic Messages API format.
        Handles role alternation and tool result grouping.
        """
        system_prompt = None
        anthropic_msgs: List[Dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role")

            if role == "system":
                system_prompt = msg.get("content", "")

            elif role == "user":
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
        """Convert Anthropic Messages response → Chat Completions format."""
        content_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        for block in data.get("content", []):
            block_type = block.get("type")
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
    ) -> Dict[str, Any]:
        """Send a chat request using the configured API format."""
        if self.config.api_format == "anthropic":
            return await self._chat_anthropic(messages, tools, stream)
        if self.config.api_format == "chat_completions":
            return await self._chat_completions(messages, tools, stream)
        return await self._chat_responses(messages, tools, stream)

    async def _chat_responses(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a request to the Responses API."""
        instructions, input_items = self._convert_messages_to_input(messages)

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

        url = self.config.api_base.rstrip("/")

        if stream:
            return await self._stream_chat(url, payload)

        return await self._post_with_retries(url, payload, self._convert_response)

    async def _chat_anthropic(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a request to the Anthropic Messages API."""
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt
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
    ) -> Dict[str, Any]:
        """Send a request to a Chat Completions API endpoint."""
        payload = self._build_cc_payload(messages, tools, stream)
        url = self.config.api_base.rstrip("/")

        if stream:
            return await self._stream_chat_completions(url, payload)

        return await self._post_with_retries(url, payload, self._convert_cc_response)

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

                if event_type == "content_block_start":
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
    ) -> AsyncGenerator[str, None]:
        """Yield text content token-by-token for live display."""
        if self.config.api_format == "anthropic":
            async for token in self._stream_anthropic_tokens(messages, tools):
                yield token
            return

        if self.config.api_format == "chat_completions":
            async for token in self._stream_chat_completions_tokens(
                messages, tools
            ):
                yield token
            return

        instructions, input_items = self._convert_messages_to_input(messages)

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

        url = self.config.api_base.rstrip("/")

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
                        if data.get("type") == "response.output_text.delta":
                            yield data.get("delta", "")
                    except json.JSONDecodeError:
                        continue

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

    async def _stream_chat_completions_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text tokens from a Chat Completions streaming response."""
        payload = self._build_cc_payload(messages, tools, stream=True)
        url = self.config.api_base.rstrip("/")

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
                        choices = data.get("choices", [])
                        if choices:
                            content = choices[0].get("delta", {}).get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

    async def _stream_anthropic_tokens(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text tokens from an Anthropic Messages streaming response."""
        system_prompt, anthropic_msgs = self._convert_messages_to_anthropic(messages)

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": anthropic_msgs,
            "max_tokens": self.config.max_tokens,
            "stream": True,
        }

        if system_prompt:
            payload["system"] = system_prompt
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if tools:
            payload["tools"] = self._convert_anthropic_tools(tools)

        url = self.config.api_base.rstrip("/")

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
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
                    except json.JSONDecodeError:
                        continue

        except asyncio.CancelledError:
            raise
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

    async def close(self):
        await self.client.aclose()
