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
    MAX_RETRIES = 3

    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
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

    # ── API calls ──

    async def chat(
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

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                response = await self.client.post(
                    url, headers=self.headers, json=payload
                )
                response.raise_for_status()
                return self._convert_response(response.json())

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code not in self.RETRYABLE_STATUS_CODES:
                    raise RuntimeError(
                        f"HTTP {e.response.status_code}: {e.response.text}"
                    )
                # Retryable — wait with exponential backoff
                delay = 2 ** attempt
                await asyncio.sleep(delay)

            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                last_error = e
                delay = 2 ** attempt
                await asyncio.sleep(delay)

            except Exception as e:
                raise RuntimeError(f"Request failed: {e}")

        # All retries exhausted
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

    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Yield text content token-by-token for live display."""
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

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"HTTP {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")

    async def close(self):
        await self.client.aclose()
