"""LLM client for OpenAI-compatible APIs."""

import json
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator
import httpx

from .config import Config


class LLMClient:
    """Client for OpenAI-compatible v1/chat/completions API."""
    
    def __init__(self, config: Config):
        """Initialize LLM client with configuration."""
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0)
        )
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        }
    
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send chat completion request."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": stream,
        }
        
        if tools:
            payload["tools"] = tools
        
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        
        try:
            if stream:
                return await self._stream_chat(url, payload)
            else:
                response = await self.client.post(
                    url,
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
        
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}")
    
    async def _stream_chat(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle streaming chat response."""
        async with self.client.stream(
            "POST",
            url,
            headers=self.headers,
            json=payload
        ) as response:
            response.raise_for_status()
            
            # Collect streaming response
            full_content = ""
            tool_calls = []
            
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            
                            # Handle content
                            if "content" in delta and delta["content"]:
                                full_content += delta["content"]
                            
                            # Handle tool calls
                            if "tool_calls" in delta:
                                for tool_call_delta in delta["tool_calls"]:
                                    index = tool_call_delta.get("index", 0)
                                    
                                    # Extend tool_calls list if needed
                                    while len(tool_calls) <= index:
                                        tool_calls.append({
                                            "id": "",
                                            "type": "function",
                                            "function": {"name": "", "arguments": ""}
                                        })
                                    
                                    if "id" in tool_call_delta:
                                        tool_calls[index]["id"] = tool_call_delta["id"]
                                    
                                    if "function" in tool_call_delta:
                                        func_delta = tool_call_delta["function"]
                                        if "name" in func_delta:
                                            tool_calls[index]["function"]["name"] = func_delta["name"]
                                        if "arguments" in func_delta:
                                            tool_calls[index]["function"]["arguments"] += func_delta["arguments"]
                    
                    except json.JSONDecodeError:
                        continue
            
            # Construct response in OpenAI format
            message = {"role": "assistant"}
            
            if full_content:
                message["content"] = full_content
            
            if tool_calls:
                message["tool_calls"] = tool_calls
            
            return {
                "choices": [{
                    "message": message,
                    "finish_reason": "stop"
                }]
            }
    
    async def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream chat response token by token."""
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_completion_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        
        if tools:
            payload["tools"] = tools
        
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"
        
        try:
            async with self.client.stream(
                "POST",
                url,
                headers=self.headers,
                json=payload
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                delta = choice.get("delta", {})
                                
                                if "content" in delta and delta["content"]:
                                    yield delta["content"]
                        
                        except json.JSONDecodeError:
                            continue
        
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP {e.response.status_code}: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Stream request failed: {e}")
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()