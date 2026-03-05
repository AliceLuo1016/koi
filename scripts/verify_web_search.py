#!/usr/bin/env python3
"""Verify web search availability across Koi presets.

Usage:
  python scripts/verify_web_search.py --preset 1 --key YOUR_KEY
  python scripts/verify_web_search.py --preset 2 --key YOUR_KEY
  python scripts/verify_web_search.py --preset 3 --key YOUR_KEY

Set the API key directly in --key or edit DEFAULT_KEYS below.
"""

import argparse
import sys
from typing import Any

import httpx

# Optional: fill these in locally if you don't want to pass --key each time.
DEFAULT_KEYS = {
    "1": "",  # Preset 1 key
    "2": "",  # Preset 2 key
    "3": "",  # Preset 3 key
}

MODEL_PRESETS = {
    "1": {
        "name": "GPT-5.2 Codex",
        "model": "openai/openai/gpt-5.2-codex",
        "api_base": "https://inference-api.nvidia.com/v1/responses",
        "api_format": "responses",
    },
    "2": {
        "name": "Claude Opus 4.6 (NVIDIA chat_completions)",
        "model": "aws/anthropic/bedrock-claude-opus-4-6",
        "api_base": "https://inference-api.nvidia.com/v1/chat/completions",
        "api_format": "chat_completions",
    },
    "3": {
        "name": "Claude Opus 4 (Anthropic Messages)",
        "model": "claude-opus-4-6",
        "api_base": "https://api.anthropic.com/v1/messages",
        "api_format": "anthropic",
    },
}


def build_request(preset: dict[str, Any], key: str) -> dict[str, Any]:
    """Return (url, headers, payload)."""
    api_format = preset["api_format"]
    model = preset["model"]
    url = preset["api_base"].rstrip("/")

    if api_format == "responses":
        # OpenAI-style Responses API with web_search tool
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        payload = {
            "model": model,
            "input": [
                {
                    "role": "user",
                    "content": ("Search the web for today's top tech headline and cite sources."),
                }
            ],
            "tools": [{"type": "web_search"}],
            "max_output_tokens": 200,
        }
        return url, headers, payload

    if api_format == "chat_completions":
        # Chat Completions models generally do NOT support
        # built-in web search unless using search models.
        # We still send a normal request and mark web search
        # as NOT supported if the model lacks it.
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": ("Try to search the web for today's top tech headline and cite sources."),
                }
            ],
            "max_tokens": 200,
        }
        return url, headers, payload

    if api_format == "anthropic":
        # Anthropic Messages API with built-in web search tool
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": 2048,
            "messages": [
                {
                    "role": "user",
                    "content": ("Search the web for today's top tech headline and cite sources."),
                }
            ],
            "tools": [
                {
                    "type": TOOL_TYPE,
                    "name": "web_search",
                    "max_uses": 3,
                }
            ],
        }
        if FORCE_TOOL:
            payload["tool_choice"] = {"type": "tool", "name": "web_search"}
        return url, headers, payload

    raise ValueError(f"Unknown api_format: {api_format}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", required=True, choices=["1", "2", "3"], help="Preset number")
    parser.add_argument("--key", default="", help="API key for this preset")
    parser.add_argument(
        "--tool-type",
        default="web_search_20260209",
        help="Anthropic web search tool type (preset 3 only)",
    )
    parser.add_argument(
        "--force-tool",
        action="store_true",
        help="Force tool_choice=web_search (Anthropic only)",
    )
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.preset]
    key = args.key or DEFAULT_KEYS.get(args.preset, "")
    if not key:
        print("ERROR: No API key provided. Use --key or set DEFAULT_KEYS.")
        return 2

    global TOOL_TYPE, FORCE_TOOL
    TOOL_TYPE = args.tool_type
    FORCE_TOOL = args.force_tool

    url, headers, payload = build_request(preset, key)

    print(f"Testing preset {args.preset}: {preset['name']}")
    print(f"Endpoint: {url}")

    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, headers=headers, json=payload)
        print(f"HTTP {resp.status_code}")
        if resp.status_code >= 400:
            print(resp.text[:2000])
            return 1

        data = resp.json()
        # Heuristic: look for tool usage and any actual tool results
        conclusion = "INCONCLUSIVE"
        if preset["api_format"] == "responses":
            # Responses API: tool calls appear in output items
            output = data.get("output", [])
            tool_items = [
                item for item in output if item.get("type") in ("web_search_call", "tool_call", "function_call")
            ]
            used_tool = bool(tool_items)
            print("Tool usage detected." if used_tool else "No tool usage detected.")
            if tool_items:
                print("--- Tool call details ---")
                for item in tool_items:
                    name = item.get("name") or item.get("tool") or item.get("function", {}).get("name")
                    args = item.get("arguments") or item.get("parameters") or item.get("function", {}).get("arguments")
                    if isinstance(args, str) and len(args) > 500:
                        args = args[:500] + "..."
                    print(f"type={item.get('type')} name={name}")
                    if args:
                        print(f"args={args}")
            # Check for tool results or citations if provided by the provider
            if output:
                print("--- Output item types ---")
                print(", ".join(item.get("type", "?") for item in output))
            has_result = any(
                item.get("type") in ("web_search_result", "tool_result", "function_call_output") for item in output
            )
            if has_result:
                conclusion = "PASS: web search executed (results present)."
            elif used_tool:
                conclusion = "FAIL: model requested web search, but no results were returned."
            else:
                conclusion = "FAIL: no web search tool call detected."
        elif preset["api_format"] == "anthropic":
            # Anthropic: tool_use blocks indicate use of web search
            content = data.get("content", [])
            used_tool = any(block.get("type") == "server_tool_use" for block in content)
            print("Tool usage detected." if used_tool else "No tool usage detected.")
            if used_tool:
                print("--- Tool call details ---")
                for block in content:
                    if block.get("type") == "server_tool_use":
                        name = block.get("name")
                        tool_input = block.get("input")
                        print(f"name={name} input={tool_input}")
            has_result = any(
                block.get("type") == "web_search_tool_result"
                or (block.get("type") == "server_tool_use" and block.get("name") == "web_search")
                for block in content
            )
            if has_result:
                conclusion = "PASS: web search executed (results/citations present)."
            elif used_tool:
                conclusion = "FAIL: model requested web search, but no results were returned."
            else:
                conclusion = "FAIL: no web search tool call detected."
        else:
            # Chat Completions: no web search unless using special models
            print(
                "Note: Chat Completions endpoint does not"
                " support built-in web search unless"
                " using search-enabled models."
            )
            conclusion = "FAIL: built-in web search not supported on this endpoint/model."

        # Print a short snippet
        text = None
        if preset["api_format"] == "anthropic":
            for block in data.get("content", []):
                if block.get("type") == "text":
                    text = block.get("text")
                    break
        else:
            choices = data.get("choices") or []
            if choices:
                msg = choices[0].get("message", {})
                text = msg.get("content")
        if text:
            print("--- Response snippet ---")
            print(text[:500])

        print("--- Conclusion ---")
        print(conclusion)

        return 0

    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
