"""Tests for llm module — Responses API conversion logic."""

from unittest.mock import MagicMock

import pytest

from koi.config import Config
from koi.llm import LLMClient


@pytest.fixture
def client():
    config = Config(
        api_base="https://api.example.com/v1/responses",
        api_key="test-key",
        model="test-model",
    )
    return LLMClient(config)


# ── _convert_messages_to_input ──


def test_system_message_becomes_instructions(client):
    messages = [{"role": "system", "content": "You are helpful."}]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions == "You are helpful."
    # Also injected as a developer message
    assert len(input_items) == 1
    assert input_items[0]["role"] == "developer"
    assert input_items[0]["content"] == "You are helpful."


def test_user_message(client):
    messages = [{"role": "user", "content": "Hello"}]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions is None
    assert input_items == [{"role": "user", "content": "Hello"}]


def test_assistant_text_message(client):
    messages = [{"role": "assistant", "content": "Hi there"}]
    _, input_items = client._convert_messages_to_input(messages)
    assert input_items == [{"role": "assistant", "content": "Hi there"}]


def test_assistant_tool_calls_become_function_call_items(client):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.txt"}',
                    },
                }
            ],
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert len(input_items) == 1
    assert input_items[0] == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "read_file",
        "arguments": '{"path": "test.txt"}',
    }


def test_assistant_with_content_and_tool_calls(client):
    messages = [
        {
            "role": "assistant",
            "content": "Let me check that.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"command": "ls"}',
                    },
                }
            ],
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert len(input_items) == 2
    assert input_items[0] == {"role": "assistant", "content": "Let me check that."}
    assert input_items[1]["type"] == "function_call"
    assert input_items[1]["name"] == "exec_command"


def test_tool_result_becomes_function_call_output(client):
    messages = [
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file contents here",
        }
    ]
    _, input_items = client._convert_messages_to_input(messages)
    assert input_items == [
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "file contents here",
        }
    ]


def test_full_conversation_roundtrip(client):
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Read test.txt"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "test.txt"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello world"},
        {"role": "assistant", "content": "The file says hello world."},
        {"role": "user", "content": "Thanks"},
    ]
    instructions, input_items = client._convert_messages_to_input(messages)
    assert instructions == "Be helpful."
    # developer msg + user + function_call + function_call_output + assistant + user
    assert len(input_items) == 6
    assert input_items[0]["role"] == "developer"
    assert input_items[1]["role"] == "user"
    assert input_items[2]["type"] == "function_call"
    assert input_items[3]["type"] == "function_call_output"
    assert input_items[4]["role"] == "assistant"
    assert input_items[5]["role"] == "user"


# ── _convert_tools ──


def test_convert_tools_flattens_function(client):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]
    converted = client._convert_tools(tools)
    assert len(converted) == 1
    assert converted[0] == {
        "type": "function",
        "name": "read_file",
        "description": "Read a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }
    # Must NOT have nested "function" key
    assert "function" not in converted[0]


def test_convert_tools_multiple(client):
    from koi.tools import get_tool_definitions

    tools = get_tool_definitions()
    converted = client._convert_tools(tools)
    assert len(converted) == len(tools)
    for c in converted:
        assert "name" in c
        assert "function" not in c


# ── _convert_response ──


def test_convert_text_response(client):
    api_response = {
        "id": "resp_1",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello!"
    assert "tool_calls" not in msg


def test_convert_tool_call_response(client):
    api_response = {
        "id": "resp_2",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "read_file",
                "arguments": '{"path": "x.py"}',
            }
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_abc"
    assert tc["function"]["name"] == "read_file"
    assert tc["function"]["arguments"] == '{"path": "x.py"}'


def test_convert_multiple_tool_calls(client):
    api_response = {
        "id": "resp_3",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path": "a.py"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "read_file",
                "arguments": '{"path": "b.py"}',
            },
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert len(msg["tool_calls"]) == 2


def test_convert_mixed_text_and_tool_calls(client):
    api_response = {
        "id": "resp_4",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Let me check."}],
            },
            {
                "type": "function_call",
                "call_id": "call_x",
                "name": "exec_command",
                "arguments": '{"command": "ls"}',
            },
        ],
    }
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["content"] == "Let me check."
    assert len(msg["tool_calls"]) == 1


def test_convert_empty_output(client):
    api_response = {"id": "resp_5", "output": []}
    result = client._convert_response(api_response)
    msg = result["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert "content" not in msg
    assert "tool_calls" not in msg


# ── chat() request construction ──


async def test_chat_builds_correct_payload(client):
    """Verify the payload sent to the API matches Responses API format."""
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]

    captured_payload = {}

    async def fake_post(url, headers=None, json=None):
        captured_payload.update(json)
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {
            "id": "resp_1",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Hi!"}],
                }
            ],
        }
        return mock_resp

    client.client.post = fake_post

    result = await client.chat(messages, tools=tools)

    # Verify payload structure
    assert captured_payload["model"] == "test-model"
    assert captured_payload["instructions"] == "You are helpful."
    # input includes developer message + user message
    assert captured_payload["input"] == [
        {"role": "developer", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
    ]
    # Tools should be flattened
    assert captured_payload["tools"][0]["name"] == "read_file"
    assert "function" not in captured_payload["tools"][0]
    # No stream for non-streaming call
    assert "stream" not in captured_payload

    # Verify response was converted back
    assert result["choices"][0]["message"]["content"] == "Hi!"


async def test_chat_url_uses_api_base_directly(client):
    """Verify we POST to api_base, not api_base + /chat/completions."""
    captured_url = None

    async def fake_post(url, **kwargs):
        nonlocal captured_url
        captured_url = url
        mock_resp = MagicMock()
        mock_resp.raise_for_status = lambda: None
        mock_resp.json.return_value = {"id": "r", "output": []}
        return mock_resp

    client.client.post = fake_post
    await client.chat([{"role": "user", "content": "test"}])

    assert captured_url == "https://api.example.com/v1/responses"
    assert "chat/completions" not in captured_url
