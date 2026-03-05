"""Unified streaming event protocol for LLM responses."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """A single event from an LLM streaming response.

    Event types:
    - text_start: Text content block started (contentIndex)
    - text_delta: Text content chunk (delta)
    - text_end: Text content block finished (content = full accumulated text)
    - thinking_start: Thinking block started (contentIndex)
    - thinking_delta: Thinking content chunk (delta)
    - thinking_end: Thinking block finished (content = full thinking text)
    - toolcall_start: Tool call started (contentIndex, tool_name, tool_call_id)
    - toolcall_delta: Tool call argument chunk (delta)
    - toolcall_end: Tool call finished
      (tool_name, tool_call_id, arguments = full JSON string)
    - usage: Token usage update (usage dict)
    - done: Stream completed (finish_reason)
    - error: Stream error (error message)
    """

    type: str
    content_index: int = 0
    delta: str = ""
    content: str = ""
    tool_name: str = ""
    tool_call_id: str = ""
    arguments: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    error: str = ""
