"""Preemptive context pruning for koi agent.

Two-phase pruning of old tool results to manage context window usage
before LLM compaction kicks in. Inspired by OpenClaw's approach.

Phase 1 (soft trim): Keep head + tail of large old tool results.
Phase 2 (hard clear): Replace entire old tool results with placeholder.
"""

from typing import Any

# --- Constants ---

CHARS_PER_TOKEN = 4

# Start soft-trimming old tool results at 30% of context window
SOFT_TRIM_RATIO = 0.3
# Start hard-clearing old tool results at 50% of context window
HARD_CLEAR_RATIO = 0.5

# Never prune tool results within the last N assistant messages
KEEP_LAST_ASSISTANTS = 3

# Soft trim parameters
SOFT_TRIM_MAX_CHARS = 4000
SOFT_TRIM_HEAD_CHARS = 1500
SOFT_TRIM_TAIL_CHARS = 1500

# Minimum total prunable chars before hard clear activates
MIN_PRUNABLE_CHARS = 50000

HARD_CLEAR_PLACEHOLDER = "[compacted: tool output removed to free context]"

# Tool results from these tools are eligible for pruning
PRUNABLE_TOOLS = {
    "read_file",
    "exec_command",
    "web_fetch",
    "web_search",
    "glob_files",
    "grep_files",
}


def estimate_message_chars(msg: dict[str, Any]) -> int:
    """Estimate the character count of a single message."""
    role = msg.get("role", "")

    if role == "user":
        content = msg.get("content", "")
        return len(content) if isinstance(content, str) else 0

    if role == "assistant":
        chars = 0
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                func = tc.get("function", {})
                args = func.get("arguments", "")
                chars += len(args) if isinstance(args, str) else 0
        return chars

    if role == "tool":
        content = msg.get("content", "")
        return len(content) if isinstance(content, str) else 0

    # system or unknown
    content = msg.get("content", "")
    return len(content) if isinstance(content, str) else 0


def estimate_context_chars(messages: list[dict[str, Any]]) -> int:
    """Estimate total character count across all messages."""
    return sum(estimate_message_chars(m) for m in messages)


def _find_first_user_index(messages: list[dict[str, Any]]) -> int | None:
    """Find the index of the first user message."""
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            return i
    return None


def _find_assistant_cutoff_index(
    messages: list[dict[str, Any]], keep_last: int
) -> int | None:
    """Find the index of the Nth-from-last assistant message.

    Everything from this index onward is protected from pruning.
    Returns None if there aren't enough assistant messages.
    """
    if keep_last <= 0:
        return len(messages)

    remaining = keep_last
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            remaining -= 1
            if remaining == 0:
                return i
    return None


def _get_tool_name_for_result(
    messages: list[dict[str, Any]], tool_msg_index: int
) -> str | None:
    """Look up the tool name for a tool result message.

    Walks backward from the tool result to find the preceding assistant
    message with tool_calls, then matches by tool_call_id.
    """
    tool_msg = messages[tool_msg_index]
    target_id = tool_msg.get("tool_call_id")
    if not target_id:
        return None

    for i in range(tool_msg_index - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("id") == target_id:
                    return tc.get("function", {}).get("name")
            break
    return None


def _soft_trim_content(content: str) -> str | None:
    """Soft-trim a tool result's content: keep head + tail, replace middle.

    Returns None if no trimming needed (content is small enough).
    """
    if len(content) <= SOFT_TRIM_MAX_CHARS:
        return None

    head = content[:SOFT_TRIM_HEAD_CHARS]
    tail = content[-SOFT_TRIM_TAIL_CHARS:]
    trimmed = f"{head}\n...\n{tail}"
    note = (
        f"\n\n[Tool result trimmed: kept first {SOFT_TRIM_HEAD_CHARS} chars "
        f"and last {SOFT_TRIM_TAIL_CHARS} chars of {len(content)} chars.]"
    )
    return trimmed + note


def prune_context(
    messages: list[dict[str, Any]], context_window_tokens: int
) -> list[dict[str, Any]]:
    """Prune old tool results to manage context window usage.

    Phase 1 (soft trim): When context exceeds SOFT_TRIM_RATIO, trim large
    old prunable tool results to head + tail.

    Phase 2 (hard clear): When context exceeds HARD_CLEAR_RATIO, replace
    entire old prunable tool results with a placeholder.

    Protections:
    - Never prune messages before the first user message
    - Never prune the last KEEP_LAST_ASSISTANTS assistant messages or
      anything after them
    - Only prune tool results from PRUNABLE_TOOLS

    Returns a new list (does not mutate the original).
    """
    if not messages or context_window_tokens <= 0:
        return list(messages)

    char_window = context_window_tokens * CHARS_PER_TOKEN
    total_chars = estimate_context_chars(messages)
    ratio = total_chars / char_window

    if ratio < SOFT_TRIM_RATIO:
        return list(messages)

    # Determine protected ranges
    cutoff_index = _find_assistant_cutoff_index(messages, KEEP_LAST_ASSISTANTS)
    if cutoff_index is None:
        return list(messages)

    first_user = _find_first_user_index(messages)
    prune_start = first_user if first_user is not None else len(messages)

    # Build list of prunable tool result indices
    prunable_indices: list[int] = []
    for i in range(prune_start, cutoff_index):
        msg = messages[i]
        if msg.get("role") != "tool":
            continue
        tool_name = _get_tool_name_for_result(messages, i)
        if tool_name and tool_name in PRUNABLE_TOOLS:
            prunable_indices.append(i)

    if not prunable_indices:
        return list(messages)

    # Make a shallow copy of the list; we'll replace individual dicts as needed
    result = list(messages)

    # Phase 1: soft trim
    for i in prunable_indices:
        content = result[i].get("content", "")
        if not isinstance(content, str):
            continue
        trimmed = _soft_trim_content(content)
        if trimmed is not None:
            old_chars = estimate_message_chars(result[i])
            result[i] = {**result[i], "content": trimmed}
            new_chars = estimate_message_chars(result[i])
            total_chars += new_chars - old_chars

    ratio = total_chars / char_window
    if ratio < HARD_CLEAR_RATIO:
        return result

    # Phase 2: hard clear — check minimum prunable chars
    prunable_chars = sum(estimate_message_chars(result[i]) for i in prunable_indices)
    if prunable_chars < MIN_PRUNABLE_CHARS:
        return result

    for i in prunable_indices:
        if ratio < HARD_CLEAR_RATIO:
            break
        old_chars = estimate_message_chars(result[i])
        result[i] = {**result[i], "content": HARD_CLEAR_PLACEHOLDER}
        new_chars = estimate_message_chars(result[i])
        total_chars += new_chars - old_chars
        ratio = total_chars / char_window

    return result
