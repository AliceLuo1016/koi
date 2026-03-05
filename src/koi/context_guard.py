"""Context window guard — last line of defense before every LLM call.

Ensures messages fit in the context window by:
1. Capping individual tool results at 50% of window
2. Compacting oldest tool results when total exceeds 75% of window
"""

import math
from typing import Any

# --- Constants ---

CHARS_PER_TOKEN = 4
CONTEXT_INPUT_HEADROOM = 0.75  # 25% headroom for tokenizer variance
SINGLE_TOOL_RESULT_SHARE = 0.5  # No single tool result > 50% of window
TOOL_RESULT_CHARS_PER_TOKEN = 2  # Tool output is denser than prose

TRUNCATION_NOTICE = "[truncated: output exceeded context limit]"
COMPACTION_PLACEHOLDER = "[compacted: tool output removed to free context]"


def estimate_message_chars(msg: dict[str, Any]) -> int:
    """Estimate the character cost of a single message.

    Tool messages are weighted by CHARS_PER_TOKEN / TOOL_RESULT_CHARS_PER_TOKEN
    because tool output is denser (fewer chars per token) than prose.
    """
    role = msg.get("role", "")

    if role == "user":
        content = msg.get("content", "")
        return len(content) if isinstance(content, str) else 0

    if role == "assistant":
        chars = 0
        content = msg.get("content")
        if isinstance(content, str):
            chars += len(content)
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            args = func.get("arguments", "")
            if isinstance(args, str):
                chars += len(args)
        return chars

    if role == "tool":
        content = msg.get("content", "")
        raw_chars = len(content) if isinstance(content, str) else 0
        # Weight: tool output is denser, so it costs more tokens per char
        return math.ceil(raw_chars * (CHARS_PER_TOKEN / TOOL_RESULT_CHARS_PER_TOKEN))

    # system or unknown
    content = msg.get("content", "")
    return len(content) if isinstance(content, str) else 0


def estimate_context_chars(messages: list[dict[str, Any]]) -> int:
    """Estimate total weighted character count across all messages."""
    return sum(estimate_message_chars(m) for m in messages)


def truncate_tool_result(msg: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Truncate a tool result's content if it exceeds max_chars.

    Tries to break at a newline boundary within the last 30% of the budget.
    Returns a new dict (does not mutate the original).
    """
    if msg.get("role") != "tool":
        return msg

    content = msg.get("content", "")
    if not isinstance(content, str):
        return msg

    # Compare raw content length against max_chars (max_chars is in raw chars
    # for tool results, caller accounts for weighting)
    if len(content) <= max_chars:
        return msg

    if max_chars <= 0:
        return {**msg, "content": TRUNCATION_NOTICE}

    suffix = "\n" + TRUNCATION_NOTICE
    body_budget = max(0, max_chars - len(suffix))
    if body_budget <= 0:
        return {**msg, "content": TRUNCATION_NOTICE}

    # Try to break at a newline within the last 30% of the body budget
    cut_point = body_budget
    search_start = int(body_budget * 0.7)
    newline_pos = content.rfind("\n", search_start, body_budget)
    if newline_pos > search_start:
        cut_point = newline_pos

    return {**msg, "content": content[:cut_point] + suffix}


def compact_oldest_tool_results(
    messages: list[dict[str, Any]], chars_needed: int
) -> list[dict[str, Any]]:
    """Replace oldest tool results with COMPACTION_PLACEHOLDER until enough chars freed.

    Returns a new list (does not mutate the original).
    """
    if chars_needed <= 0:
        return list(messages)

    result = list(messages)
    freed = 0

    for i, msg in enumerate(result):
        if freed >= chars_needed:
            break
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        # Skip already-compacted results
        if content == COMPACTION_PLACEHOLDER:
            continue

        before = estimate_message_chars(msg)
        result[i] = {**msg, "content": COMPACTION_PLACEHOLDER}
        after = estimate_message_chars(result[i])
        freed += before - after

    return result


def enforce_context_budget(
    messages: list[dict[str, Any]], context_window_tokens: int
) -> list[dict[str, Any]]:
    """Enforce context window budget on messages.

    Two-step enforcement:
    1. Cap individual tool results at 50% of window
    2. If total still over 75% budget, compact oldest tool results

    Returns a new list (does not mutate the original).
    """
    if not messages or context_window_tokens <= 0:
        return list(messages)

    context_budget_chars = int(
        context_window_tokens * CHARS_PER_TOKEN * CONTEXT_INPUT_HEADROOM
    )
    max_single_tool_chars = int(
        context_window_tokens * TOOL_RESULT_CHARS_PER_TOKEN * SINGLE_TOOL_RESULT_SHARE
    )

    # Step 1: Truncate any individual tool result exceeding max_single_tool_chars
    result = list(messages)
    for i, msg in enumerate(result):
        if msg.get("role") != "tool":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > max_single_tool_chars:
            result[i] = truncate_tool_result(msg, max_single_tool_chars)

    # Step 2: If total context exceeds budget, compact oldest tool results
    total_chars = estimate_context_chars(result)
    if total_chars > context_budget_chars:
        result = compact_oldest_tool_results(result, total_chars - context_budget_chars)

    return result
