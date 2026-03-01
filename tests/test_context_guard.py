"""Tests for context_guard module."""

import copy

from koi.context_guard import (
    CHARS_PER_TOKEN,
    COMPACTION_PLACEHOLDER,
    CONTEXT_INPUT_HEADROOM,
    SINGLE_TOOL_RESULT_SHARE,
    TOOL_RESULT_CHARS_PER_TOKEN,
    TRUNCATION_NOTICE,
    compact_oldest_tool_results,
    enforce_context_budget,
    estimate_context_chars,
    estimate_message_chars,
    truncate_tool_result,
)


# --- estimate_message_chars ---


class TestEstimateMessageChars:
    def test_user_message(self):
        msg = {"role": "user", "content": "hello world"}
        assert estimate_message_chars(msg) == len("hello world")

    def test_assistant_message_with_content(self):
        msg = {"role": "assistant", "content": "I will help you."}
        assert estimate_message_chars(msg) == len("I will help you.")

    def test_assistant_message_with_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "Let me check.",
            "tool_calls": [
                {
                    "id": "tc_1",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/test.txt"}',
                    },
                }
            ],
        }
        expected = len("Let me check.") + len('{"path": "/tmp/test.txt"}')
        assert estimate_message_chars(msg) == expected

    def test_tool_message_weighted(self):
        content = "x" * 100
        msg = {"role": "tool", "content": content}
        # Tool messages are weighted: raw_chars * (CHARS_PER_TOKEN / TOOL_RESULT_CHARS_PER_TOKEN)
        # 100 * (4 / 2) = 200
        expected = 100 * CHARS_PER_TOKEN // TOOL_RESULT_CHARS_PER_TOKEN
        assert estimate_message_chars(msg) == expected

    def test_system_message(self):
        msg = {"role": "system", "content": "You are helpful."}
        assert estimate_message_chars(msg) == len("You are helpful.")

    def test_empty_content(self):
        assert estimate_message_chars({"role": "user", "content": ""}) == 0

    def test_missing_content(self):
        assert estimate_message_chars({"role": "user"}) == 0

    def test_non_string_content(self):
        assert estimate_message_chars({"role": "user", "content": 123}) == 0


# --- estimate_context_chars ---


class TestEstimateContextChars:
    def test_multiple_messages(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "data"},
        ]
        user_chars = len("hi")
        assistant_chars = len("hello")
        tool_chars = len("data") * CHARS_PER_TOKEN // TOOL_RESULT_CHARS_PER_TOKEN
        assert estimate_context_chars(messages) == user_chars + assistant_chars + tool_chars

    def test_empty_list(self):
        assert estimate_context_chars([]) == 0


# --- truncate_tool_result ---


class TestTruncateToolResult:
    def test_preserves_small_result(self):
        msg = {"role": "tool", "content": "short output", "tool_call_id": "tc_1"}
        result = truncate_tool_result(msg, 1000)
        assert result["content"] == "short output"

    def test_truncates_oversized_result(self):
        content = "x" * 5000
        msg = {"role": "tool", "content": content, "tool_call_id": "tc_1"}
        result = truncate_tool_result(msg, 1000)
        assert len(result["content"]) < len(content)
        assert result["content"].endswith(TRUNCATION_NOTICE)

    def test_breaks_at_newline_boundary(self):
        # Build content with a newline in the last 30% of the budget
        budget = 1000
        suffix_len = len("\n" + TRUNCATION_NOTICE)
        body_budget = budget - suffix_len
        # Place a newline at ~80% of body_budget (within the 70-100% search window)
        newline_pos = int(body_budget * 0.8)
        content = "a" * newline_pos + "\n" + "b" * (5000 - newline_pos - 1)
        msg = {"role": "tool", "content": content, "tool_call_id": "tc_1"}
        result = truncate_tool_result(msg, budget)
        # Should cut at the newline
        assert result["content"].startswith("a" * newline_pos)
        assert TRUNCATION_NOTICE in result["content"]

    def test_does_not_mutate_original(self):
        content = "x" * 5000
        msg = {"role": "tool", "content": content, "tool_call_id": "tc_1"}
        original_content = msg["content"]
        truncate_tool_result(msg, 1000)
        assert msg["content"] == original_content

    def test_non_tool_message_passthrough(self):
        msg = {"role": "user", "content": "x" * 5000}
        result = truncate_tool_result(msg, 10)
        assert result is msg  # Same object, unchanged

    def test_zero_budget(self):
        msg = {"role": "tool", "content": "x" * 100, "tool_call_id": "tc_1"}
        result = truncate_tool_result(msg, 0)
        assert result["content"] == TRUNCATION_NOTICE

    def test_preserves_other_fields(self):
        msg = {"role": "tool", "content": "x" * 5000, "tool_call_id": "tc_1"}
        result = truncate_tool_result(msg, 100)
        assert result["tool_call_id"] == "tc_1"
        assert result["role"] == "tool"


# --- compact_oldest_tool_results ---


class TestCompactOldestToolResults:
    def test_replaces_oldest_first(self):
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "first tool output " * 100, "tool_call_id": "tc_1"},
            {"role": "tool", "content": "second tool output " * 100, "tool_call_id": "tc_2"},
            {"role": "tool", "content": "third tool output " * 100, "tool_call_id": "tc_3"},
        ]
        # Request enough chars to compact the first tool result
        result = compact_oldest_tool_results(messages, 1)
        assert result[1]["content"] == COMPACTION_PLACEHOLDER
        assert result[2]["content"] != COMPACTION_PLACEHOLDER
        assert result[3]["content"] != COMPACTION_PLACEHOLDER

    def test_stops_when_enough_freed(self):
        messages = [
            {"role": "tool", "content": "a" * 1000, "tool_call_id": "tc_1"},
            {"role": "tool", "content": "b" * 1000, "tool_call_id": "tc_2"},
            {"role": "tool", "content": "c" * 1000, "tool_call_id": "tc_3"},
        ]
        # Freeing from first message alone should be enough for a small request
        result = compact_oldest_tool_results(messages, 100)
        assert result[0]["content"] == COMPACTION_PLACEHOLDER
        # Second should be untouched if first freed enough
        assert result[1]["content"] == "b" * 1000

    def test_skips_already_compacted(self):
        messages = [
            {"role": "tool", "content": COMPACTION_PLACEHOLDER, "tool_call_id": "tc_1"},
            {"role": "tool", "content": "b" * 1000, "tool_call_id": "tc_2"},
        ]
        result = compact_oldest_tool_results(messages, 100)
        # First was already compacted, so second gets compacted
        assert result[0]["content"] == COMPACTION_PLACEHOLDER
        assert result[1]["content"] == COMPACTION_PLACEHOLDER

    def test_does_not_mutate_original(self):
        messages = [
            {"role": "tool", "content": "a" * 1000, "tool_call_id": "tc_1"},
        ]
        original = copy.deepcopy(messages)
        compact_oldest_tool_results(messages, 100)
        assert messages[0]["content"] == original[0]["content"]

    def test_no_compaction_needed(self):
        messages = [
            {"role": "tool", "content": "small", "tool_call_id": "tc_1"},
        ]
        result = compact_oldest_tool_results(messages, 0)
        assert result[0]["content"] == "small"


# --- enforce_context_budget ---


class TestEnforceContextBudget:
    def test_individual_tool_result_capped_at_50_percent(self):
        # context_window = 1000 tokens
        # max_single_tool_chars = 1000 * 2 * 0.5 = 1000 chars
        context_window = 1000
        max_single = int(context_window * TOOL_RESULT_CHARS_PER_TOKEN * SINGLE_TOOL_RESULT_SHARE)
        oversized_content = "x" * (max_single * 3)
        messages = [
            {"role": "user", "content": "go"},
            {"role": "tool", "content": oversized_content, "tool_call_id": "tc_1"},
        ]
        result = enforce_context_budget(messages, context_window)
        assert len(result[1]["content"]) <= max_single + len("\n" + TRUNCATION_NOTICE)
        assert TRUNCATION_NOTICE in result[1]["content"]

    def test_total_context_capped_at_75_percent(self):
        # context_window = 10000 tokens
        # budget_chars = 10000 * 4 * 0.75 = 30000
        # max_single = 10000 * 2 * 0.5 = 10000 chars (raw) per tool result
        context_window = 10000
        budget_chars = int(context_window * CHARS_PER_TOKEN * CONTEXT_INPUT_HEADROOM)
        # 10 tool results each with 5000 raw chars => 5000 * 2 = 10000 weighted each
        # Total tool weighted = 100000, way over 30000 budget
        messages = [{"role": "user", "content": "go"}]
        for i in range(10):
            messages.append({
                "role": "tool",
                "content": "d" * 5000,
                "tool_call_id": f"tc_{i}",
            })
        result = enforce_context_budget(messages, context_window)
        total = estimate_context_chars(result)
        assert total <= budget_chars

    def test_no_changes_when_under_budget(self):
        context_window = 100000  # huge window
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "tool", "content": "small result", "tool_call_id": "tc_1"},
        ]
        result = enforce_context_budget(messages, context_window)
        for orig, new in zip(messages, result):
            assert orig["content"] == new["content"]

    def test_returns_new_list_doesnt_mutate_original(self):
        context_window = 10  # tiny window to force changes
        messages = [
            {"role": "tool", "content": "x" * 10000, "tool_call_id": "tc_1"},
        ]
        original_content = messages[0]["content"]
        result = enforce_context_budget(messages, context_window)
        # Original must be unchanged
        assert messages[0]["content"] == original_content
        # Result should be a different list
        assert result is not messages

    def test_compacts_oldest_when_total_over_budget(self):
        # context_window = 5000 tokens
        # budget_chars = 5000 * 4 * 0.75 = 15000
        # max_single = 5000 * 2 * 0.5 = 5000 raw chars per tool result
        context_window = 5000
        budget_chars = int(context_window * CHARS_PER_TOKEN * CONTEXT_INPUT_HEADROOM)
        # 5 tool results each with 3000 raw chars => 6000 weighted each
        # Total = 2 (user) + 30000 (tools) = 30002, over 15000 budget
        messages = [{"role": "user", "content": "go"}]
        for i in range(5):
            messages.append({
                "role": "tool",
                "content": "z" * 3000,
                "tool_call_id": f"tc_{i}",
            })
        result = enforce_context_budget(messages, context_window)
        # At least some old tool results should be compacted
        compacted_count = sum(
            1 for m in result if m.get("role") == "tool" and m["content"] == COMPACTION_PLACEHOLDER
        )
        assert compacted_count > 0
        # Total should now be under budget
        assert estimate_context_chars(result) <= budget_chars

    def test_handles_empty_messages(self):
        result = enforce_context_budget([], 1000)
        assert result == []

    def test_handles_messages_with_no_tool_results(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        result = enforce_context_budget(messages, 100000)
        assert len(result) == 3
        for orig, new in zip(messages, result):
            assert orig["content"] == new["content"]

    def test_zero_context_window(self):
        messages = [{"role": "user", "content": "hi"}]
        result = enforce_context_budget(messages, 0)
        assert result == [{"role": "user", "content": "hi"}]

    def test_negative_context_window(self):
        messages = [{"role": "user", "content": "hi"}]
        result = enforce_context_budget(messages, -100)
        assert result == [{"role": "user", "content": "hi"}]


# --- System prompt preservation ---


class TestSystemPromptPreservation:
    def test_context_guard_preserves_system_prompt(self):
        """System prompt is never compacted or truncated by context guard."""
        system_content = "You are koi. Follow all safety rules and tool guidance."
        context_window = 5000
        budget_chars = int(context_window * CHARS_PER_TOKEN * CONTEXT_INPUT_HEADROOM)

        messages = [{"role": "system", "content": system_content}]
        for i in range(10):
            messages.append({
                "role": "tool",
                "content": "z" * 3000,
                "tool_call_id": f"tc_{i}",
            })

        result = enforce_context_budget(messages, context_window)

        # System prompt must be preserved exactly
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_content

    def test_compact_oldest_tool_results_skips_system(self):
        """compact_oldest_tool_results never touches system messages."""
        system_content = "You are koi."
        messages = [
            {"role": "system", "content": system_content},
            {"role": "tool", "content": "a" * 1000, "tool_call_id": "tc_1"},
            {"role": "tool", "content": "b" * 1000, "tool_call_id": "tc_2"},
        ]
        result = compact_oldest_tool_results(messages, 5000)

        # System message unchanged
        assert result[0]["role"] == "system"
        assert result[0]["content"] == system_content
        # Tool messages get compacted
        assert result[1]["content"] == COMPACTION_PLACEHOLDER
