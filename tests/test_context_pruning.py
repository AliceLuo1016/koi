"""Tests for context pruning and compaction improvements."""

import asyncio
import copy
from unittest.mock import MagicMock, patch

from koi.compaction import ContextCompactor, compact_with_timeout
from koi.context_pruning import (
    HARD_CLEAR_PLACEHOLDER,
    PRUNABLE_TOOLS,
    estimate_context_chars,
    estimate_message_chars,
    prune_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call(tool_id: str, name: str, arguments: str = "{}"):
    return {
        "id": tool_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assistant_with_tool_calls(tool_calls):
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}


def _tool_result(tool_call_id: str, content: str):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _user_msg(content: str = "hello"):
    return {"role": "user", "content": content}


def _assistant_msg(content: str = "Sure, I can help."):
    return {"role": "assistant", "content": content}


def _make_conversation_with_tool_results(
    num_tool_rounds: int,
    tool_name: str = "read_file",
    result_size: int = 5000,
):
    """Build a realistic conversation with tool call/result pairs.

    Structure: user, then N rounds of (assistant+tool_call, tool_result),
    ending with a final assistant message.
    """
    msgs = [_user_msg("Read some files for me")]

    for i in range(num_tool_rounds):
        tc_id = f"tc_{i}"
        msgs.append(_assistant_with_tool_calls([_make_tool_call(tc_id, tool_name)]))
        msgs.append(_tool_result(tc_id, "x" * result_size))

    msgs.append(_assistant_msg("Done!"))
    return msgs


# ---------------------------------------------------------------------------
# estimate_message_chars
# ---------------------------------------------------------------------------


class TestEstimateMessageChars:
    def test_user_message(self):
        msg = _user_msg("hello world")
        assert estimate_message_chars(msg) == len("hello world")

    def test_assistant_message_with_content(self):
        msg = _assistant_msg("some reply text")
        assert estimate_message_chars(msg) == len("some reply text")

    def test_assistant_message_with_tool_calls(self):
        args = '{"path": "/tmp/foo.txt"}'
        msg = _assistant_with_tool_calls([_make_tool_call("tc1", "read_file", args)])
        # content is "" (0 chars) + arguments length
        assert estimate_message_chars(msg) == len(args)

    def test_tool_result_message(self):
        msg = _tool_result("tc1", "file content here")
        assert estimate_message_chars(msg) == len("file content here")

    def test_system_message(self):
        msg = {"role": "system", "content": "system prompt text"}
        assert estimate_message_chars(msg) == len("system prompt text")

    def test_empty_content(self):
        msg = {"role": "user", "content": ""}
        assert estimate_message_chars(msg) == 0

    def test_missing_content(self):
        msg = {"role": "user"}
        assert estimate_message_chars(msg) == 0


# ---------------------------------------------------------------------------
# estimate_context_chars
# ---------------------------------------------------------------------------


class TestEstimateContextChars:
    def test_full_conversation(self):
        msgs = [
            _user_msg("hello"),  # 5
            _assistant_msg("world"),  # 5
        ]
        assert estimate_context_chars(msgs) == 5 + 5

    def test_empty(self):
        assert estimate_context_chars([]) == 0


# ---------------------------------------------------------------------------
# prune_context — soft trim
# ---------------------------------------------------------------------------


class TestSoftTrim:
    def test_old_tool_results_trimmed_above_soft_ratio(self):
        """Old large tool results should be soft-trimmed when ratio > 0.3."""
        big_content = "A" * 6000  # > SOFT_TRIM_MAX_CHARS
        msgs = [
            _user_msg("u"),
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", big_content),
            # 3 more recent assistant messages to protect
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        # Total chars: 1 + 0 + 6000 + 2 + 2 + 2 = ~6007
        context_tokens = 5000
        result = prune_context(msgs, context_tokens)

        # The tool result at index 2 should be trimmed
        tool_content = result[2]["content"]
        assert "..." in tool_content
        assert "Tool result trimmed" in tool_content
        assert len(tool_content) < len(big_content)

    def test_recent_tool_results_not_trimmed(self):
        """Tool results within the last 3 assistant messages are protected."""
        big_content = "B" * 6000
        msgs = [
            _user_msg("u"),
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", big_content),
            _assistant_msg("a2"),
            _assistant_with_tool_calls([_make_tool_call("tc1", "read_file")]),
            _tool_result("tc1", big_content),
            _assistant_msg("a3"),
        ]
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        # Index 2 should be trimmed (before cutoff protection zone)
        assert "..." in result[2]["content"]
        # Index 5 should NOT be trimmed (in protected zone)
        assert result[5]["content"] == big_content

    def test_non_prunable_tools_not_trimmed(self):
        """Tools not in PRUNABLE_TOOLS should never be trimmed."""
        big_content = "C" * 6000
        msgs = [
            _user_msg("u"),
            _assistant_with_tool_calls([_make_tool_call("tc0", "update_memory")]),
            _tool_result("tc0", big_content),
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        # update_memory is not prunable — content should be unchanged
        assert result[2]["content"] == big_content

    def test_small_tool_result_not_trimmed(self):
        """Tool results smaller than SOFT_TRIM_MAX_CHARS stay untouched."""
        small_content = "D" * 100  # well under 4000
        msgs = [
            _user_msg("u" * 3000),
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", small_content),
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        # Make ratio > 0.3 by using a small context window
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        assert result[2]["content"] == small_content


# ---------------------------------------------------------------------------
# prune_context — hard clear
# ---------------------------------------------------------------------------


class TestHardClear:
    def test_old_tool_results_cleared_above_hard_ratio(self):
        """Old tool results replaced with placeholder when ratio > 0.5."""
        big_content = "E" * 20000
        msgs = [
            _user_msg("u" * 5000),
        ]
        for i in range(20):
            tc_id = f"tc_{i}"
            msgs.append(
                _assistant_with_tool_calls([_make_tool_call(tc_id, "exec_command")])
            )
            msgs.append(_tool_result(tc_id, big_content))

        # Protected tail
        msgs.append(_assistant_msg("a1"))
        msgs.append(_assistant_msg("a2"))
        msgs.append(_assistant_msg("a3"))

        # Small context window to push ratio well above 0.5
        context_tokens = 10000  # 40000 char window
        result = prune_context(msgs, context_tokens)

        # At least some tool results should be hard-cleared
        cleared_count = sum(
            1
            for m in result
            if m.get("role") == "tool" and m.get("content") == HARD_CLEAR_PLACEHOLDER
        )
        assert cleared_count > 0

    def test_hard_clear_respects_min_prunable_chars(self):
        """Hard clear should not activate if prunable chars < MIN_PRUNABLE_CHARS."""
        small_content = "F" * 2000
        msgs = [
            _user_msg("u" * 10000),
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", small_content),
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        context_tokens = 8000
        result = prune_context(msgs, context_tokens)

        # Tool result should NOT be hard-cleared (prunable chars < 50000)
        assert result[2]["content"] != HARD_CLEAR_PLACEHOLDER


# ---------------------------------------------------------------------------
# prune_context — protections
# ---------------------------------------------------------------------------


class TestProtections:
    def test_messages_before_first_user_not_pruned(self):
        """Messages before the first user message are never pruned."""
        big_content = "G" * 6000
        msgs = [
            # Imagine a tool result appeared before user (unusual but possible)
            {"role": "tool", "tool_call_id": "tc_pre", "content": big_content},
            _user_msg("first user message"),
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        # The pre-user tool result should be untouched
        assert result[0]["content"] == big_content

    def test_last_3_assistant_messages_protected(self):
        """The last 3 assistant messages and their tool results are protected."""
        big_content = "H" * 6000
        msgs = [
            _user_msg("u"),
            # Old assistant + tool
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", big_content),
            # Protected tail: 3 assistants, the last one has a tool call
            _assistant_msg("a1"),
            _assistant_with_tool_calls([_make_tool_call("tc1", "read_file")]),
            _tool_result("tc1", big_content),
            _assistant_msg("a3"),
        ]
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        # Index 2 (old tool result) should be trimmed
        assert "..." in result[2]["content"]
        # Index 5 (within last 3 assistants) should NOT be trimmed
        assert result[5]["content"] == big_content


# ---------------------------------------------------------------------------
# prune_context — no-op cases
# ---------------------------------------------------------------------------


class TestNoOp:
    def test_small_context_no_pruning(self):
        """No pruning when context is below SOFT_TRIM_RATIO."""
        msgs = [
            _user_msg("hi"),
            _assistant_msg("hello"),
        ]
        # Huge context window so ratio is tiny
        context_tokens = 1_000_000
        result = prune_context(msgs, context_tokens)
        assert result == msgs

    def test_empty_messages(self):
        result = prune_context([], 100000)
        assert result == []

    def test_returns_new_list(self):
        """prune_context must return a new list, not mutate the original."""
        big_content = "I" * 6000
        msgs = [
            _user_msg("u"),
            _assistant_with_tool_calls([_make_tool_call("tc0", "read_file")]),
            _tool_result("tc0", big_content),
            _assistant_msg("a1"),
            _assistant_msg("a2"),
            _assistant_msg("a3"),
        ]
        original = copy.deepcopy(msgs)
        context_tokens = 3000
        result = prune_context(msgs, context_tokens)

        # Original messages should be unchanged
        assert msgs[2]["content"] == original[2]["content"]
        # Result should be a different list object
        assert result is not msgs


# ---------------------------------------------------------------------------
# Compaction threshold
# ---------------------------------------------------------------------------


class TestCompactionThreshold:
    def test_threshold_is_0_6(self):
        """Compaction threshold should be 60% of context window."""
        mock_llm = MagicMock()
        compactor = ContextCompactor(mock_llm, context_window=10000)

        # Mock estimate_tokens to return exactly 60% + 1
        with patch.object(compactor, "estimate_tokens", return_value=6001):
            assert compactor.needs_compaction([]) is True

        # At exactly 60% it should not trigger
        with patch.object(compactor, "estimate_tokens", return_value=6000):
            assert compactor.needs_compaction([]) is False

        # Below 60%
        with patch.object(compactor, "estimate_tokens", return_value=5000):
            assert compactor.needs_compaction([]) is False


# ---------------------------------------------------------------------------
# Compaction timeout
# ---------------------------------------------------------------------------


class TestCompactionTimeout:
    async def test_timeout_returns_fallback(self):
        """compact_with_timeout should return fallback on timeout."""

        async def slow_compaction():
            await asyncio.sleep(10)
            return "should not reach"

        result = await compact_with_timeout(slow_compaction, timeout=0.1)
        assert "timed out" in result.lower()

    async def test_normal_completion(self):
        """compact_with_timeout should return normally when fast enough."""

        async def fast_compaction():
            return "summary text"

        result = await compact_with_timeout(fast_compaction, timeout=5)
        assert result == "summary text"

    async def test_compact_messages_uses_timeout(self):
        """The compact_messages method should use the timeout wrapper."""
        mock_llm = MagicMock()
        compactor = ContextCompactor(mock_llm, context_window=10000)

        # Make _create_summary hang
        async def slow_summary(msgs):
            await asyncio.sleep(10)
            return "slow"

        compactor._create_summary = slow_summary

        msgs = [
            _user_msg("u1"),
            _assistant_msg("a1"),
            _user_msg("u2"),
            _assistant_msg("a2"),
        ]

        with patch("koi.compaction.COMPACTION_TIMEOUT", 0.1):
            result = await compactor.compact_messages(msgs)

        # Summary (possibly timed-out) should be at index 0
        assert len(result) > 0
        summary_msg = result[0]
        assert (
            "timed out" in summary_msg["content"].lower()
            or "summary" in summary_msg["content"].lower()
        )


# ---------------------------------------------------------------------------
# Integration: prunable tools set
# ---------------------------------------------------------------------------


class TestPrunableTools:
    def test_all_expected_tools_prunable(self):
        expected = {
            "read_file",
            "exec_command",
            "web_fetch",
            "web_search",
            "glob_files",
            "grep_files",
        }
        assert PRUNABLE_TOOLS == expected

    def test_read_skill_not_prunable(self):
        assert "read_skill" not in PRUNABLE_TOOLS

    def test_update_memory_not_prunable(self):
        assert "update_memory" not in PRUNABLE_TOOLS
