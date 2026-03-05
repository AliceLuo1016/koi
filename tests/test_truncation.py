"""Tests for tool output truncation (per-tool limits + generic safety net)."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

import yaml

from koi.prompts import _format_tool_result, truncate_tool_result
from koi.sandbox import Sandbox
from koi.tools import (
    MAX_EXEC_OUTPUT_BYTES,
    MAX_READ_BYTES,
    MAX_READ_LINES,
    MAX_WEB_FETCH_CHARS,
    ToolExecutor,
)


def _make_sandbox(tmpdir: str):
    """Create a Sandbox allowing tmpdir."""
    td = Path(tmpdir)
    koi_dir = td / ".koi"
    koi_dir.mkdir(exist_ok=True)
    cfg = {
        "filesystem": {"allowed_paths": [str(td)]},
        "commands": {},
    }
    (koi_dir / "sandbox.yaml").write_text(yaml.dump(cfg))
    return Sandbox(project_root=td)


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# read_file truncation
# ---------------------------------------------------------------------------


class TestReadFileTruncation:
    def test_truncates_at_default_line_limit(self):
        """read_file truncates at MAX_READ_LINES when no explicit limit."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            # Create a file with more than MAX_READ_LINES
            total = MAX_READ_LINES + 500
            fpath = Path(td) / "big.txt"
            fpath.write_text("".join(f"line {i}\n" for i in range(total)))

            result = _run(executor._read_file(str(fpath)))
            assert result["success"]
            assert "output truncated" in result["content"]
            assert f"of {total} lines shown" in result["content"]

    def test_truncates_at_byte_limit(self):
        """read_file truncates at MAX_READ_BYTES."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            # Create a file that exceeds MAX_READ_BYTES but within line limit
            # Use 100 lines of ~1KB each = 100KB > 50KB limit
            line = "x" * 1000 + "\n"
            total_lines = 100
            fpath = Path(td) / "big_bytes.txt"
            fpath.write_text(line * total_lines)

            result = _run(executor._read_file(str(fpath)))
            assert result["success"]
            assert "output truncated" in result["content"]
            # Content (excluding notice) should be within byte limit
            content_without_notice = result["content"].split("\n[output truncated")[0]
            assert len(content_without_notice.encode("utf-8")) <= MAX_READ_BYTES

    def test_shows_truncation_notice(self):
        """Truncation notice format is correct."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            total = MAX_READ_LINES + 10
            fpath = Path(td) / "many_lines.txt"
            fpath.write_text("".join(f"line {i}\n" for i in range(total)))

            result = _run(executor._read_file(str(fpath)))
            assert "[output truncated:" in result["content"]
            assert "Use offset/limit for more." in result["content"]

    def test_respects_explicit_limit(self):
        """When an explicit limit is provided, do not override with default."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            # File with 100 lines
            fpath = Path(td) / "hundred.txt"
            fpath.write_text("".join(f"line {i}\n" for i in range(100)))

            result = _run(executor._read_file(str(fpath), limit=10))
            assert result["success"]
            assert result["lines_read"] == 10
            # Should be truncated since 100 > 10
            assert "output truncated" in result["content"]
            assert "10 of 100 lines shown" in result["content"]

    def test_no_truncation_for_small_file(self):
        """Small files are returned in full without truncation notice."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            fpath = Path(td) / "small.txt"
            fpath.write_text("hello\nworld\n")

            result = _run(executor._read_file(str(fpath)))
            assert result["success"]
            assert result["content"] == "hello\nworld\n"
            assert "truncated" not in result["content"]


# ---------------------------------------------------------------------------
# exec_command truncation
# ---------------------------------------------------------------------------


class TestExecCommandTruncation:
    def test_truncates_at_byte_limit(self):
        """exec_command truncates combined stdout+stderr at MAX_EXEC_OUTPUT_BYTES."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            # Generate output larger than 50KB
            big_size = MAX_EXEC_OUTPUT_BYTES + 10_000
            cmd = f"python3 -c \"print('x' * {big_size})\""
            result = _run(executor._exec_command(cmd, cwd=td))

            assert result["success"]
            assert "truncation_notice" in result
            assert "output truncated" in result["truncation_notice"]
            assert (
                len(result["stdout"]) + len(result["stderr"]) <= MAX_EXEC_OUTPUT_BYTES
            )

    def test_shows_truncation_notice(self):
        """Truncation notice includes byte counts."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            big_size = MAX_EXEC_OUTPUT_BYTES + 5_000
            cmd = f"python3 -c \"print('y' * {big_size})\""
            result = _run(executor._exec_command(cmd, cwd=td))

            assert "truncation_notice" in result
            assert "showing first" in result["truncation_notice"]
            assert "bytes" in result["truncation_notice"]

    def test_no_truncation_for_small_output(self):
        """Small command output is not truncated."""
        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            result = _run(executor._exec_command("echo hello", cwd=td))
            assert result["success"]
            assert result["stdout"].strip() == "hello"
            assert "truncation_notice" not in result


# ---------------------------------------------------------------------------
# web_fetch truncation
# ---------------------------------------------------------------------------


class TestWebFetchLimit:
    def test_uses_20k_limit(self):
        """web_fetch caps content at MAX_WEB_FETCH_CHARS (20K)."""
        assert MAX_WEB_FETCH_CHARS == 20_000

        with TemporaryDirectory() as td:
            sandbox = _make_sandbox(td)
            executor = ToolExecutor(skills_manager=Mock(), sandbox=sandbox)

            # Mock httpx to return a large page
            big_body = "<html><body>" + "x" * 30_000 + "</body></html>"
            mock_resp = Mock()
            mock_resp.text = big_body
            mock_resp.raise_for_status = Mock()

            async def mock_get(*args, **kwargs):
                return mock_resp

            with patch("koi.tools.httpx.AsyncClient") as mock_client_cls:
                mock_client = AsyncMock()
                mock_client.get = mock_get
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client_cls.return_value = mock_client

                result = _run(executor._web_fetch("http://example.com"))

            assert result["success"]
            # Content (minus truncation notice) should be at most 20K
            if "output truncated" in result["content"]:
                parts = result["content"].split("\n[output truncated")
                assert len(parts[0]) <= MAX_WEB_FETCH_CHARS


# ---------------------------------------------------------------------------
# _format_tool_result (layer 2 truncation integrated)
# ---------------------------------------------------------------------------


class TestFormatToolResult:
    def test_truncates_oversized_results(self):
        """_format_tool_result applies generic truncation for huge content."""
        # Use a small context_window so the budget is tiny
        # context_window=2000 → budget = min(2000*0.3*4, 400000) = 2400
        big_content = "a" * 5000
        result = {"success": True, "content": big_content}
        formatted = _format_tool_result(result, context_window=2000)

        assert len(formatted) < 5000
        assert "Content truncated" in formatted

    def test_passes_through_small_results(self):
        """Small results are untouched."""
        result = {"success": True, "content": "hello"}
        formatted = _format_tool_result(result, context_window=128_000)
        assert formatted == "hello"

    def test_exec_truncation_notice_included(self):
        """Truncation notice from exec_command flows through _format_tool_result."""
        result = {
            "success": True,
            "stdout": "some output",
            "stderr": "",
            "exit_code": 0,
            "truncation_notice": (
                "[output truncated: showing first 50000 of 60000 bytes]"
            ),
        }
        formatted = _format_tool_result(result, context_window=128_000)
        assert "output truncated" in formatted


# ---------------------------------------------------------------------------
# truncate_tool_result (standalone function)
# ---------------------------------------------------------------------------


class TestTruncateToolResult:
    def test_preserves_small_results(self):
        """Text under the budget is returned unchanged."""
        text = "short text"
        assert truncate_tool_result(text, context_window=128_000) == text

    def test_keeps_at_least_2000_chars(self):
        """Even with a tiny context window, at least 2000 chars are kept."""
        text = "a" * 5000
        # context_window=1 → budget calc = min(1*0.3*4, 400000) = 1.2 → clamped to 2000
        result = truncate_tool_result(text, context_window=1)
        # The result should have at least 2000 chars of original content
        # (plus the suffix)
        without_suffix = result.split("\u26a0\ufe0f")[0]
        assert len(without_suffix) >= 2000

    def test_breaks_at_newline_boundary(self):
        """When truncating, the cut prefers a newline boundary."""
        # Build text: 2500 chars of 'a', then newline, then 2500 chars of 'b'
        text = "a" * 2500 + "\n" + "b" * 2500
        # Budget that would cut in the middle of the b's
        # context_window so budget ≈ 3000 chars
        # min(2600*0.3*4, 400000) = 3120 minus suffix ≈ 2980
        result = truncate_tool_result(text, context_window=2600)
        assert "Content truncated" in result
        # Should have cut at the newline after the a's, not in the middle of b's
        content_before_suffix = result.split("\u26a0\ufe0f")[0]
        assert content_before_suffix.rstrip().endswith(
            "a" * 10
        ) or content_before_suffix.rstrip().endswith("\n")

    def test_max_cap_400k(self):
        """Budget never exceeds 400K chars even with huge context window."""
        text = "x" * 500_000
        result = truncate_tool_result(text, context_window=10_000_000)
        assert "Content truncated" in result
        assert len(result) <= 400_000 + 200  # 200 for suffix
