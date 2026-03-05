"""Tests for memory search module."""

import math
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from koi.memory import Memory
from koi.memory_search import (
    MemorySearchManager,
    MemorySearchResult,
    _cosine_similarity,
    _sha256,
    chunk_markdown,
)

# ── Chunking tests ────────────────────────────────────


class TestChunkMarkdown:
    def test_empty_text(self):
        assert chunk_markdown("") == []

    def test_whitespace_only(self):
        assert chunk_markdown("   \n\n  ") == []

    def test_single_short_chunk(self):
        text = "Hello world\nThis is a test."
        chunks = chunk_markdown(text, target_tokens=1000, overlap_tokens=0)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2
        assert chunks[0].text == text
        assert chunks[0].hash == _sha256(text)

    def test_multiple_chunks(self):
        # Build text that exceeds target_tokens
        lines = [f"Line {i}: " + "word " * 20 for i in range(30)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, target_tokens=100, overlap_tokens=0)
        assert len(chunks) > 1
        # First chunk starts at line 1
        assert chunks[0].start_line == 1
        # Last chunk ends at the last line
        assert chunks[-1].end_line == 30

    def test_overlap_creates_shared_content(self):
        lines = [f"Line {i}: " + "word " * 15 for i in range(20)]
        text = "\n".join(lines)
        chunks = chunk_markdown(text, target_tokens=80, overlap_tokens=30)
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            assert chunks[1].start_line <= chunks[0].end_line + 1

    def test_chunk_hashes_are_sha256(self):
        text = "Hello world\n\nSecond paragraph."
        chunks = chunk_markdown(text, target_tokens=1000, overlap_tokens=0)
        for chunk in chunks:
            assert chunk.hash == _sha256(chunk.text)

    def test_line_tracking(self):
        text = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunk_markdown(text, target_tokens=1000, overlap_tokens=0)
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5

    def test_paragraph_boundary_splitting(self):
        # Create text with clear paragraph boundaries
        para1 = "word " * 40
        para2 = "text " * 40
        text = para1.strip() + "\n\n" + para2.strip()
        chunks = chunk_markdown(text, target_tokens=50, overlap_tokens=0)
        assert len(chunks) >= 2


# ── Cosine similarity tests ───────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(_cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(_cosine_similarity(a, b) + 1.0) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 2.0]
        assert _cosine_similarity(a, b) == 0.0

    def test_known_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        expected = 1.0 / math.sqrt(2)
        assert abs(_cosine_similarity(a, b) - expected) < 1e-6


# ── MemorySearchManager tests ─────────────────────────


def _make_embedding(dim: int = 4, base: float = 0.1) -> list[float]:
    """Create a simple deterministic embedding vector."""
    return [base * (i + 1) for i in range(dim)]


def _mock_embeddings(texts: list[str]) -> list[list[float]]:
    """Mock embeddings: each text gets a unique-ish vector based on length."""
    results = []
    for i, text in enumerate(texts):
        base = (len(text) % 10 + 1) * 0.1
        results.append([base + j * 0.01 for j in range(4)])
    return results


class TestMemorySearchManager:
    def test_unavailable_without_api_key(self):
        with TemporaryDirectory() as tmp:
            mgr = MemorySearchManager(koi_dir=Path(tmp), api_key="")
            assert not mgr.available

    def test_available_with_api_key(self):
        with TemporaryDirectory() as tmp:
            mgr = MemorySearchManager(koi_dir=Path(tmp), api_key="test-key")
            assert mgr.available
            mgr.close()

    def test_db_created(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            assert (koi_dir / "memory.sqlite").exists()
            mgr.close()

    def test_meta_table_stores_provider_model(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mgr = MemorySearchManager(
                koi_dir=koi_dir,
                api_key="test-key",
                provider="openai",
                model="text-embedding-3-small",
            )
            assert mgr._get_meta("provider") == "openai"
            assert mgr._get_meta("model") == "text-embedding-3-small"
            mgr.close()

    def test_reindex_on_model_change(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            # Create initial manager
            mgr1 = MemorySearchManager(
                koi_dir=koi_dir,
                api_key="test-key",
                provider="openai",
                model="model-a",
            )
            # Manually insert a chunk
            mgr1._db.execute(
                """INSERT INTO chunks
                   (file_path, start_line, end_line,
                    chunk_text, chunk_hash, embedding)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                ("test.md", 1, 1, "text", "hash", "[]"),
            )
            mgr1._db.commit()
            count = mgr1._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            assert count == 1
            mgr1.close()

            # Re-open with different model — should clear chunks
            mgr2 = MemorySearchManager(
                koi_dir=koi_dir,
                api_key="test-key",
                provider="openai",
                model="model-b",
            )
            count = mgr2._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            assert count == 0
            mgr2.close()


class TestMemorySearchSync:
    def test_sync_indexes_new_files(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Hello world\n\nSome memory content.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            with patch.object(mgr, "_get_embeddings", side_effect=_mock_embeddings):
                mgr.sync()

            count = mgr._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            assert count > 0
            mgr.close()

    def test_sync_removes_deleted_files(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Some content.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            with patch.object(mgr, "_get_embeddings", side_effect=_mock_embeddings):
                mgr.sync()
                assert mgr._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] > 0

                # Delete the file and re-sync
                mem_file.unlink()
                mgr.sync()
                count = mgr._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                assert count == 0
            mgr.close()

    def test_sync_reindexes_changed_files(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Original content.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            embed_calls = []

            def track_embeddings(texts):
                embed_calls.append(texts)
                return _mock_embeddings(texts)

            with patch.object(mgr, "_get_embeddings", side_effect=track_embeddings):
                mgr.sync()
                first_call_count = len(embed_calls)

                # Modify file (need to also bump mtime cache)
                import time

                time.sleep(0.05)
                mem_file.write_text("Modified content that is different.")
                mgr._file_mtimes.clear()
                mgr.sync()
                assert len(embed_calls) > first_call_count
            mgr.close()

    def test_sync_skips_unchanged_files(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Stable content.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            embed_calls = []

            def track_embeddings(texts):
                embed_calls.append(texts)
                return _mock_embeddings(texts)

            with patch.object(mgr, "_get_embeddings", side_effect=track_embeddings):
                mgr.sync()
                first_call_count = len(embed_calls)
                # Sync again without changes — no new embeddings
                mgr.sync()
                assert len(embed_calls) == first_call_count
            mgr.close()

    def test_sync_indexes_daily_logs(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_dir = koi_dir / "memory"
            mem_dir.mkdir(parents=True)
            (mem_dir / "2026-03-04.md").write_text("Daily log entry.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")
            with patch.object(mgr, "_get_embeddings", side_effect=_mock_embeddings):
                mgr.sync()

            rows = mgr._db.execute("SELECT file_path FROM chunks").fetchall()
            paths = [r[0] for r in rows]
            assert "memory/2026-03-04.md" in paths
            mgr.close()


class TestMemorySearch:
    def test_search_returns_results(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text(
                "The user prefers dark mode.\n\nThe project uses Python 3.12."
            )

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")

            # Use consistent mock embeddings
            call_count = [0]

            def mock_embed(texts):
                results = []
                for text in texts:
                    call_count[0] += 1
                    if "dark mode" in text:
                        results.append([0.9, 0.1, 0.1, 0.1])
                    elif "Python" in text:
                        results.append([0.1, 0.9, 0.1, 0.1])
                    else:
                        results.append([0.5, 0.5, 0.5, 0.5])
                return results

            with patch.object(mgr, "_get_embeddings", side_effect=mock_embed):
                results = mgr.search("dark mode preference")

            assert len(results) > 0
            assert all(isinstance(r, MemorySearchResult) for r in results)
            assert all(0 <= r.score <= 1 for r in results)
            mgr.close()

    def test_search_empty_when_unavailable(self):
        with TemporaryDirectory() as tmp:
            mgr = MemorySearchManager(koi_dir=Path(tmp), api_key="")
            results = mgr.search("anything")
            assert results == []

    def test_search_respects_min_score(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Some content.\n\nMore content.")

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")

            def mock_embed(texts):
                return [[0.1, 0.1, 0.1, 0.1]] * len(texts)

            with patch.object(mgr, "_get_embeddings", side_effect=mock_embed):
                # With very high min_score, should filter out results
                results = mgr.search("query", min_score=0.99)
                # Results should be filtered (or empty if scores < 0.99)
                for r in results:
                    assert r.score >= 0.99
            mgr.close()

    def test_search_respects_max_results(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            lines = [f"Paragraph {i}.\n" for i in range(20)]
            mem_file.write_text("\n".join(lines))

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")

            def mock_embed(texts):
                return [[0.5, 0.5, 0.5, 0.5]] * len(texts)

            with patch.object(mgr, "_get_embeddings", side_effect=mock_embed):
                results = mgr.search("query", max_results=2)
                assert len(results) <= 2
            mgr.close()

    def test_search_snippet_length(self):
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp)
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("x " * 500)

            mgr = MemorySearchManager(koi_dir=koi_dir, api_key="test-key")

            def mock_embed(texts):
                return [[0.5, 0.5, 0.5, 0.5]] * len(texts)

            with patch.object(mgr, "_get_embeddings", side_effect=mock_embed):
                results = mgr.search("query")
                for r in results:
                    assert len(r.snippet) <= 700
            mgr.close()


# ── Tool integration tests ────────────────────────────


class TestMemorySearchTool:
    @pytest.mark.asyncio
    async def test_memory_search_unavailable(self):
        from koi.tools import ToolExecutor

        executor = ToolExecutor(
            skills_manager=MagicMock(),
            memory_search_manager=None,
        )
        result = await executor._memory_search(query="test")
        assert not result["success"]
        assert "unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_memory_search_with_manager(self):
        from koi.tools import ToolExecutor

        mock_mgr = MagicMock()
        mock_mgr.available = True
        mock_mgr.search.return_value = [
            MemorySearchResult(
                path="MEMORY.md",
                start_line=1,
                end_line=5,
                score=0.85,
                snippet="test snippet",
            )
        ]

        executor = ToolExecutor(
            skills_manager=MagicMock(),
            memory_search_manager=mock_mgr,
        )
        result = await executor._memory_search(query="test")
        assert result["success"]
        assert result["count"] == 1
        assert result["results"][0]["score"] == 0.85


class TestMemoryGetTool:
    @pytest.mark.asyncio
    async def test_read_memory_file(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Line 1\nLine 2\nLine 3\n")

            executor = ToolExecutor(skills_manager=MagicMock())
            with patch("koi.tools.Path.cwd", return_value=Path(tmp)):
                result = await executor._memory_get(path="MEMORY.md")
            assert result["success"]
            assert "Line 1" in result["text"]

    @pytest.mark.asyncio
    async def test_read_with_line_range(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()
            mem_file = koi_dir / "MEMORY.md"
            mem_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\n")

            executor = ToolExecutor(skills_manager=MagicMock())
            with patch("koi.tools.Path.cwd", return_value=Path(tmp)):
                result = await executor._memory_get(
                    path="MEMORY.md", from_line=2, num_lines=2
                )
            assert result["success"]
            assert "Line 2" in result["text"]
            assert "Line 3" in result["text"]
            assert "Line 1" not in result["text"]

    @pytest.mark.asyncio
    async def test_reject_non_memory_paths(self):
        from koi.tools import ToolExecutor

        executor = ToolExecutor(skills_manager=MagicMock())
        result = await executor._memory_get(path="../etc/passwd")
        assert not result["success"]
        assert "denied" in result["error"]

    @pytest.mark.asyncio
    async def test_reject_traversal(self):
        from koi.tools import ToolExecutor

        executor = ToolExecutor(skills_manager=MagicMock())
        result = await executor._memory_get(path="memory/../../../etc/passwd")
        assert not result["success"]

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()

            executor = ToolExecutor(skills_manager=MagicMock())
            with patch("koi.tools.Path.cwd", return_value=Path(tmp)):
                result = await executor._memory_get(path="memory/2099-01-01.md")
            assert result["success"]
            assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_read_daily_log(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            mem_dir = koi_dir / "memory"
            mem_dir.mkdir(parents=True)
            daily = mem_dir / "2026-03-04.md"
            daily.write_text("Daily log content.")

            executor = ToolExecutor(skills_manager=MagicMock())
            with patch("koi.tools.Path.cwd", return_value=Path(tmp)):
                result = await executor._memory_get(path="memory/2026-03-04.md")
            assert result["success"]
            assert "Daily log content" in result["text"]


class TestUpdateMemoryTool:
    @pytest.mark.asyncio
    async def test_write_to_daily_log(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()
            mem_path = koi_dir / "MEMORY.md"

            executor = ToolExecutor(skills_manager=MagicMock())
            memory = Memory(mem_path)
            with patch("koi.memory.Memory", return_value=memory):
                result = await executor._update_memory(
                    content="daily note", target="daily"
                )
            assert result["success"]
            assert "daily log" in result["message"]

    @pytest.mark.asyncio
    async def test_write_to_long_term(self):
        from koi.tools import ToolExecutor

        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()
            mem_path = koi_dir / "MEMORY.md"

            executor = ToolExecutor(skills_manager=MagicMock())
            memory = Memory(mem_path)
            with patch("koi.memory.Memory", return_value=memory):
                result = await executor._update_memory(
                    content="important fact", target="long_term"
                )
            assert result["success"]
            assert "MEMORY.md" in result["message"]
            assert mem_path.exists()
            assert "important fact" in mem_path.read_text()

    @pytest.mark.asyncio
    async def test_triggers_reindex(self):
        from koi.tools import ToolExecutor

        mock_mgr = MagicMock()
        with TemporaryDirectory() as tmp:
            koi_dir = Path(tmp) / ".koi"
            koi_dir.mkdir()
            mem_path = koi_dir / "MEMORY.md"

            executor = ToolExecutor(
                skills_manager=MagicMock(),
                memory_search_manager=mock_mgr,
            )
            memory = Memory(mem_path)
            with patch("koi.memory.Memory", return_value=memory):
                await executor._update_memory(content="note", target="long_term")
            mock_mgr.sync.assert_called_once()


# ── Config tests ──────────────────────────────────────


class TestMemorySearchConfig:
    def test_graceful_degradation_no_api_key(self):
        with TemporaryDirectory() as tmp:
            mgr = MemorySearchManager(koi_dir=Path(tmp), api_key="")
            assert not mgr.available
            assert mgr.search("anything") == []

    def test_config_fields_loaded(self):
        from koi.config import Config

        config = Config(
            memory_search={
                "provider": "custom",
                "model": "my-model",
                "api_key": "my-key",
                "api_base": "https://example.com/v1",
            }
        )
        assert config.memory_search_provider == "custom"
        assert config.memory_search_model == "my-model"
        assert config.memory_search_api_key == "my-key"
        assert config.memory_search_api_base == "https://example.com/v1"

    def test_config_defaults(self):
        from koi.config import Config

        config = Config()
        assert config.memory_search_provider == "openai"
        assert config.memory_search_model == "text-embedding-3-small"
        assert config.memory_search_api_key == ""
        assert config.memory_search_api_base == ""


# ── Daily log tests (Memory class) ───────────────────


class TestDailyLogs:
    def test_append_daily(self):
        with TemporaryDirectory() as tmp:
            mem_path = Path(tmp) / "MEMORY.md"
            memory = Memory(mem_path)
            today = date.today()
            memory.append_daily("Test entry")
            daily_path = Path(tmp) / "memory" / f"{today.isoformat()}.md"
            assert daily_path.exists()
            assert "Test entry" in daily_path.read_text()

    def test_load_daily(self):
        with TemporaryDirectory() as tmp:
            mem_path = Path(tmp) / "MEMORY.md"
            memory = Memory(mem_path)
            today = date.today()
            memory.append_daily("First entry")
            memory.append_daily("Second entry")
            content = memory.load_daily(today)
            assert "First entry" in content
            assert "Second entry" in content

    def test_load_daily_missing(self):
        with TemporaryDirectory() as tmp:
            mem_path = Path(tmp) / "MEMORY.md"
            memory = Memory(mem_path)
            content = memory.load_daily(date(2099, 1, 1))
            assert content == ""

    def test_load_recent_daily(self):
        with TemporaryDirectory() as tmp:
            mem_path = Path(tmp) / "MEMORY.md"
            memory = Memory(mem_path)
            today = date.today()
            memory.append_daily("Today's note", today)
            content = memory.load_recent_daily()
            assert "Today's note" in content


# ── Tool definitions test ─────────────────────────────


class TestToolDefinitions:
    def test_memory_search_tool_defined(self):
        from koi.tools import get_tool_definitions

        tools = get_tool_definitions()
        names = [t["function"]["name"] for t in tools]
        assert "memory_search" in names

    def test_memory_get_tool_defined(self):
        from koi.tools import get_tool_definitions

        tools = get_tool_definitions()
        names = [t["function"]["name"] for t in tools]
        assert "memory_get" in names

    def test_update_memory_has_target_param(self):
        from koi.tools import get_tool_definitions

        tools = get_tool_definitions()
        update_memory = next(
            t for t in tools if t["function"]["name"] == "update_memory"
        )
        props = update_memory["function"]["parameters"]["properties"]
        assert "target" in props
        assert "daily" in props["target"]["enum"]
        assert "long_term" in props["target"]["enum"]
