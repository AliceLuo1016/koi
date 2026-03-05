"""Semantic memory search with embeddings and SQLite storage."""

import hashlib
import json
import logging
import math
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import httpx
import tiktoken

logger = logging.getLogger(__name__)

# Chunking parameters
TARGET_TOKENS = 400
OVERLAP_TOKENS = 80
SNIPPET_MAX_CHARS = 700


@dataclass
class MemoryChunk:
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    text: str
    hash: str  # sha256 of text


@dataclass
class MemorySearchResult:
    path: str  # e.g. "MEMORY.md" or "memory/2026-03-04.md"
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
    score: float  # 0-1 cosine similarity
    snippet: str  # ~700 chars max


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_markdown(
    text: str,
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> list[MemoryChunk]:
    """Split markdown text into overlapping chunks.

    Splits on paragraph boundaries (double newline), falls back to
    line boundaries. Each chunk tracks start/end line numbers (1-indexed).
    """
    if not text.strip():
        return []

    enc = tiktoken.get_encoding("cl100k_base")
    lines = text.split("\n")

    chunks: list[MemoryChunk] = []
    current_lines: list[str] = []
    current_tokens = 0
    chunk_start_line = 1  # 1-indexed

    def _flush(end_line: int) -> None:
        nonlocal current_lines, current_tokens, chunk_start_line
        if not current_lines:
            return
        chunk_text = "\n".join(current_lines)
        if chunk_text.strip():
            chunks.append(
                MemoryChunk(
                    start_line=chunk_start_line,
                    end_line=end_line,
                    text=chunk_text,
                    hash=_sha256(chunk_text),
                )
            )
        # Keep overlap lines from the end
        if overlap_tokens > 0:
            overlap_lines: list[str] = []
            overlap_tok_count = 0
            for line in reversed(current_lines):
                line_toks = len(enc.encode(line))
                if overlap_tok_count + line_toks > overlap_tokens:
                    break
                overlap_lines.insert(0, line)
                overlap_tok_count += line_toks
            current_lines = overlap_lines
            current_tokens = overlap_tok_count
            chunk_start_line = end_line - len(overlap_lines) + 1
        else:
            current_lines = []
            current_tokens = 0
            chunk_start_line = end_line + 1

    for i, line in enumerate(lines):
        line_num = i + 1  # 1-indexed
        line_tokens = len(enc.encode(line))

        # Check if adding this line exceeds target
        if current_tokens + line_tokens > target_tokens and current_lines:
            # Try to split at paragraph boundary (empty line)
            if line.strip() == "":
                _flush(line_num - 1)
                continue
            # Otherwise flush at this line boundary
            _flush(line_num - 1)

        current_lines.append(line)
        current_tokens += line_tokens

    # Flush remaining
    if current_lines:
        chunk_text = "\n".join(current_lines)
        if chunk_text.strip():
            chunks.append(
                MemoryChunk(
                    start_line=chunk_start_line,
                    end_line=len(lines),
                    text=chunk_text,
                    hash=_sha256(chunk_text),
                )
            )

    return chunks


class MemorySearchManager:
    """Manages semantic search over memory files using embeddings."""

    def __init__(
        self,
        koi_dir: Path | None = None,
        provider: str = "openai",
        model: str = "text-embedding-3-small",
        api_key: str = "",
        api_base: str = "",
    ):
        if koi_dir is None:
            koi_dir = Path.cwd() / ".koi"
        self.koi_dir = koi_dir
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("KOI_API_KEY", "")
        self.api_base = api_base
        self._db: sqlite3.Connection | None = None
        self._file_mtimes: dict[str, float] = {}
        self._available = bool(self.api_key)

        if self._available:
            self._init_db()

    @property
    def available(self) -> bool:
        return self._available

    def _db_path(self) -> Path:
        return self.koi_dir / "memory.sqlite"

    def _init_db(self) -> None:
        """Initialize SQLite database and tables."""
        self.koi_dir.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self._db_path()))
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )"""
        )
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_hash TEXT NOT NULL,
                embedding TEXT NOT NULL
            )"""
        )
        self._db.execute(
            """CREATE INDEX IF NOT EXISTS idx_chunks_file
               ON chunks(file_path)"""
        )
        self._db.execute(
            """CREATE INDEX IF NOT EXISTS idx_chunks_hash
               ON chunks(chunk_hash)"""
        )
        self._db.commit()

        # Check if model/provider changed — reindex if so
        stored_provider = self._get_meta("provider")
        stored_model = self._get_meta("model")
        if stored_provider != self.provider or stored_model != self.model:
            logger.info(
                "Embedding config changed (%s/%s -> %s/%s), reindexing",
                stored_provider,
                stored_model,
                self.provider,
                self.model,
            )
            self._db.execute("DELETE FROM chunks")
            self._db.commit()
            self._set_meta("provider", self.provider)
            self._set_meta("model", self.model)

    def _get_meta(self, key: str) -> str | None:
        assert self._db is not None
        row = self._db.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        assert self._db is not None
        self._db.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._db.commit()

    def _memory_files(self) -> list[Path]:
        """List all memory files (MEMORY.md + memory/*.md)."""
        files = []
        memory_md = self.koi_dir / "MEMORY.md"
        if memory_md.exists():
            files.append(memory_md)
        memory_dir = self.koi_dir / "memory"
        if memory_dir.exists():
            for f in sorted(memory_dir.glob("*.md")):
                files.append(f)
        return files

    def _rel_path(self, path: Path) -> str:
        """Get path relative to koi_dir."""
        try:
            return str(path.relative_to(self.koi_dir))
        except ValueError:
            return str(path)

    def sync(self) -> None:
        """Scan memory files and re-index changed/new/deleted ones."""
        if not self._available or self._db is None:
            return

        current_files = self._memory_files()
        current_rel_paths = {self._rel_path(f) for f in current_files}

        # Find indexed file paths
        indexed_paths = {
            row[0]
            for row in self._db.execute(
                "SELECT DISTINCT file_path FROM chunks"
            ).fetchall()
        }

        # Delete chunks for removed files
        removed = indexed_paths - current_rel_paths
        for rp in removed:
            self._db.execute("DELETE FROM chunks WHERE file_path = ?", (rp,))
        if removed:
            self._db.commit()

        # Index new or changed files
        for file_path in current_files:
            rel = self._rel_path(file_path)
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                continue

            cached_mtime = self._file_mtimes.get(rel)
            if cached_mtime is not None and mtime <= cached_mtime:
                continue

            self._index_file(file_path, rel)
            self._file_mtimes[rel] = mtime

    def _index_file(self, file_path: Path, rel_path: str) -> None:
        """Index a single memory file, only re-embedding changed chunks."""
        assert self._db is not None
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            return

        new_chunks = chunk_markdown(text)
        new_hashes = {c.hash for c in new_chunks}

        # Get existing chunk hashes for this file
        existing = {
            row[0]: row[1]
            for row in self._db.execute(
                "SELECT chunk_hash, id FROM chunks WHERE file_path = ?",
                (rel_path,),
            ).fetchall()
        }
        existing_hashes = set(existing.keys())

        # Delete chunks that no longer exist
        removed_hashes = existing_hashes - new_hashes
        for h in removed_hashes:
            self._db.execute("DELETE FROM chunks WHERE id = ?", (existing[h],))

        # Find chunks that need embedding (new ones)
        chunks_to_embed = [c for c in new_chunks if c.hash not in existing_hashes]

        if chunks_to_embed:
            texts = [c.text for c in chunks_to_embed]
            embeddings = self._get_embeddings(texts)
            if embeddings:
                for chunk, emb in zip(chunks_to_embed, embeddings):
                    self._db.execute(
                        """INSERT INTO chunks
                           (file_path, start_line, end_line,
                            chunk_text, chunk_hash, embedding)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            rel_path,
                            chunk.start_line,
                            chunk.end_line,
                            chunk.text,
                            chunk.hash,
                            json.dumps(emb),
                        ),
                    )

        # Update line numbers for existing chunks that weren't removed
        for chunk in new_chunks:
            if chunk.hash in existing_hashes:
                self._db.execute(
                    """UPDATE chunks SET start_line = ?, end_line = ?
                       WHERE file_path = ? AND chunk_hash = ?""",
                    (chunk.start_line, chunk.end_line, rel_path, chunk.hash),
                )

        self._db.commit()

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings from OpenAI-compatible API."""
        if not texts:
            return []

        base = self.api_base or "https://api.openai.com/v1"
        base = base.rstrip("/")
        url = f"{base}/embeddings"

        try:
            response = httpx.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"input": texts, "model": self.model},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            # Sort by index to ensure order matches input
            sorted_data = sorted(data["data"], key=lambda x: x["index"])
            return [item["embedding"] for item in sorted_data]
        except Exception as e:
            logger.error("Failed to get embeddings: %s", e)
            return []

    def search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Search memory using semantic similarity."""
        if not self._available or self._db is None:
            return []

        # Sync before searching
        self.sync()

        # Embed the query
        embeddings = self._get_embeddings([query])
        if not embeddings:
            return []
        query_embedding = embeddings[0]

        # Compare against all stored chunks
        rows = self._db.execute(
            "SELECT file_path, start_line, end_line, chunk_text, embedding FROM chunks"
        ).fetchall()

        scored: list[MemorySearchResult] = []
        for file_path, start_line, end_line, chunk_text, emb_json in rows:
            chunk_embedding = json.loads(emb_json)
            score = _cosine_similarity(query_embedding, chunk_embedding)
            if score < min_score:
                continue
            snippet = chunk_text[:SNIPPET_MAX_CHARS]
            scored.append(
                MemorySearchResult(
                    path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    score=round(score, 4),
                    snippet=snippet,
                )
            )

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:max_results]

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
