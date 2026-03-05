"""Semantic memory search with embeddings and SQLite storage."""

import hashlib
import json
import math
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx
import tiktoken
from loguru import logger

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


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts based on word tokens."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


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
        *,
        hybrid_enabled: bool = True,
        vector_weight: float = 0.7,
        text_weight: float = 0.3,
        temporal_decay_enabled: bool = True,
        temporal_decay_half_life_days: int = 30,
        mmr_enabled: bool = True,
        mmr_lambda: float = 0.7,
        cache_enabled: bool = True,
        cache_max_entries: int = 50000,
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
        self._has_api_key = bool(self.api_key)
        # Available means at least keyword search works (always true now)
        self._available = True

        # Hybrid search config
        self.hybrid_enabled = hybrid_enabled
        self.vector_weight = vector_weight
        self.text_weight = text_weight

        # Temporal decay config
        self.temporal_decay_enabled = temporal_decay_enabled
        self.temporal_decay_half_life_days = temporal_decay_half_life_days

        # MMR config
        self.mmr_enabled = mmr_enabled
        self.mmr_lambda = mmr_lambda

        # Embedding cache config
        self.cache_enabled = cache_enabled
        self.cache_max_entries = cache_max_entries

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

        # FTS5 full-text search table
        self._db.execute(
            """CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text, id UNINDEXED, path UNINDEXED,
                start_line UNINDEXED, end_line UNINDEXED
            )"""
        )

        # Embedding cache table
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS embedding_cache (
                hash TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                updated_at INTEGER NOT NULL
            )"""
        )

        self._db.commit()

        # Check if model/provider changed — reindex if so
        stored_provider = self._get_meta("provider")
        stored_model = self._get_meta("model")
        if stored_provider != self.provider or stored_model != self.model:
            logger.info(
                "Embedding config changed ({}/{} -> {}/{}), reindexing",
                stored_provider,
                stored_model,
                self.provider,
                self.model,
            )
            self._db.execute("DELETE FROM chunks")
            self._db.execute("DELETE FROM chunks_fts")
            self._db.commit()
            self._set_meta("provider", self.provider)
            self._set_meta("model", self.model)

    def _get_meta(self, key: str) -> str | None:
        assert self._db is not None
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
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
        indexed_paths = {row[0] for row in self._db.execute("SELECT DISTINCT file_path FROM chunks").fetchall()}

        # Delete chunks for removed files
        removed = indexed_paths - current_rel_paths
        for rp in removed:
            # Get chunk IDs before deleting for FTS cleanup
            chunk_ids = [
                row[0] for row in self._db.execute("SELECT id FROM chunks WHERE file_path = ?", (rp,)).fetchall()
            ]
            self._db.execute("DELETE FROM chunks WHERE file_path = ?", (rp,))
            for cid in chunk_ids:
                self._db.execute("DELETE FROM chunks_fts WHERE id = ?", (str(cid),))
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
            chunk_id = existing[h]
            self._db.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
            self._db.execute("DELETE FROM chunks_fts WHERE id = ?", (str(chunk_id),))

        # Find chunks that need embedding (new ones)
        chunks_to_embed = [c for c in new_chunks if c.hash not in existing_hashes]

        if chunks_to_embed:
            texts = [c.text for c in chunks_to_embed]
            embeddings = self._get_embeddings(texts)
            # If no embeddings (no API key), store empty embeddings
            if not embeddings:
                embeddings = ["[]"] * len(texts)
                use_empty = True
            else:
                use_empty = False

            for chunk, emb in zip(chunks_to_embed, embeddings):
                emb_json = emb if use_empty else json.dumps(emb)
                cursor = self._db.execute(
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
                        emb_json,
                    ),
                )
                # Insert into FTS5 table
                chunk_id = cursor.lastrowid
                self._db.execute(
                    """INSERT INTO chunks_fts (text, id, path, start_line, end_line)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        chunk.text,
                        str(chunk_id),
                        rel_path,
                        str(chunk.start_line),
                        str(chunk.end_line),
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
        """Get embeddings from OpenAI-compatible API, with caching."""
        if not texts:
            return []
        if not self._has_api_key:
            return []

        assert self._db is not None
        results: list[list[float] | None] = [None] * len(texts)
        texts_to_fetch: list[tuple[int, str, str]] = []  # (index, text, hash)

        # Check cache first
        if self.cache_enabled:
            for i, text in enumerate(texts):
                text_hash = _sha256(text)
                row = self._db.execute(
                    "SELECT embedding FROM embedding_cache WHERE hash = ?",
                    (text_hash,),
                ).fetchone()
                if row:
                    results[i] = json.loads(row[0])
                    # Update timestamp for LRU
                    self._db.execute(
                        "UPDATE embedding_cache SET updated_at = ? WHERE hash = ?",
                        (int(time.time()), text_hash),
                    )
                else:
                    texts_to_fetch.append((i, text, text_hash))
        else:
            for i, text in enumerate(texts):
                texts_to_fetch.append((i, text, _sha256(text)))

        # Fetch missing embeddings from API
        if texts_to_fetch:
            fetch_texts = [t[1] for t in texts_to_fetch]
            fetched = self._fetch_embeddings_api(fetch_texts)
            if fetched:
                now = int(time.time())
                for (idx, _text, text_hash), emb in zip(texts_to_fetch, fetched):
                    results[idx] = emb
                    # Store in cache
                    if self.cache_enabled:
                        self._db.execute(
                            """INSERT OR REPLACE INTO embedding_cache
                               (hash, embedding, updated_at)
                               VALUES (?, ?, ?)""",
                            (text_hash, json.dumps(emb), now),
                        )
                if self.cache_enabled:
                    self._prune_cache()
                    self._db.commit()
            else:
                return []

        # Filter out any None entries (shouldn't happen if API succeeded)
        final = [r for r in results if r is not None]
        if len(final) != len(texts):
            return []
        return final

    def _fetch_embeddings_api(self, texts: list[str]) -> list[list[float]]:
        """Fetch embeddings from the API (no caching)."""
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
            logger.error("Failed to get embeddings: {}", e)
            return []

    def _prune_cache(self) -> None:
        """Prune embedding cache to max_entries using LRU."""
        assert self._db is not None
        count = self._db.execute("SELECT COUNT(*) FROM embedding_cache").fetchone()[0]
        if count > self.cache_max_entries:
            excess = count - self.cache_max_entries
            self._db.execute(
                """DELETE FROM embedding_cache WHERE hash IN (
                    SELECT hash FROM embedding_cache
                    ORDER BY updated_at ASC LIMIT ?
                )""",
                (excess,),
            )

    def _search_vector(self, query: str, max_results: int) -> list[MemorySearchResult]:
        """Search using vector similarity."""
        if not self._has_api_key or self._db is None:
            return []

        embeddings = self._get_embeddings([query])
        if not embeddings:
            return []
        query_embedding = embeddings[0]

        rows = self._db.execute("SELECT file_path, start_line, end_line, chunk_text, embedding FROM chunks").fetchall()

        scored: list[MemorySearchResult] = []
        for file_path, start_line, end_line, chunk_text, emb_json in rows:
            chunk_embedding = json.loads(emb_json)
            if not chunk_embedding:
                continue
            score = _cosine_similarity(query_embedding, chunk_embedding)
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

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:max_results]

    def _search_keyword(self, query: str, max_results: int) -> list[MemorySearchResult]:
        """Search using FTS5 BM25 keyword matching."""
        if self._db is None:
            return []

        # Escape FTS5 special characters and build query
        # Split into tokens and join with space for implicit AND
        tokens = query.split()
        if not tokens:
            return []
        # Quote each token to avoid FTS5 syntax issues
        fts_query = " ".join(f'"{t}"' for t in tokens)

        try:
            rows = self._db.execute(
                """SELECT id, path, start_line, end_line, text, rank
                   FROM chunks_fts WHERE chunks_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (fts_query, max_results),
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back gracefully
            return []

        results: list[MemorySearchResult] = []
        for _id, path, start_line, end_line, text, rank in rows:
            # Convert BM25 rank to 0-1 score (rank is negative, lower = better)
            text_score = 1 / (1 + max(0, -rank))
            snippet = text[:SNIPPET_MAX_CHARS]
            results.append(
                MemorySearchResult(
                    path=path,
                    start_line=int(start_line),
                    end_line=int(end_line),
                    score=round(text_score, 4),
                    snippet=snippet,
                )
            )

        return results

    def search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Search memory using hybrid vector + keyword search."""
        if not self._available or self._db is None:
            return []

        # Sync before searching
        self.sync()

        # Gather more candidates than needed for post-processing
        fetch_count = max_results * 3

        vector_results: list[MemorySearchResult] = []
        keyword_results: list[MemorySearchResult] = []

        # Run vector search if API key available
        if self._has_api_key:
            vector_results = self._search_vector(query, fetch_count)

        # Run keyword search
        keyword_results = self._search_keyword(query, fetch_count)

        # Merge results (hybrid)
        if self.hybrid_enabled and vector_results and keyword_results:
            scored = self._merge_hybrid(vector_results, keyword_results)
        elif vector_results:
            scored = vector_results
        elif keyword_results:
            scored = keyword_results
        else:
            return []

        # Filter by min_score
        if min_score > 0:
            scored = [r for r in scored if r.score >= min_score]

        # Apply temporal decay
        if self.temporal_decay_enabled:
            scored = self._apply_temporal_decay(scored, self.temporal_decay_half_life_days)

        # Sort by score descending
        scored.sort(key=lambda r: r.score, reverse=True)

        # Apply MMR re-ranking
        if self.mmr_enabled:
            scored = self._apply_mmr(scored, max_results, self.mmr_lambda)
        else:
            scored = scored[:max_results]

        return scored

    def _merge_hybrid(
        self,
        vector_results: list[MemorySearchResult],
        keyword_results: list[MemorySearchResult],
    ) -> list[MemorySearchResult]:
        """Merge vector and keyword results with weighted scoring."""
        # Build lookup by unique key (path, start_line, end_line)
        merged: dict[tuple, MemorySearchResult] = {}

        for r in vector_results:
            key = (r.path, r.start_line, r.end_line)
            merged[key] = MemorySearchResult(
                path=r.path,
                start_line=r.start_line,
                end_line=r.end_line,
                score=round(self.vector_weight * r.score, 4),
                snippet=r.snippet,
            )

        for r in keyword_results:
            key = (r.path, r.start_line, r.end_line)
            if key in merged:
                merged[key].score = round(merged[key].score + self.text_weight * r.score, 4)
            else:
                merged[key] = MemorySearchResult(
                    path=r.path,
                    start_line=r.start_line,
                    end_line=r.end_line,
                    score=round(self.text_weight * r.score, 4),
                    snippet=r.snippet,
                )

        results = list(merged.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _apply_temporal_decay(results: list[MemorySearchResult], half_life_days: int = 30) -> list[MemorySearchResult]:
        """Apply exponential decay to scores from dated daily files."""
        today = date.today()
        decay_lambda = math.log(2) / half_life_days
        for r in results:
            m = re.match(r"memory/(\d{4}-\d{2}-\d{2})\.md$", r.path)
            if m:
                file_date = date.fromisoformat(m.group(1))
                age_days = (today - file_date).days
                r.score *= math.exp(-decay_lambda * max(0, age_days))
                r.score = round(r.score, 4)
        # Re-sort after decay
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @staticmethod
    def _apply_mmr(
        results: list[MemorySearchResult],
        max_results: int,
        lambda_param: float = 0.7,
    ) -> list[MemorySearchResult]:
        """Apply Maximal Marginal Relevance to reduce redundant results."""
        if not results:
            return results
        selected = [results[0]]
        candidates = list(results[1:])
        while candidates and len(selected) < max_results:
            best_score = -float("inf")
            best_idx = 0
            for i, cand in enumerate(candidates):
                relevance = cand.score
                max_sim = max(_jaccard_similarity(cand.snippet, s.snippet) for s in selected)
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            selected.append(candidates.pop(best_idx))
        return selected

    def close(self) -> None:
        """Close the database connection."""
        if self._db is not None:
            self._db.close()
            self._db = None
