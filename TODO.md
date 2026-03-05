# TODO

Ranked from highest to lowest priority.

---

## P0 — Memory System (OpenClaw-style)

_Give Koi semantic memory search over workspace Markdown files. Currently Koi has MEMORY.md (read/write) but no search — the agent must read the entire file every session._

### Step 1: Memory file layout
- [ ] Support `memory/YYYY-MM-DD.md` daily logs alongside `MEMORY.md`
- [ ] Auto-read today + yesterday daily logs at session startup
- [ ] `update_memory` tool writes to daily log by default, `MEMORY.md` for long-term

### Step 2: Chunking & indexing
- [ ] Chunk Markdown files (~400 tokens, ~80 token overlap)
- [ ] Store chunks in SQLite (`.koi/memory.sqlite`): chunk text, file path, line range, embedding vector
- [ ] Index on startup + watch for file changes (debounced)
- [ ] Reindex automatically when embedding model/provider changes

### Step 3: Embedding providers
- [ ] OpenAI `text-embedding-3-small` (default, remote)
- [ ] Config: `memory_search.provider`, `memory_search.api_key`, `memory_search.model`
- [ ] Fallback: if no API key, disable memory search (graceful degradation)

### Step 4: `memory_search` tool
- [ ] Semantic search: query → embed → cosine similarity against stored chunks
- [ ] Returns: snippet text (~700 chars), file path, line range, score
- [ ] Agent uses this before answering questions about prior work, decisions, preferences
- [ ] Add to system prompt: "Before answering questions about prior context, use `memory_search`"

### Step 5: `memory_get` tool
- [ ] Read specific memory file by path, with optional `from_line` and `num_lines`
- [ ] Use after `memory_search` to pull full context around a snippet

### Step 6: Hybrid search (BM25 + vector)
- [ ] Add FTS5 full-text index alongside vector index
- [ ] Weighted merge: `vectorWeight * vectorScore + textWeight * textScore`
- [ ] Handles exact tokens (IDs, code symbols) that vector search misses

### Step 7: Pre-compaction memory flush
- [ ] Before auto-compaction, inject a silent turn: "Store any durable memories now"
- [ ] Agent writes important context to daily log before it gets compacted away
- [ ] One flush per compaction cycle

### Step 8: Post-processing (optional, after basics work)
- [ ] Temporal decay: boost recent daily notes, fade old ones (half-life ~30 days)
- [ ] MMR re-ranking: reduce redundant/near-duplicate snippets
- [ ] Embedding cache: skip re-embedding unchanged chunks

---

## P1 — Safety & Reliability

### Tool Execution Safety
- [ ] Tool loop detection (prevent infinite tool calls)
- [ ] Approval tiers: auto / confirm / restricted per tool
- [ ] Audit log for every tool execution

### Context Overflow
- [ ] Detect silent overflow (provider accepts but truncates — check `usage.input > context_window`)

---

## P2 — Configuration & Model Management

- [ ] Config validation with JSON schema
- [ ] Model fallback chains
- [ ] Environment-specific overrides

---

## P3 — Integrations (Skills)

### Tier 1 — High Impact
- [ ] **GitLab / GitHub** — Browse repos, read files, create MRs/PRs, review diffs, trigger pipelines, manage issues
- [ ] **Slurm** — Submit, monitor, cancel jobs; query queue/node status/utilization
- [ ] **S3 / Object Storage** — List, upload, download, sync; handle buckets, prefixes, presigned URLs

### Tier 2 — Communication & Tracking
- [ ] **Teams / Outlook** — Read/send messages and emails, calendar integration
- [ ] **Weights & Biases** — Query runs, compare metrics, pull artifacts

### Tier 3 — Infrastructure & Data
- [ ] **Docker** — Build, run, manage containers; read logs
- [ ] **Kubernetes** — List pods, check status, scale deployments, read logs
- [ ] **CI/CD Pipelines** — Trigger builds, check status, read logs, retry failed stages
- [ ] **LanceDB / Vector Stores** — Query datasets, similarity searches
- [ ] **Google Docs / Notion** — Read and create documents, search across wikis

---

## P4 — Multi-User & Team Features

### Session Management
- [ ] Concurrent users with separate threads/contexts
- [ ] Session handoff and recovery after crashes

### Authentication & Identity
- [ ] Multi-user auth (API keys, SSO/OIDC)
- [ ] Per-user permissions and audit trails

### Credential Management
- [ ] Secure storage for API keys, tokens, service accounts
- [ ] Per-user and per-team scopes
- [ ] Rotate and revoke without downtime

### Event Bus
- [ ] Internal pub/sub for routing events (webhooks, cron, alerts)
- [ ] Support async event delivery

---

## P5 — Agent Capabilities

### Long-Running Task Management
- [ ] Track async operations (training runs, pipelines, CI) across sessions
- [ ] Resume context after hours or days

### Escalation Framework
- [ ] Configurable rules: when to act autonomously vs. ask a human
- [ ] Confidence thresholds and risk levels

### Team Knowledge Base
- [ ] Shared memory across team members
- [ ] Searchable, versioned knowledge store

### Memory & Skills Architecture
- [ ] Memory search (semantic/embedding-based)
- [ ] Skill versioning and update management

---

## P6 — Deployment

- [ ] Dockerfile with all dependencies
- [ ] Helm chart / K8s manifests (scaling, health checks)
- [ ] Multi-instance coordination via backing store (Redis, Postgres)
- [ ] API server mode (REST/gRPC endpoints for web UIs, bots)
