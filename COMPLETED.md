# Completed

## 2026-03-04
- [x] Fix ACP dependency: `acp-sdk` (IBM) → `agent-client-protocol` (correct package)
- [x] Session branching: in-place via parentId chains (replaced copy-based fork)
- [x] Clean up: remove unused mock/ dir, stale scripts
- [x] Docs overhaul: slim README, add QUICK-START.md, update wiki (sessions, usage pages), fix stale info
- [x] Lint: resolve all ruff errors across src/ and tests/, add pre-commit hooks
- [x] AGENTS.md: positive framing, deduplicate into single constant, add `uv run` guidance

## 2026-03-03
- [x] Session fork: `/fork` command (Codex-style)
- [x] Session Persistence: SessionManager + Agent integration + CLI flags (--resume, --no-session)
- [x] Unified Streaming Event Protocol: StreamEvent dataclass, 3 adapters, stream_chat() dispatches
- [x] Context Overflow auto-compact-and-retry (max 1 retry)
- [x] Error Handling & Recovery: typed hierarchy, classify_http_error, extract_retry_delay, 48 tests

## 2026-03-02
- [x] Ctrl+C: double-press force exit, atexit cleanup, cancellation-safe compaction
- [x] Streaming UX: spinner during tool-call argument generation
- [x] Debug transcript logger (JSONL, --debug flag)
- [x] ACP agent support (optional dependency, lazy type eval)
- [x] Webhook server + Slack integration (HTTP server, channel abstraction, Socket Mode, `koi serve`)

## 2026-03-01
- [x] Migrate remote: NVIDIA GitLab → GitHub
- [x] Consolidate 9 markdown files into README
- [x] Fix truncation tests for Python 3.14
- [x] UI improvements: colored prompt, styled tool calls, indented responses, session header card
- [x] Sub-agent completion notifications
- [x] /usage command with 7-day history, cache hit ratio
- [x] Context management: 4-layer system (truncation → pruning → compaction → guard)
- [x] Prompt caching (Anthropic)
- [x] Extended thinking (Anthropic/OpenAI native, `<think>` tag fallback)
- [x] Cron integration (system crontab)
- [x] Sub-agent orchestration (spawn, monitor, kill)
