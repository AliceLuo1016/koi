# Koi vs OpenClaw: Gap Analysis & Progress

_Generated 2026-02-27 by alibot. Last updated 2026-02-27 23:58 PST._

---

## ✅ Completed

### 1. Thinking/Reasoning Params — DONE
- Native API params for Anthropic (`extended_thinking` + `budget_tokens`), OpenAI (`reasoning_effort`), Responses API (`reasoning.effort`)
- CLI flag: `koi run --thinking high`
- 29 tests

### 2. Richer System Prompt — DONE
- 12 structured sections: tool call style, safety rules, per-tool tips, skills/memory guidance, reasoning format hints

### 3. Tool Output Truncation — DONE
- read_file: 2000 lines / 50KB, exec: 50KB, web_fetch: 20K, generic cap: 400K
- 16 tests

### 4. Model-Aware Thinking Safety — DONE
- `supports_thinking()` allowlist per model family, safe default off, error fallback + session disable
- 38 tests

### 5. Non-Native `<think>` Tag Reasoning — DONE
- Prompt injection for unsupported models, tag stripping in display, two-tier system
- 25 tests

### 6. Anthropic Prompt Caching — DONE
- System prompt + last tool result cached with `cache_control: {type: "ephemeral"}`
- ~90% savings on cached input tokens
- 16 tests

### 7. Two-Phase Context Pruning + Compaction — DONE
- Soft trim at 30%, hard clear at 50%, LLM compaction at 60% with 120s safety timeout
- Protections: last 3 assistant messages, pre-first-user messages, non-prunable tools
- 27 tests

### 8. Real Streaming to User — DONE
- Progressive token display for all 3 API formats (Responses, Chat Completions, Anthropic)
- `_last_stream_response` captures tool calls alongside streaming
- Reasoning tags buffered and stripped before display
- 11 tests

### 9. Context Window Guard — DONE
- Pre-LLM-call safety net: individual tool result cap (50% window), total context cap (75% window)
- Emergency compaction of oldest tool results, 2x char weighting for tool output density
- 31 tests

### 10. System Prompt Architectural Fix — DONE
- Stored as `self.system_prompt`, injected at API boundary per-format
- Impossible for system prompt to be accidentally compacted

### 11. max_tokens Adjusted for Thinking Budget — DONE
- `max_tokens = min(base + budget, model_max)`, always reserves 1024 for output
- `_adjust_max_tokens_for_thinking()` helper matches pi-ai's logic
- 15 tests

### 12. Thinking Budget Aligned with OpenClaw — DONE
- All levels match pi-ai: minimal=1024, low=2048, medium=8192, high=16384

### 13. Sub-Agent Spawning — DONE
- `spawn_subagent` tool: spawns isolated Koi child process in background
- `list_subagents` tool: shows active/completed sub-agents with status
- `kill_subagent` tool: terminates a running sub-agent
- Depth limits (max 3), children limits (max 5 concurrent)
- Model and thinking overrides per sub-agent
- Timeout support (kills process after N seconds)
- Result file `.koi/subagent-runs/{id}.json`, completion injected into parent conversation
- `KOI_SPAWN_DEPTH` env var for depth tracking across process boundaries
- CLI: `--result-file` and `--model` flags
- 34 tests

### 14. Pre-existing Fixes
- Streaming double LLM call eliminated (was 2x output tokens)
- Multi-provider support: Responses API + Chat Completions + Anthropic auto-detect
- New tools: glob_files, grep_files
- Async subprocess with cancellation support
- Retry logic with exponential backoff and retry-after headers

---

## 🟡 Remaining Gaps

### 15. Session Persistence — TODO
Messages stored in Python list — crash = total loss.
OpenClaw uses persistent session files with branching and rollback.

**Impact:** 🟡 Data safety.
**Effort:** High.

### 16. Model Fallback — TODO
OpenClaw supports `fallbackModel` — if primary model is overloaded (429/529), tries a different model.
Koi retries the same model up to 6 times.

**Impact:** 🟡 Resilience.
**Effort:** Medium.

### 17. Persistent Sub-Agent Sessions — TODO
OpenClaw supports `mode="session"` (persistent, thread-bound sub-agents) and `steer` (send messages to running sub-agents). Koi only has one-shot `mode="run"`.

**Impact:** 🟢 Nice to have.
**Effort:** Medium.

---

## 🟢 Minor Remaining

### 18. Anthropic Extended Context Beta — TODO
Add extended context beta header for 1M tokens on supported models. One line change.

### 19. OpenAI Responses API `store=true` — TODO
Enables server-side multi-turn state. One line change.

### 20. web_search Not Implemented — TODO
Returns placeholder. Needs Brave Search API or similar.

---

## Known Limitations (No Easy Fix)

### Full MEMORY.md in System Prompt (non-Anthropic)
Entire file dumped every turn. Anthropic prompt caching makes this free; non-Anthropic pays full price. Would need server-side caching.

### Tool Schemas Uncached for Non-Anthropic
All tool schemas sent every turn. Free with Anthropic caching; no fix for other providers.

---

## Summary

| Status | Count |
|--------|-------|
| ✅ Completed | 14 |
| 🟡 Remaining | 3 |
| 🟢 Remaining (minor) | 3 |
| **Total tests** | **494 (0 failures)** |

### Original Questions — Status

**"Koi doesn't think deep enough"** → ✅ RESOLVED
- Native thinking for Anthropic/OpenAI/Google
- `<think>/<final>` fallback for unsupported models
- Model detection + error fallback
- Budget aligned with OpenClaw, max_tokens adjusted correctly
- Richer system prompt with structured guidance

**"Koi burns tokens quickly"** → ✅ RESOLVED
- 4-layer context management (truncation → pruning → compaction → guard)
- Anthropic prompt caching (~90% savings)
- System prompt separated from compactable messages
- Streaming (no double LLM calls)
- Tool output truncation at source

### Session Timeline (2026-02-27)
All 14 items implemented in a single evening session using Claude Code sub-agents:
1. Thinking/reasoning params (29 tests)
2. Tool output truncation (16 tests)
3. Richer system prompt
4. Model-aware thinking safety (38 tests)
5. Non-native `<think>` tag reasoning (25 tests)
6. Anthropic prompt caching (16 tests)
7. Two-phase context pruning + compaction (27 tests)
8. Real streaming to user (11 tests)
9. Context window guard (31 tests)
10. System prompt architectural fix
11. max_tokens + budget alignment (15 tests)
12. System prompt compaction bug fix
13. Sub-agent spawning (34 tests)
14. Various test fixes and regressions resolved
