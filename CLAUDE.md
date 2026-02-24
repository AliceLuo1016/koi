# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Koi is a terminal-based AI agent (Python 3.9+) with persistent memory, tool calling, extensible skills, sandbox security, and system cron integration. It uses an OpenAI-compatible Responses API backend.

## Build & Install

```bash
pip install .              # Install the package
pip install ".[dev]"       # Install with dev dependencies
```

Uses **uv** as the package manager and **hatchling** as the build backend. Source is in `src/koi/`. The `skills/` directory is force-included as `koi/bundled_skills` in the wheel.

## Testing

```bash
pytest tests/ -v                    # Run all tests
pytest tests/test_config.py -v      # Run a single test file
pytest tests/test_config.py::test_function_name -v  # Run a single test
```

Tests use pytest with pytest-asyncio for async tests. Test files use `TemporaryDirectory` for isolation and `unittest.mock` for mocking.

## Lint & Format

```bash
black src/ tests/          # Format (line-length 88, target py39)
ruff check src/ tests/     # Lint (rules: E, F, W, I, N, UP)
```

## Architecture

**Core flow:** `cli.py` (Click CLI) → `Agent` (async conversation loop) → `LLMClient` (Responses API) + `ToolExecutor` (tool dispatch)

Key modules in `src/koi/`:

- **cli.py** — Click CLI commands: `init`, `run`, `cron`, `skills`, `config`, `memory`
- **agent.py** — Main `Agent` class with async conversation loop, prompt_toolkit input, and tool execution cycle
- **llm.py** — `LLMClient` translates between internal Chat Completions message format and the Responses API wire format; includes retry with exponential backoff
- **tools.py** — 14 tool definitions (OpenAI function calling format) and `ToolExecutor` dispatch class
- **sandbox.py** — `Sandbox` class with three security layers: filesystem ACL, env variable allowlist, command pattern blocking/confirmation. Loaded from `.koi/sandbox.yaml`
- **memory.py** — Persistent memory via `.koi/MEMORY.md`
- **skills.py** — `SkillsManager` discovers `SKILL.md` files from configured paths, loads on demand via `read_skill` tool
- **config.py** — `Config` class loads `.koi/config.json`, supports `KOI_API_KEY` env var override
- **compaction.py** — `ContextCompactor` uses tiktoken for token estimation; auto-summarizes at 70% context usage
- **prompts.py** — Assembles system prompt from base instructions + tools + skills + project instructions + memory + alerts
- **cron.py** — `CronManager` manages system crontab entries with launcher scripts in `.koi/cron-scripts/`

**Key patterns:**
- Async throughout (agent loop, tool execution, LLM calls)
- Internal message format is OpenAI Chat Completions dicts (`role`/`content`/`tool_calls`), converted to Responses API format only at the LLM boundary in `llm.py`
- Sandbox is consulted before every file read/write (`check_read`/`check_write`) and command execution (`check_command`)
- Skills are markdown-based `SKILL.md` files loaded on-demand; bundled skills are copied to `.koi/skills/` on `koi init`

## Per-Project Structure (.koi/)

Created by `koi init` in any project directory:
- `config.json` — API settings (api_base, api_key, model, max_tokens, context_window, skills_paths, temperature)
- `sandbox.yaml` — Security rules (filesystem, environment, commands)
- `MEMORY.md` / `AGENTS.md` — Persistent memory and project-specific instructions
- `credentials/` — Credential files loaded as env vars for sandboxed commands
- `alerts/` — Structured alert markdown files
- `crontab.json` + `cron-scripts/` + `cron-logs/` — Cron job metadata, launcher scripts, and output logs

## Workflow Orchestration

### 1. Plan Mode Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately – don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes – don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests – then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.
