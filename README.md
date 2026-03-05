# Koi 🐠

Terminal-based AI agent with persistent memory, tool calling, extensible skills, sandbox security, and cron integration. Built for developers who want a powerful AI assistant in their terminal.

## Features

- **Multi-Provider Support** — OpenAI Responses API, Chat Completions, and Anthropic Claude (auto-detected)
- **Extended Thinking** — Native reasoning for Anthropic/OpenAI, `<think>` tag fallback for others
- **Tool Calling** — File ops, shell commands, web fetch, memory, alerts, and more
- **Persistent Memory** — Context that survives across sessions
- **Skills System** — Markdown-based extensible capabilities
- **Sandbox Security** — File access control, env scrubbing, command filtering
- **Cron Integration** — Schedule AI tasks via system crontab
- **Context Management** — 4-layer system: truncation → pruning → compaction → guard
- **Prompt Caching** — ~90% input token savings on Anthropic
- **Sub-Agents** — Spawn isolated child processes for parallel work
- **ACP Agent Support** — Spawn external coding agents (Claude Code, Codex, Gemini, etc.) via Agent Client Protocol
- **Session Management** — Persistent sessions with in-place branching via parentId chains
- **Streaming** — Real-time token display for all API formats
- **Rich Terminal UI** — Markdown rendering, multi-line input (Escape+Enter)

## Quick Start

See **[BEST-PRACTICES.md](BEST-PRACTICES.md)** for a step-by-step getting started guide, configuration details, and practical examples.

## Commands

**CLI:**

| Command | Description |
|---------|-------------|
| `koi init` | Initialize `.koi/` directory |
| `koi run` | Start interactive session |
| `koi cron add "schedule" "task"` | Schedule a recurring task |
| `koi cron list` / `koi cron remove <id>` | Manage cron jobs |
| `koi skills` | List available skills |
| `koi config` | Show configuration |
| `koi memory` | Show memory |

**Chat (during `koi run`):**

`/help` · `/exit` · `/quit` · `/memory` · `/remember TEXT` · `/skills` · `/compact` · `/stats` · `/usage` · `/new` · `/fork`

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read files (with offset/limit for large files) |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical find-and-replace |
| `remove_file` | Delete files |
| `exec_command` | Execute shell commands (sandboxed) |
| `glob_files` | Find files by pattern |
| `grep_files` | Search file contents |
| `web_search` | Search the web |
| `web_fetch` | Fetch web pages as markdown |
| `update_memory` | Persist info across sessions |
| `read_skill` | Load skill definitions |
| `create_alert` / `list_alerts` / `resolve_alert` | Structured alert management |
| `add_cron_job` / `list_cron_jobs` / `remove_cron_job` | Cron job management |
| `spawn_subagent` / `send_to_subagent` / `list_subagents` / `kill_subagent` | Sub-agent orchestration |
| `list_available_agents` | Discover installed ACP agents |

## Sandbox Security

Three layers of protection configured in `.koi/sandbox.yaml`:

1. **File Access Control** — Allowed/blocked path lists. Blocks `~/.aws`, `~/.ssh`, `~/.config` by default.
2. **Environment Scrubbing** — Only allowlisted env vars passed to commands. API keys and tokens stripped.
3. **Command Filtering** — Dangerous commands blocked (`rm -rf /`, `DROP TABLE`). Risky commands require confirmation.

## Skills

Markdown-based skill files in `.koi/skills/<name>/SKILL.md`. Discovered automatically from `skills_paths` and loaded on demand.

**Creating skills:** Use the built-in `skill-creator` tool — walk Koi through your workflow, then ask it to package it as a skill.

## Memory & Context

- **MEMORY.md** — Long-term memory persisted to disk
- **Context compaction** — Auto-summarizes at 60% context usage (with 120s safety timeout)
- **Pruning** — Soft trim at 30%, hard clear at 50%
- **Context guard** — Pre-call safety net caps individual tool results (50% window) and total context (75% window)
- **Prompt caching** — System prompt + last tool result cached on Anthropic (~90% savings)

## Cron

```bash
koi cron add "0 * * * *" "Read logs and create alerts for errors"
koi cron add "0 9 * * 1-5" "Review yesterday's commits and plan today"
```

Output goes to `.koi/cron-logs/`. Full `koi` path is resolved automatically for cron's minimal environment.

## Architecture

```
src/koi/
├── cli.py             # Click CLI (init, run, cron, skills, config, memory)
├── agent.py           # Async conversation loop + tool execution
├── llm.py             # Multi-provider LLM client (Responses/Chat Completions/Anthropic)
├── tools.py           # Tool definitions + sandboxed executor
├── sandbox.py         # 3-layer security enforcement
├── memory.py          # Persistent memory via .koi/MEMORY.md
├── skills.py          # Skill discovery and loading
├── config.py          # Configuration management
├── compaction.py      # Context compaction via LLM summarization
├── context_guard.py   # Pre-call context window safety net
├── context_pruning.py # Two-phase message pruning
├── subagent.py        # Sub-agent spawning and orchestration
├── acp_client.py      # ACP agent client (Claude Code, Codex, Gemini, etc.)
├── acp_registry.py    # ACP agent discovery and configuration
├── session_manager.py # Session persistence with branching support
├── sessions.py        # Session listing and selection
├── stream_events.py   # Streaming event handling
├── server.py          # Optional HTTP/Slack server interface
├── transcript.py      # Session transcript utilities
├── usage.py           # Token usage tracking
├── errors.py          # Error types
├── prompts.py         # System prompt assembly (12 structured sections)
└── cron.py            # System crontab management
```

**Key patterns:**
- Async throughout (agent loop, tool execution, LLM calls)
- Internal message format is OpenAI Chat Completions dicts, converted at the LLM boundary
- Sandbox consulted before every file read/write and command execution

## Development

```bash
pip install ".[dev]"          # Install with dev deps
pytest tests/ -v              # Run tests (721 tests, 0 failures)
black src/ tests/             # Format (line-length 88, target py39)
ruff check src/ tests/        # Lint (rules: E, F, W, I, N, UP)
```

Uses **uv** as package manager, **hatchling** as build backend. Python 3.9+. Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `koi: command not found` | `pip install .` and ensure PATH includes pip's bin dir |
| API key not found | Set `KOI_API_KEY` env var, or add to `.koi/config.json`, or run `claude auth` for Anthropic |
| Context window exceeded | Use `/compact`, reduce `context_window`, or start a fresh session |
| Sandbox blocks file/command | Edit `.koi/sandbox.yaml` to adjust allowed paths or command patterns |
| Cron jobs not running | Check `crontab -l | grep koi` and logs in `.koi/cron-logs/` |
