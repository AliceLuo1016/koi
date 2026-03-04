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
- **Streaming** — Real-time token display for all API formats
- **Rich Terminal UI** — Markdown rendering, multi-line input (Escape+Enter)

## Quick Start

See **[BEST-PRACTICES.md](BEST-PRACTICES.md)** for a step-by-step getting started guide with practical examples.

### Install

```bash
cd ~/koi
pip install .
```

### Initialize a Project

```bash
cd your-project
koi init
```

### Configure

Edit `.koi/config.json`:

```json
{
  "api_base": "https://api.example.com/v1/responses",
  "api_key": "your-api-key-here",
  "model": "your-model-name",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": [".koi/skills"],
  "temperature": 0.7
}
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_base` | string | — | API endpoint URL (required) |
| `api_key` | string | — | API key, or set `KOI_API_KEY` env var |
| `model` | string | `openai/openai/gpt-5.2-codex` | Model identifier |
| `api_format` | string | auto-detected | `responses`, `chat_completions`, or `anthropic` |
| `max_tokens` | int | 4096 | Max tokens per response |
| `context_window` | int | 128000 | Model's context window size |
| `temperature` | float | model default | Sampling temperature (0–2) |
| `skills_paths` | array | `[".koi/skills"]` | Directories to search for skills |

### Run

```bash
koi run
koi run --thinking high              # Enable extended thinking
koi run --task "..." --non-interactive  # One-shot for scripts/cron
```

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

`/help` · `/exit` · `/memory` · `/remember TEXT` · `/skills` · `/compact` · `/stats`

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read files (with offset/limit for large files) |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical find-and-replace |
| `exec_command` | Execute shell commands (sandboxed) |
| `glob_files` | Find files by pattern |
| `grep_files` | Search file contents |
| `web_fetch` | Fetch web pages as markdown |
| `update_memory` | Persist info across sessions |
| `read_skill` | Load skill definitions |
| `create_alert` / `list_alerts` / `resolve_alert` | Structured alert management |
| `spawn_subagent` / `list_subagents` / `kill_subagent` | Sub-agent orchestration |

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
├── cli.py           # Click CLI (init, run, cron, skills, config, memory)
├── agent.py         # Async conversation loop + tool execution
├── llm.py           # Multi-provider LLM client (Responses/Chat Completions/Anthropic)
├── tools.py         # Tool definitions + sandboxed executor
├── sandbox.py       # 3-layer security enforcement
├── memory.py        # Persistent memory via .koi/MEMORY.md
├── skills.py        # Skill discovery and loading
├── config.py        # Configuration management
├── compaction.py    # Context compaction via LLM summarization
├── context_guard.py # Pre-call context window safety net
├── context_pruning.py # Two-phase message pruning
├── subagent.py      # Sub-agent spawning and orchestration
├── usage.py         # Token usage tracking
├── prompts.py       # System prompt assembly (12 structured sections)
└── cron.py          # System crontab management
```

**Key patterns:**
- Async throughout (agent loop, tool execution, LLM calls)
- Internal message format is OpenAI Chat Completions dicts, converted at the LLM boundary
- Sandbox consulted before every file read/write and command execution

## Development

```bash
pip install ".[dev]"          # Install with dev deps
pytest tests/ -v              # Run tests (494 tests, 0 failures)
black src/ tests/             # Format (line-length 88, target py39)
ruff check src/ tests/        # Lint (rules: E, F, W, I, N, UP)
```

Uses **uv** as package manager, **hatchling** as build backend. Python 3.9+. Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`.

## Example Workflows

- **Batch-analyze failed jobs** — Scan logs across N failed jobs, summarize common failures with fixes
- **Automated cluster monitoring** — Cron jobs to watch utilization and auto-submit work
- **Custom skills** — Walk Koi through your workflow step by step, then package it with skill-creator

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `koi: command not found` | `pip install .` and ensure PATH includes pip's bin dir |
| API key not found | Set `KOI_API_KEY` env var, or add to `.koi/config.json`, or run `claude auth` for Anthropic |
| Context window exceeded | Use `/compact`, reduce `context_window`, or start a fresh session |
| Sandbox blocks file/command | Edit `.koi/sandbox.yaml` to adjust allowed paths or command patterns |
| Cron jobs not running | Check `crontab -l | grep koi` and logs in `.koi/cron-logs/` |
