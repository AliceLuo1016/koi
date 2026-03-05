# Koi 🐠

Terminal-based AI agent with persistent memory, tool calling, extensible skills, sandbox security, and cron integration. Built for developers who want a powerful AI assistant in their terminal.

## Features

- **Multi-Provider Support** — OpenAI Responses API, Chat Completions, and Anthropic Claude
- **Extended Thinking** — Native reasoning for Anthropic/OpenAI, `<think>` tag fallback for others
- **Tool Calling** — File ops, shell commands, web search/fetch, memory, alerts, and more
- **Persistent Memory** — Context that survives across sessions
- **Skills System** — Markdown-based extensible capabilities
- **Sandbox Security** — File access control, env scrubbing, command filtering
- **Context Management** — 4-layer system: truncation → pruning → compaction → guard
- **Sub-Agents & ACP** — Spawn child agents or external coding agents (Claude Code, Codex, Gemini)
- **Session Branching** — In-place forking via parentId chains
- **Cron Integration** — Schedule AI tasks via system crontab
- **Streaming** — Real-time token display for all API formats
- **Rich Terminal UI** — Markdown rendering, multi-line input (Escape+Enter)

## Quick Start

See **[QUICK-START.md](QUICK-START.md)** for installation, configuration, adding skills, and example workflows.

## Documentation

📖 **[docs/wiki/index.md](docs/wiki/index.md)** — Full wiki with architecture, tools, skills, context management, security, and more.

## Development

```bash
pip install ".[dev]"          # Install with dev deps
pytest tests/ -v              # Run tests (721 tests, 0 failures)
black src/ tests/             # Format (line-length 88, target py39)
ruff check src/ tests/        # Lint (rules: E, F, W, I, N, UP)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `koi: command not found` | `pip install .` and ensure PATH includes pip's bin dir |
| API key not found | Set `KOI_API_KEY` env var, or add to `.koi/config.json` |
| Context window exceeded | Use `/compact`, reduce `context_window`, or start a fresh session |
| Sandbox blocks file/command | Edit `.koi/sandbox.yaml` to adjust allowed paths or command patterns |
| Cron jobs not running | Check `crontab -l \| grep koi` and logs in `.koi/cron-logs/` |
