# Koi 🐠

Terminal-based AI agent with memory, tool calling, skills, sandbox security, and system cron integration. Built for developers who want a powerful AI assistant that lives in their terminal.

## Features

- **Conversational AI**: Chat naturally with an AI agent that can think and use tools
- **Tool Integration**: Read/write files, execute commands, fetch web content, and more
- **Persistent Memory**: Remember important context between sessions
- **Skills System**: Extensible capabilities through markdown-based skill definitions
- **Alerts System**: Structured alerts with severity levels and desktop notifications
- **Sandbox Security**: Credential protection, env scrubbing, and file access control
- **Cron Integration**: Schedule AI tasks to run automatically
- **Context Management**: Smart conversation compaction to stay within token limits
- **Multi-line Input**: Escape+Enter for newlines, multi-line paste support via prompt_toolkit
- **Rich Terminal UI**: Beautiful output with colored text

## Quick Start

### Installation

```bash
cd ~/koi
pip install .
```

### Initialize a Project

```bash
cd your-project
koi init
```

This creates a `.agent/` directory with:
- `config.json` — API settings and configuration
- `sandbox.yaml` — Security sandbox rules
- `MEMORY.md` — Persistent memory file
- `AGENTS.md` — Project-specific instructions
- `cron-logs/` — Directory for scheduled task logs

### Configure API Access

Edit `.agent/config.json`:

```json
{
  "api_base": "https://api.example.com/v1/responses",
  "api_key": "your-api-key-here",
  "model": "your-model-name",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": [".agent/skills"],
  "temperature": 0.7
}
```

The `api_base` should point to an OpenAI-compatible Responses API endpoint. You can also set `KOI_API_KEY` as an environment variable instead of putting your key in the config file.

### Start Chatting

```bash
koi run
```

## Commands

### CLI

| Command | Description |
|---------|-------------|
| `koi init` | Initialize `.agent/` directory |
| `koi run` | Start interactive session |
| `koi run --task "..." --non-interactive` | Run a task and exit (for cron/scripts) |
| `koi cron add "0 9 * * *" "Check emails"` | Schedule a recurring task |
| `koi cron list` | List scheduled tasks |
| `koi cron remove <id>` | Remove a scheduled task |
| `koi skills` | List available skills |
| `koi config` | Show current configuration |
| `koi memory` | Show current memory |

### Chat Commands

During a `koi run` session:

| Command | Description |
|---------|-------------|
| `/exit`, `/quit` | Exit the session |
| `/help` | Show help |
| `/memory` | Show current memory |
| `/remember TEXT` | Add text to memory |
| `/skills` | List available skills |
| `/compact` | Force conversation compaction |
| `/stats` | Show context usage statistics |

**Input:** Enter submits, Escape+Enter inserts a newline. Multi-line paste is supported.

## Tools

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents (with offset/limit) |
| `write_file` | Create or overwrite files |
| `edit_file` | Surgical find-and-replace edits |
| `exec_command` | Execute shell commands (sandboxed) |
| `web_fetch` | Fetch and convert web pages to markdown |
| `web_search` | Web search (placeholder — TODO) |
| `update_memory` | Persist info across sessions |
| `read_skill` | Load skill definitions by name |
| `create_alert` | Create structured alerts with desktop notifications |
| `list_alerts` | List alerts filtered by status |
| `resolve_alert` | Approve or dismiss alerts |

## Sandbox Security

Koi includes a sandbox that prevents accidental credential leaks and destructive actions. Configuration lives in `.agent/sandbox.yaml` and is loaded automatically on every run.

### Three Layers of Protection

**1. File Access Control**

```yaml
filesystem:
  allowed_paths:
    - "."              # Only the project directory
  blocked_paths:
    - "~/.aws"         # AWS credentials
    - "~/.ssh"         # SSH keys
    - "~/.config"      # App configs & tokens
```

File reads/writes outside allowed paths are denied. Blocked paths are denied even if you expand allowed paths later.

**2. Environment Scrubbing**

```yaml
environment:
  allowlist:
    - PATH
    - HOME
    - USER
    - SHELL
    - LANG
    - TERM
    # ... safe vars only
```

Shell commands only receive allowlisted environment variables. API keys, cloud credentials, and tokens are stripped — so even `echo $OPENAI_API_KEY` returns nothing.

**3. Command Blocking**

```yaml
commands:
  blocked_patterns:
    - 'rm\s+-rf\s+/'          # Nuke from orbit
    - 'DROP\s+TABLE'           # SQL destruction
    - 'aws\s+iam'              # IAM changes
  confirm_patterns:
    - 'rm\s+'                  # Needs confirmation
    - 'git\s+push\s+.*--force' # Force push
```

Dangerous commands are hard-blocked. Risky commands are flagged for user confirmation.

### Customization

Edit `.agent/sandbox.yaml` to adjust rules for your project. The defaults are secure but not overly restrictive.

## Alerts System

Koi can create, track, and resolve structured alerts — useful for monitoring tasks.

```
koi> Read the logs and create alerts for any errors
🔧 read_file mock/logs/latest.log
🔧 create_alert "DB pool overloaded" severity=critical
🔧 create_alert "Auth retry storm" severity=high
```

Alerts are saved as markdown files in `.agent/alerts/` and trigger desktop notifications (macOS/Linux). On startup, koi shows a count of pending alerts so nothing gets missed.

### Alert Workflow

1. **Create** — Koi detects issues and creates alerts with severity + proposed fix
2. **Review** — `list_alerts` shows pending alerts
3. **Resolve** — `resolve_alert` approves or dismisses; approved alerts return their fix command

## Skills System

Skills are markdown files that teach koi how to handle specific tasks:

```
skills/
└── log-monitor/
    └── SKILL.md
```

Skills are discovered from paths in `config.skills_paths` and listed in the system prompt. The agent loads full skill content on demand via `read_skill`.

## Memory System

- **MEMORY.md** — Long-term memory that persists between sessions
- **Context Compaction** — When conversations get long, koi summarizes older messages to stay within the context window
- **`/remember`** — Quick way to save important info during a session
- **`update_memory`** — Tool the agent can use proactively

## Cron Integration

Schedule AI tasks to run automatically via system crontab:

```bash
# Check logs every hour
koi cron add "0 * * * *" "Read mock/logs/latest.log and create alerts for errors"

# Daily standup
koi cron add "0 9 * * 1-5" "Review yesterday's commits and plan today"
```

Cron jobs use the full path to `koi` (resolved via `shutil.which`) so they work correctly in cron's minimal environment. Output goes to `.agent/cron-logs/`.

## Architecture

```
src/koi/
├── __main__.py      # Entry point
├── cli.py           # CLI commands (init, run, cron, etc.)
├── agent.py         # Main conversation loop
├── llm.py           # OpenAI-compatible Responses API client
├── tools.py         # Tool definitions and execution
├── sandbox.py       # Security sandbox enforcement
├── memory.py        # Persistent memory
├── skills.py        # Skill discovery and loading
├── cron.py          # System crontab integration
├── config.py        # Configuration management
├── compaction.py    # Context window management
└── prompts.py       # System prompt building
```

### Per-Project Structure

```
your-project/
├── .agent/
│   ├── config.json      # API settings
│   ├── sandbox.yaml     # Security rules
│   ├── MEMORY.md        # Persistent memory
│   ├── AGENTS.md        # Project instructions
│   ├── alerts/          # Alert files
│   ├── crontab.json     # Cron job metadata
│   ├── skills/          # Skills (bundled + custom)
│   └── cron-logs/       # Scheduled task logs
└── ...
```

## Development

```bash
# Install with dev dependencies
pip install ".[dev]"

# Run tests
pytest tests/

# Format & lint
black src/ tests/
ruff check src/ tests/
```

## License

MIT

---

Built with ❤️ for developers who want AI assistance that stays out of the way until you need it.
