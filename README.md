# Koi 🐠

Terminal-based AI agent with memory, tool calling, skills, and system cron integration. Built for developers who want a powerful AI assistant that lives in their terminal.

## Features

- **Conversational AI**: Chat naturally with an AI agent that can think and use tools
- **Tool Integration**: Read/write files, execute commands, fetch web content, and more
- **Persistent Memory**: Remember important context between sessions
- **Skills System**: Extensible capabilities through markdown-based skill definitions
- **Cron Integration**: Schedule AI tasks to run automatically
- **Context Management**: Smart conversation compaction to stay within token limits
- **Rich Terminal UI**: Beautiful output with streaming responses and colored text

## Quick Start

### Installation

```bash
cd ~/koi
pip install -e .
```

### Initialize a Project

```bash
cd your-project
koi init
```

This creates a `.agent/` directory with:
- `config.json` - API settings and configuration
- `MEMORY.md` - Persistent memory file
- `AGENTS.md` - Project-specific instructions
- `cron-logs/` - Directory for scheduled task logs

### Configure API Access

Edit `.agent/config.json`:

```json
{
  "api_base": "https://api.openai.com/v1",
  "api_key": "your-api-key-here",
  "model": "gpt-5.2",
  "max_tokens": 4096,
  "context_window": 128000,
  "skills_paths": ["./skills"],
  "temperature": 0.7
}
```

### Start Chatting

```bash
koi run
```

## Commands

### CLI Commands

- `koi init` - Initialize .agent directory
- `koi run` - Start interactive session
- `koi run --task "..." --non-interactive` - Run specific task (for cron)
- `koi cron add "0 9 * * *" "Check emails"` - Schedule task
- `koi cron list` - List scheduled tasks
- `koi cron remove <id>` - Remove scheduled task
- `koi skills` - List available skills
- `koi config` - Show current configuration
- `koi memory` - Show current memory

### Chat Commands

During a `koi run` session:

- `/exit`, `/quit` - Exit the session
- `/memory` - Show current memory
- `/remember TEXT` - Add text to memory
- `/skills` - List available skills
- `/compact` - Force conversation compaction
- `/stats` - Show context usage statistics
- `/help` - Show help

## Architecture

### Core Components

- **Agent** (`agent.py`) - Main conversation loop and coordination
- **LLM Client** (`llm.py`) - OpenAI-compatible API client with streaming
- **Tools** (`tools.py`) - Function definitions and execution
- **Memory** (`memory.py`) - Persistent memory management
- **Skills** (`skills.py`) - Skill discovery and loading
- **Cron** (`cron.py`) - System crontab integration
- **Compaction** (`compaction.py`) - Context window management

### Available Tools

1. **read_file** - Read file contents (with offset/limit for large files)
2. **write_file** - Create or overwrite files
3. **edit_file** - Make surgical edits to files
4. **exec_command** - Execute shell commands (with safety checks)
5. **web_search** - Search the web (placeholder - TODO)
6. **web_fetch** - Fetch and convert web pages to markdown
7. **read_skill** - Load skill definitions by name

## Skills System

Skills are defined in `SKILL.md` files that the agent can discover and load:

```markdown
# Example Skill

This skill demonstrates how to do something useful.

## Usage

Explain how to use this skill...

## Examples

Show examples...
```

Skills are discovered from paths in `config.skills_paths` and listed in the system prompt. The agent can read full skill content using the `read_skill` tool when needed.

## Memory System

Koi maintains persistent memory across sessions:

- **MEMORY.md** - Long-term memory that persists between sessions
- **Context** - Current conversation context with automatic compaction
- **Memory Commands** - Use `/remember` to add important information

## Cron Integration

Schedule AI tasks to run automatically:

```bash
# Check emails every morning
koi cron add "0 9 * * *" "Check my emails and summarize any urgent ones"

# Weekly project status
koi cron add "0 17 * * 5" "Review this week's commits and create a status report"
```

Cron jobs run `koi run --task "..." --non-interactive` and log output to `.agent/cron-logs/`.

## Configuration

### Environment Variables

- `OPENAI_API_KEY` - API key (can also be set in config.json)

### Config Options

- `api_base` - API endpoint URL
- `api_key` - API key for authentication
- `model` - Model to use (e.g., "gpt-5.2", "gpt-4o", "claude-3-sonnet")
- `max_tokens` - Maximum tokens per response
- `context_window` - Total context window size
- `skills_paths` - Directories to search for skills
- `temperature` - Response randomness (0.0-1.0)

## Project Structure

```
koi/
├── src/koi/
│   ├── __init__.py
│   ├── __main__.py      # CLI entry point
│   ├── cli.py           # CLI command definitions
│   ├── agent.py         # Main agent loop
│   ├── llm.py           # LLM client
│   ├── tools.py         # Tool definitions
│   ├── memory.py        # Memory management
│   ├── skills.py        # Skills system
│   ├── cron.py          # Cron integration
│   ├── config.py        # Configuration
│   ├── compaction.py    # Context management
│   └── prompts.py       # System prompt building
├── tests/               # Test suite
├── pyproject.toml       # Project metadata
└── README.md           # This file
```

## Per-Project Structure

When you run `koi init` in a project:

```
your-project/
├── .agent/
│   ├── config.json      # Project-specific config
│   ├── MEMORY.md        # Persistent memory
│   ├── AGENTS.md        # Project instructions
│   ├── crontab.json     # Cron job metadata
│   └── cron-logs/       # Scheduled task logs
└── skills/              # Project-specific skills (optional)
```

## Safety

Koi includes safety features:

- **Command Validation** - Dangerous commands require manual confirmation
- **Sandboxing** - Operates within current working directory by default
- **Audit Trail** - All tool calls and results are logged in conversation
- **Memory Control** - You control what gets remembered

## Development

### Setup Development Environment

```bash
cd ~/koi
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

## Examples

### Basic Usage

```bash
$ koi run
🐠 Koi Agent - Ready to help!

koi> What files are in this directory?
🔧 exec_command...
This directory contains:
- README.md
- pyproject.toml  
- src/
- tests/

koi> Create a Python script that prints "Hello, World!"
🔧 write_file...
I've created hello.py with a simple "Hello, World!" script.

koi> /remember This project uses koi for automation
✅ Added to memory
```

### Scheduled Tasks

```bash
# Daily standup reminder
koi cron add "0 9 * * 1-5" "Review yesterday's work and plan today's tasks"

# Weekly cleanup
koi cron add "0 18 * * 5" "Clean up temporary files and organize the workspace"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Changelog

### v0.1.0 - Initial Release

- Core agent functionality
- Tool system with file operations and command execution
- Memory management
- Skills system
- Cron integration
- Context compaction
- Rich terminal UI

---

Built with ❤️ for developers who want AI assistance that stays out of the way until you need it.