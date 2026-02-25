# Koi Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                             │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ Terminal UI │  │ Chat Commands│  │ System Cron Integration│ │
│  │  (rich)     │  │  (/help etc) │  │ (koi cron commands)    │ │
│  └──────┬──────┘  └──────┬───────┘  └───────────┬────────────┘ │
└─────────┼────────────────┼──────────────────────┼──────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   cli.py    │  │ Command      │  │   Cron Manager         │ │
│  │ (click)     │──│ Routing      │──│   (cron.py)            │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Agent Core                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  agent.py   │  │ Conversation │  │  Context Compaction    │ │
│  │            ←┼──│ Loop         │──│  (compaction.py)       │ │
│  └──────┬──────┘  └──────────────┘  └────────────────────────┘ │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────────────────────────────┐                       │
│  │        LLM Client (llm.py)          │                       │
│  │  ┌─────────────┐ ┌────────────────┐│                       │
│  │  │  Responses  │ │   Anthropic    ││                       │
│  │  │   Format    │ │    Format      ││                       │
│  │  └─────────────┘ └────────────────┘│                       │
│  └─────────────────────────────────────┘                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Tool & Skills Layer                           │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   tools.py  │  │   Tool       │  │   Skills Manager       │ │
│  │            ←┼──│  Executor    ├──│   (skills.py)          │ │
│  └─────────────┘  └──────┬───────┘  └────────────────────────┘ │
│                           │                                      │
│  ┌────────────────────────┼────────────────────────────────┐   │
│  │                 Sandbox Security                         │   │
│  │  ┌─────────────┐ ┌────┴───────┐ ┌────────────────────┐│   │
│  │  │ File Access │ │ Environment│ │ Command Filtering  ││   │
│  │  │  Control    │ │  Scrubbing │ │ & Confirmation     ││   │
│  │  └─────────────┘ └────────────┘ └────────────────────┘│   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Storage Layer                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Memory     │  │    Config    │  │      Project Files     │ │
│  │ (MEMORY.md) │  │ (config.json)│  │  (.koi/ directory)     │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### User Interface Layer
- **Terminal UI**: Rich-based interface with markdown rendering
- **Chat Commands**: Built-in commands like /help, /memory, /skills
- **Cron Integration**: System crontab management for scheduled tasks

### CLI Layer
- **cli.py**: Click-based command-line interface
- **Command Routing**: Maps CLI commands to agent actions
- **Cron Manager**: Handles crontab entries and job scheduling

### Agent Core
- **agent.py**: Main conversation loop with async operations
- **LLM Client**: Supports multiple providers (OpenAI-compatible, Anthropic)
- **Context Compaction**: Manages conversation history within token limits

### Tool & Skills Layer
- **Tool Executor**: Runs built-in tools with sandbox protection
- **Skills Manager**: Discovers and loads markdown-based skills
- **Sandbox Security**: Three-layer protection for safe execution

### Storage Layer
- **Memory**: Persistent storage in MEMORY.md
- **Config**: API settings and preferences in config.json
- **Project Files**: All koi data in .koi/ directory

## Data Flow

1. **User Input** → CLI parses command
2. **CLI** → Agent receives message
3. **Agent** → Builds context with memory & skills
4. **Agent** → Sends to LLM with tools
5. **LLM** → Returns response with tool calls
6. **Tool Executor** → Runs tools with sandbox checks
7. **Results** → Back to LLM for processing
8. **Final Response** → Displayed to user
9. **Memory Updates** → Persisted to disk

## Security Model

```
┌─────────────────────────────────────┐
│         User Request                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│     Sandbox Security Check          │
│  ┌─────────────────────────────┐   │
│  │ 1. File Access Control      │   │
│  │    - Allowed paths          │   │
│  │    - Blocked paths          │   │
│  │    - Read-only paths        │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ 2. Environment Scrubbing    │   │
│  │    - Allowlist only         │   │
│  │    - Strip credentials      │   │
│  │    - Add from .koi/creds    │   │
│  └─────────────────────────────┘   │
│  ┌─────────────────────────────┐   │
│  │ 3. Command Filtering        │   │
│  │    - Block patterns         │   │
│  │    - Confirm patterns       │   │
│  │    - User approval          │   │
│  └─────────────────────────────┘   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│      Execute if Approved            │
└─────────────────────────────────────┘
```

## Extension Points

### Adding Tools
1. Define in `get_tool_definitions()`
2. Implement in `ToolExecutor.execute()`
3. Add sandbox checks if needed

### Adding Providers
1. Update `Config` for auto-detection
2. Extend `LLMClient` for API format
3. Add response parsing logic

### Custom Skills
1. Create `.koi/skills/name/SKILL.md`
2. Follow markdown structure
3. Test with skill-creator

## Performance Considerations

- **Async Operations**: All I/O is async for responsiveness
- **Streaming Responses**: LLM responses stream to user
- **Context Management**: Automatic compaction prevents overflows
- **Caching**: Skills are loaded once per session

## Error Handling

- **Graceful Degradation**: Tools fail individually
- **Retry Logic**: Exponential backoff for API calls
- **User Feedback**: Clear error messages
- **Interrupt Support**: Ctrl+C cancels operations

## Future Architecture

Planned improvements:
- Plugin system for dynamic tool loading
- Multi-agent coordination
- Distributed memory storage
- Web API for remote access