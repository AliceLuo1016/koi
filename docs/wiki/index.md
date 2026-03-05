# Koi Wiki 🐠

Koi is a terminal-based AI agent with persistent memory, tool calling, extensible skills, sandbox security, and multi-provider LLM support. Built for developers who want a powerful AI assistant in their terminal.

## Architecture at a Glance

```
CLI (cli.py)
  └── Agent (agent.py) — conversation loop, commands, streaming display
        ├── LLMClient (llm.py) — 3 API formats, thinking, caching, streaming
        ├── ToolExecutor (tools.py) — 20+ built-in tools
        ├── SkillsManager (skills.py) — markdown-based extensible skills
        ├── SubagentManager (subagent.py) — parallel child agents + ACP agents
        ├── SessionManager (session_manager.py) — persistent sessions with branching
        ├── ContextCompactor (compaction.py) — LLM-based summarization
        ├── Sandbox (sandbox.py) — file/command security
        └── TranscriptLogger (transcript.py) — debug logging
```

## Wiki Pages

### Core
- [Architecture Overview](architecture.md) — Layers, agent loop, message flow
- [LLM Client](llm-client.md) — Multi-provider streaming, thinking, caching
- [Streaming Protocol](streaming.md) — How token streaming works across providers

### Features
- [Tool System](tools.md) — Built-in tools, execution, result formatting
- [Skills System](skills.md) — Markdown-based extensible capabilities
- [Sub-Agents](subagents.md) — Spawning parallel child agents and ACP agents
- [Session Management](sessions.md) — Persistent sessions with in-place branching
- [Usage Tracking](usage.md) — Token counting, cost estimation, usage logs
- [Context Management](context-management.md) — 4-layer context budget system
- [Cron Integration](cron.md) — Scheduling AI tasks

### Operations
- [Configuration](config.md) — All config fields and environment variables
- [Sandbox Security](sandbox.md) — File access, env scrubbing, command filtering
- [Debug Transcript](transcript.md) — JSONL logging for debugging

## Codebase Stats

| Component | File | Lines |
|-----------|------|-------|
| LLM Client | `llm.py` | ~1,340 |
| Tools | `tools.py` | ~1,100 |
| CLI | `cli.py` | ~990 |
| Agent | `agent.py` | ~950 |
| Sub-Agents | `subagent.py` | ~550 |
| Session Manager | `session_manager.py` | ~345 |
| Prompts | `prompts.py` | ~320 |
| ACP Client | `acp_client.py` | ~275 |
| Cron | `cron.py` | ~245 |
| Context Pruning | `context_pruning.py` | ~230 |
| Config | `config.py` | ~230 |
| Errors | `errors.py` | ~225 |
| Usage Tracking | `usage.py` | ~215 |
| Compaction | `compaction.py` | ~190 |
| Context Guard | `context_guard.py` | ~170 |
| Skills | `skills.py` | ~125 |
| Sandbox | `sandbox.py` | ~120 |
| Sessions | `sessions.py` | ~115 |
| ACP Registry | `acp_registry.py` | ~90 |
| Transcript | `transcript.py` | ~70 |
| Stream Events | `stream_events.py` | ~35 |
| **Total** | | **~8,100** |
