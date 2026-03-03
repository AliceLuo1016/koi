# Debug Transcript

Koi includes a lightweight JSONL transcript logger for post-mortem debugging. When enabled, every LLM message (sent and received) is recorded to `.koi/transcript.jsonl`.

## Enabling

Set `debug: true` in `.koi/config.json` or pass `--debug` on the CLI:

```bash
koi chat --debug
```

The `TranscriptLogger` is initialized in `Agent.__init__()`:

```python
self.transcript = TranscriptLogger(koi_dir, enabled=config.debug)
```

When disabled (`enabled=False`), all logging methods are no-ops.

## JSONL Format

Each line is a self-contained JSON object:

```json
{"timestamp": "2026-03-01T18:30:00.123456+00:00", "type": "session_start", "data": {"model": "claude-opus-4", "api_format": "anthropic", "system_prompt_hash": "a1b2c3d4e5f6g7h8"}}
{"timestamp": "2026-03-01T18:30:05.000000+00:00", "type": "user_message", "data": {"role": "user", "content": "read the config file"}}
{"timestamp": "2026-03-01T18:30:07.500000+00:00", "type": "assistant_message", "data": {"role": "assistant", "tool_calls": [...]}}
```

All timestamps are UTC ISO 8601.

## Event Types

| Event Type | When Logged | Data |
|------------|-------------|------|
| `session_start` | Agent initialization | `model`, `api_format`, `system_prompt_hash` |
| `user_message` | User sends a message | Full message dict |
| `assistant_message` | LLM responds | Full message dict (including `tool_calls`) |
| `tool_call` | Tool is called | Tool call details |
| `tool_result` | Tool returns | Tool result details |
| `compaction` | Context compacted | `before_message_count`, `after_message_count` |
| `pruning` | Context pruned | `before_chars`, `after_chars` |

## `TranscriptLogger` API

```python
class TranscriptLogger:
    def __init__(self, koi_dir: Path, enabled: bool = False)
    def log_event(self, event_type: str, data: Dict) -> None
    def log_session_start(self, metadata: Dict) -> None
    def log_message(self, event_type: str, message: Dict) -> None
    def log_compaction(self, before_count: int, after_count: int) -> None
    def log_pruning(self, before_chars: int, after_chars: int) -> None
    def close(self) -> None
```

All methods are no-ops when `enabled=False`.

### Core: `log_event()`

```python
def log_event(self, event_type, data):
    if not self._enabled:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "data": data,
    }
    self._file.write(json.dumps(record, default=str) + "\n")
    self._file.flush()
```

The file is opened in append mode and flushed after every write for crash resilience. `default=str` handles non-serializable objects (e.g., `Path`, `datetime`).

## File Location

```
.koi/
└── transcript.jsonl
```

The `.koi/` directory is created if it doesn't exist. The file grows unboundedly — no rotation is implemented. Users should clean it up manually or use standard log rotation.

## Usage in Agent

The `Agent` class logs at two points:

1. **Session start** — in `__init__()`:
   ```python
   self.transcript.log_session_start({
       "model": config.model,
       "api_format": config.api_format,
       "system_prompt_hash": hashlib.sha256(
           self.system_prompt.encode()
       ).hexdigest()[:16],
   })
   ```

2. **User messages** — in `run_interactive()`:
   ```python
   self.transcript.log_message("user_message", user_msg)
   ```

The system prompt itself is not logged — only a SHA-256 hash prefix (16 hex chars) for identification without bloating the transcript.

## Analyzing Transcripts

Since the file is JSONL, it's easy to work with standard tools:

```bash
# Count events by type
jq -r '.type' .koi/transcript.jsonl | sort | uniq -c

# View all user messages
jq 'select(.type == "user_message") | .data.content' .koi/transcript.jsonl

# View compaction events
jq 'select(.type == "compaction")' .koi/transcript.jsonl

# Pretty-print the last 5 events
tail -5 .koi/transcript.jsonl | jq .
```

## Related Pages

- [Configuration](config.md) — `debug` config field
- [Context Management](context-management.md) — Compaction and pruning events
- [Architecture Overview](architecture.md) — Agent initialization
