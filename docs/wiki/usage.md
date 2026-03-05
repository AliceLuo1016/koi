# Usage Tracking

Koi tracks token usage per session and estimates costs based on model pricing.

## TokenUsage

The `TokenUsage` dataclass accumulates counts across LLM calls:

```python
@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    requests: int = 0
```

- `add(input, output, cache_read, cache_creation)` — accumulate from an API response
- `total_tokens` — property returning `input + output`
- `summary(model)` — formatted string with counts, cache stats, and estimated cost
- `to_dict()` / `from_dict()` — serialization for logging

## Cost Estimation

`estimate_cost(model, input, output, cache_read, cache_creation)` uses built-in pricing tables:

| Model | Input (per 1M) | Output (per 1M) | Cache Read | Cache Write |
|-------|----------------|------------------|------------|-------------|
| Claude Opus 4 | $15.00 | $75.00 | $1.50 | $18.75 |
| Claude Sonnet 4 | $3.00 | $15.00 | $0.30 | $3.75 |
| GPT-5.2 / GPT-4o | $2.50 | $10.00 | — | — |
| O3 | $10.00 | $40.00 | — | — |

Unknown models return $0.00.

## Usage Logging

`log_usage(koi_dir, session_id, model, usage)` appends a JSONL entry to `.koi/usage.log`:

```json
{"timestamp": "2026-03-04T16:00:00Z", "session_id": "abc123", "model": "claude-sonnet-4", "input_tokens": 5000, "output_tokens": 1200, "requests": 3, "estimated_cost_usd": 0.021}
```

## Usage History

`get_usage_history(koi_dir, hours=24)` reads the log and returns recent entries for the `/usage` command.

## Chat Commands

| Command | Description |
|---------|-------------|
| `/usage` | Show detailed token usage and estimated cost for current session |
| `/status` / `/stats` | Show summary including usage |

## LLM Client Integration

Usage is extracted from API responses in `LLMClient`:
- **Anthropic** — `response.usage.input_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`
- **Chat Completions** — `response.usage.prompt_tokens`, `completion_tokens` (with `stream_options` for streaming)
- **Responses API** — `response.usage.input_tokens`, `output_tokens`
- **Streaming fallback** — If stream doesn't provide usage, estimates via tiktoken

## Related Pages

- [LLM Client](llm-client.md) — Where usage is extracted from API responses
- [Configuration](config.md) — Model settings that affect pricing
